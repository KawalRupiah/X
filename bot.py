import os
import math
import pytz
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tweepy
from openai import OpenAI
from supabase import create_client, Client

# --- SECURELY LOAD ENVIRONMENT VARIABLES ---
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
X_API_KEY = os.environ.get("X_API_KEY")
X_API_SECRET = os.environ.get("X_API_SECRET")
X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN")
X_ACCESS_SECRET = os.environ.get("X_ACCESS_SECRET")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def main():
    print("🚀 Initiating hourly execution checks...")
    
    # 1. Short-circuit if market is closed (Weekend Check)
    if is_market_closed():
        print("💤 Market is currently closed for the weekend (Sabtu 04:00 - Senin 04:00 WIB). Exiting safely.")
        return

    # 2. Fetch live market data via Twelve Data
    try:
        data_df = fetch_market_data_hourly()
        print('head', data_df.head())
        print('tail', data_df.tail())
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # 3. Extract the latest rates and previous hour rates safely
    latest_usd = normalize_rate(data_df['USDIDR=X'].iloc[-1])
    latest_sgd = normalize_rate(data_df['SGDIDR=X'].iloc[-1])
    latest_myr = normalize_rate(data_df['MYRIDR=X'].iloc[-1])

    if not all(value is not None for value in (latest_usd, latest_sgd, latest_myr)):
        print("❌ Error: Invalid numeric values received. Skipping this run.")
        return
    
    # Get previous hour data (1 hour = 12 candles of 5m intervals)
    # With 5m data: -1 is current hour, -13 is 1 hour ago
    prev_usd = normalize_rate(data_df['USDIDR=X'].iloc[-13])
    prev_sgd = normalize_rate(data_df['SGDIDR=X'].iloc[-13])
    prev_myr = normalize_rate(data_df['MYRIDR=X'].iloc[-13])
    
    # Get the datetime from the dataframe (already in Asia/Jakarta from API)
    current_dt = data_df.index[-1]
    tz = pytz.timezone('Asia/Jakarta')
    if current_dt.tzinfo is None:
        # Naive datetime from API is already in Asia/Jakarta, localize it
        current_dt = tz.localize(current_dt)

    # 4. Track All-Time Highs (ATH) from 24-hour data
    ath_broken = check_and_update_ath(data_df)

    # 5. Generate Dynamic AI Intro & Grafana-Style Charts
    dynamic_intro = generate_ai_intro(ath_broken)
    image_paths = generate_chart(data_df, None)  # Returns list of 3 image paths

    # 6. Format Tweet Body
    tweet_text = format_tweet(dynamic_intro, latest_usd, latest_sgd, latest_myr, prev_usd, prev_sgd, prev_myr, current_dt)
    print(f"📝 Prepared Tweet Content:\n\n{tweet_text}\n")
    
    # 7. Post To X
    post_to_x(tweet_text, image_paths)
    
    # Cleanup local images
    for img_path in image_paths:
        if os.path.exists(img_path):
            os.remove(img_path)


# ==========================================
# DATA FETCHING & CHARTING
# ==========================================

def fetch_market_data_hourly() -> pd.DataFrame:
    """Fetches hourly market data using start_date and end_date.
    Gets last 24 hours for ATH comparison and change calculation."""
    print("🔄 Fetching hourly market data from Twelve Data...")
    
    if not TWELVEDATA_API_KEY:
        raise ValueError("TWELVEDATA_API_KEY is missing from environment variables!")

    # Calculate date range: 24 hours ago to now (both at minute 00)
    tz = pytz.timezone('Asia/Jakarta')
    end_time = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - pd.Timedelta(hours=24)
    
    # Format dates for API (ISO 8601 with timezone info)
    start_date = start_time.strftime('%Y-%m-%dT%H:%M:%S')
    end_date = end_time.strftime('%Y-%m-%dT%H:%M:%S')

    symbols = "USD/IDR,SGD/IDR,MYR/IDR"
    url = f"https://api.twelvedata.com/time_series?symbol={symbols}&interval=5min&apikey={TWELVEDATA_API_KEY}&start_date={start_date}&end_date={end_date}&timezone=Asia/Jakarta"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"HTTP Error: {response.status_code}")
        
    data = response.json()
    if 'status' in data and data['status'] == 'error':
        raise ValueError(f"Twelve Data Error: {data['message']}")

    df_list = []
    for symbol in ["USD/IDR", "SGD/IDR", "MYR/IDR"]:
        if symbol not in data or 'values' not in data[symbol]:
            continue
            
        temp_df = pd.DataFrame(data[symbol]['values'])
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
        temp_df.set_index('datetime', inplace=True)
        temp_df = temp_df[['close']].astype(float)
        
        # Rename column for our standard formatting
        clean_col_name = symbol.replace("/", "") + "=X"
        temp_df.rename(columns={'close': clean_col_name}, inplace=True)
        df_list.append(temp_df)

    if not df_list:
        raise ValueError("No valid data processed from API.")

    combined_df = pd.concat(df_list, axis=1)
    combined_df.sort_index(inplace=True)

    # ffill/bfill prevents any gaps from ruining the graph
    clean_data = combined_df.ffill().bfill()
    return clean_data

def generate_chart(df, output_path: str):
    """Generates three separate Grafana-style charts, one for each currency. Returns list of image paths."""
    print("📊 Generating three separate currency charts...")
    
    # Grafana Color Palette
    bg_color = '#161719'        
    grid_color = '#2c3235'      
    text_color = '#c7d0d9'      
    color_usd = '#5794f2'       
    color_sgd = '#ff780a'       
    color_myr = '#73bf69'       

    # Get previous and current rates from the DataFrame
    # With 5m intervals: -1 is current hour, -13 is 1 hour ago
    prev_rates = {
        'usd': normalize_rate(df['USDIDR=X'].iloc[-13]),
        'sgd': normalize_rate(df['SGDIDR=X'].iloc[-13]),
        'myr': normalize_rate(df['MYRIDR=X'].iloc[-13])
    }
    current_rates = {
        'usd': normalize_rate(df['USDIDR=X'].iloc[-1]),
        'sgd': normalize_rate(df['SGDIDR=X'].iloc[-1]),
        'myr': normalize_rate(df['MYRIDR=X'].iloc[-1])
    }

    currencies = [
        ('USDIDR=X', 'USD/IDR', color_usd, 'usd', 'rupiah_usd.png'),
        ('SGDIDR=X', 'SGD/IDR', color_sgd, 'sgd', 'rupiah_sgd.png'),
        ('MYRIDR=X', 'MYR/IDR', color_myr, 'myr', 'rupiah_myr.png')
    ]

    # Convert index to WIB (UTC+7)
    df_wib = df.copy()
    df_wib.index = df_wib.index.tz_localize('UTC').tz_convert('Asia/Jakarta')

    image_paths = []
    
    for col, label, color, key, img_filename in currencies:
        # Create individual figure for each currency
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Plot the line with WIB timezone
        ax.plot(df_wib.index, df[col], label=label, color=color, linewidth=2)
        ax.fill_between(df_wib.index, df[col], alpha=0.1, color=color)

        # Draw dashed reference line for previous value
        if prev_rates[key]:
            ax.axhline(y=prev_rates[key], color=color, linestyle='--', alpha=0.3, linewidth=1.5, label='Sebelumnya')

        # Auto-scale y-axis individually for each currency
        ymin, ymax = df[col].min(), df[col].max()
        padding = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - padding, ymax + padding)

        # Build title with change
        change = current_rates[key] - prev_rates[key] if prev_rates[key] else 0
        title = f"{label}\n{change:+.0f}" if prev_rates[key] else label
        ax.set_title(title, color=text_color, fontsize=13, fontweight='bold', pad=10)

        ax.set_xlabel("Waktu (WIB)", color=text_color, fontsize=10, labelpad=8)
        ax.set_ylabel("Kurs (Rp)", color=text_color, fontsize=10, labelpad=8)
        ax.tick_params(colors=text_color, labelsize=9)
        
        # Format x-axis labels in HH:MM format and rotate for readability
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)

        if prev_rates[key]:
            ax.legend(loc="upper left", frameon=True, facecolor=bg_color, edgecolor=grid_color, fontsize=9, labelcolor=text_color)

        plt.tight_layout()
        plt.savefig(img_filename, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close()
        image_paths.append(img_filename)
    
    return image_paths


# ==========================================
# HELPER & FORMATTING FUNCTIONS
# ==========================================

def is_market_closed() -> bool:
    tz = pytz.timezone('Asia/Jakarta')
    now = datetime.now(tz)
    day, hour = now.weekday(), now.hour
    if day == 5 and hour >= 4: return True
    if day == 6: return True
    if day == 0 and hour < 4: return True
    return False

def is_valid_rate(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)

def format_idr_number(num) -> str:
    """Formats an integer to Indonesian currency style (using dots)"""
    if not is_valid_rate(num): return "nan"
    return f"{int(num):,}".replace(",", ".")

def normalize_rate(value):
    try:
        rate = float(value)
        if not math.isfinite(rate): return None
        return int(round(rate))
    except (TypeError, ValueError):
        return None

def get_change_text(new_rate: int, old_rate: int) -> str:
    if not old_rate or not new_rate: return ""
    change = int(new_rate - old_rate)
    if change > 0: return f" (+{format_idr_number(change)})"
    elif change < 0: return f" ({format_idr_number(change)})" 
    return ""

def format_tweet(intro: str, usd: int, sgd: int, myr: int, prev_usd: int, prev_sgd: int, prev_myr: int, dt) -> str:
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    
    date_str = f"{days[dt.weekday()]}, {dt.day} {months[dt.month-1]} {dt.year}"
    time_str = dt.strftime("%H:%M")

    usd_change = get_change_text(usd, prev_usd)
    sgd_change = get_change_text(sgd, prev_sgd)
    myr_change = get_change_text(myr, prev_myr)

    intro_block = f"{intro}\n\n" if intro else ""
    return (f"{intro_block}Kurs Rupiah hari {date_str} jam {time_str} WIB:\n"
            f"- USD/IDR: Rp{format_idr_number(usd)}{usd_change}\n"
            f"- SGD/IDR: Rp{format_idr_number(sgd)}{sgd_change}\n"
            f"- MYR/IDR: Rp{format_idr_number(myr)}{myr_change}")


# ==========================================
# EXTERNAL APIS & SUPABASE
# ==========================================



def check_and_update_ath(df: pd.DataFrame) -> list:
    """Check if any currencies in the data hit new all-time highs.
    Updates only the existing ath_records row and includes updated_at timestamp."""
    ath_broken = []
    try:
        # Find maximum values from the data (convert to int)
        max_usd = int(round(df['USDIDR=X'].max()))
        max_sgd = int(round(df['SGDIDR=X'].max()))
        max_myr = int(round(df['MYRIDR=X'].max()))

        # Get existing ATH records
        response = supabase.table("ath_records").select("*").eq("id", 1).execute()
        
        # Initialize or get existing ATH data
        if response.data:
            ath_data = response.data[0]
        else:
            ath_data = {"id": 1, "usd": 0, "sgd": 0, "myr": 0}
        
        # Prepare updates with current timestamp
        now = datetime.utcnow().isoformat() + 'Z'
        updates = {"id": 1, "updated_at": now}

        # Check each currency against stored ATH
        if max_usd > ath_data.get("usd", 0):
            ath_broken.append(f"USD ({max_usd})")
            updates["usd"] = max_usd
        else:
            updates["usd"] = ath_data.get("usd", max_usd)
            
        if max_sgd > ath_data.get("sgd", 0):
            ath_broken.append(f"SGD ({max_sgd})")
            updates["sgd"] = max_sgd
        else:
            updates["sgd"] = ath_data.get("sgd", max_sgd)
            
        if max_myr > ath_data.get("myr", 0):
            ath_broken.append(f"MYR ({max_myr})")
            updates["myr"] = max_myr
        else:
            updates["myr"] = ath_data.get("myr", max_myr)

        # Use upsert to create record if it doesn't exist, or update if it does
        supabase.table("ath_records").upsert(updates, on_conflict="id").execute()
        if ath_broken:
            print(f"🚀 New ATH(s) found for: {', '.join(ath_broken)}")
    except Exception as e:
        print(f"⚠️ Error tracking ATH: {e}")
    return ath_broken



def generate_ai_intro(ath_broken: list) -> str:
    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY not set. Skipping AI intro generation.")
        return ""
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    ai_context = f"Peringatan! Mata uang berikut baru saja mencapai harga tertinggi sepanjang masa terhadap Rupiah: {', '.join(ath_broken)}!" if ath_broken else "Hanya update pasar yang rutin setiap jam. Pergerakan pasar normal."
    prompt = f"Anda adalah expert currency bot di X (Twitter) yang melacak Rupiah Indonesia. Konteks: {ai_context}\nTulis pembukaan singkat dan menarik dalam 1 kalimat bahasa Indonesia untuk tweet update kurs setiap jam. Jika ada All-Time High (ATH) yang terlampaui, bunyi alarm dan buat suasana yang menegangkan! Jika tidak, tetap profesional dan singkat. JANGAN sertakan harga sebenarnya atau hashtag dalam kalimat Anda, hanya set mood-nya."
    
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        if response and response.choices and len(response.choices) > 0:
            intro = response.choices[0].message.content.strip()
            print(f"✨ AI Intro Generated: {intro}")
            return intro
        return ""
    except Exception as e:
        print(f"⚠️ Error generating AI intro: {e}")
        return ""

def post_to_x(text: str, image_paths) -> bool:
    try:
        auth = tweepy.OAuth1UserHandler(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET)
        api_v1 = tweepy.API(auth)
        
        # Handle both single image (string) and multiple images (list)
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # Upload all media
        media_ids = []
        for img_path in image_paths:
            media = api_v1.media_upload(img_path)
            media_ids.append(media.media_id)
        
        client_v2 = tweepy.Client(consumer_key=X_API_KEY, consumer_secret=X_API_SECRET, access_token=X_ACCESS_TOKEN, access_token_secret=X_ACCESS_SECRET)
        response = client_v2.create_tweet(text=text, media_ids=media_ids)
        print(f"✅ Tweet successfully posted! Tweet ID: {response.data['id']}")
        return True
    except Exception as e:
        print(f"❌ Error posting to X: {e}")
        return False

if __name__ == "__main__":
    main()