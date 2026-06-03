import os
import pytz
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import tweepy
from openai import OpenAI
from supabase import create_client, Client

# --- SECURELY LOAD ENVIRONMENT VARIABLES ---
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
    print("🔄 Fetching market data from yfinance...")
    tickers = ["USDIDR=X", "SGDIDR=X", "MYRIDR=X"]
    
    data = yf.download(tickers, period="1d", interval="1m", progress=False)
    if data.empty:
        print("❌ Error: No data received from yfinance.")
        return
        
    last_hour_data = data['Close'].tail(60)
    latest_usd = float(last_hour_data['USDIDR=X'].iloc[-1])
    latest_sgd = float(last_hour_data['SGDIDR=X'].iloc[-1])
    latest_myr = float(last_hour_data['MYRIDR=X'].iloc[-1])
    
    current_timestamp = int(last_hour_data.index[-1].timestamp())

    # Check for duplicates
    if check_if_already_posted(current_timestamp):
        print(f"⏳ Data for timestamp {current_timestamp} already posted. Skipping.")
        return

    # Track ATH and previous rates
    ath_broken = check_and_update_ath(latest_usd, latest_sgd, latest_myr)
    prev_rates = load_previous_rates()

    # Generate Intro & Chart
    dynamic_intro = generate_ai_intro(ath_broken)
    image_path = "rupiah_hourly.png"
    generate_chart(last_hour_data, image_path)

    # Format and Send Tweet
    tweet_text = format_tweet(dynamic_intro, latest_usd, latest_sgd, latest_myr, prev_rates, current_timestamp)
    print(f"📝 Prepared Tweet:\n\n{tweet_text}\n")
    
    tweet_success = post_to_x(tweet_text, image_path)
    save_current_rates(current_timestamp, latest_usd, latest_sgd, latest_myr, tweet_success)
    
    if os.path.exists(image_path):
        os.remove(image_path)

# --- HELPER FUNCTIONS ---

def check_if_already_posted(timestamp: int) -> bool:
    try:
        response = supabase.table("currency").select("is_posted").eq("timestamp", timestamp).execute()
        return response.data[0].get("is_posted", False) if response.data else False
    except Exception:
        return False

def check_and_update_ath(usd: float, sgd: float, myr: float) -> list:
    ath_broken = []
    try:
        response = supabase.table("ath_records").select("*").eq("id", 1).execute()
        if not response.data: return ath_broken
            
        ath_data = response.data[0]
        updates = {}

        if usd > ath_data.get("usd", 0.0):
            ath_broken.append("USD"); updates["usd"] = usd
        if sgd > ath_data.get("sgd", 0.0):
            ath_broken.append("SGD"); updates["sgd"] = sgd
        if myr > ath_data.get("myr", 0.0):
            ath_broken.append("MYR"); updates["myr"] = myr

        if updates:
            supabase.table("ath_records").update(updates).eq("id", 1).execute()
            print(f"🚀 New ATH recorded in database for: {', '.join(ath_broken)}")
    except Exception as e:
        print(f"⚠️ Error tracking ATH: {e}")
    return ath_broken

def load_previous_rates() -> dict:
    try:
        response = supabase.table("currency").select("*").order("timestamp", desc=True).limit(1).execute()
        if response.data:
            return {"usd": response.data[0].get("usd"), "sgd": response.data[0].get("sgd"), "myr": response.data[0].get("myr")}
    except Exception:
        pass
    return {"usd": None, "sgd": None, "myr": None}

def generate_ai_intro(ath_broken: list) -> str:
    if not OPENROUTER_API_KEY: return ""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    ai_context = f"Alert! The following currencies just hit an all-time high against the Rupiah: {', '.join(ath_broken)}!" if ath_broken else "Just another routine hourly market update. Normal market movements."
    prompt = f"You are an expert currency bot on X (Twitter) tracking the Indonesian Rupiah. Context: {ai_context}\nWrite a very short, engaging 1-sentence opening announcement for the hourly update tweet. If an All-Time High (ATH) was broken, sound the alarm and make it exciting! If not, keep it professional and snappy. Do NOT include the actual prices or hashtags in your sentence, just set the mood."
    
    try:
        response = client.chat.completions.create(model="openrouter/auto", messages=[{"role": "user", "content": prompt}], max_tokens=50)
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

def generate_chart(df, output_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['USDIDR=X'], label="USD/IDR", color="#1f77b4")
    plt.plot(df.index, df['SGDIDR=X'], label="SGD/IDR", color="#ff7f0e")
    plt.plot(df.index, df['MYRIDR=X'], label="MYR/IDR", color="#2ca02c")
    plt.title("Rupiah Movement (Last Hour)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (UTC)", fontsize=10)
    plt.ylabel("Rupiah (Rp)", fontsize=10)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def get_change_text(new_rate: float, old_rate: float) -> str:
    if not old_rate or not new_rate: return ""
    change = new_rate - old_rate
    return f" (+{change:,.2f})" if change > 0 else f" ({change:,.2f})" if change < 0 else ""

def format_tweet(intro: str, usd: float, sgd: float, myr: float, prev: dict, timestamp: int) -> str:
    tz = pytz.timezone('Asia/Jakarta')
    dt = datetime.fromtimestamp(timestamp, tz)
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    
    date_str = f"{days[dt.weekday()]}, {dt.day} {months[dt.month-1]} {dt.year}"
    time_str = dt.strftime("%H:%M")

    usd_change = get_change_text(usd, prev.get("usd"))
    sgd_change = get_change_text(sgd, prev.get("sgd"))
    myr_change = get_change_text(myr, prev.get("myr"))

    intro_block = f"{intro}\n\n" if intro else ""
    return (f"{intro_block}Kurs Rupiah hari {date_str} jam {time_str} WIB:\n"
            f"- USD/IDR: Rp{usd:,.2f}{usd_change}\n"
            f"- SGD/IDR: Rp{sgd:,.2f}{sgd_change}\n"
            f"- MYR/IDR: Rp{myr:,.2f}{myr_change}")

def post_to_x(text: str, image_path: str) -> bool:
    try:
        auth = tweepy.OAuth1UserHandler(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET)
        api_v1 = tweepy.API(auth)
        media = api_v1.media_upload(image_path)
        
        client_v2 = tweepy.Client(consumer_key=X_API_KEY, consumer_secret=X_API_SECRET, access_token=X_ACCESS_TOKEN, access_token_secret=X_ACCESS_SECRET)
        response = client_v2.create_tweet(text=text, media_ids=[media.media_id])
        print(f"✅ Tweet successfully posted! Tweet ID: {response.data['id']}")
        return True
    except Exception as e:
        print(f"❌ Error posting to X: {e}")
        return False

def save_current_rates(timestamp: int, usd: float, sgd: float, myr: float, is_posted: bool):
    try:
        data = {"timestamp": timestamp, "usd": usd, "sgd": sgd, "myr": myr, "is_posted": is_posted}
        supabase.table("currency").upsert(data, on_conflict="timestamp").execute()
    except Exception as e:
        print(f"⚠️ Error saving state to Supabase: {e}")

if __name__ == "__main__":
    main()