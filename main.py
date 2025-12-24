import os
import logging
import requests
import google.generativeai as genai
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse

# --- CONFIGURATION ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Security: Only allow your phone number
ALLOWED_NUMBER = os.environ.get('ALLOWED_NUMBER')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logging.error("GEMINI_API_KEY is missing!")

# --- HELPER FUNCTIONS ---

def get_coordinates(place_name):
    """Uses OpenStreetMap (Nominatim) to find lat/long. Free, no key required."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': place_name, 'format': 'json', 'limit': 1}
        # User-agent is required by Nominatim policy
        headers = {'User-Agent': 'BackcountryBot/1.0'}
        response = requests.get(url, params=params, headers=headers).json()
        if response:
            return float(response[0]['lat']), float(response[0]['lon']), response[0]['display_name']
        return None, None, None
    except Exception as e:
        logging.error(f"Geocoding error: {e}")
        return None, None, None

def get_weather_data(lat, lon):
    """Fetches weather from Open-Meteo (Free, no key)."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m",
            "hourly": "temperature_2m,precipitation_probability",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "forecast_days": 1
        }
        r = requests.get(url, params=params).json()

        curr = r['current']
        # Simple string formatting for SMS
        report = (
            f"NOW: {curr['temperature_2m']}F, Wind {curr['wind_speed_10m']}mph. "
            f"Precip: {curr['precipitation']}in. "
        )
        return report
    except Exception as e:
        logging.error(f"Weather API error: {e}")
        return "Error fetching weather data."

def ask_gemini(prompt):
    """Sends prompt to Gemini with a 'brief' system instruction."""
    try:
        # System prompt instructions are prepended to ensure brevity
        full_prompt = (
            "You are a backcountry assistant. Answer purely in text (no markdown). "
            "Be concise, prioritizing safety and facts. "
            "Keep answers under 300 characters if possible. "
            f"User Query: {prompt}"
        )
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Gemini is currently unreachable."

# --- MAIN ROUTE ---

@app.route('/sms', methods=['POST'])
def sms_reply():
    """Handles incoming SMS from Twilio."""

    # 1. Security Check
    sender = request.values.get('From')
    if sender != ALLOWED_NUMBER:
        logging.warning(f"Blocked unauthorized access from {sender}")
        return abort(403)

    # 2. Parse Incoming Message
    incoming_msg = request.values.get('Body', '').strip()
    incoming_msg_lower = incoming_msg.lower()

    resp = MessagingResponse()
    reply_text = ""

    # 3. Routing Logic

    # CASE A: Weather Command (e.g., "Wx Rainier" or "Weather Moab")
    if incoming_msg_lower.startswith(('wx', 'weather')):
        # Extract location (remove first word)
        location_query = ' '.join(incoming_msg.split()[1:])
        if not location_query:
            reply_text = "Please specify a location (e.g., 'Wx Mt Rainier')"
        else:
            lat, lon, name = get_coordinates(location_query)
            if lat:
                wx_data = get_weather_data(lat, lon)
                reply_text = f"@{name[:20]}: {wx_data}"
            else:
                reply_text = f"Could not find location: {location_query}"

    # CASE B: Preset Commands
    elif incoming_msg_lower == 'ping':
        reply_text = "Pong! System online. 🏔️"

    elif incoming_msg_lower == 'checkin':
        # Placeholder for logging logic (e.g., append to Google Sheet)
        reply_text = "Check-in logged (simulation)."

    # CASE C: Default to Gemini AI
    else:
        reply_text = ask_gemini(incoming_msg)

    # 4. Smart Segmentation (Handling Long Responses)
    # Twilio handles standard splitting, but we can force breaks for clarity
    # if the AI writes a wall of text.

    chunk_size = 300 # conservative safe size for satellite links

    if len(reply_text) > chunk_size:
        # Split into chunks of 300 chars
        chunks = [reply_text[i:i+chunk_size] for i in range(0, len(reply_text), chunk_size)]
        for i, chunk in enumerate(chunks):
            # Add (1/3) type counters for clarity
            msg_body = f"({i+1}/{len(chunks)}) {chunk}"
            resp.message(msg_body)
    else:
        resp.message(reply_text)

    return str(resp)

# --- PRODUCTION SERVER CONFIG ---
if __name__ == "__main__":
    # Cloud Run injects the PORT environment variable.
    # Default to 8080 for local testing.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)