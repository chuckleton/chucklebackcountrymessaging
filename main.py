import os
import time
import logging
import requests
import google.generativeai as genai
import google.cloud.logging
from datetime import datetime, timezone
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse

# --- GLOBAL STATE ---
# This variable survives between requests if the container stays warm.
# If it's None, we know this is a "Cold Start".
is_warm = False

# --- CONFIGURATION ---
app = Flask(__name__)

# --- LOGGING SETUP (The New Part) ---
# This checks if we are running in Cloud Run (vs local) to avoid local errors
if os.environ.get("K_SERVICE"):
    log_client = google.cloud.logging.Client()
    log_client.setup_logging()
else:
    # Fallback for local testing
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

def log_structured(message, **kwargs):
    """
    Helper to send a structured JSON log entry.
    Usage: log_structured("Weather checked", location="Rainier", user="...555")
    """
    payload = {"message": message}

    # Add the application-level timestamp (UTC)
    # format: YYYY-MM-DDTHH:MM:SS.mmmmm+00:00
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    payload.update(kwargs)
    # Using 'extra' passes this dictionary to the jsonPayload field in GCP
    logger.info(message, extra={"json_fields": payload})

# --- TIMER HELPER ---
class ExecutionTimer:
    def __init__(self):
        self.start = time.time()

    def stop(self):
        return int((time.time() - self.start) * 1000) # Returns ms

# Security: Only allow your phone number
ALLOWED_NUMBER = os.environ.get('ALLOWED_NUMBER')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.error("GEMINI_API_KEY is missing!")

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
        logger.error(f"Geocoding error: {e}")
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
        logger.error(f"Weather API error: {e}")
        return "Error fetching weather data."

def ask_gemini(prompt):
    """Sends prompt to Gemini with a 'brief' system instruction."""
    timer = ExecutionTimer()
    try:
        # System prompt instructions are prepended to ensure brevity
        full_prompt = (
            "You are a backcountry assistant. Answer purely in text (no markdown). "
            "Be concise, prioritizing safety and facts. "
            "Keep answers under 300 characters if possible. "
            f"User Query: {prompt}"
        )
        response = model.generate_content(full_prompt)
        duration = timer.stop()

        # Extract token usage if available (Gemini API provides usage_metadata)
        # Note: Check API docs for exact field structure as it updates often
        t_in = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        t_out = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        log_structured("Gemini Call",
                       event_type="api_call",
                       service="gemini",
                       duration_ms=duration,
                       tokens_input=t_in,
                       tokens_output=t_out)

        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Gemini is currently unreachable."

# --- MAIN ROUTE ---

@app.route('/sms', methods=['POST'])
def sms_reply():
    """Handles incoming SMS from Twilio."""
    global is_warm
    request_timer = ExecutionTimer()

    # Cold Start Check
    cold_start = not is_warm
    is_warm = True # Set to True for next time

    # 1. Security Check
    sender = request.values.get('From')
    incoming_msg = request.values.get('Body', '').strip()

    # Log the incoming event
    # We redact the phone number for privacy logs, keeping only the last 4 digits
    safe_sender = f"...{sender[-4:]}" if sender and len(sender) > 4 else "Unknown"
    log_structured(
        "Incoming SMS",
        event_type="sms_received",
        sender_suffix=safe_sender,
        msg_length=len(incoming_msg)
    )

    if sender != ALLOWED_NUMBER:
        log_structured("Security Block", sender=sender)
        return abort(403)

    # 2. Parse Incoming Message
    resp = MessagingResponse()
    reply_text = ""
    incoming_msg_lower = incoming_msg.lower()
    intent = "unknown"

    # 3. Routing Logic
    extra_log_kwargs = {}

    # CASE A: Weather Command (e.g., "Wx Rainier" or "Weather Moab")
    if incoming_msg_lower.startswith(('wx', 'weather')):
        intent = "weather"

        # Extract location (remove first word)
        location_query = ' '.join(incoming_msg.split()[1:])

        extra_log_kwargs['weather: location_query'] = location_query

        if not location_query:
            reply_text = "Please specify a location (e.g., 'Wx Mt Rainier')"
        else:
            lat, lon, name = get_coordinates(location_query)
            extra_log_kwargs['weather: resolved_name'] = name
            extra_log_kwargs['weather: lat'] = lat
            extra_log_kwargs['weather: lon'] = lon
            if lat:
                wx_data = get_weather_data(lat, lon)
                reply_text = f"@{name[:20]}: {wx_data}"
            else:
                reply_text = f"Could not find location: {location_query}"

    # CASE B: Preset Commands
    elif incoming_msg_lower == 'ping':
        intent = "ping"
        reply_text = "Pong! System online. 🏔️"

    elif incoming_msg_lower == 'checkin':
        intent = "checkin"
        # Placeholder for logging logic (e.g., append to Google Sheet)
        reply_text = "Check-in logged (simulation)."

    # CASE C: Default to Gemini AI
    else:
        intent = "ai_chat"
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

    log_structured(
        "Request Complete",
        event_type="request_summary",
        intent=intent,
        total_duration_ms=request_timer.stop(),
        is_cold_start=cold_start,
        response_segments=len(chunks),
        response_char_count=len(reply_text),
        **extra_log_kwargs
    )

    return str(resp)

# --- PRODUCTION SERVER CONFIG ---
if __name__ == "__main__":
    # Cloud Run injects the PORT environment variable.
    # Default to 8080 for local testing.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)