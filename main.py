import os
import logging
import requests
import google.generativeai as genai
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Switch logging to debug for better visibility in Cloud Logs
logging.basicConfig(level=logging.INFO)

# --- ENVIRONMENT VARIABLES ---
# We retrieve these from the Cloud Run configuration
ALLOWED_NUMBER = os.environ.get('ALLOWED_NUMBER')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logging.error("GEMINI_API_KEY is missing!")

# ... [Keep the helper functions: get_coordinates, get_weather_data, ask_gemini] ...
# ... [Keep the @app.route('/sms') logic] ...

# --- PRODUCTION SERVER CONFIG ---
if __name__ == "__main__":
    # Cloud Run injects the PORT environment variable.
    # Default to 8080 for local testing.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)