import requests
import pywhatkit
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("whatsapp_motivator.log"), logging.StreamHandler()]
)

# List of motivational quotes
quotes = [
    "Paise de diijiye",
    "pay karde bhai",
    "final payment kabtak ayegi?",
    "paise nahi the to kaam kyu karaya",
    "you are gareeb"
]

# Function to fetch a random quote from the list
def get_random_quote():
    quote = random.choice(quotes)
    logging.info(f"Selected quote: {quote}")
    return quote

# Function to send a WhatsApp message
def send_whatsapp_message(phone_number, message):
    try:
        # Send the message using PyWhatKit
        pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=15)
        logging.info(f"WhatsApp message sent successfully: {message}")
    except Exception as e:
        logging.error(f"Error sending WhatsApp message: {e}")

# Main function to start the AI Agent
def main():
    logging.info("WhatsApp Motivator AI Agent activated.")         

    # Fetch a random quote
    quote = get_random_quote()

    # Phone number (include country code, e.g., +91 for India)
    phone_number = "+919958389484"  # Replace with the recipient's phone number

    # Send the quote via WhatsApp
    send_whatsapp_message(phone_number, f"Mr.Client:\n\n\"{quote}\"")

if __name__ == "__main__":
    main()
