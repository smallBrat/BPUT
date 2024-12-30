import sys
import os
import cv2
import base64
import google.generativeai as genai
import pyttsx3
from datetime import datetime

# Configure the API key
genai.configure(api_key="AIzaSyC4G9h8STxQCtsC6ysiXuLgvzNtpRoLPsY")

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty("rate", 180)

def speak(text):
    """Convert text to speech."""
    print(text)
    engine.say(text)
    engine.runAndWait()

def query_gemini(prompt):
    """Send a text prompt to the Gemini API and get a response."""
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content([prompt])
    return response.text

def query_gemini_with_image(prompt, image_path):
    """Send a text prompt and an image to the Gemini API and get a response."""
    # Read and encode the image in Base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content([
        {'mime_type': 'image/jpeg', 'data': image_data},
        prompt
    ])
    return response.text

def get_latest_photo(folder_path):
    """Get the latest photo from the specified folder."""
    photos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not photos:
        return None
    latest_photo = max(photos, key=os.path.getctime)
    return latest_photo

def main():
    if len(sys.argv) < 2:
        print("No text input provided.")
        return

    input_text = sys.argv[1]
    print(f"Received input: {input_text}")

    # Check if the input text indicates an image-related query
    image_related_keywords = ["photo", "image", "picture", "next to", "beside", "above", "below"]
    if any(keyword in input_text.lower() for keyword in image_related_keywords):
        photos_folder = "photos"  # Folder where images are stored
        latest_photo = get_latest_photo(photos_folder)

        if latest_photo:
            print(f"Using the latest photo: {latest_photo}")
            response = query_gemini_with_image(input_text, latest_photo)
        else:
            response = "I couldn't find any photos to analyze. Please add a photo to the 'photos' folder."
    else:
        response = query_gemini(input_text)

    speak(response)

if __name__ == "__main__":
    main()