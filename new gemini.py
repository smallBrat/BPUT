import cv2
import base64
import google.generativeai as genai
import mtranslate
import pyttsx3
import time
import os

# Configure the API key
genai.configure(api_key="AIzaSyC4G9h8STxQCtsC6ysiXuLgvzNtpRoLPsY")

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty("rate", 180)

def speak(audio):
    translated_audio = mtranslate.translate(audio, to_language="te-IN", from_language="te-IN")
    print(translated_audio)
    engine.say(translated_audio)
    engine.runAndWait()

# Function to capture an image from the phone camera
def capture_image():
    video_url = "http://192.168.1.4:8080/video"
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Error: Could not access phone camera.")
        return None

    print("Capturing photo in 5 seconds...")
    time.sleep(5)  # Wait for 5 seconds

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        cap.release()
        return None

    cap.release()
    cv2.destroyAllWindows()

    # Save the image in the 'photos' folder
    photos_folder = "photos"
    if not os.path.exists(photos_folder):
        os.makedirs(photos_folder)

    image_path = os.path.join(photos_folder, f"photo_{int(time.time())}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image saved at {image_path}")

    return frame

# Main function to process the captured image and query the API
def main():
    frame = capture_image()
    if frame is None:
        return

    # Encode the image in Base64
    _, buffer = cv2.imencode(".jpg", frame)
    image_data = base64.b64encode(buffer).decode('utf-8')

    # Prompt for the AI model
    prompt = "What is there in this picture?"

    # Initialize the Generative Model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Generate content
    response = model.generate_content([
        {'mime_type': 'image/jpeg', 'data': image_data},
        prompt
    ])

    # Print and speak the response
    print(response.text)
    speak(response.text)

if __name__ == "__main__":
    main()