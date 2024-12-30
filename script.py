import subprocess
import speech_recognition as sr

def recognize_command():
    """Recognize voice commands using the microphone."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that. Please try again.")
        except sr.RequestError:
            print("There was an issue with the speech recognition service.")
        except sr.WaitTimeoutError:
            print("No command detected. Please speak clearly.")
        return None

def main():
    print("Welcome! I am here to assist you. Say 'bye' to exit.")
    first_process = None
    second_process = None

    while True:
        command = recognize_command()
        if command is None:
            continue  # Skip to the next iteration if no valid command was recognized

        if command == "detect":
            if first_process is None:
                first_process = subprocess.Popen(["python", "face object distance.py"])  # Replace with your file name
                print("Viewing mode activated.")
            else:
                print("Viewing mode is already active.")

        elif command == "capture":
            if second_process is None:
                second_process = subprocess.Popen(["python", "new gemini.py"])  # Replace with your file name
                print("Capturing what is ahead.")
            else:
                print("Detection mode is already active.")

        elif command == "escape":
            if first_process:
                first_process.terminate()
                first_process = None
                print("Exited viewing mode.")
            else:
                print("Viewing mode is not active.")

        elif command == "skip":
            if second_process:
                second_process.terminate()
                second_process = None

                print("Skipped detection.")
            else:
                print("Detection mode is not active.")

        elif command == "bye":
            if first_process:
                first_process.terminate()
            if second_process:
                second_process.terminate()
            print("Goodbye!")
            break

        else:
            # Pass undefined commands to qna.py
            print(f"Forwarding command to QnA: {command}")
            subprocess.Popen(["python", "qna.py", command])

if __name__ == "__main__":
    main()
