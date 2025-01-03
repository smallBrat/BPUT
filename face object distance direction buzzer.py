import cv2
import pygame
import numpy as np
import os
import time
import pyttsx3
#import playsound
import supervision as sv
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speaking rate

# Paths and thresholds
dataset_path = "./data/"
CONFIDENCE_THRESHOLD = 0.6  # For face recognition confidence
last_announcement = {}

# Camera calibration constants
KNOWN_DISTANCE = 30  # cm
KNOWN_WIDTH = 5  # cm
FOCAL_LENGTH = 800  # Calibrated focal length

# Vehicle classes for buzzer alert
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Face recognition initialization
faceData = []
labels = []
nameMap = {}
offset = 40
classId = 0

# Load existing face data
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]
        dataItem = np.load(dataset_path + f)

        # Convert to grayscale if necessary
        if dataItem.shape[1] == 30000:  # Check if the data is in color (3 channels)
            dataItem = np.mean(dataItem.reshape(-1, 100, 100, 3), axis=-1).reshape(-1, 10000)

        faceData.append(dataItem)
        m = dataItem.shape[0]
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

if faceData:
    XT = np.concatenate(faceData, axis=0)
    yT = np.concatenate(labels, axis=0).reshape((-1, 1))
else:
    XT = np.empty((0, 10000))  # Placeholder for grayscale data
    yT = np.empty((0, 1))

# Helper functions

# Initialize pygame mixer for sound playback
try:
    pygame.mixer.init()
    pygame_mixer_initialized = True
except pygame.error as e:
    print(f"Error initializing pygame mixer: {e}")
    pygame_mixer_initialized = False

def calculate_distance(known_width, focal_length, pixel_width):
    """Calculate distance to an object using its pixel width."""
    return (known_width * focal_length) / pixel_width

def calculate_clock_direction(center_x, frame_width):
    """Calculate direction relative to the camera using clock positions."""
    relative_position = (center_x / frame_width) * 12
    if relative_position <= 1 or relative_position > 11:
        return "12 o'clock"
    elif 1 < relative_position <= 3:
        return "2 o'clock"
    elif 3 < relative_position <= 5:
        return "4 o'clock"
    elif 5 < relative_position <= 7:
        return "6 o'clock"
    elif 7 < relative_position <= 9:
        return "8 o'clock"
    else:
        return "10 o'clock"

def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []
    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))
    dlist = sorted(dlist, key=lambda x: x[0])
    labels = [label for _, label in dlist[:k]]

    # Calculate label probabilities
    labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / k

    # Find the label with the highest probability
    idx = probabilities.argmax()
    pred_label = labels[idx]
    pred_confidence = probabilities[idx]

    return int(pred_label), pred_confidence

def play_sound(sound_file):
    """Play sound and pause detection while it's playing."""
    if not pygame_mixer_initialized:
        print("Pygame mixer not initialized. Skipping sound playback.")
        return

    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

    # Wait until the sound has finished playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# YOLO initialization
yolo_model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator(
    thickness=2, text_thickness=2, text_scale=1
)

# Haar Cascade initialization
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    success, frame = cam.read()
    if not success:
        print("Camera read failed.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Process detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = gray_frame[max(0, y - offset): min(y + h + offset, gray_frame.shape[0]),
                                  max(0, x - offset): min(x + w + offset, gray_frame.shape[1])]

        try:
            cropped_face_resized = cv2.resize(cropped_face, (100, 100))
        except:
            continue

        cropped_face_flattened = cropped_face_resized.flatten()
        current_time = time.time()

        # Face recognition prediction
        if XT.size > 0:
            classPredictedId, confidence = knn(XT, yT, cropped_face_flattened)

            if confidence >= CONFIDENCE_THRESHOLD:
                namePredicted = nameMap.get(classPredictedId, "Unknown")

                if namePredicted != "Unknown":
                    pixel_width = w
                    center_x = x + w // 2
                    face_distance = int(calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width))
                    direction = calculate_clock_direction(center_x, frame.shape[1])

                    cv2.putText(frame, f"{namePredicted} {face_distance}cm {direction}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if namePredicted not in last_announcement or current_time - last_announcement[namePredicted] > 10:
                        engine.say(f"{namePredicted} is {face_distance} centimeters away at your {direction}.")
                        engine.runAndWait()
                        last_announcement[namePredicted] = current_time

    # YOLO object detection
    result = yolo_model(frame, agnostic_nms=True)[0]  # Ensure YOLO processes the frame correctly
    current_time = time.time()  # Define current_time here for YOLO detections
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = map(int, xyxy)
        width = x2 - x1
        center_x = (x1 + x2) // 2
        obj_name = yolo_model.model.names[class_id]

        # Skip "person" objects entirely
        if obj_name == "person":
            continue

        object_distance = int(calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, width))
        direction = calculate_clock_direction(center_x, frame.shape[1])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{obj_name} {object_distance}cm {direction}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Trigger buzzer for vehicles within 10 meters
        if obj_name in VEHICLE_CLASSES and object_distance <= 1000:
            # Play sound and pause detection
            play_sound("buzzer.mp3")

        # Announce object direction and distance
        if obj_name not in last_announcement or current_time - last_announcement[obj_name] > 10:
            engine.say(f"{obj_name} is {object_distance} centimeters away at your {direction}.")
            engine.runAndWait()
            last_announcement[obj_name] = current_time

    # Display frame
    cv2.imshow("Face & Object Detection with Distance and Direction", frame)

    if cv2.waitKey(30) == 27:  # Press 'Esc' to exit
        break

cam.release()
cv2.destroyAllWindows()