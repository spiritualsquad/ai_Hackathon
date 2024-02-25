import cv2
import numpy as np
import pygame

# Load the known face images
known_face1 = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)
known_face2 = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)

# Load the Haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize pygame mixer
pygame.mixer.init()

# Load ding sound
ding_sound = pygame.mixer.Sound("ding.mp3")  
ding_sound2 = pygame.mixer.Sound("welcome2.mp3")

cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the size of the known faces
    frame_gray_resized1 = cv2.resize(frame_gray, (known_face1.shape[1], known_face1.shape[0]))
    frame_gray_resized2 = cv2.resize(frame_gray, (known_face2.shape[1], known_face2.shape[0]))

    # Compute Mean Squared Error (MSE) for both known faces
    mse1 = np.sum((known_face1 - frame_gray_resized1) ** 2) / float(known_face1.shape[0] * known_face1.shape[1])
    mse2 = np.sum((known_face2 - frame_gray_resized2) ** 2) / float(known_face2.shape[0] * known_face2.shape[1])

    # Set a threshold for MSE
    mse_threshold = 100  # Adjust this threshold based on your requirements and images

    # Check if the MSE is below the threshold for any known face
    if mse1 < mse_threshold or mse2 < mse_threshold:
        name = "High Value Customer"
        color = (0, 255, 0)  # green color for high value

        # Play ding sound
        ding_sound.play()

        # Pause the program until the audio finishes playing
        pygame.time.wait(int(ding_sound.get_length() * 1000))  # Wait in milliseconds
    else:
        name = "Normal customer"
        color = (255, 255, 255)  # white color for normal
        # Play ding sound
        ding_sound2.play()

        # Pause the program until the audio finishes playing
        pygame.time.wait(int(ding_sound2.get_length() * 1000))  # Wait in milliseconds

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the face region
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Display the name
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (10, 30), font, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
