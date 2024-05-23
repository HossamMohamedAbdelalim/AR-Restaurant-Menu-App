import cv2
import os
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images and labels for training
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}
    current_id = 0

    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = os.path.splitext(filename)[0]

            if label not in label_dict:
                label_dict[label] = current_id
                current_id += 1

            images.append(image)
            labels.append(label_dict[label])

    return images, labels, label_dict

# Load known faces
known_faces_dir = 'known_faces'
images, labels, label_dict = load_images_from_folder(known_faces_dir)

# Train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(roi_gray)

        for name, id in label_dict.items():
            if id == label_id:
                label_name = name

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
