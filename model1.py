import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
from keras.utils import plot_model

# Mediapipe components
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create directories if they don't exist
os.makedirs('./Fatigue Subjects', exist_ok=True)
os.makedirs('./Active Subjects', exist_ok=True)

# Helper function to draw landmarks
def draw_landmarks(image, face_landmarks, category, index):
    imgW, imgH = image.shape[1], image.shape[0]
    img_copy = image.copy()

    # Draw facial landmarks
    mp_drawing.draw_landmarks(
        image=img_copy,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
    )

    if category == 'Fatigue Subjects':
        cv2.imwrite(f'./Fatigue Subjects/{index}.jpg', img_copy)
    else:
        cv2.imwrite(f'./Active Subjects/{index}.jpg', img_copy)

    return cv2.resize(img_copy, (145, 145))

# Main function to process images and extract landmarks
def process_images(direc, face_cas_path):
    yaw_no = []
    face_cascade = cv2.CascadeClassifier(face_cas_path)
    categories = ["Fatigue Subjects", "Active Subjects"]

    for category in categories:
        path = os.path.join(direc, category)
        label = categories.index(category)

        for idx, img_name in enumerate(os.listdir(path)):
            image = cv2.imread(os.path.join(path, img_name))
            faces = face_cascade.detectMultiScale(image, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = image[y:y + h, x:x + w]

                with mp_facemesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            resized_image = draw_landmarks(roi, face_landmarks, category, idx)
                            yaw_no.append([resized_image, label])

    return yaw_no

# Data preparation
yawn_data = process_images("../input/drowsiness-prediction-dataset/0 FaceImages", "../input/prediction-images/haarcascade_frontalface_default.xml")
X, y = zip(*yawn_data)

# Label encoding and train-test split
X = np.array(X).reshape(-1, 145, 145, 3)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
train_gen = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30).flow(X_train, y_train)
test_gen = ImageDataGenerator(rescale=1/255).flow(X_test, y_test)

# Model definition
model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(32, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, 10, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 12, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
model.summary()

# Training
history = model.fit(train_gen, epochs=70, validation_data=test_gen)

# Evaluation and saving the model
model.evaluate(test_gen)
model.save('my_model.h5')