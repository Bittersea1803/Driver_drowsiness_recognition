import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Initialize mediapipe modules
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Create directories
os.makedirs('Fatigue Subjects', exist_ok=True)
os.makedirs('Active Subjects', exist_ok=True)

# Constants
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))

all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))

all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
IMG_SIZE = 145

# Dummy directory paths (you should replace these)
direc = "./"
face_cas_path = "haar cascade files/haarcascade_frontalface_alt.xml"

# Functions
def draw(*, n, img_dt, cat, face_landmarks):
    imgH, imgW, _ = img_dt.shape
    image_drawing_tool = img_dt
    connections_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(255, 255, 255))
    mp_drawing.draw_landmarks(image=image_drawing_tool, landmark_list=face_landmarks, 
                              connections=mp_facemesh.FACEMESH_TESSELATION, 
                              connection_drawing_spec=connections_drawing_spec)
    
    landmarks = face_landmarks.landmark
    for landmark_idx, landmark in enumerate(landmarks):
        pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
        if landmark_idx in all_idxs:
            cv2.circle(image_drawing_tool, pred_cord, 3, (255, 255, 255), -1)
            
    if cat == 'Fatigue Subjects':
        cv2.imwrite(f'./Fatigue Subjects/{n}.jpg', image_drawing_tool)
    else:
        cv2.imwrite(f'./Active Subjects/{n}.jpg', image_drawing_tool)
        
    resized_array = cv2.resize(image_drawing_tool, (IMG_SIZE, IMG_SIZE))
    return resized_array

def landmarks(image, category, i):
    resized_array = []
    image = np.ascontiguousarray(image)
    imgH, imgW, _ = image.shape
    with mp_facemesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, 
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                resized_array = draw(n=i, img_dt=image.copy(), cat=category, face_landmarks=face_landmarks)
    return resized_array

# Your data collection part
# Note: Replace this part with your actual data collection code
data = []
labels = []

# Your data preprocessing and model training
# Note: Replace this part with your actual preprocessing and model training code
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(data, integer_labels, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, validation_split=0.1, epochs=10)