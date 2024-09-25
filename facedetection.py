import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import normalize

# Load pre-trained models
face_detector = MTCNN()
face_recognizer = load_model('path_to_your_face_recognition_model.h5')

def detect_faces(image):
    return face_detector.detect_faces(image)

def extract_face(image, face_info, required_size=(160, 160)):
    x, y, width, height = face_info['box']
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, required_size)
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

def get_embedding(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    embedding = face_recognizer.predict(face)
    embedding = normalize(embedding).flatten()
    return embedding

def recognize_face(embedding, database):
    min_distance = float('inf')
    identity = None
    for name, stored_embedding in database.items():
        distance = np.linalg.norm(embedding - stored_embedding)
        if distance < min_distance:
            min_distance = distance
            identity = name
    if min_distance > 0.7:  # Threshold for unknown faces
        return "Unknown"
    return identity

def process_image(image_path, database):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = detect_faces(image_rgb)
    
    for face_info in faces:
        face = extract_face(image_rgb, face_info)
        embedding = get_embedding(face)
        identity = recognize_face(embedding, database)
        
        x, y, width, height = face_info['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(image, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    return image

# Example usage
database = {
    "Person1": np.random.rand(128),  # Replace with actual embeddings
    "Person2": np.random.rand(128),
    # Add more known faces to the database
}

result = process_image('path_to_your_image.jpg', database)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()