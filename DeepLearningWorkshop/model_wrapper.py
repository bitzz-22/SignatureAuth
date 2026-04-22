import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Sequential, Model
import numpy as np
import cv2

def create_siamese_model(input_shape):
    # Define the base convolutional neural network (CNN) model (Embedding Model)
    embedding_model = Sequential(name="embedding_model")
    embedding_model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    embedding_model.add(MaxPooling2D(2, 2))
    embedding_model.add(Dropout(0.25))
    embedding_model.add(Conv2D(32, (3, 3), activation='relu'))
    embedding_model.add(MaxPooling2D(2, 2))
    embedding_model.add(Dropout(0.25))
    embedding_model.add(Conv2D(64, (3, 3), activation='relu'))
    embedding_model.add(MaxPooling2D(2, 2))
    embedding_model.add(Dropout(0.25))
    embedding_model.add(Conv2D(128, (3, 3), activation='relu'))
    embedding_model.add(MaxPooling2D(2, 2))
    embedding_model.add(Dropout(0.25))
    embedding_model.add(Flatten())
    embedding_model.add(Dense(512, activation='relu'))
    embedding_model.add(Dense(128, activation='relu'))
    
    # Define inputs for the siamese network
    input1 = Input(input_shape, name="image1")
    input2 = Input(input_shape, name="image2")

    # Obtain embeddings for both inputs using the base CNN model
    embedding1 = embedding_model(input1)
    embedding2 = embedding_model(input2)

    # Define a lambda layer to calculate the Manhattan distance between embeddings
    manhattan_distance_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))

    # Calculate the Manhattan distance between embeddings
    manhattan_distance = manhattan_distance_layer([embedding1, embedding2])

    # Define the final output layer to predict similarity (0 for dissimilar, 1 for similar)
    output = Dense(1, activation='sigmoid')(manhattan_distance)

    # Create the siamese model with inputs and output
    siamese_model = Model(inputs=[input1, input2], outputs=output, name="siamese_model")

    return siamese_model, embedding_model

def preprocess_image(img_path):
    # Read image file
    img = tf.io.read_file(img_path)
    # Decode image and convert to grayscale
    img = tf.image.decode_image(img, channels=1) 
    # Resize image to desired dimensions
    img = tf.image.resize(img, (128, 128))
    # Convert image data type to uint8
    img = tf.cast(img, tf.uint8) 

    # Apply Canny edge detection using OpenCV
    img_cv2 = cv2.Canny(img.numpy(), 20, 220) 
    
    # Normalize pixel values to [0, 1]
    img_cv2 = tf.cast(img_cv2, tf.float32) / 255.0
    
    # Reshape for model input (batch_size, h, w, c)
    img_cv2 = tf.reshape(img_cv2, (1, 128, 128, 1))
    
    return img_cv2

class SignatureModel:
    def __init__(self, model_path):
        input_shape = (128, 128, 1)
        self.siamese_model, self.embedding_model = create_siamese_model(input_shape)
        self.siamese_model.load_weights(model_path)
        print("Model loaded successfully.")

    def get_embedding(self, img_path):
        processed_img = preprocess_image(img_path)
        embedding = self.embedding_model.predict(processed_img)
        return embedding[0] # Return the 128-d vector
