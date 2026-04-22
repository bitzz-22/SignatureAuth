# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:35:26.817384Z","iopub.execute_input":"2025-11-08T19:35:26.817679Z","iopub.status.idle":"2025-11-08T19:35:37.909Z","shell.execute_reply.started":"2025-11-08T19:35:26.817647Z","shell.execute_reply":"2025-11-08T19:35:37.908306Z"}}
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import cv2
# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:35:37.91006Z","iopub.execute_input":"2025-11-08T19:35:37.910707Z","iopub.status.idle":"2025-11-08T19:35:37.965566Z","shell.execute_reply.started":"2025-11-08T19:35:37.910676Z","shell.execute_reply":"2025-11-08T19:35:37.964927Z"}}
# Load train and test datasets from CSV files
train_dataset = pd.read_csv('/kaggle/input/signature-verification-dataset/sign_data/train_data.csv', header = None)
test_dataset = pd.read_csv('/kaggle/input/signature-verification-dataset/sign_data/test_data.csv', header = None)  
# Define directories for train and test data
train_dir = "/kaggle/input/signature-verification-dataset/sign_data/train"
test_dir = "/kaggle/input/signature-verification-dataset/sign_data/test"

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:35:37.967308Z","iopub.execute_input":"2025-11-08T19:35:37.96758Z","iopub.status.idle":"2025-11-08T19:35:37.977115Z","shell.execute_reply.started":"2025-11-08T19:35:37.967559Z","shell.execute_reply":"2025-11-08T19:35:37.976354Z"}}
class DataLoader:
    def __init__(self, dataset, batch_size, dir):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dir = dir
    
    def shuffle(self):
        # Shuffling the dataset
        return self.dataset.sample(frac=1)
    
    def preprocess_image(self, img_path):
        # Read image file
        img = tf.io.read_file(img_path)
        # Decode image and convert to grayscale
        img = tf.image.decode_png(img, channels=1) 
        # Resize image to desired dimensions
        img = tf.image.resize(img, (128, 128))
        # Convert image data type to uint8
        img = tf.cast(img, tf.uint8) 

        # Apply Canny edge detection using OpenCV
        img_cv2 = cv2.Canny(img.numpy(), 20, 220) 
        
        # Normalize pixel values to [0, 1]
        img_cv2 = tf.cast(img_cv2, tf.float32) / 255.0
        
        return img_cv2
    
    def datagen(self):
        num_samples = len(self.dataset)
        while True:
            # Shuffle dataset for each epoch
            self.dataset = self.shuffle()
            for batch in range(0, num_samples, self.batch_size):
                # Get batch samples
                image1_batch_samples = [self.dir + "/" + img for img in self.dataset.iloc[batch:batch + self.batch_size, 0]]
                image2_batch_samples = [self.dir + "/" + img for img in self.dataset.iloc[batch:batch + self.batch_size, 1]]
                label_batch_samples = self.dataset.iloc[batch:batch + self.batch_size, 2]
                Image1, Image2, Label = [], [], []
                for image1, image2, label in zip(image1_batch_samples, image2_batch_samples, label_batch_samples):
                    # Preprocess image1
                    image1_data = self.preprocess_image(image1)
                    # Preprocess image2
                    image2_data = self.preprocess_image(image2)
                    # Append preprocessed images and labels to lists
                    Image1.append(image1_data)
                    Image2.append(image2_data)
                    Label.append(label)

                # Convert lists to numpy arrays
                Image1 = np.asarray(Image1)
                Image2 = np.asarray(Image2)
                Label = np.asarray(Label)
                # Yield batch data
                yield {"image1": Image1, "image2": Image2}, Label


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:35:37.977998Z","iopub.execute_input":"2025-11-08T19:35:37.978225Z","iopub.status.idle":"2025-11-08T19:35:37.995798Z","shell.execute_reply.started":"2025-11-08T19:35:37.978207Z","shell.execute_reply":"2025-11-08T19:35:37.994976Z"}}
train_set, val_set = train_test_split(train_dataset, test_size=0.25)

# Create train, val and test 
train_gen = DataLoader(train_set, 256, train_dir)
val_gen = DataLoader(val_set, 256, train_dir)
test_gen = DataLoader(test_dataset, 256, test_dir)  

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:35:37.996844Z","iopub.execute_input":"2025-11-08T19:35:37.997086Z","iopub.status.idle":"2025-11-08T19:36:05.134024Z","shell.execute_reply.started":"2025-11-08T19:35:37.997063Z","shell.execute_reply":"2025-11-08T19:36:05.133281Z"}}

# Create data generator
generator = train_gen.datagen()

# Plot 5 1st images in batch
for i in range(5):
    batch_data, label = next(generator)
    
    pair_data = batch_data["image1"], batch_data["image2"]
    
    print(f"Pair {i+1}:")
    print("Label:", label[0]) 
    
    plt.figure(figsize=(10, 5))
    for j in range(2):
        plt.subplot(1, 2, j+1)
        plt.imshow(pair_data[j][0], cmap='gray')
        plt.title('Image {}'.format(j+1))
        plt.axis('off')
    
    plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:36:05.135314Z","iopub.execute_input":"2025-11-08T19:36:05.135669Z","iopub.status.idle":"2025-11-08T19:36:05.145198Z","shell.execute_reply.started":"2025-11-08T19:36:05.13564Z","shell.execute_reply":"2025-11-08T19:36:05.144429Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:36:05.146325Z","iopub.execute_input":"2025-11-08T19:36:05.146905Z","iopub.status.idle":"2025-11-08T19:36:05.157427Z","shell.execute_reply.started":"2025-11-08T19:36:05.146877Z","shell.execute_reply":"2025-11-08T19:36:05.156737Z"}}
# Define the shape of input images: height 128 pixels, width 128 pixels, and 1 channel (grayscale)
input_shape = (128, 128, 1)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:36:05.158324Z","iopub.execute_input":"2025-11-08T19:36:05.158585Z","iopub.status.idle":"2025-11-08T19:36:05.347913Z","shell.execute_reply.started":"2025-11-08T19:36:05.158566Z","shell.execute_reply":"2025-11-08T19:36:05.347313Z"}}
# Create the siamese model using the specified input shape
siamese_model, embedding_model = create_siamese_model(input_shape)

# Print a summary of the siamese model architecture
siamese_model.summary()

# Compile the siamese model with binary cross-entropy loss, Adam optimizer, and accuracy metric
siamese_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:36:05.351387Z","iopub.execute_input":"2025-11-08T19:36:05.35162Z","iopub.status.idle":"2025-11-08T19:36:05.355641Z","shell.execute_reply.started":"2025-11-08T19:36:05.351601Z","shell.execute_reply":"2025-11-08T19:36:05.354774Z"}}
# Define callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

callbacks_list = [checkpoint, early_stopping]

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T19:36:05.356752Z","iopub.execute_input":"2025-11-08T19:36:05.357011Z","iopub.status.idle":"2025-11-08T20:49:20.677854Z","shell.execute_reply.started":"2025-11-08T19:36:05.356984Z","shell.execute_reply":"2025-11-08T20:49:20.677114Z"}}
# Model training
history = siamese_model.fit(x=train_gen.datagen(),
                            steps_per_epoch=len(train_set) // 256,
                            epochs=100,
                            validation_data=val_gen.datagen(),
                            validation_steps=len(val_set) // 256,
                            callbacks=callbacks_list)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:49:20.679085Z","iopub.execute_input":"2025-11-08T20:49:20.679339Z","iopub.status.idle":"2025-11-08T20:49:20.943759Z","shell.execute_reply.started":"2025-11-08T20:49:20.679319Z","shell.execute_reply":"2025-11-08T20:49:20.9429Z"}}
# Train and val model accuracy per epoch
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:49:20.945324Z","iopub.execute_input":"2025-11-08T20:49:20.94567Z","iopub.status.idle":"2025-11-08T20:49:21.261725Z","shell.execute_reply.started":"2025-11-08T20:49:20.94564Z","shell.execute_reply":"2025-11-08T20:49:21.260981Z"}}

# load weights
siamese_model.load_weights("best_model.keras")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:49:21.262585Z","iopub.execute_input":"2025-11-08T20:49:21.262828Z","iopub.status.idle":"2025-11-08T20:49:21.272414Z","shell.execute_reply.started":"2025-11-08T20:49:21.262808Z","shell.execute_reply":"2025-11-08T20:49:21.2716Z"}}
mytest_dir = "/kaggle/input/signature-verification-dataset/sign_data/test"
mytest_dataset = pd.read_csv('/kaggle/input/signature-verification-dataset/sign_data/test_data.csv', header = None)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:49:21.273518Z","iopub.execute_input":"2025-11-08T20:49:21.273719Z","iopub.status.idle":"2025-11-08T20:50:33.440935Z","shell.execute_reply.started":"2025-11-08T20:49:21.273702Z","shell.execute_reply":"2025-11-08T20:50:33.439983Z"}}
# Testing model on test dataset
test_loss, test_accuracy = siamese_model.evaluate(test_gen.datagen(),
                                          steps=len(test_dataset) // 256)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:33.442152Z","iopub.execute_input":"2025-11-08T20:50:33.442425Z","iopub.status.idle":"2025-11-08T20:50:36.504864Z","shell.execute_reply.started":"2025-11-08T20:50:33.442402Z","shell.execute_reply":"2025-11-08T20:50:36.504103Z"}}
# Getting image pairs and their labels for the test dataset
test_pairs, test_labels = next(test_gen.datagen())

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:36.505921Z","iopub.execute_input":"2025-11-08T20:50:36.506187Z","iopub.status.idle":"2025-11-08T20:50:37.6367Z","shell.execute_reply.started":"2025-11-08T20:50:36.506166Z","shell.execute_reply":"2025-11-08T20:50:37.635782Z"}}
# Getting predictions for the test dataset
test_predictions = siamese_model.predict(test_pairs)

# Conversion of predictions to binary format (0 or 1)
binary_predictions = (test_predictions > 0.5).astype(int)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:37.637944Z","iopub.execute_input":"2025-11-08T20:50:37.638254Z","iopub.status.idle":"2025-11-08T20:50:42.120089Z","shell.execute_reply.started":"2025-11-08T20:50:37.63823Z","shell.execute_reply":"2025-11-08T20:50:42.119233Z"}}
import random

# Getting random 20 pair indices for output
random_indices = random.sample(range(len(test_pairs['image1'])), 20)

# Output random 20 pairs of images, their true labels and predicted labels
for i in random_indices:
    
    image1 = test_pairs['image1'][i]
    image2 = test_pairs['image2'][i]
    
    # Getting the true label and predicted label
    true_label = test_labels[i]
    predicted_label = binary_predictions[i]
    
    # output
    print("Pair", i+1)
    print("True Label:", true_label)
    print("Predicted Label:", predicted_label)
    
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1.squeeze(), cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2.squeeze(), cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    plt.show()

# %% [markdown]
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.121166Z","iopub.execute_input":"2025-11-08T20:50:42.121468Z","iopub.status.idle":"2025-11-08T20:50:42.417576Z","shell.execute_reply.started":"2025-11-08T20:50:42.121445Z","shell.execute_reply":"2025-11-08T20:50:42.416943Z"}}

# Loading the best model
siamese_model.load_weights("best_model.keras")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.418812Z","iopub.execute_input":"2025-11-08T20:50:42.419204Z","iopub.status.idle":"2025-11-08T20:50:42.430567Z","shell.execute_reply.started":"2025-11-08T20:50:42.419174Z","shell.execute_reply":"2025-11-08T20:50:42.42982Z"}}
# Define the directory containing test data
mytest_dir = "/kaggle/input/signature-verification-dataset/sign_data/test"

# Load test dataset from a CSV file
# Specifying encoding='latin1' to handle non-UTF-8 encoded characters, if any
# Specifying header=None since CSV file may not contain column names
mytest_dataset = pd.read_csv('/kaggle/input/signature-verification-dataset/sign_data/test_data.csv', encoding='latin1', header=None)


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.431795Z","iopub.execute_input":"2025-11-08T20:50:42.432035Z","iopub.status.idle":"2025-11-08T20:50:42.442413Z","shell.execute_reply.started":"2025-11-08T20:50:42.432014Z","shell.execute_reply":"2025-11-08T20:50:42.441425Z"}}
class CDataLoader:
    def __init__(self, dataset, batch_size, dir):
        # Initialize the DataLoader with dataset, batch_size, and directory
        self.dataset = dataset
        self.batch_size = batch_size
        self.dir = dir
    
    def shuffle(self):
        # Shuffle the dataset
        return self.dataset.sample(frac=1)
    
    def preprocess_image(self, img_path):
        # Read image file
        img = tf.io.read_file(img_path)
        # Decode image (assuming JPEG format) and convert to grayscale
        img = tf.image.decode_jpeg(img, channels=1) 
        # Resize image to 128x128 pixels
        img = tf.image.resize(img, (128, 128))
        # Convert image data type to uint8
        img = tf.cast(img, tf.uint8) 

        # Apply Canny edge detection using OpenCV
        img_cv2 = cv2.Canny(img.numpy(), 20, 220) 
        
        # Normalize pixel values to [0, 1]
        img_cv2 = tf.cast(img_cv2, tf.float32) / 255.0
        
        return img_cv2
    
    def datagen(self):
        num_samples = len(self.dataset)
        while True:
            # Shuffle dataset for each epoch
            self.dataset = self.shuffle()
            # Iterate over dataset in batches
            for batch in range(0, num_samples, self.batch_size):
                # Get batch samples
                image1_batch_samples = [self.dir + "/" + img for img in self.dataset.iloc[batch:batch + self.batch_size, 0]]
                image2_batch_samples = [self.dir + "/" + img for img in self.dataset.iloc[batch:batch + self.batch_size, 1]]
                label_batch_samples = self.dataset.iloc[batch:batch + self.batch_size, 2]
                Image1, Image2, Label = [], [], []
                # Preprocess batch samples
                for image1, image2, label in zip(image1_batch_samples, image2_batch_samples, label_batch_samples):
                    image1_data = self.preprocess_image(image1)
                    image2_data = self.preprocess_image(image2)
                    Image1.append(image1_data)
                    Image2.append(image2_data)
                    Label.append(label)
                # Convert lists to numpy arrays
                Image1 = np.asarray(Image1)
                Image2 = np.asarray(Image2)
                Label = np.asarray(Label)
                # Yield batch data
                yield {"image1": Image1, "image2": Image2}, Label


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.44356Z","iopub.execute_input":"2025-11-08T20:50:42.444127Z","iopub.status.idle":"2025-11-08T20:50:42.455763Z","shell.execute_reply.started":"2025-11-08T20:50:42.444097Z","shell.execute_reply":"2025-11-08T20:50:42.455068Z"}}
# Instantiate a CDataLoader object for generating batches of test data
mytestgen = CDataLoader(mytest_dataset, 2, mytest_dir)


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.456901Z","iopub.execute_input":"2025-11-08T20:50:42.457136Z","iopub.status.idle":"2025-11-08T20:50:42.492617Z","shell.execute_reply.started":"2025-11-08T20:50:42.457117Z","shell.execute_reply":"2025-11-08T20:50:42.491965Z"}}
# Generate a data generator from the CDataLoader object for obtaining batches of test data
generator = mytestgen.datagen()

# Get the next batch of data and corresponding labels from the generator
batch_data, labels = next(generator)


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.493634Z","iopub.execute_input":"2025-11-08T20:50:42.493934Z","iopub.status.idle":"2025-11-08T20:50:42.498925Z","shell.execute_reply.started":"2025-11-08T20:50:42.493907Z","shell.execute_reply":"2025-11-08T20:50:42.497991Z"}}
print("Image1 data shape:", batch_data["image1"].shape)
print("Image2 data shape:", batch_data["image2"].shape)
print("Label data shape:", labels.shape)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.499881Z","iopub.execute_input":"2025-11-08T20:50:42.50014Z","iopub.status.idle":"2025-11-08T20:50:42.945343Z","shell.execute_reply.started":"2025-11-08T20:50:42.500107Z","shell.execute_reply":"2025-11-08T20:50:42.94448Z"}}
for i in range(len(batch_data["image1"])):
    image1 = batch_data["image1"][i]
    image2 = batch_data["image2"][i]
    label = labels[i]
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    plt.suptitle('Label: ' + str(label))
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:50:42.946377Z","iopub.execute_input":"2025-11-08T20:50:42.946627Z","iopub.status.idle":"2025-11-08T20:51:57.495647Z","shell.execute_reply.started":"2025-11-08T20:50:42.946605Z","shell.execute_reply":"2025-11-08T20:51:57.494678Z"}}
# Evaluate the model using the test data generator
# Steps parameter is set to the number of batches in the test dataset
mytest_loss, mytest_accuracy = siamese_model.evaluate(mytestgen.datagen(), steps=len(mytest_dataset) // 2)


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:51:57.496846Z","iopub.execute_input":"2025-11-08T20:51:57.497105Z","iopub.status.idle":"2025-11-08T20:51:57.501907Z","shell.execute_reply.started":"2025-11-08T20:51:57.497083Z","shell.execute_reply":"2025-11-08T20:51:57.501006Z"}}
print("Test Loss:", mytest_loss)
print("Test Accuracy:", mytest_accuracy)
# I messed up my labels in .csv file, the model is 100% accurate though

# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:51:57.505598Z","iopub.execute_input":"2025-11-08T20:51:57.505934Z","iopub.status.idle":"2025-11-08T20:51:57.538746Z","shell.execute_reply.started":"2025-11-08T20:51:57.505914Z","shell.execute_reply":"2025-11-08T20:51:57.537877Z"}}
# Generate a batch of test data and corresponding labels using the test data generator
test_pairs, test_labels = next(mytestgen.datagen())


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:51:57.53977Z","iopub.execute_input":"2025-11-08T20:51:57.540012Z","iopub.status.idle":"2025-11-08T20:51:58.02028Z","shell.execute_reply.started":"2025-11-08T20:51:57.539992Z","shell.execute_reply":"2025-11-08T20:51:58.019349Z"}}
# Predict labels for the test data pairs using the trained model
test_predictions = siamese_model.predict(test_pairs)

# Convert predicted probabilities to binary predictions using a threshold of 0.5
binary_predictions = (test_predictions > 0.5).astype(int)


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T20:51:58.021868Z","iopub.execute_input":"2025-11-08T20:51:58.022161Z","iopub.status.idle":"2025-11-08T20:51:58.458676Z","shell.execute_reply.started":"2025-11-08T20:51:58.022138Z","shell.execute_reply":"2025-11-08T20:51:58.45786Z"}}
import random

# Generate random indices to select pairs of images
random_indices = random.sample(range(len(test_pairs['image1'])), 2)

# Iterate over random indices to display image pairs
for i in random_indices:
    # Get image pair and corresponding labels
    image1 = test_pairs['image1'][i]
    image2 = test_pairs['image2'][i]
    true_label = test_labels[i]
    predicted_label = binary_predictions[i]
    
    # Print pair information
    print("Pair", i+1)
    print("True Label:", true_label)
    print("Predicted Label:", predicted_label)
    
    # Plot image pair
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1.squeeze(), cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2.squeeze(), cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-11-08T21:10:30.832343Z","iopub.execute_input":"2025-11-08T21:10:30.832692Z","iopub.status.idle":"2025-11-08T21:11:41.004202Z","shell.execute_reply.started":"2025-11-08T21:10:30.832666Z","shell.execute_reply":"2025-11-08T21:11:41.003203Z"}}
# ==================================================
# Code for Metrics and Confusion Matrix
# ==================================================

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

# Load the best model
siamese_model.load_weights("best_model.keras")

# Create test data generator
test_metrics_gen = CDataLoader(mytest_dataset, 256, mytest_dir)
test_generator = test_metrics_gen.datagen()

# Collect all predictions and true labels
all_predictions = []
all_true_labels = []

# Number of steps needed to cover all test data
steps = len(mytest_dataset) // 256

for i in range(steps):
    batch_data, batch_labels = next(test_generator)
    batch_predictions = siamese_model.predict(batch_data)
    binary_predictions = (batch_predictions > 0.5).astype(int)
    
    all_predictions.extend(binary_predictions.flatten())
    all_true_labels.extend(batch_labels)

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predictions)

# Calculate metrics
accuracy = accuracy_score(all_true_labels, all_predictions)
precision = precision_score(all_true_labels, all_predictions)
recall = recall_score(all_true_labels, all_predictions)
f1 = f1_score(all_true_labels, all_predictions)

# Print metrics
print("\n" + "="*50)
print("Comprehensive Metrics Results")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(all_true_labels, all_predictions, target_names=['Non-Matching (0)', 'Matching (1)']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Matching (0)', 'Matching (1)'], 
            yticklabels=['Non-Matching (0)', 'Matching (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.show()

# Calculate additional metrics
TN, FP, FN, TP = conf_matrix.ravel()

print("\n" + "="*50)
print("Detailed Confusion Matrix Analysis")
print("="*50)
print(f"True Negative (TN): {TN} - Correct non-matching predictions")
print(f"False Positive (FP): {FP} - Incorrect matching predictions")
print(f"False Negative (FN): {FN} - Incorrect non-matching predictions")
print(f"True Positive (TP): {TP} - Correct matching predictions")

# Calculate error rates
false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"\nFalse Positive Rate: {false_positive_rate:.4f}")
print(f"False Negative Rate: {false_negative_rate:.4f}")

# Calculate accuracy per class
class_0_accuracy = TN / (TN + FP) if (TN + FP) > 0 else 0
class_1_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"\nClass 0 Accuracy (Non-Matching): {class_0_accuracy:.4f}")
print(f"Class 1 Accuracy (Matching): {class_1_accuracy:.4f}")

# Plot metrics graph
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title('Model Performance Metrics')
plt.ylim(0, 1)
plt.ylabel('Value')

# Add values on bars
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom')

plt.grid(axis='y', alpha=0.3)
plt.show()

print("\n" + "="*50)
print("Metrics and Confusion Matrix Analysis Completed")
print("="*50)

# %% [code]
# ==================================================
# Embedding Extraction and Verification
# ==================================================

print("\n" + "="*50)
print("Embedding Extraction Verification")
print("="*50)

# Get a single image from the test set
batch_data, _ = next(mytestgen.datagen())
single_image = batch_data["image1"][0:1] # Keep batch dimension (1, 128, 128, 1)

# Get embedding using the standalone embedding_model
embedding = embedding_model.predict(single_image)

print(f"Input image shape: {single_image.shape}")
print(f"Generated embedding shape: {embedding.shape}")
print(f"First 10 values of embedding:\n{embedding[0][:10]}")

# Verification of weight sharing:
# Let's compare the output of embedding_model with the internal output of siamese_model
# We can create a temporary model that maps siamese_model inputs to the output of its base CNN

# The base CNN (embedding_model) is the 3rd layer in siamese_model (after 2 inputs)
# Or we can just check if the weights are literally the same objects
siamese_base_weights = siamese_model.get_layer("embedding_model").get_weights()
embedding_model_weights = embedding_model.get_weights()

weights_match = all(np.array_equal(w1, w2) for w1, w2 in zip(siamese_base_weights, embedding_model_weights))
print(f"\nWeight sharing verification: {'SUCCESS' if weights_match else 'FAILED'}")

if weights_match:
    print("The Embedding Model successfully shares weights with the Siamese Network.")
    print("You can now use `embedding_model.predict(image)` to get embeddings for your database.")

print("="*50)
