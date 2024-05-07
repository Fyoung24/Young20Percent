# FY 2024
#CS AT Fracture Detector
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential,Model
from pandas import read_csv
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Define the number of classes
NUM_CLASSES = 10  # Replace with the actual number of classes in your dataset

# Define the CNN architecture
model = Sequential([
    # Convolutional and pooling layers for feature extraction
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # Flatten to prepare for fully connected layers
    Flatten(),
    # Fully connected layers for segmentation
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Using softmax for multi-class segmentation
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Using categorical cross-entropy for multi-class segmentation
              metrics=['accuracy'])

# Print the model summary
model.summary()
train_csv_path = '/Users/fyoung24/Downloads/FracAtlas 2/Utilities/Fracture Split/train.csv'
valid_csv_path = '/Users/fyoung24/Downloads/FracAtlas 2/Utilities/Fracture Split/valid.csv'
test_csv_path = '/Users/fyoung24/Downloads/FracAtlas 2/Utilities/Fracture Split/test.csv'
# Read the CSV files containing the data paths
# Read the CSV files
train_data_paths = pd.read_csv(train_csv_path)
val_data_paths = pd.read_csv(valid_csv_path)
test_data_paths = pd.read_csv(test_csv_path)

# Print the first few rows of each DataFrame
print("Train data:")
print(train_data_paths.head())
print("\nValidation data:")
print(val_data_paths.head())
print("\nTest data:")
print(test_data_paths.head())
try:
    train_data_paths = pd.read_csv(train_csv_path)
    val_data_paths = pd.read_csv(valid_csv_path)
    test_data_paths = pd.read_csv(test_csv_path)
except FileNotFoundError:
    print("One or more CSV files not found. Please check the file paths.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the CSV files. Please check the file format.")
    exit()

# Set the current directory to the location of the CSV files
os.chdir(os.path.dirname(os.path.abspath('/Users/fyoung24/Downloads/FracAtlas 2/Utilities/Fracture Split/train.csv')))

# Extract the file paths from the CSV files
train_data_dir = train_data_paths['image_id'][0]
val_data_dir = val_data_paths['image_id'][0]
test_data_dir = test_data_paths['image_id'][0]

# Data Augmentation
# Access the class label column by index

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                               rotation_range=20,
                                                               width_shift_range=0.2,
                                                               height_shift_range=0.2,
                                                               horizontal_flip=True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create the data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data_paths,
    x_col='file_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data_paths,
    x_col='file_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_data_paths,
    x_col='file_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                               rotation_range=20,
                                                               width_shift_range=0.2,
                                                               height_shift_range=0.2,
                                                               horizontal_flip=True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create the data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = val_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Model Selection
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Model Training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator)

# Evaluation
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)