import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import time

# Directory paths
data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\dataset_blood_group'
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'

# Image dimensions and settings
img_width, img_height = 96, 103
batch_size = 32
num_classes = 8  # Blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
target_epochs = 50

# Function to initialize data generators
def initialize():
    global train_datagen, train_generator, validation_generator

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.3,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Split for training/validation
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        color_mode='grayscale'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        color_mode='grayscale'
    )

# Initialize data generators
initialize()

# Function to build the model
def build_model():
    model = Sequential([
        layers.Input(shape=(img_width, img_height, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Load or build the model
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded model from checkpoint.")
else:
    model = build_model()
    print("Initialized a new model.")

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for efficient training
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Function to train the model for a specified number of epochs
def train_with_time_limit():
    start_time = time.time()
    elapsed_time = 0
    current_epoch = 0
    while current_epoch < target_epochs:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=current_epoch + 1,
            initial_epoch=current_epoch,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=[checkpoint, reduce_lr, early_stopping],
            verbose=1
        )
        current_epoch += 1
        elapsed_time = time.time() - start_time
        print(f"Completed epoch {current_epoch}/{target_epochs}. Elapsed time: {elapsed_time:.2f} seconds.")

# Function to train until interrupted by the user
def train_until_interrupted():
    start_time = time.time()
    current_epoch = 0
    try:
        while True:
            train_with_time_limit()
            elapsed_time = time.time() - start_time
            print(f"Completed epoch {current_epoch}/{target_epochs}. Elapsed time: {elapsed_time:.2f} seconds.")
            model.save(model_path)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the final model...")
        model.save(model_path)
        print("Model saved. Exiting training.")

train_until_interrupted()
