import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import time

# Directory paths
data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\dataset_blood_group'
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'  # Replace with your model's path

# Updated image dimensions and settings
img_width, img_height = 96, 103
batch_size = 32
num_classes = 8  # 8 blood group classes (A+, A-, B+, B-, AB+, AB-, O+, O-)
target_epochs = 50  # Total number of epochs you want to train

# Set up data augmentation and automatic train/test split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of data for validation
)

# Creating training and validation generators
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # Train set
    color_mode='grayscale'  # Load images in grayscale for .bmp format
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Validation set
    color_mode='grayscale'  # Load images in grayscale for .bmp format
)

# Build the model (if not loaded from a checkpoint)
def build_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (11, 11), activation='relu', input_shape=(img_width, img_height, 1)))  # Grayscale images
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Load or build the model
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded model from checkpoint.")
else:
    model = build_model()
    print("Initialized a new model.")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for saving the model and managing training
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Training function with time limit (1 hour at a time)
def train_with_time_limit():
    start_time = time.time()
    elapsed_time = 0
    current_epoch = 0
    while current_epoch < target_epochs:
        # Fit the model for the next epoch
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
        elapsed_time = time.time() - start_time  # Update elapsed time
        print(f"Completed epoch {current_epoch}/{target_epochs}. Elapsed time: {elapsed_time:.2f} seconds.")

        # Check if 1 hour has passed
        if elapsed_time >= 3600:
            print("1 hour of training completed. Saving the model...")
            model.save(model_path)  # Save the model
            break

# Run training with checkpointing every hour
train_with_time_limit()
