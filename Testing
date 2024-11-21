import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Directory paths
data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\dataset_blood_group'
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'
cwd = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection'

# Image dimensions and settings
img_width, img_height = 96, 103
batch_size = 32
num_classes = 8  # Blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
target_epochs = 110

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
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (7, 7), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

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

# Function to plot training/validation accuracy and loss
def plot_accuracy_loss(history):
    if not history.history:
        print("No training history found to plot.")
        return

    train_acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    if not train_acc or not val_acc or not train_loss or not val_loss:
        print("Incomplete training history; unable to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(cwd + r'\accuracy_loss_plot.png')
    plt.close()
    print("Accuracy and loss plot saved as 'accuracy_loss_plot.png'.")

# Function to visualize feature maps
def visualize_feature_maps(model, layer_name, image):
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    if not model.built:
        _ = model.predict(image)

    try:
        layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        feature_maps = layer_model.predict(image)

        num_filters = feature_maps.shape[-1]
        plt.figure(figsize=(15, 15))
        for i in range(min(num_filters, 64)):
            plt.subplot(8, 8, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')

        plt.savefig('feature_maps.png')
        plt.close()
        print("Feature maps visualization saved as 'feature_maps.png'.")
    except Exception as e:
        print(f"Error during feature map visualization: {e}")

# Function to train with a time limit
def train_with_time_limit():
    global history
    start_time = time.time()
    elapsed_time = 0
    current_epoch = 0
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size ,
        epochs=target_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, reduce_lr, early_stopping],
        verbose=1
    )
    current_epoch += 1
    elapsed_time = time.time() - start_time
    print(f"Completed epoch {current_epoch}/{target_epochs}. Elapsed time: {elapsed_time:.2f} seconds.")
    plot_accuracy_loss(history)

# Function to train until interrupted
def train_until_interrupted():
    start_time = time.time()
    current_epoch = 0
    try:
        train_with_time_limit()
    except KeyboardInterrupt:
        print("Training interrupted. Saving the final model...")
    finally:
        model.save(model_path)
        print("Model saved. Exiting training.")
        if 'history' in globals() and history is not None:
            plot_accuracy_loss(history)
        try:
            sample_image, _ = next(validation_generator)
            sample_image = sample_image[0]
            if sample_image.shape[-1] != 1:
                sample_image = np.expand_dims(sample_image[:, :, 0], axis=-1)
            visualize_feature_maps(model, 'conv2d', sample_image)
        except Exception as e:
            print(f"Error during feature map visualization: {e}")

train_until_interrupted()
