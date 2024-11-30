import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths
data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\dataset_blood_group'
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the dataset
batch_size = 32  # Adjust as per your hardware capabilities
img_size = (96, 103)  # Match image dimensions in the dataset

# Create dataset and extract class names
raw_test_dataset = image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False  # Keep order for accurate label matching
)

# Save class names (should be ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'])
class_names = raw_test_dataset.class_names
print(f"Class names: {class_names}")

# Convert to grayscale and normalize
def convert_to_grayscale(images, labels):
    """Converts images to grayscale."""
    grayscale_images = tf.image.rgb_to_grayscale(images)
    return grayscale_images, labels

test_dataset = raw_test_dataset
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Get true labels and predictions
true_labels = np.concatenate([y.numpy() for _, y in test_dataset])
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))

# Save the confusion matrix to a file
output_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\confusion_matrix.png'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Blood Group Detection')
plt.savefig(output_path)
plt.close()

print(f"Confusion matrix saved to {output_path}")
