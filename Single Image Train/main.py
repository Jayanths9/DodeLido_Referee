# Importing Library Files
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers
from helper import *


"""

Dodelido Training Single Image

Workflow:
1. Data collection
2. Data preparation
3. Create a fast input pipeline in TensorFlow
4. Build up the model
5. Stack a multi-label neural network classifier on top
6. Model training and evaluation
7. Understand the role of macro soft-F1 loss
8. Export and save tf.keras models

"""

# Model Parameters
IMG_SIZE = 224  # Specify height and width of image
CHANNELS = 3  # Keep RGB color channels
BATCH_SIZE = 256  # Big enough to measure an F1-score
LR = 1e-4  # Keep it small when transfer learning
EPOCHS = 200
PATIENCE = 5

dir_path = ""
image_dir = "/Dataset/"
feature_extractor_url = \
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
images = []
buffer = 20

# Reading the JSON file
data = pd.read_json(f"{dir_path}/labels.json").T
data = data.sort_index(ascending=True)
data = data.reset_index()
data["animal_color"] = data.apply(lambda row: [row[0], row[1]], axis=1)
data['labels'] = list(zip(data[0], data[1]))

# Splitting the dataset into Test and Train
X_train, X_val, y_train, y_val = train_test_split(
    data['index'], data['animal_color'], test_size=0.2, random_state=44)

print("Number of images for training: ", len(X_train))
print("Number of images for validation: ", len(X_val))

# The targets should be a list of strings to fit a
# binarizer (multi-hot encoding).
y_train = list(y_train)
y_val = list(y_val)

# Fit the multi-label binarizer on the training set
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Loop over all labels and print them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# Transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Create dataset
train_ds1 = create_dataset(X_train, y_train_bin, CHANNELS,
                           IMG_SIZE, image_dir, BATCH_SIZE)
val_ds1 = create_dataset(X_val, y_val_bin, CHANNELS,
                         IMG_SIZE, image_dir, BATCH_SIZE)

# Extract Features and define the model
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,
                                                      IMG_SIZE,
                                                      CHANNELS))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential(
    [feature_extractor_layer,
     layers.Dense(1024, activation='relu', name='hidden_layer'),
     layers.Dense(N_LABELS, activation='sigmoid', name='output')])

model.summary()

early_stop = EarlyStopping(monitor = 'val_loss', patience=PATIENCE)
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_f1])

# Train the model
history = model.fit(train_ds1,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val_bin,
                                                   CHANNELS, IMG_SIZE,
                                                   image_dir, BATCH_SIZE,
                                                   is_training=False),
                    callbacks=[early_stop])

# Save the model
tf.saved_model.save(model, f'{dir_path}/latest_model.h5')

# Evaluating the output from model
X_val, Y_val = val_ds1
images = X_val[0]
labels = np.array(X_val[1])
predictions = np.array(model.predict(images))

# Define the threshold for predictions
threshold = 0.5
# Convert values to binary array based on the threshold
binary_array = np.where(np.array(predictions) > threshold, 1, 0)
predicted_labels = mlb.inverse_transform(binary_array)
actual_labels = mlb.inverse_transform(labels)
print("Predicted Labels:", predicted_labels)

tf.saved_model.save(model, f'{dir_path}/latest_model.h5')
path = "/content/drive/MyDrive/dodelido/three/IMG_1998.JPG"
cv2_image = cv2.imread(path)

path = "/content/drive/MyDrive/dodelido/three/IMG_2017.JPG"
target_width = 1024
target_height = 1024
image = cv2.imread(path)
# image = cv2.resize(cv2_image, (target_width, target_height))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Use Hough Circle Transform to detect circles
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=500,         # Adjust based on expected circle density
    param1=100,         # Higher threshold for Canny edge detector
    param2=30,          # Lower accumulator threshold for circle detection
    minRadius=100,       # Adjust based on the smallest expected circle size
    maxRadius=450       # Adjust based on the largest expected circle size
)

if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = circles.astype(int)
    min_dist_between_circles = 500  # Adjust this value as needed
    filtered_circles = filter_overlapping_circles(circles,
                                                  min_dist_between_circles)

    # Loop over all detected circles
    for circle in filtered_circles:
        # Extract the coordinates and radius of the circle
        x, y, r = circle
        # Draw the bounding box around the circle
        x1 = max(0, x - r - buffer)
        y1 = max(0, y - r - buffer)
        x2 = min(image.shape[1], x + r + buffer)
        y2 = min(image.shape[0], y + r + buffer)

        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the bounding box on the original image
        new_image = image[y1:y2, x1:x2]
        images.append(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

for i in images:
    read_image(i, IMG_SIZE, model, mlb, threshold)
