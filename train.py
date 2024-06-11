<a href="https://colab.research.google.com/github/Jayanths9/Dodelido_opencv/blob/main/Jay_Single_Image_Train.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# Importing Library Files
import albumentations as A
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import joblib
import json
#from google.colab.patches import cv2_imshow
image_dir="./Dataset/Images/"
data = pd.read_json("./Dataset/labels.json").T

data = data.sort_index(ascending=True)
data = data.reset_index()

data["animal_color"] = data.apply(lambda row: [row[0], row[1]], axis=1)
data['labels'] = list(zip(data[0], data[1]))

# Splitting the dataset

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    data['index'], data['animal_color'], test_size=0.2, random_state=44)

print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))
# The targets should be a list of list of strings to fit a binarizer (multi-hot encoding).

y_train = list(y_train)
y_val = list(y_val)

print(y_val[:4])


from sklearn.preprocessing import MultiLabelBinarizer

# Fit the multi-label binarizer on the training set
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
# transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Print example of images and their binary targets
for i in range(3):
    print(X_train[i], y_train_bin[i])
import tensorflow as tf

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 256 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations

def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # filename=image_dir+filename
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label

def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    filenames=image_dir+filenames
    

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (x, y, print(x[0], flush=True)))

        

    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
train_ds1 = create_dataset(X_train, y_train_bin)
val_ds1 = create_dataset(X_val, y_val_bin)


# Each batch will be a pair of arrays (one that holds the features and another one that holds the labels).
# The features array will be of shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS).
# The labels array will be of shape (BATCH_SIZE, N_LABELS) where N_LABELS is the maximum number of labels.


import tensorflow_hub as hub
from keras import layers

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
feature_extractor_layer.trainable = False
model = tf.keras.Sequential(
    [feature_extractor_layer,
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dense(N_LABELS, activation='sigmoid', name='output')])

model.summary()
@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1
LR = 1e-4 # Keep it small when transfer learning
EPOCHS = 100
PATIENCE=5
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', patience = PATIENCE)
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_f1])
# Train the model
history = model.fit(train_ds1,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val_bin,
                                                   is_training=False),callbacks = [early_stop])
# Check Keras version (should be 2.10.x or higher)
from tensorflow.keras import __version__
print(f"Keras version: {__version__}")

# Define your custom metrics (macro_soft_f1 and macro_f1)

# Assuming you have a model trained with these custom metrics
tf.keras.models.save_model('./Dataset/dodeLidoModel', custom_objects={'macro_soft_f1': macro_soft_f1, 'macro_f1': macro_f1})