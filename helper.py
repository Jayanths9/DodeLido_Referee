import tensorflow as tf
import cv2
import numpy as np

from collections import Counter

AUTOTUNE = tf.data.experimental.AUTOTUNE  # Adapt preprocessing
SHUFFLE_BUFFER_SIZE = 1024  # Shuffle the training data by a chunck of 1024.

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across
    all labels). Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation
                                of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations
    (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of
                                shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def parse_function(filename, label, channels, img_size):
    """Function that returns a tuple of normalized image array and labels
    array.

    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # filename=image_dir+filename
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [img_size,  img_size])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


def create_dataset(filenames, labels, channels,
                   img_size, img_dir, batch_size, is_training=True):
    """Load and parse dataset.

    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    filenames = img_dir + filenames

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels,
                                                  channels, img_size))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    if is_training is True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(batch_size)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def read_image(image_object, img_size, model, mlb, threshold):
    image_resized = tf.image.resize(image_object,[img_size, img_size])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    expanded_tensor = tf.expand_dims(image_normalized, axis=0)
    ouptut = model.predict(expanded_tensor)
    result_list = list(ouptut)
    binary_array = np.where(np.array(result_list) > threshold, 1, 0)
    result = mlb.inverse_transform(binary_array)
    print(result)


def filter_overlapping_circles(circles, min_dist_between_circles):
    filtered_circles = []
    for circle in circles[0]:
        x, y, r = circle
        overlap = False
        for fc in filtered_circles:
            fx, fy, fr = fc
            distance = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if distance < min_dist_between_circles:
                overlap = True
                break
        if not overlap:
            filtered_circles.append(circle)

    return filtered_circles


def load_image(VideoCapture=False):
    """Returns either saved image or real time camera image
    if VideoCapture is True.

        Args:
            VideoCapture: Boolean; True/False

        Returns:
            cap: [w x h x n] N-dimensional image
        """

    if VideoCapture == True:
       image = cv2.VideoCapture(0, cv2.CAP_DSHOW)

       if not cap.isOpened():
          print("Error opening camera")
          exit()
    else:
       image = cv2.imread('./Dataset/trialImage3.JPG')
       image = cv2.resize(image, (1280, 720))

    return image


def get_dominant_color(frame, x, y, r):
    """
    Calculates the average color (BGR) within a circle region.

    Args:
        frame: The original color frame.
        x, y, r: Circle center coordinates and radius.

    Returns:
        A tuple containing the average BGR color values.
    """
    top_left = (x - r, y - r)
    bottom_right = (x + r, y + r)
    circle_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cv2.mean(circle_roi)


def calculate_dodelido_output(output_list):
    """
    Calculates the output of Dodelido game.

    Args:
        output_list: 3x2 n-D Array. Predicted outputs from the model.

    Returns:
        Output: String value containing the output if dodelido game.
    """
    element_counts = Counter(output_list)

    if "Alarm" in output_list:
        return "Alarm"

    sloth_count = element_counts["Sloth"]

    if sloth_count > 0:

        if sloth_count == 1:
            dode_output = "Oh-"
        elif sloth_count == 2:
            dode_output = "Oh-Oh-"
        elif sloth_count == 3:
            dode_output = "Oh-Oh-Oh-"
        # Remove Sloth from list

    else:
        dode_output = ""

    element_counts.subtract(["Sloth"])
    max_count = max(element_counts.values())

    # Find the element(s) with the maximum count (considering first instance)
    max_value_elements = [element for element, count in element_counts.items()
                          if count == max_count and element_counts[
                              element] == count]

    # Print the maximum count and element(s) (if there are multiple)
    if max_value_elements:

        if len(max_value_elements) > 1:
            if max_count >= len(max_value_elements):
                element = "DODELIDO"
            elif max_count == 1:
                element = "Nothing"
            else:
                element = max_value_elements
        else:
            element = max_value_elements[0]

        print(f"Element: {element} (Maximum count: {max_count})")

    else:
        element = "Nothing"
        print("No elements were repeated")

    return dode_output + str(element)