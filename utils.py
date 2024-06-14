import tensorflow as tf
import tensorflow_hub as hub
import joblib
import cv2
import numpy as np
from collections import Counter


IMG_SIZE = 224  # image size input for mobilenet
CHANNELS = 3  # RGB channels
AUTOTUNER = (
    tf.data.experimental.AUTOTUNE
)  # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024  # Shuffle the training data


def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


def create_dataset(
    filenames, labels, image_dir, batchsize: int = 100, is_training: bool = True
):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    filenames = image_dir + filenames

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNER)

    if is_training is True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(batchsize)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNER)

    return dataset


def show_image(image, msg: str = "Loaded Image"):
    target_width, target_height = 1024, 1024
    image_copy = image.copy()
    image_resize = cv2.resize(image_copy, (target_width, target_height))
    cv2.imshow(msg, image_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_dodelido_output(output_list):
    """
    Calculates the output of Dodelido game.

    Args:
        output_list: 3x2 n-D Array. Predicted outputs from the model.

    Returns:
        Output: String value containing the output if dodelido game.
    """
    element_counts = Counter(output_list)
    if "alarm" in output_list:
        return "Alarm"

    sloth_count = element_counts["sloth"]

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

    element_counts.subtract(["sloth"])
    max_count = max(element_counts.values())

    # Find the element(s) with the maximum count (considering first instance)
    max_value_elements = [
        element
        for element, count in element_counts.items()
        if count == max_count and element_counts[element] == count
    ]

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

    else:
        element = "Nothing"

    return dode_output + str(element)


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


def load_model_lite(litemodel: str,classifier:str):
    interpreter = tf.lite.Interpreter(model_path=str(litemodel))
    interpreter.allocate_tensors()
    mlb=joblib.load(classifier)
    return interpreter , mlb

def load_model(model_path: str,classifier:str):
    custom_objects = {'KerasLayer': hub.KerasLayer}
    new_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    mlb=joblib.load(classifier)
    return new_model , mlb


def model_predict(model, image, classifier, threshold: float = 0.5):
    image_resized = tf.image.resize(image, [224, 224])
    image_normalized = image_resized / 255.0
    expanded_tensor = tf.expand_dims(image_normalized, axis=0)
    output = tf.sigmoid(model.predict(expanded_tensor,verbose=0))
    binary_array = np.where(np.array(output) > threshold, 1, 0)
    result = classifier.inverse_transform(binary_array)
    return list(result[0])



def model_lite_predict(interpreter, image, classifier, threshold: float = 0.5):
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    expanded_tensor = np.expand_dims(image_normalized, axis=0)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, expanded_tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    output = tf.sigmoid(predictions)
    binary_array = np.where(np.array(output) > threshold, 1, 0)
    result = classifier.inverse_transform(binary_array)
    return list(result[0])
