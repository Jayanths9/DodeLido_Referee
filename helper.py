import tensorflow as tf
import cv2
import numpy as np
import joblib

from collections import Counter

autotuner = tf.data.experimental.AUTOTUNE
shuffle_buffer_size = 1024
image_size = 224
channels = 3


def parse_function(filename, label):
    """Function that returns tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [image_size, image_size])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


def create_dataset(
        filenames, labels, image_dir, batchsize: int = 100,
        is_training: bool = True):
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
    dataset = dataset.map(parse_function, num_parallel_calls=autotuner)

    if is_training is True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Batch the data for multiple steps
    dataset = dataset.batch(batchsize)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=autotuner)

    return dataset


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
    circle_roi = frame[top_left[1]:bottom_right[1],
                       top_left[0]:bottom_right[0]]
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
    print(output_list)
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
            element_counts = []
        # Remove Sloth from list

    else:
        dode_output = ""

    element_counts.subtract(["Sloth"])   
    if len(element_counts) > 0:
        max_count = max(element_counts.values())

        max_value_elements = [element for element,
                              count in element_counts.items()
                              if count == max_count and
                              element_counts[element] == count]
    
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
    else:
        element = "Nothing"
    return dode_output + str(element)


def load_model_lite(litemodel: str, classifier: str):
    interpreter = tf.lite.Interpreter(model_path=str(litemodel))
    interpreter.allocate_tensors()
    mlb = joblib.load(classifier)
    return interpreter, mlb


def model_lite_predict(interpreter, image, classifier):
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    expanded_tensor = np.expand_dims(image_normalized, axis=0)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, expanded_tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)

    if len(predictions.shape) == 1:
        predictions = np.expand_dims(predictions, axis=0)
    class_indices = np.argsort(predictions[0])[::-1]
    confidence_scores = predictions[0][class_indices]

    predicted_classes = []
    for index in class_indices:
        class_name = classifier.classes_[index]
        predicted_classes.append(class_name)

    return predicted_classes, confidence_scores
