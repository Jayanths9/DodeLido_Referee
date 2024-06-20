import cv2
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras import layers
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from helper import *
import pathlib

seed = 99
np.random.seed(seed)
tf.random.set_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Configure the script parameters.")
    parser.add_argument("--label", type=str, default="./resources/labels.json", help="Path to the labels file.")
    parser.add_argument("--image", type=str, default="./resources/Images/", help="Directory containing the images.")
    parser.add_argument("--batchsize", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for training.")
    args = parser.parse_args()

    label_path = args.label
    image_dir = args.image
    BATCH_SIZE = args.batchsize
    PATIENCE = args.patience
    LR = args.lr

    # load labels and image names
    data = pd.read_json(label_path).T
    data = data.reset_index()
    data["labels"] = list(zip(data[0], data[1]))

    X_train, X_temp, y_train, y_temp = train_test_split(
        data["index"], data["labels"], test_size=0.2, random_state=seed
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )

    print("Number of training images: ", len(X_train))
    print("Number of test images: ", len(X_test))
    print("Number of validation images: ", len(X_val))

    # Fit the multi-label binarizer on the training set
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.fit_transform(y_test)
    y_val_bin = mlb.fit_transform(y_val)

    # number of unique labels
    N_LABELS = len(mlb.classes_)

    # Save the MultiLabelBinarizer
    with open('./resources/classifier.pkl', 'wb') as f:
        joblib.dump(mlb, f)  # Use joblib.dump for this code

    print("MultiLabelBinarizer saved as classifier.pkl")


    # create dataset
    train = create_dataset(
        X_train, y_train_bin, image_dir=image_dir, batchsize=BATCH_SIZE
    )
    test = create_dataset(
        X_test, y_test_bin, image_dir=image_dir, batchsize=len(y_test_bin)
    )
    val = create_dataset(X_val, y_val_bin, image_dir=image_dir, batchsize=BATCH_SIZE)

    # import mobilenetv-2 model
    feature_extractor_url = (
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    )
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_url, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)
    )
    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential(
        [
            feature_extractor_layer,
            layers.Dense(1024, activation="relu", name="hidden_layer"),
            layers.Dense(N_LABELS, name="output_layer"),
        ]
    )
    #print model
    model.summary()

    EPOCHS = 25

    bce_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=0.5, axis=-1, reduction="sum_over_batch_size"
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=bce_loss, metrics=[bce_loss]
    )
    # Train the model
    model.fit(train, epochs=EPOCHS, validation_data=val, callbacks=[early_stop])
    # save the trained model
    model.save('./Dataset/trained_model.keras')
    print(f"Trained model saved as /trained_model.keras")

    # quantize model and save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("./Dataset")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"trained_model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    print(f"Quantized model saved as trained_model.tflite")

    # test model accuracy
    for data in test:
        images=data[0]
        labels=data[1].numpy().astype(bool)
    predictions=tf.sigmoid(model.predict(images))
    threshold=0.5
    predictions_bin = np.where(np.array(predictions) > threshold, 1, 0)
    accuracy=accuracy_score(labels, predictions_bin)
    print(f"Model Accuracy : {accuracy*100}%")

if __name__ == "__main__":
    main()