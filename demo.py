# Importing library files

import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk
from ttkbootstrap.constants import *

from helper import *

# Defining the class names
class_names = ["Alarm", "Blue", "Flamingo", "Giraffe", "Green", "Grey", "Lion",
               "Monkey", "Pink", "Sloth", "Snake", "Yellow"]
# Defining the color for bounding boxes
color = (0, 255, 0)  # Green


def open_camera():
    """
     This function captures video frames from a camera, detects circles,
        classifies them using a model, and displays the results on a label.

        It performs the following steps:

        1. Captures a frame from the camera (assuming `cap` is a configured
           VideoCapture object).
        2. Converts the frame from BGR to grayscale for circle detection.
        3. Applies Gaussian blurring for noise reduction.
        4. Detects circles using Hough Circle Transform with specified
           parameters.
        5. Iterates over detected circles:
            - Isolates the Region of Interest (ROI) around each circle in
              color.
            - Preprocesses the ROI (resize, normalize) for the classification
              model.
            - Makes a prediction using the model and gets class probabilities.
            - Sorts predictions in descending order and retrieves top two
              classes.
            - Draws a bounding box around the detected circle.
            - Calculates the dominant color within the circle.
            - Displays the top two class names, probabilities, and dominant
              color on the frame.
        6. Updates a label with the combined class and color information.
        7. Calculates an output using the `calculate_dodelido_output` function
           (replace with your implementation).
        8. Updates another label with the `dodelido` output.
        9. Converts the frame to a PIL Image object.
        10. Creates a PhotoImage object from the Image.
        11. Updates the label widget with the new PhotoImage object.
        12. Schedules the function to be called again after 10 seconds for
            continuous processing.
        13. If no circles are detected, prints a message.
    """

    # _, frame = cap.read()   #Uncomment this to read from Live Camera
    frame = cap
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))

    # Circle detection in color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=200,  # Adjust based on expected circle density
        param1=120,  # Higher threshold for Canny edge detector
        param2=50,  # Lower accumulator threshold for circle detection
        minRadius=100,  # Adjust based on the smallest expected circle size
        maxRadius=170  # Adjust based on the largest expected circle size
    )

    if circles is not None:
        circles = circles.astype(np.int32)
        text_class = []
        text_color = []
        text_class_prob = []

        for i, (x, y, r) in enumerate(circles[0, :]):

            # Isolate the ROI in color
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)
            roi = frame[top_left[1]:bottom_right[1],
                  top_left[0]:bottom_right[0]]

            # Preprocess the ROI for the model (resize, normalize)
            roi = cv2.resize(roi, (
            224, 224))  # Assuming your model expects 224x224 input
            roi = roi.astype(
                'float32') / 255.0  # Normalize pixel values between 0-1

            # Expand dimensions for model input (batch size of 1)
            roi = np.expand_dims(roi, axis=0)

            # Make prediction using the model
            predictions = new_model.predict(roi)[
                0]  # Get probabilities for all classes

            # Sort predictions in descending order and get top two classes
            sorted_indices = np.argsort(predictions)[::-1][:2]
            top_two_classes = [class_names[i] for i in sorted_indices]

            # Draw bounding box
            cv2.rectangle(frame, top_left, bottom_right, color, 2)

            # Get dominant color within circle
            bgr_color = get_dominant_color(frame, r, y, x)

            # Display top two class names, probabilities, dominant color (BGR)
            y_offset = 20  # Adjust offset based on font size
            for j, (class_name, probability) in enumerate(
                    zip(top_two_classes, predictions[sorted_indices])):
                if probability > 0.5:
                    text_class1 = f"{class_name}: {probability:.2f}"
                    text_class2 = f"{class_name}"
                    text_color1 = f"BGR: ({int(bgr_color[0])}, " \
                                  f"{int(bgr_color[1])}, " \
                                  f"{int(bgr_color[2])})"
                    cv2.putText(frame, text_class1, (top_left[0], top_left[1]
                                                     - y_offset * (j + 1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                    text_class_prob.append(text_class1)
                    text_class.append(text_class2)
                    text_color.append(text_color1)

        # Update label text with both class and color information
        classes_color_all = []
        classes_color_all_prob = []

        for classes, prob in zip(text_class, text_class_prob):
            classes_color_all_prob.append(prob)
            classes_color_all.append(classes)

        classes_color_all_reshaped = [classes_color_all_prob[i:i + 2] for i in
                                      range(0, len(classes_color_all_prob), 2)]

        text_label.config(text=f"{classes_color_all_reshaped}")

        # ------------------------------------------------------------#

        output_dodelido = calculate_dodelido_output(classes_color_all)
        output_label.config(text=f"{output_dodelido}")

        # ------------------------------------------------------------#

        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(frame)

        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)

        # Displaying photoimage in the label
        label_widget.photo_image = photo_image

        # Configure image in the label
        label_widget.configure(image=photo_image, padx=10, pady=10)

        # Repeat the same process after every 10 seconds
        label_widget.after(10, open_camera)
    else:
        print("No Circles Detected")


def close_window():
    cv2.destroyAllWindows()
    app.destroy()  # Destroys the main window


# Load the pre-trained model
new_model = load_model('./Dataset/dodeLidoModel', custom_objects={
    'macro_soft_f1': macro_soft_f1, 'macro_f1': macro_f1})

# Load either camera image or saved image
cap = load_image(VideoCapture=False)

# Create a GUI app 
app = tk.Tk(className=" Dode Lido Referee") 
  
# Bind the app with Escape keyboard to 
# quit app whenever pressed 
app.bind('<Escape>', lambda e: app.quit()) 
app.geometry("1280x850")
  
photo = tk.PhotoImage(file='./Dataset/WelcomeScreen.png', master=app)
label_widget = Label(app, padx=10, pady= 10, image=photo)
label_widget.pack()
text_label = Label(app, text="Dode Lido Referee")
text_label.pack(side=BOTTOM, anchor="se", pady=5, padx=5)

# Create buttons with good font and padding
button_font = ("Roboto", 10)  # Adjust font as desired
button_width = 50

button1 = ttk.Button(app, text="Open Camera", command=open_camera,
                     width=button_width, bootstyle=SUCCESS, padding=5)
button2 = ttk.Button(app, text="Close Window", command=close_window,
                     width=button_width, bootstyle=(INFO, OUTLINE), padding=5)


# Create a label widget for displaying text
output_label = Label(app, text="Dode Lido Referee",
                     font=("Roboto", 14, "bold"), padx=10, pady=10,
                     width=25, bg="light cyan")
output_label.pack(side="right", padx=10, pady=10)

# Place buttons at the bottom, side-by-side
button2.pack(side=BOTTOM, anchor="sw", pady=5, padx=10)
button1.pack(side=BOTTOM, anchor="sw", pady=5, padx=10)

  
# Create an infinite loop for displaying app on screen 
app.mainloop()
# Release capture object and close all windows
cv2.destroyAllWindows()
