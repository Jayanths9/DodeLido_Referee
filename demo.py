# functions.py
import argparse
import cv2
from helper import *
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import *
from PIL import Image, ImageTk
from ttkbootstrap.constants import *


class_names = ["Alarm", "Blue", "Flamingo", "Giraffe", "Green", "Grey", "Lion",
               "Monkey", "Pink", "Sloth", "Snake", "Yellow"]
color = (0, 255, 0)  # Green
stop_camera = False


def close_window():
    cv2.destroyAllWindows()
    app.destroy()  # Destroys the main window


def close_window_video(video_capture, app):
    video_capture.release()
    app.quit()


def image_input(model, classifier, frame):
    interpreter, mlb = load_model_lite(model, classifier)
    # frame = cv2.imread(image)
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    results = []
    buffer = 0
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = circles.astype(int)
        min_dist_between_circles = 500  # Adjust this value as needed
        filtered_circles = filter_overlapping_circles(circles, min_dist_between_circles)
        text_class = []
        text_color = []
        text_class_prob = []


        # Loop over all detected circles
        for circle in filtered_circles:

            # Extract the coordinates and radius of the circle
            x, y, r = circle

            # Draw the bounding box around the circle
            x1 = x - r-buffer
            y1 = y - r-buffer
            x2 = x + r+buffer
            y2 = y + r+buffer

            # crop out the card image
            roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            roi = cv2.resize(roi, (224, 224))  # Assuming your model expects 224x224 input

            # predit the class of cropped card image
            class_names, confidence_scores = model_lite_predict(interpreter=interpreter, image=roi, classifier=mlb)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2,y2), color, 2)
            
            # Sort predictions in descending order and get top two classes
            top_two_classes = class_names[:2]
            top_two_confidences = confidence_scores[:2]


            # Get dominant color within circle
            bgr_color = get_dominant_color(roi, r, y, x)

            # Display top two class names, probabilities, dominant color (BGR)
            y_offset = 20  # Adjust offset based on font size
            for j, (class_name, confidence) in enumerate(zip(top_two_classes, top_two_confidences)):
                if confidence > 0.5:
                    text_class1 = f"{class_name}: {confidence:.2f}"
                    text_class2 = f"{class_name}"
                    text_color1 = f"BGR: ({int(bgr_color[0])}, {int(bgr_color[1])}, {int(bgr_color[2])})"
                    cv2.putText(frame, text_class1, (x1, y1 - y_offset * (j + 1)),
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

        classes_color_all_reshaped = [classes_color_all_prob[i:i + 2] for i in range(0, len(classes_color_all_prob), 2)]
        text_label.config(text=f"{classes_color_all_reshaped}")
        # ------------------------------------------------------------#

        output_dodelido = calculate_dodelido_output(classes_color_all)
        output_label.config(text=f"{output_dodelido}")

        # ------------------------------------------------------------#

        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(frame)

        captured_image = captured_image.resize((1280, 720))

        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)

        # Displaying photoimage in the label
        label_widget.photo_image = photo_image

        # Configure image in the label
        label_widget.configure(image=photo_image, padx=10, pady=10)

    else:
        print("No Circles Detected")


def camera_input(model, classifier, cap):
    ret, frame = cap.read()
    interpreter, mlb = load_model_lite(model, classifier)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use Hough Circle Transform to detect circles
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=150)

    results = []
    buffer = 10
        
    if circles is not None:
        print("Loop Entered")
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = circles.astype(int)
        min_dist_between_circles = 100  # Adjust this value as needed
        filtered_circles = filter_overlapping_circles(circles, min_dist_between_circles)
        text_class = []
        text_color = []
        text_class_prob = []
                
        # Loop over all detected circles
        for circle in filtered_circles:

            # Extract the coordinates and radius of the circle
            x, y, r = circle

            # Draw the bounding box around the circle
            x1 = x - r-buffer
            y1 = y - r-buffer
            x2 = x + r+buffer
            y2 = y + r+buffer

            # crop out the card image
            new_image = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            # predit the class of cropped card image
            class_names, confidence_scores = model_lite_predict(interpreter=interpreter, image=new_image, classifier=mlb)

            # Draw the bounding box on the original image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Sort predictions in descending order and get top two classes
            top_two_classes = class_names[:2]
            top_two_confidences = confidence_scores[:2]

            # Get dominant color within circle
            bgr_color = get_dominant_color(new_image, r, y, x)

            # Display top two class names, probabilities, dominant color (BGR)
            y_offset = 20  # Adjust offset based on font size
            for j, (class_name, confidence) in enumerate(zip(top_two_classes, top_two_confidences)):
                if confidence > 0.3:
                    text_class1 = f"{class_name}: {confidence:.2f}"
                    text_class2 = f"{class_name}"
                    text_color1 = f"BGR: ({int(bgr_color[0])}, {int(bgr_color[1])}, {int(bgr_color[2])})"
                    cv2.putText(frame, text_class1, (x1, y1 - y_offset * (j + 1)),
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
        # ------------------------------------------------------------#
        print(classes_color_all)
        output_dodelido = calculate_dodelido_output(classes_color_all)
        output_label.config(text=f"{output_dodelido}")

        # ------------------------------------------------------------#
        # Capture the latest frame and transform to 

    else:
        text_label.config(text="No circles detected")
    
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))

    captured_image = Image.fromarray(frame)
    captured_image = captured_image.resize((1280, 720))

    # Convert captured image to photoimage
    photo_image = ImageTk.PhotoImage(image=captured_image)

    # Displaying photoimage in the label
    label_widget.photo_image = photo_image

    # Configure image in the label
    label_widget.configure(image=photo_image, padx=10, pady=10)
    label_widget.after(10, camera_input, model, classifier, cap)
   
                 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a specific function with arguments.")
    subparsers = parser.add_subparsers(dest="function", help="Function to run")

    # Subparser for camera
    parser_camera = subparsers.add_parser("camera", help="Run with camera input")
    parser_camera.add_argument("--model", type=str, default="./Dataset/trained_model.tflite",help="Path to trained model for image function")
    parser_camera.add_argument("--classifier", type=str, default="./Dataset/classifier.pkl", help="Path to classifier for image function")

    # Subparser for image
    parser_image = subparsers.add_parser("image", help="Run on local image")
    parser_image.add_argument("--model", type=str, default="./Dataset/trained_model.tflite",help="Path to trained model for image function")
    parser_image.add_argument("--classifier", type=str, default="./Dataset/classifier.pkl", help="Path to classifier for image function")
    parser_image.add_argument("--imagepath", type=str, default="./Dataset/trialImage3.jpg", help="Path to image for image function")

    args = parser.parse_args()

    if args.function == "camera":
        # Create a GUI app
        app = tk.Tk(className=" Dode Lido Referee")
        app.bind('<Escape>', lambda e: app.quit())
        app.geometry("1280x850")

        photo = tk.PhotoImage(file='./Dataset/WelcomeScreen.png',
                              master=app)
        label_widget = Label(app, padx=10, pady=10, image=photo)
        label_widget.pack()
        text_label = Label(app, text="Dode Lido Referee")
        text_label.pack(side=BOTTOM, anchor="se", pady=5, padx=5)

        # Create buttons with good font and padding
        button_font = ("Roboto", 10)  # Adjust font as desired
        button_width = 50

        
        def input_camera_call(): 
            cap= cv2.VideoCapture(0)     
            return camera_input(args.model, args.classifier, cap)
        
        def input_camera_close():
            cap= cv2.VideoCapture(0)
            return close_window_video(cap, app)


        button1 = ttk.Button(app, text="Open Camera", command=input_camera_call,
                             width=button_width, bootstyle=SUCCESS,
                             padding=5)
        button2 = ttk.Button(app, text="Close Window",
                             command=input_camera_close,
                             width=button_width, bootstyle=(INFO, OUTLINE),
                             padding=5)

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

        # camera_input(args.model, args.classifier)

    elif args.function == "image":
        if args.imagepath:
            # Create a GUI app
            app = tk.Tk(className=" Dode Lido Referee")
            app.bind('<Escape>', lambda e: app.quit())
            app.geometry("1280x850")

            photo = tk.PhotoImage(file='./Dataset/WelcomeScreen.png',
                                  master=app)
            label_widget = Label(app, padx=10, pady=10, image=photo)
            label_widget.pack()
            text_label = Label(app, text="Dode Lido Referee")
            text_label.pack(side=BOTTOM, anchor="se", pady=5, padx=5)

            # Create buttons with good font and padding
            button_font = ("Roboto", 10)  # Adjust font as desired
            button_width = 50

            def input_image_call():
                image = cv2.imread(args.imagepath)
                return image_input(args.model, args.classifier, image)

            button1 = ttk.Button(app, text="Open Camera",
                                 command=input_image_call,
                                 width=button_width, bootstyle=SUCCESS,
                                 padding=5)
            button2 = ttk.Button(app, text="Close Window",
                                 command=close_window,
                                 width=button_width, bootstyle=(INFO, OUTLINE),
                                 padding=5)

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
    else:
        print("Invalid function argument")
