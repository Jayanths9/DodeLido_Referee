# functions.py
import argparse
import cv2
from utils import *

def image_input(model, classifier,imagepath):
    interpreter,mlb=load_model_lite(model,classifier)
    image=cv2.imread(imagepath)
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

    results=[]
    buffer=20
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = circles.astype(int)
        min_dist_between_circles = 500  # Adjust this value as needed
        filtered_circles = filter_overlapping_circles(circles, min_dist_between_circles)

        # Loop over all detected circles
        for circle in filtered_circles:

            # Extract the coordinates and radius of the circle
            x, y, r = circle

            # Draw the bounding box around the circle
            x1 = max(0, x - r-buffer)
            y1 = max(0, y - r-buffer)
            x2 = min(image.shape[1], x + r+buffer)
            y2 = min(image.shape[0], y + r+buffer)

            # Draw the bounding box on the original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # crop out the card image
            new_image= cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            # predit the class of cropped card image
            result=model_lite_predict(interpreter=interpreter, image=new_image, classifier=mlb)
            results.append(result)

            # add image labels for each cards
            cv2.putText(image, str(result), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2)
        # use the labels each cards to predict the dodelido
        output=str(calculate_dodelido_output(sum(results, [])))
        # add the dodelido results on the image
        cv2.putText(image,output, (int(image.shape[0]/3),int(image.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0) , 5)
    # display the image
    show_image(image)

def camera_input(model, classifier):
    interpreter,mlb=load_model_lite(model,classifier)
    video_capture = cv2.VideoCapture(0)
    while True:
        # read image from camera
        success, image = video_capture.read()
        # if input from camera
        if success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use Hough Circle Transform to detect circles
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,         # Adjust based on expected circle density
                param1=100,         # Higher threshold for Canny edge detector
                param2=30,          # Lower accumulator threshold for circle detection
                minRadius=10,       # Adjust based on the smallest expected circle size
                maxRadius=150       # Adjust based on the largest expected circle size
            )

            results=[]
            buffer=10
            if circles is not None:
                # Convert the (x, y) coordinates and radius of the circles to integers
                circles = circles.astype(int)
                min_dist_between_circles = 100  # Adjust this value as needed
                filtered_circles = filter_overlapping_circles(circles, min_dist_between_circles)

                # Loop over all detected circles
                for circle in filtered_circles:

                    # Extract the coordinates and radius of the circle
                    x, y, r = circle

                    # Draw the bounding box around the circle
                    x1 = max(0, x - r-buffer)
                    y1 = max(0, y - r-buffer)
                    x2 = min(image.shape[1], x + r+buffer)
                    y2 = min(image.shape[0], y + r+buffer)

                    # Draw the bounding box qon the original image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    # crop out the card image
                    new_image= cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

                    # predit the class of cropped card image
                    result=model_lite_predict(interpreter=interpreter, image=new_image, classifier=mlb)
                    results.append(result)

                    # add image labels for each cards
                    cv2.putText(image, str(result), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)

            output=str(calculate_dodelido_output(sum(results, [])))
            # use the labels each cards to predict the dodelido
            cv2.putText(image,output, (int(image.shape[0]/2),int(image.shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0) , 2)
            # display the image
            cv2.imshow("Cards Image", image)
            # press q to exit the cv2 window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a specific function with arguments.")
    subparsers = parser.add_subparsers(dest="function", help="Function to run")

    # Subparser for camera
    parser_camera = subparsers.add_parser("camera", help="Run with camera input")
    parser_camera.add_argument("--model", type=str,default="trained_model.tflite",help="Path to trained model for image function")
    parser_camera.add_argument("--classifier", type=str,default="classifier.pkl", help="Path to classifier for image function")

    # Subparser for image
    parser_image = subparsers.add_parser("image", help="Run on local image")
    parser_image.add_argument("--model", type=str,default="trained_model.tflite",help="Path to trained model for image function")
    parser_image.add_argument("--classifier", type=str,default="classifier.pkl", help="Path to classifier for image function")
    parser_image.add_argument("--imagepath", type=str, help="Path to image for image function")

    args = parser.parse_args()

    if args.function == "camera":
        camera_input(args.model, args.classifier)
    elif args.function == "image":
        if args.imagepath:
            image_input(args.model, args.classifier, args.imagepath)
        else:
            parser_image.error("--imagepath is required for 'image' function.")
