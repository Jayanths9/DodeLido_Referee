# DodeLido Referee- A Deep Learning Project

DodeLido is an amazing card game that keeps you on your toes. The game is simple, each player will put one card in turn in a triangle format and then the player has to call out what is the majority, either the colour or the animal and in case there is a clash, the player has to call DodeLido. The player who finishes all the cards in their hand, Wins! Simple! And there is sloth and alarm also. 

<p align="center">
  <img src="https://github.com/Jayanths9/Dodelido_opencv/assets/9052405/d8003dd2-108d-4b64-97fd-904f702442a0" width="500">
</p>

Link to the card game: [Schmidt Spiele GmbH](https://www.dreimagier.de/details/produkt/dodelido-extreme.html)

P.S. Yes, we have merged the basic DodeLido game with the DodeLido Extreme version because we like it that way!

## Installation
1. Download the repository either as a zip and extract it to a folder or use the below command:
   ```
   git clone https://github.com/Jayanths9/DodeLido_Referee.git
   ```
2. Install the required dependencies (for Linux machines):
   ```
   cd DodeLido_Referee
   ./install_dependencies.sh
   ```
3. To use an actual camera, run the command:
   ```
   python demo.py camera 
   ```
4. To use default image, run the command:
   ```
   python demo.py image --imagepath "./resources/trialImage3.jpg" 
   ```
## Implementation

### Dataset Preparation:
  <p align="center">
  <img src="https://github.com/Jayanths9/DodeLido_Referee/assets/9052405/b8e37230-2b1a-4104-b406-f4dc64405166" width="500">
  </p>
To prepare the dataset, we took images of all the cards and used OpenCV to extract each image, followed by augmentations. 
  
### Training:
1. **Data preparation:** We loaded labels and image data, split them into training, validation and testing sets, and performed multi-label binarization to convert textual labels into binary vectors for the model.
3. **Dataset creation:** We created functions to define training, validation and testing datasets that load images and their corresponding labels in batches during training.
4. **Model definition:** We imported a pre-trained mobilenetv2 model as a feature extractor and froze its weights. We then added custom dense layers on top for classification, defining the overall model architecture.
5. **Model training:** We compiled the model with an Adam optimizer, a custom binary cross-entropy loss function with label smoothing, and early stopping for regularization. We trained the model on the training dataset and monitored performance on the validation set.
6. **Model evaluation and saving:** After training, we evaluated the model's accuracy on the testing set and saved the trained model (both Keras and TFLite formats) for future use.

### Working
1. **Function Selection:** We created two functions, camera_input and image_input, to handle processing for live camera feeds and static images, respectively. Both rely on helper functions like load_model_lite to handle loading the TensorFlow Lite model and classifier.
2. **Circle Detection:**  Within both functions, we implemented circle detection using OpenCV's cv2.HoughCircles function. This helps us isolate potential objects of interest in the frame or image.
3. **Preprocessing and Prediction:**  For each detected circle, we crop the region of interest (ROI) around it.  We then pre-process the ROI by resizing it and converting it to the format expected by our model.  Finally, we use the loaded TFLite model to make predictions on the preprocessed ROI.
4. **Result Display and Text Update:** After predictions are complete, we draw a bounding box around the detected circle and display the predicted class names, confidence scores, and the dominant colour of the circle on the frame.  We use OpenCV's text rendering functions for this purpose.
5. **GUI Integration:**  We built a basic graphical user interface (GUI) with Tkinter. This GUI provides buttons to start and stop the camera feed, displays the processed video with detections, and allows the user to close the application window.

### Output
Below you can find predictions from the trained model for a few images:
  <p align="center">
  <img src="https://github.com/Jayanths9/DodeLido_Referee/assets/9052405/1c972f70-9a59-4770-8415-9d666caa5d30" width="600">
  </p>


## Suggestions: 
- We are open to suggestions or any bugs that you find. You can contact us through the Issues tab in this repository. 


## Copyright:
- We do not own any copyright or make money from this project. We just love the game and decided to build a personal project around it. 


Till then DodeLido !!!!!
