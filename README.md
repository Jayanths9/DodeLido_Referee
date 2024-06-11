# DodeLido Referee- A Deep Learning Project

DodeLido is an amazing card game that keeps you on your toes. The game is simple, each player will put one card in turn in a triangle format and then the player has to tell what is the majority, either the colour or the animal and in case there is a clash, you have to say DodeLido. Simple! And there is sloth and alarm also. 

![gitImage](https://github.com/Jayanths9/Dodelido_opencv/assets/9052405/d8003dd2-108d-4b64-97fd-904f702442a0)


Link to the card game: [Schmidt Spiele GmbH](https://www.dreimagier.de/details/produkt/dodelido-extreme.html)

P.S. Yes, we have merged the basic DodeLido game with the DodeLido Extreme version because we like it that way!

## Suggestions: 
- We are open to suggestions or any bugs that you find. You can contact us through the Issues tab in this repository. 


## Copyright:
- We do not own any copyright or make money from this project. We just love the game and decided to build a personal project around it. 


Till then DodeLido !!!!!

------

### Link to Dataset : [Click Here](https://drive.google.com/file/d/1t_2G9JKH61lNrjjcBCieCkRvlT0-tz4p/view?usp=sharing)

### TODO
- [X] Capture images of DodeLido cards. 
- [x] Resize and label images.
- [X] Augmented images and create coressponding labels.
- [X] Train neural network to detect single single card.
- [X] Write python logic to guess the outcome based on three cards, e.g Blue, Lion , Dodelio etc.
- [ ] Close Window Button not working 


### Detect Single card
- Images are resized to 256x256x3.
- Each card has two labels i.e. colour of the card & animal on the card.
- Alarm and Sloth cards are labelled as ('alarm, 'alarm') & ('sloth', 'sloth').
- Neural network has to be trained to classify the card on two labels.
