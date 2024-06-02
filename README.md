# DodeLido - A Deep Learning Project
### link to Dataset : [click here](https://drive.google.com/file/d/1t_2G9JKH61lNrjjcBCieCkRvlT0-tz4p/view?usp=sharing)


### TODO
- [X] Capture images of DodeLido cards. 
- [x] Resize and label images.
- [X] Augmented images and create coressponding labels.
- [X] Train neural network to detect single single card.
- [ ] Write python logic to guess the outcome based on three cards, e.g Blue, Lion , Dodelio etc.


### Detect Single card
- Images are resized to 256x256x3.
- Each card has two labels i.e. colour of the card & animal on the card.
- Alarm and Sloth cards are labelled as ('alarm, 'alarm') & ('sloth', 'sloth').
- Neural network has to be trained to classify the card on two labels.
