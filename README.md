# Dodelido_openv
### link to Dataset : [click here](https://drive.google.com/file/d/1t_2G9JKH61lNrjjcBCieCkRvlT0-tz4p/view?usp=sharing)

### TODO
- [X] Capture images of dodelido cards. 
- [x] Resize and label images.
- [X] Augmented images and create coressponding labels.
- [ ] Train neural network to detect single single card.
- [ ] Train neural network to detect three cards.
- [ ] Write python logic to guess the outcome based on three cards, e.g Blue, Lion , Dodelio etc.

### Detect Single card
- Images are reseized to 256x256x3.
- Each card has two labels i.e color of card & animal in the card.
- Alarm and Soth cards are labels as ('alaram','alaram') & ('sloth','sloth').
- Neural network has to be trained to classify the card on two labels.
