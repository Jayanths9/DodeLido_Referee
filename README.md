# DodeLido - A Deep Learning Project
### link to Dataset : [click here](https://drive.google.com/file/d/1neRs9rxjZBj3Mgrhrn2yiH2G3o-cbTla/view?usp=sharing)

### TODO
- [X] Capture images of DodeLido cards. 
- [x] Resize and label images.
- [ ] Augment images and create corresponding labels.
- [ ] Train neural network to detect cards one by one.
- [ ] Train neural network to detect three cards.
- [ ] Write Python logic to guess the outcome based on three cards, e.g. Blue, Lion, Dodelio etc.

### Detect Single card
- Images are resized to 256x256x3.
- Each card has two labels i.e. colour of the card & animal on the card.
- Alarm and Sloth cards are labelled as ('alarm, 'alarm') & ('sloth', 'sloth').
- Neural network has to be trained to classify the card on two labels.
