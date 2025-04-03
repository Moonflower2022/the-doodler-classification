# the-doodler

Classification models for the quick, draw! data created by google.

Includes a training file (`train_model.ipynb`), a loading file (`load_model.ipynb`), a visualization file (`visualization.py`), and a utilities file (`utils.py`) that is used in both the loading and training files.

best classification model is getting ~92% accuracy on a handpicked 95 class dataset

## todo
ways to improve classifier
* more random deep learning things to add to the model
* more data
* optimize class list more
  * remove similar stuff (duck, bird)
  * add unique stuff thats easy to classify

ways to improve interactive_classifier
* add info button to show
  * class list
  * tips for getting it to work
    * draw bigger (fill more of the allotted area)
    * look at what the training data looked like