# Lymphoma-subtype-classification-
We implemented several CNN'S in order to distinguish among 3 classes of lymphoma, using keras.


Our implementation is based on different .py modules:
\\-Main.py: this is the main of our work. It imports the other scripts as a libraries and call all methods needed from here.
\\-Visualize_images: Small script that allow us to visualize images and patches, defining two methods based on MatplotLib.
\\-Process_images: From here we are able to collect images from the dataset and make all the preprocessing that we need in order to build our algorithms(Divide in patches, rescale images, split the dataset for training)
\\-4 modules NN_HDA.py: This is the core of our work, where the CNN's are defined, compiled and trained, saved as .h5 files. In the final part of this module you can also find a method for make a prediction novel datas.

\\All modules are divided into methods that implement specific operations. We chose to do everything in this way in order to facilitate reusability and readability of the code.

All rights reserved @Davide Colussi, Filippo Canderle.
