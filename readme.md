I will not spend much time on explaining YOLOV1 paper or architecture, it is quite thorougly covered by several articles online.
Some are here: 
Originbal Paper: https://arxiv.org/pdf/1506.02640

Other good sources: https://www.youtube.com/watch?v=9s_FpMpdYW8

                    https://www.youtube.com/watch?v=n9_XyCGr-MI

This repository is designed to help you quickly set up a simple YOLO model. It includes only 5 files:

1. main.py: This script serves as the entry point and is responsible for initializing the model, loss function, and data loader.

2. loss_fn.py: This file contains the implementation of the loss function. While I didn't write it myself, I utilized the implementation from this source: GitHub link.

3. models.py: Here, you'll find the model architecture built from scratch. It closely resembles the image shown in the original paper. The model itself is quite straightforward, with all the layers hardcoded to maintain simplicity.

4. utils.py: This file provides various helper functions, such as file reading and converting bounding box coordinates into the proper format. It's nothing extravagant, just some handy utilities.

5. 'debug' folder: Unzip the zip file. A sample folder with 10 images and labels to show you how the script expects the images and labels to be.


With these files, you'll be on your way to running a YOLO model in no time!

One vital thing to note is how I'm normalizing the ground truth bounding boxes before using them. Basically, I adjust the center coordinates of each bounding box to be an offset from the top-left coordinate of the grid it falls into. Also, I normalize the width and height of the bounding boxes with respect to the entire image. If I've misunderstood something, just let me know, and I'll make sure to address it.
