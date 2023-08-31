# Transfer-Learning-With-TensorFlow-Object-Detection-API
This Repo shows you how you can fine-tune an object detection model from TensorFlow's Object Detection API using only code, and no terminal commands. 
This repo is only applicable for datasets with only 1 class (only one object to detect).

## Usage 
The object_detection.py file has the code you need to train an object detection model from the Object Detection API collection. You can use any model you prefer, ( I used the SSD MobileNet v2 320x320 model), and all you need is a folder of all your training and testing images, and two CSV files specifying the order in which the training data set and the testing data set should be opened with and the ground truth bounding boxes and classes in the object_detection.py file. 



### CSV Format
**filename, image_width, image_height, class, xmin, ymin, xmax, ymax**
image_1
image_2
image_3
.
.
.
image_n
