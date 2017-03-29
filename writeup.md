**Traffic Sign Recognition** 
Please find the code in 'Traffic_Sign_Classifier.ipynb' file as part of this repo. The outputs of all the cells can be found in the 'Traffic_Sign_Classifier.html' file. I am also including the test images which are in the 'german_traffic_signs' directory.

**Build a Traffic Sign Recognition Project**
The goal of this project was to classify images of various German traffic signs.
The following steps were taken:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize various layers of the neural network to gain insight
* Summarize the results with a written report


##Data Set Summary & Exploration
A separate training, validation and test datasets were provided by Udacity as pickle files. Each dataset contained images and labels. Below is a summary of the size and shape of the dataset:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (for the 3 RGB channels)
* The number of unique classes/labels in the data set is 43

Investigating the number of samples in the training set for each of the 43 labels, it became apparent that not all classes are represented equally in the training sample. This could introduce a bias for the network as it will see certain types of traffic signs more often than others. In order to fix this issue, in the next section, I augmented the training set by adding modified images to the classes which were under-represented. The first step to achieve this was to identify the number of images that are needed to be added per class in order to arrive at an equally represented sample. For example label 2 (speed limit 50km/h) is the most represented class, with 2010 examples in the training set. However label 13 (Yield)  has only 1920 samples in the training set. So in order to have equal samples in both case, I need to add an extra 90 Yield sign examples. This is done by randomly taking 90 exisiting Yield images and modify them one way or another and adding them to the training pool. Below I explain the different modification methods that were used for this purpose. 


In order to augment the training dataset, I used ? differnet approaches:
 * Adding Gaussian Noise
 * Converting to gray scale 
 * Adding blurr
 * Contrast enhancement
 * Translation (shifting the image by a small amount in x and y direction)

The result of each of these functions is a normalized image, so I also normalize the original training set before combining them all together into a large set.

For each image that needs to be added to the training sample, I then randomly choose one the above agumentation methods.
The original training set contained 34799 images, and the augmented set contains 51631 images, for a total of 86430 training samples which are evenly distributed accross the 43 classes.
Next, I compute the mean of all images in the training set and subtract that from each image in the training and valdation (and later test dataset).
Finally I shuffle the training dataset and have it ready to be fed to the network.

##Design and Test a Model Architecture

The model architecture I used is LeNet which consist of the following layers:
1. Input layer which is the 32x32x3 image
2. Convolution layer (5x5) with stride of 1x1  and output depth 6
3. Relu activation
4. Max Pooling layer (2X2) with stride 2x2
5. Convolution layer (5x5) with stride 1x1 and output depth 16
6. Relu activation
7. Max Pooling layer (2x2) with stride 2x2
8. and finally two fully connected layers

As for the hyper parameters, after some trial and error, I setteled for the following values:
learning rate = 0.001
batch size = 64
number of ephocs = 24 (since I achive 93% validation accuracy earlier than 24)
The accuracy values for each ephoc for both training and validation sets are shown as output in the notebook.
My best training accuracy is 0.995 and highest validation accuracy is 0.953 which were achieved at epoch 21.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
