## **Traffic Sign Recognition** 
Please find the code in 'Traffic_Sign_Classifier.ipynb' file as part of this repo. The outputs of all the cells can be found in the 'Traffic_Sign_Classifier.html' file. I am also including the test images which are in the 'german_traffic_signs' directory.

## **Build a Traffic Sign Recognition Project**
The goal of this project was to classify images of various German traffic signs.
The following steps were taken:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize various layers of the neural network to gain insight
* Summarize the results with a written report


### Data Set Summary & Exploration
A separate training, validation and test datasets were provided by Udacity as pickle files. Each dataset contained images and labels. Below is a summary of the size and shape of the dataset:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (for the 3 RGB channels)
* The number of unique classes/labels in the data set is 43

Investigating the number of samples in the training set for each of the 43 labels, it became apparent that not all classes are represented equally in the training sample. This could introduce a bias for the network as it will see certain types of traffic signs more often than others. In order to fix this issue, in the next section, I augmented the training set by adding modified images to the classes which were under-represented. The first step to achieve this was to identify the number of images that are needed to be added per class in order to arrive at an equally represented sample. For example label 2 (speed limit 50km/h) is the most represented class, with 2010 examples in the training set. However label 13 (Yield)  has only 1920 samples in the training set. So in order to have equal samples in both case, I need to add an extra 90 Yield sign examples. This is done by randomly taking 90 exisiting Yield images and modify them one way or another and adding them to the training pool. Below I explain the different modification methods that were used for this purpose. 


In order to augment the training dataset, I used 5 differnet approaches:
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

### Design and Test a Model Architecture

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
My best training accuracy is 0.993 and highest validation accuracy is 0.944 which were achieved at epoch 18.
I used LeNet archtecture because I thought the problem at hand is very similar to the alphabet classification that we have worked on earlier in the course. And from the accuracy obtained on the test images I believe that it was a good choice. As for the hyper parameters, I started with a higher learning rate and larger batch size but did not achive as good a result as with the values listed above.

### Test a Model on New Images
As for testing I found several images for German traffic signs on the web. I downloaded them and cropped them to square shapes and then loaded and resized them to 32x32 as the rest of the images for this network.
I then normalzied and subtracted the mean of the training data set from them. The before and after of this process is visialuzed in the notebook for all 7 images. The test data includes the following signs: No entry, Road work, General caution, Roundabout mandatory, Yield, Speed Limit (120km/h) and Wild animals crossing.

After using the trained network to predict the relevant class, the result show that for 5 out of 7 cases the prediction is correct. In case of the Speed Limit (120km/h) image, the prediction is Speed limit (20km/h) which is interesting. At least the network figured that the sign is a speed limit sign, and had trouble identiying the actual value and the correct class has the second highest probability.
In case of the Wild animals crossing sign, the network fails and predict the sign to be Dangerous curve to the right. In this case again the correct class has the second highest probability.

Following this I used the optional visualization section to visualize various stages of the netwrok and in the case of this image, it seems that the network is successful in picking up the triangular shape of the sign but almost doesn't identify anything interesting in the center. Inspecting the test image in this case carefully, shows a spiral watermark overlaid the sign and I wonder if that is the cause of confusion for the network.

As a final step I plot the top 5 softmax probabilities for each of the test images. I have chosen a log scale to show the other lower probabilities better, since the network is very confident with the predictions every time and the highest class is much more probable than the other four. 

I have also tested the accuracy of the predictions on the test dataset which shows to be 88.65% which is higher than the test image examples I found on the web with accuracy of 71.43%. The occasions that the model preformed suboptimally on new images can be contributed in one case to watermark on the image. Also comparing results on 5(7 in my case) arbitrary images is not a good indication of statistics.
