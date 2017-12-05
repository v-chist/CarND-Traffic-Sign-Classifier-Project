# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[before_normalize1]: ./illustrations/before_normalize1.jpg "Image 1 before normalization"
[before_normalize2]: ./illustrations/before_normalize2.jpg "Image 2 before normalization"
[after_normalize1]: ./illustrations/after_normalize1.jpg "Image 1 before normalization"
[after_normalize2]: ./illustrations/after_normalize2.jpg "Image 2 before normalization"
[new_sign1]: ./illustrations/1.jpg "New sign 1"
[new_sign2]: ./illustrations/2.jpg "New sign 2"
[new_sign3]: ./illustrations/3.jpg "New sign 3"
[new_sign4]: ./illustrations/4.jpg "New sign 4"
[new_sign5]: ./illustrations/5.jpg "New sign 5"
[ahead]: ./illustrations/ahead.jpg "Ahead"
[train_dist]: ./illustrations/train_dist.jpg "Sign distribution in train set"
[valid_dist]: ./illustrations/valid.jpg "Sign distribution in validation set"
[test_dist]: ./illustrations/test_dist.jpg "Sign distribution in test set"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/v-chist/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Here are bar charts showing how the data is distributed in training, validation and test datasets:


![alt text][train_dist]

![alt text][valid_dist]

![alt text][test_dist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

I used cv2.normalize function using min-max normalization, since images in the dataset have different brightness. I didn`t convert images in grayscale, since I supposed that color of traffic sign can give additional information to neural network.

Here are two images before normalization:

![alt text][before_normalize1] 
![alt text][before_normalize2] 

Here are these images after normalization:


![alt text][after_normalize1] 
![alt text][after_normalize2] 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (slightly modified LeNet architechture adopted to output of 43 classes, inout of 3 layer RGB image and incresed number of output layers on convolutional operation):


| Layer         		|     Description	                			|
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolutional     	| 5x5 stride, valid padding, output = 28x28x18 	|
| RELU Activation		|												|
| Max pooling	      	| Input = 28x28x18. Output = 14x14x18			|
| Convolution   	    | 5x5 stride, valid padding, output = 10x10x24  |
| RELU Activation		|												|
| Max pooling	      	| Input = 10x10x24. Output = 5x5x24 			|
| Flatten   	      	| Input = 5x5x24. Output = 600      			|
| Fully connected		| Input = 600. Output = 200 					|
| RELU Activation		|												|
| Fully connected		| Input = 200. Output = 86 			    		|
| RELU Activation		|												|
| Fully connected		| Input = 86. Output = 43  				    	|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

Type of optimizer: AdamOptimizer 

Number of epochs = 9

Batch size = 128

Learning rate = 0.0008

Parameters used for initial weights in LeNet function:

	mu = 0
	sigma = 0.05


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.935 
* test set accuracy of 0.927

First architecture that was chosen was Lenet architecture since it is widely used for analysis and classification of images.

Initially i converted images into grayscaled to use original Lenet architecture. However this model didn`t give needed accuracy on validation set, thus i have chosen to use rgb images, increase the number of output layer of convolution operation and normalize initial images.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new_sign1] ![new_sign2] ![alt text][new_sign3] 
![alt text][new_sign4] ![alt text][new_sign5]

The "speed limit 70" image may be hard to classify because there are many different traffic sign types with speed limits.

"Staraight ahead" sign may be difficult to classify, since shape of the arrow is slightly different from the shape in test set:

![alt text][new_sign2] ![alt text][ahead]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Model was correct for all of the traffic signs (accuracy 100%). However for "speed limit 70" it was very close to make a mistake and consider it "speed limit 30". This approximately matches with the accuracy 0.927 on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Prediction was successful for all images, softmax probability is close to 1 for the correct answer for all images. Most probably it is caused by quite good quality of the images taken from the web. The most difficult sign to classify: "speed limit 70"

Softmax probabilities:

Top 5 predictions for **"speed limit 70"**:

    1) Speed limit 70				 Probability: 0.973
    2) Speed limit 20				 Probability: 0.027
    3) Speed limit 30				 Probability: ~1e-06
    4) General caution				 Probability: ~1e-10
    5) Speed limit 120				 Probability: ~1e-11
    
Top 5 predictions for **"straight ahead":**

    1) Straignt ahead				 Probability: 1
    2) Go straight or right			 Probability: ~1.e-12
    3) Turn left ahead			 	 Probability: ~1.e-13
    4) Turn right ahead				 Probability: ~1.e-14
    5) Go straight or left			 Probability: ~1.e-16
    
Top 5 predictions for **"general caution"**:

    1) General caution							 Probability: 1
    2) Pedestrians 								 Probability: ~1.e-9
    3) Traffic signals							 Probability: ~1.e-13
    4) Right-of-way at the next intersection	 Probability: ~1.e-15
    5) Road narrows on the right				 Probability: ~1.e-17
    
Top 5 predictions for **"yield"**:

    1) Yield						 				Probability: 1
    2) Double curve					 				Probability: ~1.e-11
    3) Road work					 				Probability: ~1.e-12
    4) Speed limit (50km/h)			 				Probability: ~1.e-14
    5) No passing for vehicles over 3.5 metric tons	Probability: ~1.e-14
 
    
Top 5 predictions for **"children crossing"**:

    1) Children crossing			 Probability: 1
    2) Dangerous curve to the right	 Probability: ~1.e-6
    3) Beware of ice/snow 			 Probability: ~1.e-8
    4) Road narrows on the right	 Probability: ~1.e-8
    5) Speed limit (120km/h)		 Probability: ~1.e-8






