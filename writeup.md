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

[image1]: ./examples/All_Label_Images.png "All Label Images"
[image2]: ./examples/images_from_web.png "Images from web"
[image3]: ./examples/image_transformations.png "Image Transformations"
[image4]: ./examples/Label_count_After_augmentation.png "Label Count After Augmentation"
[image5]: ./examples/Label_count_before_augmentation.png "Label Count Before Augmentation"
[image6]: ./examples/normalization.png "Normalization"
[image7]: ./examples/valid_sample_dist.png "Valid Sample Distribution"
[image8]: ./examples/test_sample_dist.png "Test Sample Distribution"
[image9]: ./examples/validation_acc.png "Validation Accuracy by Epoch"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic signs data set:

* The size of training set before Augmentation = 34799
* The size of the validation set = 4410
* The size of test set = 12630
* The shape of a traffic sign image = (32x32x3)
* The number of unique classes/labels in the data set = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

We have 43 different traffic signs. Below is a view of one image per traffic sign in the dataset

![alt text][image1]

The distribution of labels in each of the datasets is shown below.

![image5]

![image7]

![image8]

It can be observed that there are many classes where the sample size is too small. To give equal chance to all the labels I decided to increase the samples using image augmentation techniques. This increased the sample size by 11681 images to make sure that each of the label type has atleast 800 samples. I chose this number arbitarily.

I used Scaling, Translation and Rotation for augmentation. An example is shown below.

![image3]

The size of training set after Augmentation is - 46480

The distribution of the training set after augmentation is shown below. 

![image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It is observed that neural networks work better if the input distribution have a mean close to zero. To do this as a first step, I decided to convert the images to grayscale. After that I normalized the data by dividing each of the input dataset by 255 so that its range is between 0-1.

Here is an example of a traffic sign image before and after preprocessing.

![image6]

Also it is observed that the training dataset is sorted by image label. To not give the model any bias I shuffled the training dataset before using it.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The starting model was LeNet by Udacity. The model worked well with recognition of hand written / print digits. So i started with this model as the base. First I modified this to take the 3 layer RGB input image instead of the 1 layer grayscale image in the MNIST dataset. Also I modified the output classes from 10 to 43 as per the traffic sign recognition dataset.

With this model where the EPOCH were 10 and a BATCH_SIZE of 128 my accuracy was between 85-90 %

To improved this I started making the convolutions deeper and increase the size of the fully connected layer. With these changes my validation set accuracy was 93%. To increase the accuracy further I added two dropout layers with 0.5 keep probability and increased the training epochs to 20. 

My final model consisted of the following layers:

|Layer | Description|Output|
|------|------------|------|
|Input | RGB image| 32x32x3|
|Convolutional Layer 1 | 1x1 strides, valid padding | 28x28x16|
|RELU| | |
|Max Pool| 2x2 | 14x14x16|
|Convolutional Layer 2 | 1x1 strides, valid padding | 10x10x64|
|RELU| | |
|Max Pool | 2x2 | 5x5x64|
|Fatten| To connect to fully-connected layers |
|Fully-connected Layer 1| | 1600|
|RELU| | |
|Dropout| 0.5 keep probability ||
|Fully-connected Layer 2| | 240
|RELU| | |
|Dropout| 0.5 keep probability||
|Fully-connected Layer 3| | 43
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a learning rate of 0.001. I started with 10 epochs and after few trial runs settled at 20 epochs with a batch size of 128. 

The validation accuracy vs the epoch count can be seen below. 

![image9]



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy: 99.5%
* Validation set accuracy: 96.2%
* Test set accuracy: 94.6%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have chosen TEN German traffic signs that I found on the web:

![alt text][image2]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

After preprocessing is applied I loaded the model that was used on the test set and used it to classify the above images. 


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| Turn left ahead   			| Turn left ahead						|
| Speed limit (30km/h)					| Speed limit (30km/h)											|
| Stop	      		| Stop					 				|
| Keep right			| Keep right      							|
| Yield			| Yield      							|
| General caution			| General caution      							|
| Speed limit (60km/h)			| Speed limit (60km/h)      							|
| Bumpy road			| Bumpy road      							|
| Road work			| Road narrows on the right      							|


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 94.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 32nd cell of the Ipython notebook.

### Image 1
|Actual Image Label: No entry|  |
|:-------------------------------|------------------------------|
|Top 5 predictions with their probability percentage:|  |
|Prediction 1 - No entry                                 | 100%|
|Prediction 2 - Keep left                                | 0%|
|Prediction 3 - Stop                                     | 0%|
|Prediction 4 - Keep right                               | 0%|
|Prediction 5 - Go straight or right                     | 0%|
|Prediction Matching! | |
### Image 2

|Actual Image Label: Turn left ahead||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Turn left ahead                          | 100%|
|Prediction 2 - Keep right                               | 0%|
|Prediction 3 - Stop                                     | 0%|
|Prediction 4 - Keep left                                | 0%|
|Prediction 5 - Turn right ahead                         | 0%|
|Prediction Matching!||
### Image 3
|Actual Image Label: Speed limit (30km/h)||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Speed limit (30km/h)                     | 100%
|Prediction 2 - Speed limit (20km/h)                     | 0%
|Prediction 3 - Speed limit (50km/h)                     | 0%
|Prediction 4 - End of speed limit (80km/h)              | 0%
|Prediction 5 - Speed limit (70km/h)                     | 0%
|Prediction Matching!
### Image 4
|Actual Image Label: Stop ||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Stop                                     | 100%
|Prediction 2 - Keep right                               | 0%
|Prediction 3 - Turn left ahead                          | 0%
|Prediction 4 - Keep left                                | 0%
|Prediction 5 - Speed limit (60km/h)                     | 0%
|Prediction Matching!
### Image 5
|Actual Image Label: Keep right||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Keep right                               | 100%
|Prediction 2 - Keep left                                | 0%
|Prediction 3 - Yield                                    | 0%
|Prediction 4 - Turn left ahead                          | 0%
|Prediction 5 - Stop                                     | 0%
|Prediction Matching!
### Image 6
|Actual Image Label: Yield||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Yield                                    | 100%
|Prediction 2 - No vehicles                              | 0%
|Prediction 3 - Keep left                                | 0%
|Prediction 4 - Bicycles crossing                        | 0%
|Prediction 5 - Bumpy road                               | 0%
|Prediction Matching!
### Image 7
|Actual Image Label: General caution||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - General caution                          | 100%
|Prediction 2 - Traffic signals                          | 0%
|Prediction 3 - Pedestrians                              | 0%
|Prediction 4 - Dangerous curve to the right             | 0%
|Prediction 5 - Keep left                                | 0%
|Prediction Matching!
### Image 8
|Actual Image Label: Speed limit (60km/h)||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Speed limit (60km/h)                     | 100%
|Prediction 2 - Speed limit (50km/h)                     | 0%
|Prediction 3 - No passing                               | 0%
|Prediction 4 - No passing for vehicles over 3.5 metric tons | 0%
|Prediction 5 - Speed limit (80km/h)                     | 0%
|Prediction Matching!
### Image 9
|Actual Image Label: Bumpy road||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Bumpy road                               | 100%
|Prediction 2 - Bicycles crossing                        | 0%
|Prediction 3 - Children crossing                        | 0%
|Prediction 4 - Dangerous curve to the right             | 0%
|Prediction 5 - Keep left                                | 0%
|Prediction Matching!
### Image 10
|Actual Image Label: Road work||
|----------------------------|-----------------------------------|
|Top 5 predictions with their probability percentage:||
|Prediction 1 - Road narrows on the right                | 100%|
|Prediction 2 - Bicycles crossing                        | 0%|
|Prediction 3 - Traffic signals                          | 0%|
|Prediction 4 - Bumpy road                               | 0%|
|Prediction 5 - Speed limit (20km/h)                     | 0%|
|Prediction Not Matching!||


## Questions:

Its been a wonderful project to get introduction to many topics such as Neural Nets, Deep Learning, Tensor Flow etc.,

But I leave the project with the following questions in mind. 

1.  When I train my model from scratch, accuracy differs by 1-2% each time. The interesting thing is that the accuracy of the images downloaded from web also varies sometimes. What is causing this variation even thought the training set and the model architecture are same each time?

    I believe the randomization of weights and shuffling of data can cause this. Are there any other factors that effect the accuracy?

2. For a self driving car, dont we need 100% accuracy in determining anything. As there are lives at stake? 

    My thoughts are that multiple algorithms run together to back each others desicion to arrive at a 100% accuracy before taking any decisions. 

3. In this CNN we have taken cropped images in a specific size and resolution to determine the traffic sign. But in the real world we might have to process these signs in real time from the video frames like the lane finding projects. 

    Will be looking forward to learn techniques on doing the same from a video feed instead of static images. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


