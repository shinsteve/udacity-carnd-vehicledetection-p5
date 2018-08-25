
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./example_images/car_notcar.png
[image2]: ./example_images/hog_car_notcar.png
[image3]: ./example_images/slide_window.png
[image4]: ./example_images/test3_found.jpg
[image5]: ./example_images/test4_found.jpg
[image6]: ./example_images/test4_heatmap.jpg
[image7]: ./example_images/test4_filtered.jpg
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in [train_classifier.py](train_classifier.py) and [feature_tools.py](feature_tools.py).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and selected the values with which the learning perfomance was best. 

| Parameter           | Value   | 
|:-------------------:|:-------------:| 
| color_space         | YCrCb        | 
| HOG orientations    | 9      |
| HOG pixels per cell | 8      |
| HOG cells per block | 2       |
| HOG channel         | ALL    |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in [train_classifier.py](train_classifier.py).

##### Data set

The data set I used for training is shown in the below table. They are shuffled and splitted into data set for training (80%) and data set for testing (20%).

|           | Num of images   | 
|:----------:|:------:| 
| Vehicle     | 8792 |
| Non Vehicle | 8968 |

##### Training method

Regarding the method for training, Linear SVC is used.

##### The features extracted from image

The following features are extracted from image and used for training.

* HOG features shown in the above section.
* Spacial color binning feature with dimensions (32, 32)
* Color Histogram of 32 bin

The feature values are scaled by StandardScaler of scikit_learn and that scaler is applied when the trained classifier is used for detecting.

##### Performance of the training

The test accuracy of the model was **99.01%**.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the function find_car_multiscale() of [find_cars.py](find_cars.py).

Applying various size of window for all over the image requires huge computing cost. So I considered the following technique to reduce the computing cost.

* Search only bottom half of the image since there should be no cars flying above a road.
* In case of small scaled windows, search is performed only for 'far' area. This is because the longer the distance of other car, the smaller it appears on the image.

The picture below illustrates the windows for scale=1.0 and scale=2.0. The overlap between next window is 75%. In case of scale=1.0, window size is 64x64 pixel and the offest for adjacent window is 16 pix.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. The variation of scales of window is : `1.0, 1.12, 1.25, 1.5, 2.0`.
Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in [bbox_filter.py](bbox_filter.py) and [find_cars.py](find_cars.py).

I recorded the positions of positive detections in each frame of the video. These positions are collected for several frames as "history". From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I set the number of frames for history to **10**. The minimum threshold to filter out is **40**.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Performance is not enough for real time processing
  * It took about 50 minutes to process project_video.mp4 on my desktop PC which has only Intel GPU integrated in CPU.
  * Improvement by algorithm
    * Once the car is detected, we can estimate the velocity of it and predict the position in next frame. Considering that point will reduce the unncesarry search with window.
  * Improvement by computing
    * If OpenCV/numpy is accelerated with GPU, the performance might be much better.


