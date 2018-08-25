import os
import numpy as np
import cv2
import glob
import pickle
import time
import pprint
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from feature_tools import *


# Read in cars and notcars from dataset
dataset_dir = os.getenv('UDACITY_DATASET_PATH', '.') + '/vehicle_detection/'
cars = glob.glob(dataset_dir + 'vehicles/*/*.png')
notcars = glob.glob(dataset_dir + 'non-vehicles/*/*.png')
# cars = glob.glob(dataset_dir + 'vehicles_smallset/*/*.jpeg')
# notcars = glob.glob(dataset_dir + 'non-vehicles_smallset/*/*.jpeg')

print('Dataset:  cars:{}, notcars:{}'.format(len(cars), len(notcars)))

# Hyper parameters
PARAMS = {
    "color_space": 'YCrCb',      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    "orient": 9,                 # HOG orientations
    "pix_per_cell": 8,           # HOG pixels per cell
    "cell_per_block": 2,         # HOG cells per block
    "hog_channel": 'ALL',        # Can be 0, 1, 2, or "ALL"
    "spatial_size": (32, 32),    # Spatial binning dimensions
    "hist_bins": 32,             # Number of histogram bins
    "spatial_feat": True,        # Spatial features on or off
    "hist_feat": True,           # Histogram features on or off
    "hog_feat": True,            # HOG features on or off
}

car_features = extract_features(cars, **PARAMS)
notcar_features = extract_features(notcars, **PARAMS)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Parameters: ')
pprint.pprint(PARAMS)
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

# Save model in pickle
dist_picle = PARAMS
dist_picle['svc'] = svc
dist_picle['scaler'] = X_scaler

with open('svc_pickle.p', mode='wb') as f:
    pickle.dump(dist_picle, f)
