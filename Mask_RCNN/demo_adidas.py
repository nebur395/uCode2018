import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import cv2
import visualize

NUM_FRAMES = 5

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print COCO_MODEL_PATH
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Referenced by index will be the filter used for each class
custom_filters = ['cell phone', 'backpack', 'sports ball']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
#%matplotlib inline 

cap = cv2.VideoCapture(0)

# Array to track object coords 
coords_x = list()
coords_y = list()
frames_tracked = 0
plot_trajectory = 0
zipped = None

c_f = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    c_f += 1
    if c_f == NUM_FRAMES:
        # Run detection
        results = model.detect([frame], verbose=1)
        # Visualize results
        r = results[0]
        tracking, center_x, center_y = visualize.display_instances(frame, custom_filters, r['rois'], r['masks'], r['class_ids'], class_names, class_names.index('sports ball'), zipped, r['scores'])
        if (tracking):
            frames_tracked += 1
            coords_x.append(center_x)
            coords_y.append(center_y)
            if(frames_tracked == 3):
                print(coords_x)
                print(coords_y)
                plot_trajectory = 1
                regressor = np.polyfit(coords_x, coords_y, 2)
                if coords_x[0] <= coords_x[2]:
                    x_list = range(coords_x[0],coords_x[2])
                else:
                    x_list = range(coords_x[2],coords_x[0])
                traj_pts = np.polyval(regressor,x_list)
                print(x_list)
                print(traj_pts)
                zipped = [[x_list[i],int(traj_pts[i])] for i in range(abs(coords_x[0] - coords_x[2]))]
                coords_x = list()
                coords_y = list()
                frames_tracked = 0
                print(traj_pts)
        c_f = 0
    
    # Display the resulting frame
    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#image = skimage.io.imread(os.path.join(IMAGE_DIR, '7933423348_c30bd9bd4e_z.jpg'))#random.choice(file_names)))

