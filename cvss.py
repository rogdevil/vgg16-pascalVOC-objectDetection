# force tensorflow to use the CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from imutils.object_detection import non_max_suppression
from cv2 import cv2
import numpy as np
import argparse
import imutils
import time
import matplotlib.pyplot as plt
import imageio, os 
from keras.models import load_model
import skimage.transform
import numpy as np
import selective_search as ss
import seaborn as sns
from keras.applications import VGG16
from keras import models
import tensorflow as tf
import time

#construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-m", "--mode", choices = ["fast", "slow"], required = True, help = "selective search mode")
ap.add_argument("-c", "--min_conf", type = float, default = 0.9, help = "threshold to filter out weak predictions")
args = vars(ap.parse_args())

# initialize necessary constants
WIDTH = 600
INPUT_SHAPE = (224, 224)
BS = 128

# spped up opencv using multiple threads
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# load the image and resize it to the appropriate dimensions
img = cv2.imread(args["image"])
img = imutils.resize(img, width = WIDTH)

# initialize the selective search object using the default parameters and set the input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)

# switch to the appropriate mode
if args["mode"] == "fast":
    ss.switchToSelectiveSearchFast()
else:
    ss.switchToSelectiveSearchQuality()

# get the region proposals
start = time.time()
rects = ss.process()
end = time.time()
print(f"[INFO] total number of region proposals = {len(rects)}")
print(f"[INFO] selective search took {(end - start):0.5f} seconds")

def get_regions(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects

# initialize a list to store the regions of interest
rois = []
locs = []

# loop through the region proposals
for rect in rects:
    # extract the ROI coordinates
    x, y, w, h = rect

    # extract the region of interest from the image
    roi = img[y: y + h, x: x + w]

    # preprocess the ROI
    roi = cv2.resize(roi, INPUT_SHAPE, interpolation = cv2.INTER_AREA)
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # append the current ROI to the ROIs list
    rois.append(roi)

    # modify the bbox coordinates and append them to the locs list
    locs.append((x, y, x + w, y + h))

# convert the ROI list to a numpy array
rois = np.array(rois)


def plot_cadidate_regions_in_training(image1,title):
    fig = plt.figure(figsize=(12,12))
    fig.subplots_adjust(hspace=0.0001,
                        wspace=0.0001,
                        left=0,right=1,bottom=0, top=1)
    print(title)
    nw, nh = 10, 10
    count = 1
    for irow in range(100):#np.random.choice(len(image1),nw*nh):
        im  = image1[irow]
        ax  = fig.add_subplot(nh,nw,count)
        ax.imshow(im)
        ax.axis("off")
        count += 1
    plt.show()
plot_cadidate_regions_in_training(rois,title="plot warped cadidate regions with a Cardf_anno_Car object intraining ")

# modelvgg16 = VGG16(include_top=True,weights='imagenet')
# model = models.Model(inputs  =  modelvgg16.inputs, 
#                         outputs = modelvgg16.layers[-3].output)
# ## show the deep learning model
# model.summary()

# start   = time.time()
# feature = model.predict(rois, verbose=1, batch_size=32)
# end     = time.time()
# print("TIME TOOK: {:5.4f}MIN".format((end-start)/60.0))
# feature.shape

# def plt_rectangle(plt,label,x1,y1,x2,y2,color = "yellow", alpha=0.5):
#     linewidth = 3
#     if type(label) == list:
#         linewidth = len(label)*3 + 2
#         label = ""
        
#     plt.text(x1,y1,label,fontsize=20,backgroundcolor=color,alpha=alpha)
#     plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
#     plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
#     plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color, alpha=alpha)
#     plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color, alpha=alpha)



# dir_result = "result"
# classifier = load_model(os.path.join(dir_result,"classifier.h5"))
# classifier.summary()
# y_pred = classifier.predict(feature)

# def plot_selected_regions_with_estimated_prob(y_pred,
#                                               method="highest",
#                                               upto=5):
#     ## increasing order
#     irows = np.argsort(y_pred[:,0])
#     if method == "highest":
#         irows = irows[::-1]
#     count = 1
#     const = 4
#     fig = plt.figure(figsize=(5*const,np.ceil(upto/5)*const))
#     fig.subplots_adjust(hspace=0.13,wspace=0.0001,
#                         left=0,right=1,bottom=0, top=1)
#     for irow in irows:
#         prob = y_pred[irow,0]
#         r    = rects[irow]
#         origx , origy , width, height = r["rect"]
        
#         ax = fig.add_subplot(np.ceil(upto/5),5,count)
#         ax.imshow(img)
#         ax.axis("off")
#         plt_rectangle(ax,label="Hydrent",
#                       x1=origx,
#                       y1=origy,
#                       x2=origx + width,
#                       y2=origy+height,color = "yellow", alpha=0.5)
        
#         #candidate_region = img[origy:origy + height,
#         #                      origx:origx + width]       
#         #ax.imshow(candidate_region)
#         ax.set_title("Prob={:4.3f}".format(prob))
#         count += 1
#         if count > upto:
#             break
#     plt.show()

# print("The most likely candidate regions")    
# plot_selected_regions_with_estimated_prob(y_pred,method="highest",upto=10)
# print(y_pred)