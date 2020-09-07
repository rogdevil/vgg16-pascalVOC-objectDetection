import os
import tensorflow as tf 
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import imageio 
import seaborn as sns
import scipy.misc
import skimage.segmentation
import skimage.feature
from copy import copy
import sys 
import random
import skimage
from keras.applications import VGG16
import pickle
import time
from keras import models
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from selective_search import *
from cv2 import cv2



dir_anno = 'D:\\dataset\\Cars Completed\\Car-PascalVOC-export\\Annotations'
img_dir = 'D:\\dataset\\Cars Completed\\Car-PascalVOC-export\\images'

# function to extract the data from annotations
def extract_single_xml_file(tree):
    Nobj = 0
    row  = OrderedDict()
    for elems in tree.iter():

        if elems.tag == "size":
            for elem in elems:
                row[elem.tag] = int(elem.text)
        if elems.tag == "object":
            for elem in elems:
                if elem.tag == "name":
                    row["bbx_{}_{}".format(Nobj,elem.tag)] = str(elem.text)              
                if elem.tag == "bndbox":
                    for k in elem:
                        row["bbx_{}_{}".format(Nobj,k.tag)] = float(k.text)
                    Nobj += 1
    row["Nobj"] = Nobj
    return(row)

df_anno = []
for fnm in os.listdir(dir_anno):  
    if not fnm.startswith('.'): ## do not include hidden folders/files
        tree = ET.parse(os.path.join(dir_anno,fnm))
        row = extract_single_xml_file(tree)
        row["fileID"] = fnm.split(".")[0]
        df_anno.append(row)
df_anno = pd.DataFrame(df_anno)

maxNobj = np.max(df_anno["Nobj"])

dir_preprocessed = "D:\\detection\\"
df_anno.to_csv(os.path.join(dir_preprocessed,"df_anno.csv"),index=False)

df_anno_Car = df_anno


#function to plot rectangle in images
def plt_rectangle(plt,label,x1,y1,x2,y2):
    '''
    == Input ==
    
    plt   : matplotlib.pyplot object
    label : string containing the object class name
    x1    : top left corner x coordinate
    y1    : top left corner y coordinate
    x2    : bottom right corner x coordinate
    y2    : bottom right corner y coordinate
    '''
    linewidth = 3
    color = "yellow"
    plt.text(x1,y1,label,fontsize=20,backgroundcolor="magenta")
    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color)
    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color)
    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color)
    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color)



# start = time.time()            
# Car_found_vec = []
# for irow in range(df_anno_Car.shape[0]): ## run over each frame
#     row  = df_anno_Car.iloc[irow,:]
#     path = os.path.join(img_dir,row["fileID"] + ".png")
#     img  = imageio.imread(path)
    
#     ## calculate region proposal
#     regions = get_region_proposal(img,min_size=40)
    
#     for ibb in range(row["Nobj"]): ## go over each of the true annotated object
#         print("frameID = {:04.0f}/{}, BBXID = {},  N region proposals = {}".format(
#             irow, df_anno_Car.shape[0], ibb, len(regions)))
#         name = row["bbx_{}_name".format(ibb)]
#         if name != "Car":
#             continue 
#         ## bounding box of the Car     
#         true_xmin   = row["bbx_{}_xmin".format(ibb)]
#         true_ymin   = row["bbx_{}_ymin".format(ibb)]
#         true_xmax   = row["bbx_{}_xmax".format(ibb)]
#         true_ymax   = row["bbx_{}_ymax".format(ibb)]   
#         Car_found_TF = 0
#         for r in regions:  ## go over each region proposal and calculate the IoU
            
#             prpl_xmin, prpl_ymin, prpl_width, prpl_height = r["rect"]
#             IoU = get_IOU(prpl_xmin, prpl_ymin, prpl_xmin + prpl_width, prpl_ymin + prpl_height,
#                           true_xmin, true_ymin, true_xmax, true_ymax)
#             if IoU > 0.2:
#                 Car_found_TF = 1
#         Car_found_vec.append(Car_found_TF)
# end = time.time()  
# print("TIME TOOK : {}MIN".format((end-start)/60))
# print("Total N of Car : {}, Total N of Car found : {}, TPR: {:4.3f}".format(
#     len(Car_found_vec),
#     np.sum(Car_found_vec),
#     np.mean(Car_found_vec)))


modelvgg16 = VGG16(include_top=True,weights='imagenet')
model = models.Model(inputs  =  modelvgg16.inputs, 
                        outputs = modelvgg16.layers[-3].output)
## show the deep learning model
model.summary()

        
def get_regions(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    print(len(rects))
    return rects

# program to resize image
def warp(img, newsize):
    '''
    warp image 
    
    
    img     : np.array of (height, width, Nchannel)
    newsize : (height, width)
    '''
    img_resize = skimage.transform.resize(img,newsize)
    img_resize = np.float32(img_resize)
    return(img_resize)

dir_result = 'D:\\detection\\result\\'
IoU_cutoff_object     = 0.5
IoU_cutoff_not_object = 0.3

cols_bbx = []
# here we look for the all the images which have Car
for colnm in df_anno.columns:
    if "name" in colnm:
        cols_bbx.append(colnm)
bbx_has_CarTF = df_anno[cols_bbx].values == "Car"
pick = np.any(bbx_has_CarTF,axis=1)
df_anno_Car = df_anno.loc[pick,:]

IoU_cutoff_object     = 0.5
IoU_cutoff_not_object = 0.3
objnms = ["image0","info0","image1","info1"]  
dir_result = "result"

start = time.time()   
# the "rough" ratio between the region candidate with and without objects.
N_img_without_obj = 2 
newsize = (800, 800, 3) ## hack
image0, image1, info0,info1 = [], [], [], [] 
for irow in range(df_anno_Car.shape[0]):
    ## extract a single frame that contains at least one Cardf_anno_Car object
    row  = df_anno_Car.iloc[irow,:]
    ## read in the corresponding frame
    path = os.path.join(img_dir,row["fileID"] + ".png")
    img  = imageio.imread(path)
    orig_h, orig_w, _ = img.shape
    ## to reduce the computation speed,
    ## I will do a small hack here. I will resize all the images into newsize = (200,250)    
    img  = warp(img, newsize)
    orig_nh, orig_nw, _ = img.shape
    ## region candidates for this frame
    regions = get_regions(img)
    
    ## for each object that exists in the data,
    ## find if the candidate regions contain the Cardf_anno_Car
    for ibb in range(row["Nobj"]): 

        name = row["bbx_{}_name".format(ibb)]
        if name != "Car": ## if this object is not Cardf_anno_Car, move on to the next object
            continue 
        if irow % 50 == 0:
            print("frameID = {:04.0f}/{}, BBXID = {:02.0f},  N region proposals = {}, N regions with an object gathered till now = {}".format(
                    irow, df_anno_Car.shape[0], ibb, len(regions), len(image1)))
        
        ## extract the bounding box of the Cardf_anno_Car object  
        multx, multy  = orig_nw/orig_w, orig_nh/orig_h 
        true_xmin     = row["bbx_{}_xmin".format(ibb)]*multx
        true_ymin     = row["bbx_{}_ymin".format(ibb)]*multy
        true_xmax     = row["bbx_{}_xmax".format(ibb)]*multx
        true_ymax     = row["bbx_{}_ymax".format(ibb)]*multy
        
        
        Cardf_anno_Car_found_TF = 0
        _image1 = None
        _image0, _info0  = [],[]
        ## for each candidate region, find if this Cardf_anno_Car object is included
        for r in regions:
            
            prpl_xmin, prpl_ymin, prpl_width, prpl_height = r
            ## calculate IoU between the candidate region and the object
            IoU = get_IOU(prpl_xmin, prpl_ymin, prpl_xmin + prpl_width, prpl_ymin + prpl_height,
                             true_xmin, true_ymin, true_xmax, true_ymax)
            ## candidate region numpy array
            img_bb = np.array(img[prpl_ymin:prpl_ymin + prpl_height,
                                  prpl_xmin:prpl_xmin + prpl_width])
            
            info = [irow, prpl_xmin, prpl_ymin, prpl_width, prpl_height]
            if IoU > IoU_cutoff_object:
                _image1 = img_bb
                _info1  = info
                break
            elif IoU < IoU_cutoff_not_object:
                _image0.append(img_bb) 
                _info0.append(info) 
        if _image1 is not None:
            # record all the regions with the objects
            image1.append(_image1)
            info1.append(_info1)
            if len(_info0) >= N_img_without_obj: ## record only 2 regions without objects
                # downsample the candidate regions without object 
                # so that the training does not have too much class imbalance. 
                # randomly select N_img_without_obj many frames out of all the sampled images without objects.
                pick = np.random.choice(np.arange(len(_info0)),N_img_without_obj)
                image0.extend([_image0[i] for i in pick ])    
                info0.extend( [_info0[i]  for i in pick ])  

        
end = time.time()  
print("TIME TOOK : {}MIN".format((end-start)/60))

### Save image0, info0, image1, info1 
objs   = [image0,info0,image1,info1]        
for obj, nm in zip(objs,objnms):
    with open(os.path.join(dir_result ,'{}.pickle'.format(nm)), 'wb') as handle:
        pickle.dump(obj, 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

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
plot_cadidate_regions_in_training(image1,title="plot warped cadidate regions with a Cardf_anno_Car object intraining ")
plot_cadidate_regions_in_training(image0,title="plot warped cadidate regions without a Cardf_anno_Car object intraining ")

objnms = ["image0","info0","image1","info1"] 
objs  = []
for nm in objnms:
    with open(os.path.join(dir_result,'{}.pickle'.format(nm)), 'rb') as handle:
        objs.append(pickle.load(handle))
image0,info0,image1,info1 = objs 
assert len(image0) == len(info0)
assert len(image1) == len(info1)


print("N candidate regions that has IoU > {} = {}".format(IoU_cutoff_object,len(image1)))
print("N candidate regions that has IoU < {} = {}".format(IoU_cutoff_not_object,len(image0)))


def warp_and_create_cnn_feature(image,model):
    '''
    image  : np.array of (N image, shape1, shape2, Nchannel )
    shape 1 and shape 2 depend on each image
    '''
    print("-"*10)
    print("warp_and_create_cnn_feature")
    start = time.time()
    print("len(image)={}".format(len(image)))
    print("**warp image**")
    for irow in range(len(image)):
        image[irow] = warp(image[irow], (224, 224, 3))
    image = np.array(image)
    print("**create CNN features**")
    feature = model.predict(image)
    print("DONE!")
    print("  feature.shape={}".format(feature.shape))
    end = time.time()
    print("  Time Took = {:5.2f}MIN".format((end - start)/60.0))
    print("")
    return(feature)

feature1 = warp_and_create_cnn_feature(image1,model)
print(feature1)
feature0 = warp_and_create_cnn_feature(image0,model)
print(feature0)

N_obj = len(feature1)
## stack the two set of data
## the first Nobj rows contains the Cardf_anno_Car objects
X = np.concatenate((feature1,feature0))
y = np.zeros((X.shape[0],1))
y[:N_obj,0] = 1


## Save data
print("X.shape={}, y.shape={}".format(X.shape,y.shape))
np.save(file = os.path.join(dir_result,"X.npy"),arr = X)
np.save(file = os.path.join(dir_result,"y.npy"),arr = y)


prop_train = 0.8

## shuffle the order of X and y
X, y = shuffle(X, y, random_state=0)

#X, y = X, y[:,[0]]

Ntrain = int(X.shape[0]*prop_train)
X_train, y_train, X_test, y_test = X[:Ntrain], y[:Ntrain], X[Ntrain:], y[Ntrain:]


model = Sequential()
model.add(Dense(32, input_dim=4096, activation="relu"))
model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    validation_data = (X_test,y_test),
                    batch_size      = 32,
                    epochs          = 70,
                    verbose         = 1)

# fig = plt.figure(figsize=(20,5))
# ax  = fig.add_subplot(1,2,1)
# for key in ["val_loss","loss"]:
#     ax.plot(history.history[key],label=key)
# ax.set_xlabel("epoch")
# ax.set_ylabel("loss")
# plt.legend()
# ax  = fig.add_subplot(1,2,2)
# for key in ["val_acc","acc"]:
#     ax.plot(history.history[key],label=key)
# ax.set_xlabel("epoch")
# ax.set_ylabel("accuracy")
# plt.legend()

# plt.show()



model.save(os.path.join(dir_result,"classifier.h5"))
print("Saved model to disk")