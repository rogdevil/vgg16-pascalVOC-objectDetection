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
from cv2 import cv2

dir_image = "images"
img = imageio.imread(os.path.join(dir_image,"nowi.jpeg"))

newsize = (800,800,3)
img = skimage.transform.resize(img,newsize)
img = np.float32(img)
const = 4

def get_regions(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects

regions = get_regions(img)
print("N candidate regions ={}".format(len(regions)))
print("_"*10)
print("print the first 10 regions")
for r in regions[:10]:
    print(r)
print("_"*10)
print("print the last 10 regions")    
for r in regions[-10:]:
    print(r)


def plt_rectangle(plt,label,x1,y1,x2,y2,color = "yellow", alpha=0.5):
    linewidth = 3
    if type(label) == list:
        linewidth = len(label)*3 + 2
        label = ""
        
    plt.text(x1,y1,label,fontsize=20,backgroundcolor=color,alpha=alpha)
    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color, alpha=alpha)

plt.figure(figsize=(20,20))    
plt.imshow(img)
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item
    label = ['car', 'car', 'car', 'car', 'car']
    plt_rectangle(plt,label,
                  x1,
                  y1,
                  x2 = x1 + width,
                  y2 = y1 + height, 
                  color= color)
plt.show()

def warp_candidate_regions(img,regions):
    ## for each candidate region, 
    ## warp the image and extract features 
    newsize_cnn = (224, 224)
    X = []
    for i, r in enumerate(regions):
        origx , origy , width, height = r
        candidate_region = img[origy:origy + height,
                               origx:origx + width]
        img_resize = skimage.transform.resize(candidate_region,newsize_cnn)
        X.append(img_resize)

    X = np.array(X)
    print(X.shape)
    return(X)
X = warp_candidate_regions(img,regions)


modelvgg16 = VGG16(include_top=True,weights='imagenet')
model = models.Model(inputs  =  modelvgg16.inputs, 
                        outputs = modelvgg16.layers[-3].output)
## show the deep learning model
model.summary()

start   = time.time()
feature = model.predict(X, verbose=1)
end     = time.time()
print("TIME TOOK: {:5.4f}MIN".format((end-start)/60.0))
feature.shape


dir_result = "result"
classifier = load_model(os.path.join(dir_result,"classifier.h5"))
classifier.summary()
y_pred = classifier.predict(feature, verbose=1)

def plot_selected_regions_with_estimated_prob(y_pred,
                                              method="highest",
                                              upto=5):
    ## increasing order
    irows = np.argsort(y_pred[:,0])
    if method == "highest":
        irows = irows[::-1]
    count = 1
    const = 4
    fig = plt.figure(figsize=(5*const,np.ceil(upto/5)*const))
    fig.subplots_adjust(hspace=0.13,wspace=0.0001,
                        left=0,right=1,bottom=0, top=1)
    for irow in irows:
        prob = y_pred[irow,0]
        r    = regions[irow]
        origx , origy , width, height = r
        
        ax = fig.add_subplot(np.ceil(upto/5),5,count)
        ax.imshow(img)
        ax.axis("off")
        plt_rectangle(ax,label="car",
                      x1=origx,
                      y1=origy,
                      x2=origx + width,
                      y2=origy+height,color = "yellow", alpha=0.5)
        
        #candidate_region = img[origy:origy + height,
        #                      origx:origx + width]       
        #ax.imshow(candidate_region)
        ax.set_title("Prob={:4.3f}".format(prob))
        count += 1
        if count > upto:
            break
    plt.show()

print("The most likely candidate regions")    
plot_selected_regions_with_estimated_prob(y_pred,method="highest",upto=10)
print(y_pred)