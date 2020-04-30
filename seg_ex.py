#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:46:42 2019

@author: nick
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import cv2
import math
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


'''from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection'''


data_dir = '/media/nuronics4/af5c71fc-0caf-40bd-b352-560582dd8757/LCD/sample'
patients = os.listdir(data_dir)
labels = pd.read_csv('/media/nuronics4/af5c71fc-0caf-40bd-b352-560582dd8757/LCD/labels.csv', index_col=0)

def load_scan(p):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(p)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
    

def get_pixels_hu(slices):
    image = [] 
    
       
    image.append([s.pixel_array for s in slices])
    
    image = np.array(image)
    l = len(slices)
    image = np.reshape(image,[l,512,512])
   
    
    image[image <= -2000] = 0

    for slice_number in range(len(slices)):
            
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
            
        if slope != 1:
            image[0][slice_number] = slope * image[0][slice_number].astype(np.float64)
            image[0][slice_number] = image[0][slice_number].astype(np.int16)
                    
        image[0][slice_number] = image[0][slice_number] + np.int16(intercept)
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing) , dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

    
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None 
    

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = skimage.measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = skimage.measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = skimage.measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image



# =============================================================================
# def plot_3d(image, threshold=-300):
#     
#     # Position the scan upright, 
#     # so the head of the patient would be at the top facing the camera
#     p = image.transpose(2,1,0)
#     p = p[:,:,::-1]
#     
#     verts, faces = skimage.measure.marching_cubes_classic(image, threshold)
# 
#     fig = plt.figure(figsize=(10, 10))
#   #  Axes3D = Axes3D 
#     ax = fig.add_subplot(111, projection='3d')
# 
#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces], alpha=0.1)
#     face_color = [0.5, 0.5, 1]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)
# 
#     ax.set_xlim(0, image.shape[0])
#     ax.set_ylim(0, image.shape[1])
#     ax.set_zlim(0, image.shape[2])
# 
#     plt.show()
# =============================================================================



def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    
# =============================================================================
#     
# MIN_BOUND = -1000.0
# MAX_BOUND = 400.0
# =============================================================================
    
# =============================================================================
# 
# def normalize(im):
#     
#     c = im.shape[0]
#     
#     for i in range(c):
#         image = (i - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#         print(image)
#         im[image > 1] = 1.   
#         im[image<0] = 0.
#         return image
#     
# =============================================================================
    
    
def process_data(pix,patient,labels_df, hm_slices=20, visualize=False):
    
    label = labels_df.get_value(patient, 'cancer')
# =============================================================================
#     path = data_dir + '/' + patient
#     slices = [pydicom.read_file(path + '/' +s ) for s in os.listdir(path)]
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
# 
# =============================================================================
    new_slices = []

    slices = [cv2.resize(p,(100,100)) for p in pix]
    
    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
        
        
    no = 20-len(new_slices)
    
    
    if no > 0:
        for s in range(no):    
            new_slices.append(new_slices[-1])    
        
    if no < 0:
        for j in range(abs(no)):
            new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
            del new_slices[hm_slices]
            new_slices[hm_slices-1] = new_val 
    
    print(len(slices), len(new_slices))        
            
    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
    
    return np.array(new_slices),label
 

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


IMG_SIZE_PX = 100
SLICE_COUNT = 20
much_data = []

for patient in patients:
    path = data_dir + '/' + patient
    first_patient = load_scan(path)
    first_patient_pixels = get_pixels_hu(first_patient)
    
    #plot_3d(first_patient_pixels,400)
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)
    print("/n")
    
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(first_patient_pixels, True)
    
    
    img_data,label = process_data(pix_resampled,patient,labels, hm_slices=SLICE_COUNT)
    #print(img_data.shape,label)
    much_data.append([img_data,label])
    #last_pix = normalize(pix_resampled)


tr_img_data=[]
label_train=[]
s=np.array(much_data)


for i in range(0,10):
        z = np.array(s[i][0])
        tr_img_data.append(z)
        #label_train.append(s[i][1])
        
for i in range(0,10):
        m = np.array(s[i][1])
        label_train.append(m)
        
print(np.array(tr_img_data).shape,np.array(label_train).shape)    

    
d=np.array(tr_img_data)
X = tf.reshape(d, shape=[-1,IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT])
Y=np.array(label_train)
print(X.shape,Y.shape) 


# Initialising the CNN
classifier = Sequential()

classifier.add(keras.layers.InputLayer(input_shape=(100,100,1)))

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu',padding='valid'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.5))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu',padding='valid'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer

classifier.add(Conv2D(128, (3, 3), activation = 'relu',padding='valid'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#Adding dropout
classifier.add(Dropout(0.8))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#classifier.add(Dense(units = 2048, activation = 'relu'))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'sigmoid'))

print(classifier.summary())

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


list_data = []
for i in range(0,10):
    for j in range(0,20):
        list_data.append([tr_img_data[i][j], label_train[i]])
list_data = np.array(list_data)

m = []
n = []
for i in range(0,200):
    m.append(list_data[i][0])
    n.append(list_data[i][1])
m = np.array(m)
n = np.array(n)

m = np.reshape(m,[200,100,100,1])

m.shape

classifier.fit(m,n,epochs = 5)


# =============================================================================
# 
#     
# segmented_lungs = segment_lung_mask(first_patient_pixels, False)
# segmented_lungs_fill = segment_lung_mask(first_patient_pixels, True)
# 
# 
# plot_3d(first_patient_pixels,400)
# plot_3d(segmented_lungs,0)
# plot_3d(segmented_lungs_fill,0)
# plot_3d(segmented_lungs_fill - segmented_lungs, 0)
# 
# 
# 
# 
# =============================================================================
