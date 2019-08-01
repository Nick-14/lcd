import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
import math
from glob import glob 



IMG_SIZE_PX = 100
SLICE_COUNT = 20

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


def process_data(patient,labels_df,img_px_size=100, hm_slices=20, visualize=False):
    
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + '/' + patient
    slices = [pydicom.read_file(path + '/' +s ) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
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
    
    
'''    if len(new_slices) == hm_slices-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices]])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
        
    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices]])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()
'''
        
        
   

# stage 1 for real.

data_dir = '/home/nick/PROJECT/stage1'
patients = os.listdir(data_dir)
labels = pd.read_csv('/home/nick/PROJECT/stage1_labels.csv', index_col=0)

much_data = []
for num,patient in enumerate(patients):
    if num%100 == 0:
        print(num)
    try:
        img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        much_data.append([img_data,label])
    except KeyError as e:
        print('This is unlabeled data!')

#np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)


for i in patients:
    path = data_dir + '/' + i
    slices = [pydicom.read_file(path + '/' +s ) for s in os.listdir(path)]
    


tr_img_data=[]
label_train=[]
s=np.array(much_data)

for i in range(0,10):
        z = np.array(s[i][0])
        tr_img_data.append(np.array(z))
        #label_train.append(s[i][1])
        
for i in range(0,10):
        m = np.array(s[i][1])
        label_train.append(np.array(m))
        
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
#classifier.add(Dropout(0.8))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'sigmoid'))

print(classifier.summary())

# Compiling the CNN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

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

classifier.fit(m,n,epochs = 25)

ds = pydicom.dcmread("/home/nick/PROJECT/4a61e7ceb57f49ff19da103da5ce13c0.dcm")

test_data = ds.pixel_array

test_data = cv2.resize(test_data, (100,100))

test_data = np.reshape(test_data, [1,100,100,1])

pred = classifier.predict(test_data)

pred
