
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import skimage
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


'''from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection'''

path = 'N:/PROJECT/stage1'
patients = os.listdir(path)
patients.sort()

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

    image = np.reshape(image,[20,512,512])
   
    print(image.shape)
    image[image <= -2000] = 0

    for slice_number in range(len(slices)):
            
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
            
        if slope != 1:
            image[0][slice_number] = slope * image[0][slice_number].astype(np.float64)
            image[0][slice_number] = image[0][slice_number].astype(np.int16)
                    
        image[0][slice_number] = image[0][slice_number] + np.int16(intercept)
    return np.array(image, dtype=np.int16)


first_patient = load_scan(path)
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = skimage.measure.marching_cubes_classic(image, threshold)

    fig = plt.figure(figsize=(10, 10))
  #  Axes3D = Axes3D 
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])

    plt.show()
    
    
plot_3d(first_patient_pixels,400)


def plot_2d(img):
	plt.imshow(img, cmap=plt.cm.gray)
	plt.show()
    
plot_2d(first_patient_pixels)

print(matplotlib.__version__)





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


segmented_lungs = segment_lung_mask(first_patient_pixels, False)
segmented_lungs_fill = segment_lung_mask(first_patient_pixels, True)




