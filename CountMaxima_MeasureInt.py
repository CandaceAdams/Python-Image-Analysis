#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:56:36 2023

@author: candace
"""


from skimage.io import imread
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters,measure
from skimage.filters import threshold_local



# Root path - where to save

root_path="/Volumes/Candace A/CandaceImagingData/220928_antiPSD95/antiPSD95_m6_1in100_2022-09-28_14-01-05/"

green_image="488.tif"


# Paths to analyse:

pathlist=[]
for i in range(0,5):
    for k in range(0,5):
        pathlist.append(root_path +"X"+str(i)+"Y"+str(k)+"_")


# Folders to analyse:
    
def load_image(toload):
    image = imread(toload)
    return image

def z_project(image):
    
    mean_int=np.mean(image,axis=0)
  
    return mean_int

# Subtract background:
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    bg_corrected =image - background
    return bg_corrected

def threshold_image_std(input_image):
    #threshold_value=filters.threshold_otsu(input_image)  
    #threshold_value= 750
    threshold_value= input_image.mean()+5*input_image.std()
    print(threshold_value)
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

def threshold_image_standard(input_image,thresh):
     
    binary_image=input_image>thresh

    return binary_image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image
    
# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show() 
    
        
# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

Output_all = pd.DataFrame(columns=['Number green'])


for path in pathlist:

  # Load the images
    green=load_image(path+green_image)


  # z-project - get the average intensity over the range. 
    
    green_flat=np.mean(green,axis=0)


  # The excitation is not homogenous, and so need to subtract the background:
    
    green_bg_remove=subtract_bg(green_flat)
    
    
  # Threshold each channel: 
    
    thr_gr,green_binary=threshold_image_std(green_bg_remove)
    
   
  # Save the images 
    
    imsr = Image.fromarray(green_bg_remove)
    imsr.save(path+green_image+'_BG_Removed.tif')
    
    
    
    imsr = Image.fromarray(green_binary)
    imsr.save(path+green_image+'_Binary.tif')
    
    
  # Perform analysis 
   
    number_green,labelled_green=label_image(green_binary)
    print("%d feautres were detected in the green image."%number_green)
    measurements_green=analyse_labelled_image(labelled_green,green_flat)
 
     
    
# Output

    Output_all = Output_all.append({'Number green':number_green},ignore_index=True)


    Output_all.to_csv(root_path + 'All.csv', sep = '\t')    
    

 
    