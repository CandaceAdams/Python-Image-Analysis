#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:43:19 2023
@author: candace
"""

from skimage.io import imread
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters, measure
from skimage.filters import threshold_local
from skimage.transform import warp




# Root path - where to save
root_path = "/Volumes/Candace A/20230606_TwoColourNanobody/eGFP_1in1000_30minincubation_2023-06-06_16-43-57/"
matrix_path = "/Volumes/Candace A/20230606_TwoColourNanobody/Beads_2023-06-06_12-32-46/"
# These are the names of the files to image:
green_image = "488_0.tif"
red_image = "647_0.tif"

# Paths to analyse:
pathlist = []

for i in range(1, 4):
    for k in range(1, 4):
        pathlist.append(root_path + "X0Y0R" + str(i) + "W" + str(k) + "_")

# Folders to analyse:
def load_image(toload):
    image = imread(toload)
    return image

# Define the function to apply the transformation matrix to the image
def apply_transformation(image, transformation_matrix):
    transformed_image = warp(image, transformation_matrix)
    return transformed_image

def z_project(image):
    mean_int = np.mean(image, axis=0)
    return mean_int

# Subtract background:
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    bg_corrected = image - background
    return bg_corrected

def threshold_image_std(input_image):
    #threshold_value=filters.threshold_otsu(input_image)  
    threshold_value = input_image.mean() + 3* input_image.std()
    binary_image = input_image > threshold_value
    return threshold_value, binary_image

def threshold_image_standard(input_image, thresh):
    binary_image = input_image > thresh
    return binary_image

# Threshold image using Otsu method and output the filtered image along with the threshold value applied:
def threshold_image_fixed(input_image, threshold_number):
    threshold_value = threshold_number
    binary_image = input_image > threshold_value
    return threshold_value, binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image = measure.label(input_image)
    number_of_features = labelled_image.max()
    return number_of_features, labelled_image

# Function to show the particular image:
def show(input_image, color=''):
    if color == 'Red':
        plt.imshow(input_image, cmap="Reds")
        plt.show()
    elif color == 'Blue':
        plt.imshow(input_image, cmap="Blues")
        plt.show()
    elif color == 'Green':
        plt.imshow(input_image, cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show()

# Take a labelled image and the original image and measure intensities, sizes, etc.
def analyse_labelled_image(labelled_image, original_image):
    measure_image = measure.regionprops_table(labelled_image, intensity_image=original_image, properties=('area', 'perimeter', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'mean_intensity', 'max_intensity'))
    measure_dataframe = pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# Show the labelled image overlaid with the original image
def show_labelled_image(labelled_image, original_image):
    plt.imshow(original_image, cmap="gray")
    plt.imshow(labelled_image, cmap="rainbow", alpha=0.5)
    plt.show()

# Function to overlap two binary images
def feature_coincidence(image1, image2):
    image1_coords = np.where(image1)
    image2_coords = np.where(image2)
    image1_binary = np.zeros_like(image1, dtype=bool)
    image2_binary = np.zeros_like(image2, dtype=bool)
    image1_binary[image1_coords] = True
    image2_binary[image2_coords] = True

    overlap = image1_binary & image2_binary
    fraction_overlap = np.sum(overlap) / np.sum(image1_binary)

    labelled_overlap = measure.label(overlap)
    overlap_features = labelled_overlap.max()
    non_overlap1 = image1_binary ^ overlap
    non_overlap2 = image2_binary ^ overlap

    return overlap_features, overlap, fraction_overlap, overlap, non_overlap1, non_overlap2

# Rotate the image for chance
def rotate(matrix):
    temp_matrix = []
    column = len(matrix)-1
    for column in range(len(matrix)):
       temp = []
       for row in range(len(matrix)-1,-1,-1):
          temp.append(matrix[row][column])
       temp_matrix.append(temp)
    for i in range(len(matrix)):
       for j in range(len(matrix)):
          matrix[i][j] = temp_matrix[i][j]
    return matrix         

# Load the transformation matrix from the text file
transformation_matrix = np.loadtxt(matrix_path + 'transformation_matrix.txt')

Output_all = pd.DataFrame(columns=['Number green', 'Number red', 'Number coincident', 'Number chance', 'Q'])

for path in pathlist:
    # Load the images
    green = load_image(path + green_image)
    red = load_image(path + red_image)

    # Apply the transformation matrix to the green image
    transformed_green = apply_transformation(green, transformation_matrix)

    # Z-project - get the average intensity over the range.
    green_flat = z_project(transformed_green)
    red_flat = z_project(red)

    # Subtract background from the images
    green_subtracted = subtract_bg(green_flat)
    red_subtracted = subtract_bg(red_flat)

    # Threshold the images
    green_threshold_value, green_binary = threshold_image_std(green_subtracted)
    red_threshold_value, red_binary = threshold_image_std(red_subtracted)
    
   # Save the images 
     
    imsr = Image.fromarray(green_subtracted)
    imsr.save(path+green_image+'_BG_Removed.tif')
     
    imsr = Image.fromarray(red_subtracted)
    imsr.save(path+red_image+'_BG_Removed.tif')
     
     
    imsr = Image.fromarray(green_binary)
    imsr.save(path+green_image+'_Binary.tif')
     
    imsr = Image.fromarray(red_binary)
    imsr.save(path+red_image+'_Binary.tif')   

    # Perform analysis
    number_green, labelled_green = label_image(green_binary)
    print("%d features were detected in the green image." % number_green)
    measurements_green = analyse_labelled_image(labelled_green, green_flat)

    number_red, labelled_red = label_image(red_binary)
    print("%d features were detected in the red image." % number_red)
    measurements_red = analyse_labelled_image(labelled_red, red_flat)

    # Perform coincidence analysis
    overlap_features, overlap, fraction_overlap, overlap, non_overlap1, non_overlap2 = feature_coincidence(red_binary,green_binary)

    
    green_coinc_list, green_coinc_pixels, green_fraction_coinc, green_coincident_features_image, green_non_coincident_features_image, green_fract_pixels_overlap = feature_coincidence(green_binary, red_binary)
    red_coinc_list, red_coinc_pixels, red_fraction_coinc, red_coincident_features_image, red_non_coincident_features_image, red_fract_pixels_overlap = feature_coincidence(red_binary, green_binary)
    #number_of_coinc = len(np.where(green_coinc_list)[0])
    
    # Need to account for chance due to high density
    green_binary_rot = rotate(green_binary)
    chance_overlap_features, chance_overlap, chance_fraction_overlap, chance_overlap, chance_non_overlap1, chance_non_overlap2 = feature_coincidence(green_binary_rot,red_binary)
    #chance_coinc_list, chance_coinc_pixels, chance_fraction_coinc, chance_coincident_features_image, chance_non_coincident_features_image, chance_fract_pixels_overlap = feature_coincidence(green_binary_rot, red_binary)
    

    # Calculate an association quotient
    Q = (overlap_features - chance_overlap_features) / (number_green + number_red - (overlap_features - chance_overlap_features))

    imsr = Image.fromarray(green_coincident_features_image)
    imsr.save(path + green_image + '_Coincident.tif')
     
    imsr = Image.fromarray(red_coincident_features_image)
    imsr.save(path + red_image + '_Coincident.tif')

    # Output
    Output_all = Output_all.append({'Number green': number_green, 'Number red': number_red, 'Number coincident': overlap_features, 'Number chance': chance_overlap_features, 'Q': Q}, ignore_index=True)

Output_all.to_csv(root_path + 'All.csv', sep='\t')







