#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:53:30 2023

@author: candace
"""

import os

# Specify the directory path
directory = "/Volumes/Candace A/CandaceImagingData/220928_antiPSD95/antiPSD95_m6_1in100_2022-09-28_14-01-05/"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        # Extract the first 8 characters from the original filename
        new_filename = filename[:8] + ".tif"

        # Construct the full paths for the original and new filenames
        original_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(original_path, new_path)
