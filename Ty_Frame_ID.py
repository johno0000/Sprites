# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy as np
import sklearn.cluster
from sklearn.cluster import DBSCAN

def tiff_to_array(file_path):
    # Open the TIFF file with PIL
    with Image.open(file_path) as img:
        # Convert the image to a numpy array
        array = np.array(img)
    return array

def count_files_in_folder(folder_path):
    file_count = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)  # Increment count by number of files in this directory
    
    return file_count

folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Ty_Frames'
number_of_files = count_files_in_folder(folder_path)

Frame_List = []

# Set the brightness threshold
brightness_threshold = 24  # Adjust this value based on your needs (0-255 scale for grayscale images)

# for i in range(1, number_of_files + 1):  # Loop through the files
formatted_number = '%06d' % i
file_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Ty_Frames\output_' + formatted_number + '.tiff'
cropped_im = r"C:\Users\Stormberg\OneDrive\Desktop\490\test\cropped_" + formatted_number + ".tiff" 
image = Image.open(file_path)

# Define the crop box (left, upper, right, lower)
crop_box = (0, 0, 720, 387)  # Replace with your desired crop coordinates

# Perform the crop
cropped_image = image.crop(crop_box)

# Calculate the average brightness
# Convert to grayscale if the image is not already grayscale
grayscale_image = cropped_image.convert('L')  # 'L' mode converts to grayscale
brightness_array = np.array(grayscale_image)
avg_brightness = np.mean(brightness_array)

# Check if the brightness is above the threshold
if avg_brightness >= brightness_threshold:
    # Save the cropped image
    cropped_image.save(cropped_im)
    
    # Convert the TIFF to a numeric array and append to Frame_List
    tiff_array = tiff_to_array(cropped_im)
    Frame_List.append(tiff_array)
else:
    print(f"Frame {i} skipped: {avg_brightness}")
    
# DBSCAN Clustering Section
sprite_frames = []  # List to store frames with detected sprites
dbscan_eps = 2  # Maximum distance between points in a cluster
dbscan_min_samples = 5  # Minimum points required to form a cluster
#Above values were suggested starting point by Dr Ortberg

# for idx, tiff_array in enumerate(Frame_List):
#     # Identify bright pixels (potential sprite points)
bright_pixel_coords = np.argwhere(tiff_array > brightness_threshold)

#     # Skip frames with no bright pixels
#     if len(bright_pixel_coords) == 0:
#         print(f"Frame {idx + 1}: No bright pixels found.")
#         continue

#     # Apply DBSCAN clustering
    
#     clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(bright_pixel_coords)
#     labels = clustering.labels_

#     # Check if any cluster was detected (label != -1)
#     if np.any(labels != -1):
#         sprite_frames.append((idx, bright_pixel_coords, labels))
#         print(f"Frame {idx + 1}: Sprite detected with {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
#     else:
#         print(f"Frame {idx + 1}: No clusters detected.")

# Process sprite_frames for further analysis
print(f"Total frames with detected sprites: {len(sprite_frames)}")

