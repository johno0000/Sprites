# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:10:07 2025

@author: Stormberg
"""
import os
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Function to plot only bright pixels
def plot_bright_pixels(bright_pixel_coords, labels):
    if len(bright_pixel_coords) == 0:
        print("No bright pixels to plot.")
        return

    # Separate coordinates into X and Y for plotting
    y_coords, x_coords = bright_pixel_coords.T

    # Plot the bright pixels
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c='red', s=1, label='Bright Pixels')  # Reduced point size
    plt.title("Bright Pixels")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert Y-axis for image-like visualization
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to convert TIFF to numpy array
def tiff_to_array(file_path):
    with Image.open(file_path) as img:
        array = np.array(img)
    return array

# Function to count files in a folder
def count_files_in_folder(folder_path):
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

# Load folder and process files
folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\test'
number_of_files = count_files_in_folder(folder_path)
Frame_List = []

for i in range(1, number_of_files+1):
    formatted_number = f'{i:03d}'
    cropped_im = os.path.join(r"C:\Users\Stormberg\OneDrive\Desktop\490\test", f'cropped_{formatted_number}.tiff')  
    tiff_array = tiff_to_array(cropped_im)
    Frame_List.append(tiff_array)

        
# DBSCAN Clustering Section
sprite_frames = []
dbscan_eps = 2.9
dbscan_min_samples = 5

for idx, tiff_array in enumerate(Frame_List):
    bright_pixel_coords = np.argwhere(tiff_array > 50)

    if len(bright_pixel_coords) == 0:
        print(f"Frame {idx + 1}: No bright pixels found.")
        continue

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(bright_pixel_coords)
    labels = clustering.labels_

    if np.any(labels != -1):
        sprite_frames.append((idx, bright_pixel_coords, labels))
        print(f"Frame {idx + 1}: Sprite detected with {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
    else:
        print(f"Frame {idx + 1}: No clusters detected.")

print(f"Total frames with detected sprites: {len(sprite_frames)}")

for frame_idx, bright_pixel_coords, labels in sprite_frames:
    plot_bright_pixels(bright_pixel_coords, labels)