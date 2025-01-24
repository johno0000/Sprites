# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:50:43 2025

@author: Stormberg
"""

import os
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def find_highest_value(array):
    max_value = np.max(array)
    max_coords = np.argwhere(array == max_value)
    return max_value

# Function to plot only bright pixels
def plot_bright_pixels(bright_pixel_coords, labels):
    if len(bright_pixel_coords) == 0:
        print("No bright pixels to plot.")
        return

    # Separate coordinates into X and Y for plotting
    y_coords, x_coords = bright_pixel_coords.T

    # Plot the bright pixels
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c='red', s=1, label='Bright Pixels')  # Red points for bright pixels
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

# Load and process a specific TIFF file
cropped_im = r"C:\Users\Stormberg\OneDrive\Desktop\490\test\cropped_008143.tiff"
tiff_array = tiff_to_array(cropped_im)
brightness_threshold = 50

# DBSCAN parameters
dbscan_eps = 2  # Maximum distance between points in a cluster
dbscan_min_samples = 5  # Minimum points required to form a cluster

# Identify bright pixels (potential sprite points)
bright_pixel_coords = np.argwhere(tiff_array > brightness_threshold)

# Apply DBSCAN clustering
clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(bright_pixel_coords)
labels = clustering.labels_

# Check if any clusters were detected
if np.any(labels != -1):
    print(f"Sprite detected with {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
else:
    print("No clusters detected.")

# Plot the bright pixels
plot_bright_pixels(bright_pixel_coords, labels)
