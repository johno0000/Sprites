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
import pandas as pd

# Function to plot bright pixels and filter clusters by size
def plot_bright_pixels_with_filtering(bright_pixel_coords, labels):
    if len(bright_pixel_coords) == 0:
        print("No bright pixels to plot.")
        return

    # Count occurrences of each cluster label
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Define valid clusters (true if >= 70 pixels, false otherwise)
    cluster_validity = {label: count >= 70 for label, count in zip(unique_labels, counts)}
    
    plt.figure(figsize=(10, 6))

    # Assign colors based on validity
    for label in unique_labels:
        mask = labels == label
        cluster_color = 'red' if label >= 0 and cluster_validity.get(label, False) else 'black'
        plt.scatter(bright_pixel_coords[mask, 1], bright_pixel_coords[mask, 0], c=cluster_color, s=1)

    plt.title("Filtered Bright Pixels by Cluster Size")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert Y-axis for image-like visualization
    plt.show()

# Function to convert TIFF to numpy array
def tiff_to_array(file_path):
    with Image.open(file_path) as img:
        array = np.array(img)
    return array

# Function to count files in a folder
def count_files_in_folder(folder_path):
    return sum(len(files) for _, _, files in os.walk(folder_path))

# Load folder and process files
folder_path = r'C:\Users\tstro\OneDrive\Desktop\490\test\test'
number_of_files = count_files_in_folder(folder_path)
Frame_List = []

for i in range(1, number_of_files + 1):
    formatted_number = f'{i:03d}'
    cropped_im = os.path.join(folder_path, f'cropped_{formatted_number}.tiff')

    if not os.path.exists(cropped_im):
        print(f"Warning: {cropped_im} not found. Skipping.")
        continue

    tiff_array = tiff_to_array(cropped_im)
    Frame_List.append(tiff_array)

# DBSCAN Clustering Parameters
dbscan_eps = 2
dbscan_min_samples = 5

sprite_frames = []
bright_pixel_coords = []
clustering = []
labels = []
Frames = []

for idx, tiff_array in enumerate(Frame_List):
    # Extract bright pixels
    bright_pixel_coords.append(np.argwhere(tiff_array > 50))


    # Apply DBSCAN clustering
    clustering.append(DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(bright_pixel_coords[idx]))
    labels.append(clustering[idx].labels_)
    Bright_df = pd.DataFrame(bright_pixel_coords[idx], columns=['Y', 'X'])
    Bright_df.insert(2, 'Label', clustering[idx].labels_)
    Bright_df['Counts'] = Bright_df.groupby('Label')['Label'].transform('count')
    Frames.append(Bright_df)
    
    plot_bright_pixels_with_filtering(bright_pixel_coords[idx], labels[idx])
    # Count unique clusters
    # unique_labels, counts = np.unique(labels[idx], return_counts=True)
    # num_clusters = sum(count >= 70 for count in counts if count > 0)


