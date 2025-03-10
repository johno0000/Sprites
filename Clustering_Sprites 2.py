import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN

# Global sprite counter for sequential renaming
sprite_counter = 1  

# Function to plot bright pixels and filter clusters by size
def plot_bright_pixels_with_filtering(bright_pixel_coords, labels, idx, a, b, df, original_file_path, new_folder_path):
    global sprite_counter  # Use a global counter to rename files sequentially
    
    clust_num = 70  # Minimum cluster size
    sprite_lab = np.where([b > clust_num])[1] - 1
    condition = df['Label'].isin(sprite_lab)

    # Filter out bright pixels
    df_bright = df[condition]  
    df_background = df[~condition]  
    
    if len(df_bright) != 0:
        # Generate a sequential filename
        formatted_number = f'{sprite_counter:03d}'
        new_file_path = os.path.join(new_folder_path, f'Saved_{formatted_number}.tiff')

        # Move and rename the file
        shutil.move(original_file_path, new_file_path)
        print(f"File renamed: {original_file_path} -> {new_file_path}")

        # Increment sprite counter
        sprite_counter += 1

        # Scatter plots with labels
        plt.scatter(df_background['x'], df_background['y'], color='black', s=1, label="Noise")
        plt.scatter(df_bright['x'], df_bright['y'], color='red', s=1, label="Possible Sprite")

        # Add legend
        plt.legend()
        plt.title(f'Sprite {formatted_number}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.gca().invert_yaxis()
        
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
folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Full_Test'
new_folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Sprite_Frames'

# Ensure new folder exists
os.makedirs(new_folder_path, exist_ok=True)

number_of_files = count_files_in_folder(folder_path)
Frame_List = []

for i in range(1, number_of_files + 1):
    formatted_number = f'{i:03d}'
    Saved_im = os.path.join(folder_path, f'Saved_{formatted_number}.tiff')

    if not os.path.exists(Saved_im):
        print(f"Warning: {Saved_im} not found. Skipping.")
        continue

    tiff_array = tiff_to_array(Saved_im)
    tiff_array[387:460, 60:280] = 0  # Masking a region
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
    med = 1.5 * np.median(tiff_array)  # Threshold for brightness
    bright_pixel_coords.append(np.argwhere(tiff_array > med))

    clustering.append(DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(bright_pixel_coords[idx]))
    labels.append(clustering[idx].labels_)
    
    df = pd.DataFrame(bright_pixel_coords[idx], columns=['y', 'x'])
    df['Label'] = labels[idx]
    df = df[df['Label'] != -1]  # Remove noise points
    
    [a, b] = np.unique(labels[idx], return_counts=True)
    
    # Ensure that the correct files are passed
    original_file_path = os.path.join(folder_path, f'Saved_{idx+1:03d}.tiff')

    plot_bright_pixels_with_filtering(
        bright_pixel_coords[idx], labels[idx], idx, a, b, df,
        original_file_path, new_folder_path
    )
