# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import exposure

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

# Function to find the highest value in a 2D array
def find_highest_value(array):
    max_value = np.max(array)
    max_coords = np.argwhere(array == max_value)
    return max_value, max_coords

# Function to count files in a folder
def count_files_in_folder(folder_path):
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

# Load folder and process files
folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Ty_Frames'
number_of_files = count_files_in_folder(folder_path)
Frame_List = []
brightness_threshold = 83.5  # Adjust this value based on your needs

sprite_num = 1

for i in range(1, number_of_files + 1):
    formatted_number = f'{i:06d}'
    sf_number = f'{sprite_num:03d}'
    file_path = os.path.join(folder_path, f'output_{formatted_number}.tiff')
    Saved_Im = os.path.join(r"C:\Users\Stormberg\OneDrive\Desktop\490\Full_Test", f'Saved_{sf_number}.tiff')

    image = Image.open(file_path)
    # crop_box = (0, 0, 720, 387)
    # cropped_image = image.crop(crop_box)

    grayscale_image = Image.open(file_path).convert('L')
    brightness_array = np.array(grayscale_image)
    brightness_array[387:460,60:280] = 0
    brightness_array = exposure.rescale_intensity(brightness_array, in_range=(10, 50))  # Adjust range
    # brightness_array = gaussian_filter(brightness_array, sigma=1)  # Adjust sigma based on visibility
    avg_brightness = np.mean(brightness_array)
    #this step is an effort to reduce noise


    if avg_brightness >= brightness_threshold:
        print(avg_brightness)
        sprite_num += 1
        image.save(Saved_Im)
        tiff_array = tiff_to_array(Saved_Im)
        Frame_List.append(tiff_array)
    else:
        print(f"Frame {i} skipped: {avg_brightness}")

