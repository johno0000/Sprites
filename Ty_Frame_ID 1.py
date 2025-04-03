import os
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import exposure
import pandas as pd

# Function to plot only bright pixels
def plot_bright_pixels(bright_pixel_coords,sprite_num):
    if bright_pixel_coords.empty:
        print("No bright pixels to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Loop through each unique label and plot separately
    for label in sorted(bright_pixel_coords['Label'].unique()):
        cluster_points = bright_pixel_coords[bright_pixel_coords['Label'] == label]
        plt.scatter(
            cluster_points['x'], cluster_points['y'],
            s=5, label=f'Cluster {label}', alpha=0.6
        )

    plt.title("Sprite " + str(sprite_num) + " Bright Pixels by Cluster")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, 720)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()
    plt.legend(title='Clusters', markerscale=3)  # markerscale makes the legend symbols larger
    plt.grid(True)
    plt.tight_layout()
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

def is_a_star(x, y, tiff):
    if len(x) < 70:
        return False
    
    med_x = np.median(x)
    med_y = np.median(y)
    
    distances_squared = (x - med_x) ** 2 + (y - med_y) ** 2
    sum_of_squares = np.mean(distances_squared)
    
    brightness = np.mean(tiff[y, x])
    density = brightness / sum_of_squares
    
    likely_star = density > 0.25
    return likely_star

# Load folder and process files
folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Ty_Frames'
#Above is for original files
# folder_path = r'C:\Users\Stormberg\OneDrive\Desktop\490\Full_Test'
number_of_files = count_files_in_folder(folder_path)
Frame_List = []

sprite_num = 1

# DBSCAN Clustering Parameters
dbscan_eps = 2.9
dbscan_min_samples = 5

sprite_frames = []
bright_pixel_coords = []
clustering = []
labels = []
Frames = []

for i in range(1, number_of_files + 1):
    formatted_number = f'{i:06d}' #When working with original files change this to 6
    sf_number = f'{sprite_num:03d}'
    file_path = os.path.join(folder_path, f'output_{formatted_number}.tiff')
    Saved_Im = os.path.join(r"C:\Users\Stormberg\OneDrive\Desktop\490\test", f'Saved_{sf_number}.tiff')
#Change the above to Full_Test when done testing
    image = Image.open(file_path)
    # crop_box = (0, 0, 720, 387)
    # cropped_image = image.crop(crop_box)

    grayscale_image = Image.open(file_path).convert('L')
    brightness_array = np.array(grayscale_image)
    brightness_array[387:460,60:280] = 0
    # brightness_array = exposure.rescale_intensity(brightness_array, in_range=(10, 50))  # Adjust range
    # brightness_array = gaussian_filter(brightness_array, sigma=1)  # Adjust sigma based on visibility
    avg_brightness = np.median(brightness_array)
    #this step is an effort to reduce noise

    med = 1.5 * np.median(brightness_array)  # Threshold for brightness
    bright_pixel_coords = np.argwhere(brightness_array > med)

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(bright_pixel_coords)
    labels = (clustering.labels_)
    
    df = pd.DataFrame(bright_pixel_coords, columns=['y', 'x'])
    df['Label'] = labels
    df = df[df['Label'] != -1]  # Remove noise points
    
    [a, b] = np.unique(labels, return_counts=True)

    clust_num = 70  # Minimum cluster size
    sprite_lab = np.where([b > clust_num])[1] - 1
    condition = df['Label'].isin(sprite_lab)

    # Filter out bright pixels
    df_bright = df[condition]  
    df_background = df[~condition]  
    
    if not df_bright.empty:
        def evaluate_cluster(group):
            x = group['x'].to_numpy()
            y = group['y'].to_numpy()
            return not is_a_star(x, y, brightness_array)

        valid_clusters = df_bright.groupby('Label').filter(evaluate_cluster)
    
        if not valid_clusters.empty:
            image.save(Saved_Im)
            tiff_array = tiff_to_array(Saved_Im)
            Frame_List.append(tiff_array)
            plot_bright_pixels(df_bright,sprite_num)
            sprite_num += 1

    else:
        print('No Sprite Found')