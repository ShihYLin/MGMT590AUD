from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import os
import csv

# Path to the folder containing images
folder_path = '/Users/ziyun/Documents/MGMT590AUD/image'

# Create a new folder to save resized images
output_folder = '/Users/ziyun/Documents/MGMT590AUD/resized_images'
os.makedirs(output_folder, exist_ok=True)

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    try:
        # Open the image file
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # Resize the image to 100x100 pixels
        resized_img = img.resize((100, 100))

        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, filename)
        resized_img.save(output_path)

        print(f"Resized and saved {filename} successfully.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# step 3: Create a CSV file to save flattened arrays
csv_filename = '/Users/ziyun/Documents/MGMT590AUD/flattened_arrays.csv'
csv_file = open(csv_filename, 'w', newline='')

csv_writer = csv.writer(csv_file)

# Create a new folder to save histograms
histogram_folder = '/Users/ziyun/Documents/MGMT590AUD/histogram'
os.makedirs(histogram_folder, exist_ok=True)

# step 2 convert to grayscale arrays
for filename in os.listdir(output_folder):
    if filename == output_folder + '.DS_Store':
        continue
    # Open the image file
    img_path = os.path.join(output_folder, filename)
    img = Image.open(img_path)

    # Turn into array and grayscale
    gray_img = array(img.convert('L'))

    # step 3 flatten
    v_img = gray_img.flatten()

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(v_img, bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(f'Intensity Distribution - {filename}')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save histogram as an image
    histogram_path = os.path.join(histogram_folder, filename.split('.')[0] + '_histogram.png')
    plt.savefig(histogram_path)
    plt.close()

    # write the flatten array as a row in the csv file
    csv_writer.writerow(v_img)

csv_file.close()

# step 4 normalize the images and draw histogram


for filename in os.listdir(output_folder):

    # Open the image file
    img_path = os.path.join(output_folder, filename)
    img = Image.open(img_path)

    # Turn into array and grayscale
    gray_img = array(img.convert('L'))

    # flatten
    v_img = gray_img.flatten()

    # normalize
    hist, bins = histogram(v_img, 256, density=True)
    cdf = hist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    equalized_img = interp(v_img, bins[:-1], cdf)

    img_norm = equalized_img.reshape(gray_img.shape)

    # Plot histogram for the equalized image
    plt.figure(figsize=(6, 4))
    plt.hist(img_norm, bins=256, range=(0, 256), alpha=0.7)
    plt.title(f'Normalized Intensity Distribution - {filename}')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save histogram as an image
    histogram_path = os.path.join(histogram_folder, filename.split('.')[0] + '_N_histogram.png')
    plt.savefig(histogram_path)
    plt.close()

    print(f"Histogram after normalization saved for {filename} successfully.")
