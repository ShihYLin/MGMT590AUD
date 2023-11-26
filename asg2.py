from PIL import Image
from pylab import *
import os

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
        
# step 2 convert to grayscale arrays
for filename in os.listdir(output_folder):

    # Open the image file
    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path)

    # Turn into array and grayscale
    gray_img = array(img.convert('L'))
