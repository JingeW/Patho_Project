import os
from PIL import Image, ImageOps
import math

# Function to combine images with flexible row arrangement
def combine_images(image_paths, output_path, row_num=1, gap=20, frame_thickness=5):
    # Open all images and add a black frame around each one
    images = [ImageOps.expand(Image.open(image_path), border=frame_thickness, fill='black') for image_path in image_paths]
    num_images = len(images)
    
    # Determine the approximate number of images per row
    images_per_row = math.ceil(num_images / row_num)
    row_images = [images[i * images_per_row:(i + 1) * images_per_row] for i in range(row_num)]
    
    # Calculate the total width and height needed for the combined image
    total_width = max(sum(img.width for img in row) + gap * (len(row) - 1) for row in row_images)
    total_height = sum(max(img.height for img in row) for row in row_images) + gap * (len(row_images) - 1)

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # White background

    # Paste each row of images with the specified gap
    y_offset = 0
    for row in row_images:
        x_offset = 0
        for img in row:
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width + gap
        y_offset += max(img.height for img in row) + gap

    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved at {output_path}")


# Editable variables
row_num = 2  # Specify the number of rows for arranging images
gap = 10  # Gap between images
frame_thickness = 5  # Thickness of the border around each image
enlarged = True

# Directories and paths
data_dir = f'./data{"_enlarged" if enlarged else ""}'
image_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('ies')]
save_dir = f'./imageArray_{row_num}row{"_enlarged" if enlarged else ""}'

# Combine images from each directory and save
for image_dir in image_dirs:
    case_name = os.path.basename(image_dir)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    
    # Create a subfolder for each case in the save directory
    case_save_dir = os.path.join(save_dir, case_name)
    os.makedirs(case_save_dir, exist_ok=True)
    
    # Define the image name based on row number and enlargement status
    save_name = f'{row_num}row{"_enlarged" if enlarged else ""}.png'
    output_path = os.path.join(case_save_dir, save_name)
    
    # Combine images with the specified row number, gap, and frame thickness
    combine_images(image_paths, output_path, row_num=row_num, gap=gap, frame_thickness=frame_thickness)
