'''
@author: Abder-Rahman Ali, PhD
@email: aali25@mgh.harvard.edu
@date: 2024
'''

import os
import random
import shutil

source_base = "./"
destination_base = "./data/training/"

class_folders = [f"class_{i}" for i in range(1, 6)]
centroid_folders = [f"class_{i}_centroid" for i in range(1, 6)]
outlier_folders = [f"class_{i}_outliers" for i in range(1, 6)]

def copy_random_images(src_folder, num_images):
    src_files = [f for f in os.listdir(src_folder) if f.endswith(".png")]
    random.shuffle(src_files)
    for file in src_files[:num_images]:
        dst_folder = determine_destination_folder(file)
        if dst_folder is not None:
            os.makedirs(dst_folder, exist_ok=True)
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, file)
            shutil.copy(src_path, dst_path)

def determine_destination_folder(image_name):
    paths = [
        "./original_data/data/training/A",
        "./original_data/data/training/B",
        "./original_data/data/validation/A",
        "./original_data/data/validation/B",
        "./original_data/data/test/A",
        "./original_data/data/test/B"
    ]
    for path in paths:
        images = [f for f in os.listdir(path) if f.endswith(".png")]
        if image_name in images:
            return os.path.join(destination_base, os.path.basename(path))
    print(f"Skipping image: {image_name} (No corresponding image in original data)")
    return None

num_random_images = 10
for class_folder in class_folders:
    src_folder = os.path.join(source_base, class_folder)
    if os.path.exists(src_folder):
        copy_random_images(src_folder, num_random_images)

for centroid_folder in centroid_folders:
    src_folder = os.path.join(source_base, centroid_folder)
    if os.path.exists(src_folder):
        copy_random_images(src_folder, 1)

for outlier_folder in outlier_folders:
    src_folder = os.path.join(source_base, outlier_folder)
    if os.path.exists(src_folder):
        copy_random_images(src_folder, len(os.listdir(src_folder)))

print("Images copied successfully.")