'''
@author: Abder-Rahman Ali, PhD
@email: aali25@mgh.harvard.edu
@date: 2024
'''

import pandas as pd
import shutil
import os

csv_files = ['class_1.csv', 'class_2.csv', 'class_3.csv', 'class_4.csv', 'class_5.csv']

image_path = './data/unlabeled/'

for file in csv_files:
    df = pd.read_csv(file)
    
    class_num = file.split('_')[1].split('.')[0]
    
    os.makedirs(f'class_{class_num}_centroid', exist_ok=True)
    os.makedirs(f'class_{class_num}_outliers', exist_ok=True)
    os.makedirs(f'class_{class_num}', exist_ok=True)
    
    for index, row in df.iterrows():
        image_name = row['image_name']
        
        if row['is_centroid']:
            shutil.copy(image_path + image_name, f'class_{class_num}_centroid')
        elif row['is_outlier']:
            shutil.copy(image_path + image_name, f'class_{class_num}_outliers')
        else:
            shutil.copy(image_path + image_name, f'class_{class_num}')