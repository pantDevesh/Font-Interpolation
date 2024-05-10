
# This script will automatically place font images at appropriate place in the Caligraphr Template
import sys
import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
from parseq.ocr import PARSeqOCR

template_name = "./Calligraphr-Template.png"
font_folder_root_path = '/home/devesh_temp/scratch/diffae_hpc/diffae/batch_10_random_intp_output/' 
output_path = 'Prepared_10_random_Templates' 
os.makedirs(output_path, exist_ok=True)

def place_font_on_template(template_name, font_root_path, output_path, ocr_model):
    # Load the template image
    template_image = cv2.imread(template_name, 1)

    # read coordinates from csv
    df = pd.read_csv('letter_coordinates.csv')
    # convert to dict
    char_to_coords = {}
    for i in range(len(df)):
        char_to_coords[df['Letter'][i]] = (df['X1'][i], df['Y1'][i], df['X2'][i], df['Y2'][i])
        
    # Load the font images
    char_folders = os.listdir(font_root_path)
    # sort the list
    char_folders.sort()
    # if '.DS_Store' in char or 'intp' in char remove it
    char_folders = [char for char in char_folders if '.DS_Store' not in char and 'intp' not in char]
    # remove files keep ionly directories
    char_folders = [char for char in char_folders if os.path.isdir(os.path.join(font_root_path, char))]
    
    for char in char_folders:
        index = 1; curr_conf = 1.0
        for i in range(3,7): # we ignore first 3 and last 2 images as there is not much interpolation there
            img_path = os.path.join(font_root_path, char, str(i)+'.png')
            result, conf = ocr_model.run(img_path)
            conf = conf[0]; result = result[0][0]
            
            char_actual = char
            if '+' in char_actual:
                char_actual = char_actual.replace('+', '').upper()
            else:
                char_actual = char_actual.lower()
                char = char.lower()
                
            if result == char_actual:
                if conf > 0.5 and conf < curr_conf:
                    curr_conf = conf
                    index = i
                        
        img_path = os.path.join(font_root_path, char, str(index)+'.png')
        font_image = cv2.imread(img_path)
        font_image = cv2.resize(font_image, (128,128))
        template_image[char_to_coords[char_actual][1]+40:char_to_coords[char_actual][1]+168, char_to_coords[char_actual][0]+30:char_to_coords[char_actual][0]+158] = font_image

    return template_image

# load ocr
ocr_model = PARSeqOCR()

for font in tqdm(os.listdir(font_folder_root_path)):
    if '.DS_Store' in font:
        continue
    template_img = place_font_on_template(template_name, os.path.join(font_folder_root_path, font), output_path, ocr_model)
    cv2.imwrite(os.path.join(output_path, f'{font}.png'), template_img)




