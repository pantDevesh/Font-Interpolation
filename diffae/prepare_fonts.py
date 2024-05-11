# Prep for interpolating pairs of characters
# Given a font directory
# Randomly pick two fonts, create a folder and put images with proper naming convention it it.

import os
import numpy as np
from tqdm import tqdm


font_path = '/home/devesh_temp/scratch/str_kiran/font_translator_gan_hindi/font_translator_gan/datasets/Devesh_data/eng_all_imgs/'
result_path = 'prepared_new_fonts'


fonts = os.listdir(font_path)
# fonts = [font for font in fonts if font.startswith('EBGaramond-Regular')]

print(len(fonts))
already_picked = {}

counter = 0
while tqdm(counter < 1700):
    if len(os.listdir(result_path)) == 1700:
        break
    font1 = None
    font2 = None
    while font1 is None or font2 is None:
        font1 = fonts[np.random.randint(0, len(fonts))]
        font2 = fonts[np.random.randint(0, len(fonts))]
        if font1 == font2 or already_picked.get(font1) is not None or already_picked.get(font2) is not None or (font1 + '_' + font2) in os.listdir(result_path):
            font1 = None
            font2 = None
    # already_picked[font1] = True
    # already_picked[font2] = True
    new_folder = os.path.join(result_path, font1 + '_' + font2)
    os.makedirs(new_folder, exist_ok=True)
    for img in os.listdir(os.path.join(font_path, font1)):
        os.system(f'cp {os.path.join(font_path, font1, img)} {os.path.join(new_folder, font1 + "_" + img)}')
    for img in os.listdir(os.path.join(font_path, font2)):
        os.system(f'cp {os.path.join(font_path, font2, img)} {os.path.join(new_folder, font2 + "_" + img)}')
    
    counter += 1
    # print(f'Created folder {new_folder} with {len(os.listdir(new_folder))} images')