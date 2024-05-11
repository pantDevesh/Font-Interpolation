# Prep for interpolating pairs of characters
# Given a font directory
# Create font pairs for nc2 combination of fonts


import os
import random
import numpy as np
from tqdm import tqdm


font_path = '/home/devesh_temp/scratch/str_kiran/font_translator_gan_hindi/font_translator_gan/datasets/Devesh_data/eng_all_imgs/'
result_path = 'prepared_10_random_fonts'

fonts = os.listdir(font_path)
fonts = random.sample(fonts, 10)
already_picked = {}

# for the given fonts, create nc2 combinations
from itertools import combinations
combs = list(combinations(fonts, 2))

for comb in combs:
    new_folder = os.path.join(result_path, comb[0] + '_' + comb[1])
    os.makedirs(new_folder, exist_ok=True)
    
    
    
    for img in os.listdir(os.path.join(font_path, comb[0])):
        os.system(f'cp {os.path.join(font_path, comb[0], img)} {os.path.join(new_folder, comb[0] + "_" + img)}')
    for img in os.listdir(os.path.join(font_path, comb[1])):
        os.system(f'cp {os.path.join(font_path, comb[1], img)} {os.path.join(new_folder, comb[1] + "_" + img)}')
        
        