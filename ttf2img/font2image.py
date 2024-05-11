# Extracts character images from ttf files

from utils import *
import os

def get_fonts():
    files = os.listdir('/home/devesh_temp/scratch/English_data_gen/synthtiger/resources/artistic_fonts_new')   #in dir contains ttf files
    font_path_list = []
    for filename in files:
        if filename.endswith('txt'):
            try:
            # print(filename, open('latin_fonts/'+filename, 'r').readlines())
                line = open('/home/devesh_temp/scratch/English_data_gen/synthtiger/resources/artistic_fonts_new/'+filename, 'r').readlines()[0]
                flag = 0
                for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                    if a not in line:
                        flag = 1
                        break

                if flag:
                    continue
                else:
                    font_path_list.append('/home/devesh_temp/scratch/English_data_gen/synthtiger/resources/artistic_fonts_new/'+filename[:-4]+'.ttf')

            except Exception as E:
                print(E)
                continue
            
    return font_path_list


common_list_en = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# ------------------------------
# For Devanagari
unicode_min = 0x0900 
unicode_max = 0x097F
printable_glyphs = [ chr(x) for x in range(unicode_min, unicode_max+1) if chr(x).isprintable() ]
s1 = []
char_dict_hi = {}
for itr, i in enumerate(printable_glyphs):
    s1.append(itr)
    char_dict_hi[i] = itr
# ------------------------------

common_list_hi = s1
# get font files
font_path_list = get_fonts()

for font_path in font_path_list:
    image_file_hi = 'Devesh_data/hi_all_imgs/' # out dir hindi images
    image_file_eng = 'Devesh_data/eng_artistic_imgs' # out dir for english images

    os.makedirs(image_file_hi, exist_ok=True)
    os.makedirs(image_file_eng, exist_ok=True)
    try:
        print(font_path, image_file_hi, image_file_eng, common_list_en, common_list_hi)
        font2image(font_path, image_file_eng, image_file_hi, common_list_en, char_dict_hi, 64)
    except Exception as E:
        print(E)
        continue

    