import os
import cv2
from tqdm import tqdm

fonts = os.listdir('batch_intp_output')


for font in tqdm(fonts):
    try:
        if os.path.exists(f'batch_intp_output/{font}/font_combined.png'):
            print("Already exists!")
            continue
        images = []
        # keep on adding img to images vertically with a margin of 2
        chars = os.listdir(f'batchcd intp_output/{font}')
        # sort 
        chars = sorted(chars, key=lambda x: (x.endswith('+'), x.lower()))

        for char in chars:
            img = cv2.imread(f'batch_intp_output/{font}/{char}/intp.png')
            img = img[750:1250,]
            img = cv2.resize(img, (1100, 200))  # Resize each image to 1080x1080
            images.append(img)
        

        # Combine images vertically with a margin of 2
        final_image = images[0]
        for img in images[1:]:
            final_image = cv2.vconcat([final_image, img])

        # Save the final image
        cv2.imwrite(f'batch_intp_output/{font}/font_combined_new.png', final_image)
        
    except:
        continue