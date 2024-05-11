from templates import *
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_path = os.path.join(self.folder_path, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {'imgname': img_name, 'img': img, 'index': index}



# load model
device = 'cuda:0'
conf = font32_autoenc_10K()
model = LitModel(conf)
state = torch.load(f'checkpoints/font128_autoenc/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)


# inference
fonts = os.listdir('prepared_artistic_fonts')
existing_fonts = os.listdir('batch_new_artistic_intp_output')

for font in tqdm(fonts):
    try:
        if font in existing_fonts:
            print(f'{font} already exist!')
            continue
        
        if font.count('_') == 2:  # Check if '_' appears twice in the font name
            continue  # Ski
        
        
        print(f'Processing {font}')
        
        zsem_dict = {}
        xt_dict = {}
        font_path = os.path.join('prepared_artistic_fonts', font)
        dataset = CustomDataset(font_path, transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        for batch in tqdm(dataloader):
            img = batch['img'].to(device)
            cond = model.encode(img.to(device), None, None)
            xT = model.encode_stochastic(img.to(device), cond, T=10)
            for i in range(len(batch['imgname'])):
                zsem_dict[batch['imgname'][i]] = cond[i]
                xt_dict[batch['imgname'][i]] = xT[i]
                
                
        # interpolate
        font1 = font.split('_')[0]
        font2 = font.split('_')[1]
        
        pairs = {}


        # Define a function to process each image
        def process_image(args):
            imgname, font1, font2, zsem_dict, xt_dict, model = args
            if font1 in imgname:
                cond1 = zsem_dict[imgname]
                xT1 = xt_dict[imgname]
                
                imgname2 = imgname.replace(font1, font2)
                cond2 = zsem_dict[imgname2]
                xT2 = xt_dict[imgname2]
                
                alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device) 
                intp = cond1[None] * (1 - alpha[:, None])  + cond2[None] * alpha[:, None]
                
                def cos(a, b):
                    a = a.view(-1)
                    b = b.view(-1)
                    a = F.normalize(a, dim=0)
                    b = F.normalize(b, dim=0)
                    return (a * b).sum()

                theta = torch.arccos(cos(xT1, xT2))
                x_shape = xT1.shape
                intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT1.flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT2.flatten(0, 2)[None]) / torch.sin(theta)
                intp_x = intp_x.view(-1, *x_shape)
                resize_transform = transforms.Resize((128, 256))
                pred = model.render(intp_x, intp, T=100)
                pred = [resize_transform(image) for image in pred]
                write_path = f'batch_new_artistic_intp_output/{font1}_{font2}/{imgname.split("_")[-1].split(".")[0]}'
                os.makedirs(write_path, exist_ok=True)
                fig, ax = plt.subplots(1, 10, figsize=(10*10, 20))
                for i in range(len(alpha)):
                    ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
                    image = TF.to_pil_image(pred[i])
                    image.save(f'{write_path}/{i}.png')
                plt.savefig(f'{write_path}/intp.png')
                
                # close figure
                plt.close()


        # Prepare arguments for the process_image function
        args_list = [(imgname, font1, font2, zsem_dict, xt_dict, model) for imgname in zsem_dict.keys()]
        for args in args_list:
            process_image(args)
            
    except: 
        continue
    

