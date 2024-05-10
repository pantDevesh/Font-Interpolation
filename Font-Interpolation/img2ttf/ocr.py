import argparse
import torch
from PIL import Image
from .strhub.data.module import SceneTextDataModule
from .strhub.models.utils import load_from_checkpoint, parse_model_args




class PARSeqOCR:
    
    def __init__(self, checkpoint="pretrained=parseq", device='cuda'):
        self.checkpoint = checkpoint
        parser = argparse.ArgumentParser()
        self.args, unknown = parser.parse_known_args()
        kwargs = parse_model_args(unknown)

        self.model = load_from_checkpoint(checkpoint, **kwargs).eval().to(device)
        self.device = device
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
    
    
    def run(self, image_path):
        # Load image and prepare for input
        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image).unsqueeze(0).to(self.device)

        p = self.model(image).softmax(-1)
        confidence = p.max(-1).values[0]

        pred, p = self.model.tokenizer.decode(p)
        
        return pred, p[0]