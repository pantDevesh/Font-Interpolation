# Font Interpolation

<img src="https://github.com/pantDevesh/Font-Interpolation/raw/master/diffae/imgs/font_intp.png" alt="Font Interpolation" width="80%" height="80%">


Problem Statement: We aim to address the challenge of limited fonts for low-resource languages, which results in reduced Scene Text Recognition accuracy. Given two font families, our focus is on creating multiple fonts by interpolating between them. These newly generated fonts can be utilized to create synthetic datasets for model training, thereby improving OCR accuracy on real-world data. This work only targets Latin font generation.

This repository includes the code for:
   1. Generating images from TTF Fonts.
   2. Performing Font interpolation using the [DiffAE model](https://github.com/phizaz/diffae).
   3. Generating TTF fonts from interpolated images.
   4. Generating synthetic datasets from interpolated fonts.
   5. Training and benchmarking OCR model on the generated datasets.

### Requirements:
Create a conda envrionment with the given requirement.txt file.

### Instructions:
1. Generate images from TTF fonts:
   - Run [font2image.py](https://github.com/pantDevesh/IITD-Work/blob/master/Font-Interpolation/ttf2img/font2image.py) script to extract images from the ttf font.
2. Font Interpolation:
    <img src="https://github.com/pantDevesh/Font-Interpolation/raw/master/diffae/imgs/main_arch.png" alt="Model Architecure" width="40%" height="40%">
   - Various off-the-shelf GAN & diffusion models have been explored for Font Interpolation, some mentioned in the Literature section.
   - We opted for Diffusion autoencoder ([DiffAE](https://github.com/phizaz/diffae)) as the baseline due to its promising results in general image interpolation.
   - The architecture was slightly modified by adding an MLP for font classification, aiming to improve the image latent space of the font. The model was trained on a synthetic dataset generated using Synthtiger (1M images with 5 fonts). The OCR accuracy results below are from this model. However, we also experimented with training the model solely on font images from all English fonts (~200K images), observing overfitting.
   - Checkpoints: `/home/data/submit/WORKING/devesh/Font-Interpolation/diffae_checkpoints`

   Training:
       1. Run `https://github.com/pantDevesh/Font-Interpolation/blob/master/diffae/run_fonts_ddim.py` to train the model
       2. Note that we only need to train the encoder, we don't train latent DPM.
   Inference:
       1. Run [interpolate.ipynb](https://github.com/pantDevesh/Font-Interpolation/blob/master/diffae/interpolate.ipynb) to diffae interpolation.
       2. Run [interpolate_pullback.ipynb](https://github.com/pantDevesh/Font-Interpolation/blob/master/diffae/interpolate_pullback.ipynb) for interpolating through parallel transport, as mentioned in this paper(https://arxiv.org/abs/2307.12868)
   

### Difficulty in Indic Language Font Interpolation
Generating fonts from images of Devanagari characters presents a challenge, primarily due to the presence of Matras or Half Characters. These special characters require precise positioning in the template while create TTF font. For further details, please refer to: [Microsoft Typography - Devanagari Script Development.](https://learn.microsoft.com/en-us/typography/script-development/devanagari)


### Literature for Font Interpolation
1. https://www.lucasfonts.com/learn/interpolation-theory- Talks about interpolation between the fonts belonging to the same family. Focuses on generating fonts with different stroke widths.
2. Learning a Manifold of Fonts - (SIGGRAPH 2014) https://dl.acm.org/doi/pdf/10.1145/2601097.2601212
3. Attribute2Font: Creating Fonts You Want From Attributes- (SIGGRAPH 2022) https://dl.acm.org/doi/pdf/10.1145/3386569.3392456 (https://github.com/hologerry/Attr2Font)
4. Font Style Interpolation with Diffusion Models (https://arxiv.org/html/2402.14311v1)
5. SKELETON ENHANCED GAN-BASED MODEL FOR BRUSH HANDWRITING FONT GENERATION(https://arxiv.org/pdf/2204.10484.pdf)
6. Neural Transformation Fields for Arbitrary-Styled Font Generation (https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Neural_Transformation_Fields_for_Arbitrary-Styled_Font_Generation_CVPR_2023_paper.pdf) 

