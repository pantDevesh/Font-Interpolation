# Font Interpolation

<img src="https://github.com/pantDevesh/Font-Interpolation/raw/master/diffae/imgs/font_intp.png" alt="Font Interpolation" width="80%" height="80%">


We aim to address the challenge of limited fonts for low-resource languages, which results in reduced Scene Text Recognition accuracy. Given two font families, our focus is on creating multiple fonts by interpolating between them. These newly generated fonts can be utilized to create synthetic datasets for model training, thereby improving OCR accuracy on real-world data. This work only targets Latin font generation.

This repository contains the code for:
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
   - Various off-the-shelf GAN & diffusion models have been explored for Font Interpolation, some mentioned in the Literature section.
   - We opted for Diffusion autoencoder ([DiffAE](https://github.com/phizaz/diffae)) as the baseline due to its promising results in general image interpolation.

    <img src="https://github.com/pantDevesh/Font-Interpolation/raw/master/diffae/imgs/main_arch.png" alt="Model Architecure" width="40%" height="40%">


   - The architecture was slightly modified by adding an MLP for font classification, aiming to improve the image latent space of the font. The model was trained on a synthetic dataset generated using Synthtiger (1M images with 5 fonts). The OCR accuracy results below are from this model. However, we also experimented with training the model solely on font images from all English fonts (~200K images), observing overfitting.
   - Checkpoints: `/home/data/submit/WORKING/devesh/Font-Interpolation/diffae_checkpoints`

   #### Training:
       1. Run `https://github.com/pantDevesh/Font-Interpolation/blob/master/diffae/run_fonts_ddim.py` to train the model
       2. Note that we only need to train the encoder, we don't train latent DPM.
   #### Inference:
       1. Run [interpolate.ipynb](https://github.com/pantDevesh/Font-Interpolation/blob/master/diffae/interpolate.ipynb) to diffae interpolation.
       2. Run [interpolate_pullback.ipynb](https://github.com/pantDevesh/Font-Interpolation/blob/master/diffae/interpolate_pullback.ipynb) for interpolating through parallel transport, as mentioned in this paper(https://arxiv.org/abs/2307.12868)
       3. For running interpolation on all the pairs, run prepare_fonts_nc2.py to build pairs of fonts to be interpolated, run batch_intp.py to perform interpolation.

3.  Generating TTF fonts from interpolated images: <br>
      1. Each font interpolation yields 10 images (2 original and 8 interpolated). To select a character image from these, we utilize an English OCR model. Among the 3rd to 7th interpolated images, we choose the one that the model correctly predicts with the lowest confidence. This approach ensures both correctness and variability.
      2. `place_font_on_template.py` script will automatically place the font images at appropriate places in the given template (this greatly saves time to create fonts manually 😊). You also need to download checkpoint of [PARSeq] (https://github.com/baudm/parseq) OCR model. 
      3. Upload this template to https://www.calligraphr.com/ for generating ttf fonts.
     
4.  Generating synthetic datasets from interpolated fonts:
      We use [synthtiger ](https://github.com/clovaai/synthtiger/) to generate Synthetic Data from the given fonts
5.  Training and benchmarking OCR model on the generated datasets:
      We use [PARSeq] (https://github.com/baudm/parseq) model to train and benchmark OCR results. Results of OCR accuracy with model trained on 1M synthetic data generated using 10 random real fonts vs 10 + 45 addional interpolated fonts. 

      | Fonts                                     | Word Accuracy | Character Accuracy |
      |-------------------------------------------|---------------|--------------------|
      | 10 random original fonts (1M)             | 70.56         | 87.12              |
      | 10 original + 45 Interpolated fonts (1+1M)| 81.45         | 92.19              |



### Limitations and Difficulties in the Current Work:
1. The model architecture has limited novelty. Additionally, in terms of qualitative results, we don't achieve the same level as mentioned in the existing works ([Link](https://arxiv.org/html/2402.14311v1)).
2. The OCR results are satisfactory, but we are comparing results after training the model on datasets of 1M images vs 2M images. It's suggested to generate 2M images with the original 10 fonts and then compare accuracy.
3. The Pullback and parallel transport utilized in `interpolate_pullback.ipynb` isn't working as expected 😔.
4. **Difficulty in Indic Language Font Interpolation:**<br>
Generating fonts from images of Devanagari characters presents a challenge, primarily due to the presence of Matras or Half Characters. These special characters require precise positioning in the template while creating TTF font. For further details, please refer to: [Microsoft Typography - Devanagari Script Development](https://learn.microsoft.com/en-us/typography/script-development/devanagari).

### Future Work:
1. Debug `interpolate_pullback.ipynb`.
2. Explore Bezier curve direction for interpolation in curve space instead of image space.
3. Explore Indic languages, if not Devanagari, then perhaps some simpler but low-resource language.

### Literature for Font Interpolation
1. https://www.lucasfonts.com/learn/interpolation-theory- Talks about interpolation between the fonts belonging to the same family. Focuses on generating fonts with different stroke widths.
2. Learning a Manifold of Fonts - (SIGGRAPH 2014) https://dl.acm.org/doi/pdf/10.1145/2601097.2601212
3. Attribute2Font: Creating Fonts You Want From Attributes- (SIGGRAPH 2022) https://dl.acm.org/doi/pdf/10.1145/3386569.3392456 (https://github.com/hologerry/Attr2Font)
4. Font Style Interpolation with Diffusion Models (https://arxiv.org/html/2402.14311v1)
5. SKELETON ENHANCED GAN-BASED MODEL FOR BRUSH HANDWRITING FONT GENERATION(https://arxiv.org/pdf/2204.10484.pdf)
6. Neural Transformation Fields for Arbitrary-Styled Font Generation (https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Neural_Transformation_Fields_for_Arbitrary-Styled_Font_Generation_CVPR_2023_paper.pdf) 

