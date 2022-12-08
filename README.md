<p align="center">
  <img width="100" height="96" src="https://cdn.discordapp.com/attachments/999941428052500632/1000242308177993748/vitchen2.png" alt="Astro Diffusion">
</p>
<h1 align="center">Astro Diffusion v0.7</h1>

 
 
### Optimized, Customizable AI Artists
<!-- Original Deforum SD: https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb -->

1. <a href="https://colab.research.google.com/drive/1FgiGFa6rkUMCyzxUOleusYWfp1LBr5Sh?usp=sharing" alt="Deforum SDF">Deforum Stable Diffusion</a>
2. <a href="https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb" alt="Disco DF">Disco Diffusion</a>
3. 15+ more diffusion models
4. EDSR image and video super resolution
5. [DistilGPT2-powered stable diffusion prompt generator](https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion)
   * safe for work
   * finetuned on 2.03M prompts for stable diffusion v1.1 through v1.4
   * 50% faster forwardproporgation, 40% less RAM usage than other HuggingFace prompt generators
6. DistilBART-powered animation prompt generator
   * safe for work
   * given a name, describes the appearance or story of the pony, fiction, manga, or anime character
7. Fast Danbooru search engine
   * filtered, safe for work
   * allows users to fully customize anime characters: gestures, postures, scenaries, objects, etc.
 
 
### Text2Img Examples
1. [Anime Style](/art-examples/astro_anime.md)
2. [Pony Style](/art-examples/astro_pony.md)
3. [Van Gogh Style](/art-examples/astro_van_gogh.md)
 
 
### Setup for NVIDIA GPU Users
1. [Download CUDA v11.2](https://developer.nvidia.com/cuda-downloads)
2. [Join the NVIDIA Developer Program and download cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
3. Move all files in cuDNN to the CUDA folder (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin)
4. [Install Git](https://git-scm.com/downloads)
5. In a terminal, run `git clone https://github.com/FredZhang7/Astro-Diffusion.git​​` to save this repo to local
6. [Install Python ≥3.9, ≤3.10](https://www.python.org/downloads/), or `pyenv install 3.10.4` and `pyenv global 3.10.4`
7. Read the comments and instructions in `astro_image.py` to generate photos, pixelart, paintings, and more
8. In a terminal, run `git pull https://github.com/FredZhang7/Astro-Diffusion.git​​` to update to the latest version


### Setup for CPU-only
1. Start from step #4 (Install Git) in the NVIDIA section
 
 
### To-do
1. RealESRGAN super image and video resolution
2. Disco Df config, including AMD and other GPU compatibility
3. Mini Google SafeSearch - cut RAM usage of stable diffusion safety checker down to 25%
4. Deforum SD video generation
5. Disco Df video generation
6. Space Diffusion - train on 50,000+ high res images of galaxies, stars, universe, nebula, planets, etc.
7. Release to Google Colab for users to run on free A100 GPUs
 
 
### Fun Facts
1. This repo is quite new - only googleable since Nov 16th, 2022. I will constantly improve this repo for at least a few months.
2. Model checkpoints or embeddings ending in .pt aren't compatible atm, but will soon.
3. The logo for this repo was drawn by hand.
