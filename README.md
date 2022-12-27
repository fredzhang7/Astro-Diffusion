<p align="center">
  <img width="100" height="96" src="https://cdn.discordapp.com/attachments/999941428052500632/1000242308177993748/vitchen2.png" alt="Astro Diffusion">
</p>
<h1 align="center">Astro Diffusion v0.8</h1>
 
 
### Optimized, Customizable AI Artists
<!-- Original Deforum SD: https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb -->

1. [Deforum Stable Diffusion](https://colab.research.google.com/drive/1FgiGFa6rkUMCyzxUOleusYWfp1LBr5Sh?usp=sharing)
   * Text2Img, Img2Img, Inpainting, Outpainting
2. [Disco Diffusion](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb)
3. 23+ more diffusion models
   * Painting Midjourney (v1)
   * Modern Midjourney (v2)
4. Image, GIF, and video super resolution
   * R-ESRGAN General WDN 4x V3
   * R-ESRGAN 4x+ Anime6B
   * ESRGAN 4x
   * Swin2SR 4x
   * R-ESRGAN 2x+
   * PortraitSR
      * can switch between realism (0.0 uncertainty) and painting (1.0 uncertainty)
5. [DistilGPT2 Stable Diffusion v2](https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2)
   * safe for work
   * finetuned on 2,470,000+ prompts for stable diffusion v1.1 through v1.5
   * 50% faster forwardproporgation, 40% less RAM usage, 25%+ more variations, more fluent than other HuggingFace prompt generators
6. DistilBART-powered animation prompt generator
   * safe for work
   * given a name, describes the appearance or story of a pony, video game, fiction, manga, or anime character
7. [] and <> support in anime prompts
   * randomizes what's inside: gestures, postures, scenaries, objects, etc.
8. [Google Safesearch Mini](https://huggingface.co/FredZhang7/google-safesearch-mini)
   * finetuned on 2,220,000+ images classified as safe/unsafe by Google Safesearch, Reddit, and Imgur
   * 92% less RAM usage than stable diffusion safety checker
9. [Anime/Animation WebUI](./art_examples/anime_webui_preview.md)
   * Light and dark modes
   * Safe for work
   * Tab 1 - Customizations
      * fully customize your anime characters and scenes
   * Tab 2 - Search Engine
      * search for random images (including prompts and upvotes) containing give tags
      * leave tags as blank to generate random samples
10. Bilingual
    * Chinese (Simplified and Traditional)
    * English
11. Miscellaneous
    * Organized console outputs
    * Usable as a Discord Bot


<br>

### Super Res Examples
1. [Video Upscale](https://www.youtube.com/playlist?list=PLCFlAfr2X8n1oFJMEcVTCuq3df5yPcZEf)
2. [GIF Upscale](https://imgur.com/a/IEdJiyY)
3. [ImageSR and PortraitSR](https://imgur.com/a/bfRMEBt)

### Text2Img Examples
1. [Anime Models](/art_examples/astro_anime.md)
2. [Pony Model](/art_examples/astro_pony.md)
3. [Van Gogh Model](/art_examples/astro_van_gogh.md)

<br>

### Setup for NVIDIA GPU Users
1. [Download CUDA v11.2](https://developer.nvidia.com/cuda-downloads)
2. [Join the NVIDIA Developer Program and download cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
3. Move all files in cuDNN to the CUDA folder (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin)
4. [Install Git](https://git-scm.com/downloads)
5. In a terminal, run `git clone https://github.com/FredZhang7/Astro-Diffusion.git​​` to save this repo to local
6. [Install Python ≥3.9, ≤3.10](https://www.python.org/downloads/), or `pyenv install 3.10.4` and `pyenv global 3.10.4`
7. Read the comments in `astro_image.py` to generate images, `webui.py` to run anime webui, `superres.py` to upscale or beautify imgs/gifs/vids, etc.
8. In a terminal, run `git pull https://github.com/FredZhang7/Astro-Diffusion.git​​` to update to the latest version


### Setup for CPU-only
1. Start from step #4 (Install Git) in the NVIDIA section

<br>

### To-do
1. Disco Df parameters
2. New video generation methods
3. Deforum SD & Disco Df video generation
4. Space Diffusion - train on 50,000+ high res images of galaxies, stars, universe, nebula, planets, etc.
5. Midjourney
6. Release to Google Colab for users to run on free A100 GPUs


### Fun Facts
1. This repo is quite new - only googleable since Nov 16th, 2022. I will constantly improve this repo until text2video surpasses that of Google/Meta.
2. Model checkpoints or embeddings ending in .pt aren't compatible atm, but will soon.
3. The logo for this repo was drawn by hand.
