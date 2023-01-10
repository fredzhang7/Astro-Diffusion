<p align="center">
  <img width="100" height="96" src="https://media.discordapp.net/attachments/884528247998664744/1062284479613390888/astro_no_smudge_round.png" alt="Astro Diffusion">
</p>
<h1 align="center">Innovating Text-to-Video Generation with Improved Coherence and Logic</h1>

<br>

## Message From Author
This repository is similar to [Deforum Stable Diffusion](https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb) in that both are based on the image-to-image and text-to-image methods of Stable Diffusion. However, Astro Stable Diffusion differs in that it uses non-interpolation methods to create videos. In the coming months, [AMLA](https://github.com/AMLA-UBC) will introduce and release new open-source models that are not based on diffusion models or derivative work. We believe that generating coherent and logical videos longer than one minute, such as short action films and music videos, requires models that are capable of learning the relationship between vectors in videos and music, rather than relying on diffusion or interpolation techniques.

See my earlier work at [HuggingFace](https://huggingface.co/FredZhang7) and the [previous](./previous) folder.

<br>

## Description
The [Drone View V1](https://github.com/FredZhang7/Astro-Diffusion/wiki/Drone-View-V1) feature enables you to create a video from a drone's perspective by providing a description or prompt for the scene. While the drone is set to autopilot mode, you can modify its movements and responses to obstacles inside the `DroneArgs` class. See [here](https://www.youtube.com/playlist?list=PLCFlAfr2X8n2BxB9ZgKOVTG1WggWpnts0) for examples of video outputs.

Text-to-video generation for Drone View V2, Virtual Reality, Panorama Photography, and Pan Shot are currently being developed.

<br>

## Usage
Run `setup.sh` to download the required packages and repositories. Download and save a Stable Diffusion model to the `./stable-diffusion-webui/models/Stable-diffusion` folder. Lastly, launch `webui-user.bat` in `./stable-diffusion-webui` before running Astro Stable Diffusion plugins.

<br>

## Citations
```
@article{Forsgren_Martiros_2022,
  author = {Forsgren, Seth* and Martiros, Hayk*},
  title = {{Riffusion - Stable diffusion for real-time music generation}},
  url = {https://riffusion.com/about},
  year = {2022}
}
```
```
@article{DBLP:journals/corr/abs-2005-12872,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  journal   = {CoRR},
  volume    = {abs/2005.12872},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.12872},
  archivePrefix = {arXiv},
  eprint    = {2005.12872},
  timestamp = {Thu, 28 May 2020 17:38:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-12872.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```