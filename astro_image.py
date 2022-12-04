# Uncomment the following to automatically install the required libraries
# from setup import setup_environment
# setup_environment(True, True)


from types import SimpleNamespace
import os, time, random, gc, torch
from astro_diffusion import render_image_batch


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path, time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


"""
    Stable Diffusion Style
     sd-v2-0.ckpt                        (5.2 GB, latest, family-friendly, highest resolution, uses negative prompts, general artwork, medium VRAM)
     sd-v1-5.ckpt                        (4.2 GB, higher resolution, general artwork, medium VRAM)
     sd-v1-1.ckpt                        (4.2 GB, lowest accuracy, general artwork, medium VRAM)

    Animated Style
     anime-anything-v3.ckpt              (3.8 GB, highest accuracy & resolution, anime-style characters, food, animals, scenaries, etc., medium VRAM)
     anime-trinart.ckpt                  (2.1 GB, accurate, detailed, high res, anime-style characters, low VRAM)
     anime-cyberpunk.ckpt                (2.1 GB, accurate, clean-cut, high res, cyberpunk anime drawings, low VRAM)
     disney-diffusion.ckpt               (2.1 GB, high res Disney-style characters, animals, cars, & landscapes, low VRAM)
     pony-diffusion-v2.ckpt              (3.7 GB, high res pony characters, medium VRAM)
     pokemon-diffusion.ckpt              (4.2 GB, new pokedex pokemons, medium VRAM)

    Robo Style
     robo-diffusion-v1.ckpt              (4.2 GB, high-quality robot, cyborg, android drawings, medium VRAM)

    Drawing Styles
     vector-art.ckpt                     (2.1 GB, high-quality vector art, medium VRAM)
     nitro-diffusion.ckpt                (2.1 GB, mix of archer, arcane or modern disney styles, medium VRAM)
     scifipulp-diffusion.pt              (0.4 GB, sci-fi pulp art by KaliYuga-ai, low VRAM)
     portrait-diffusion.pt               (0.5 GB, portraits generator, low VRAM)
     pixelart-diffusion-expanded.pt      (0.4 GB, pixel art by KaliYuga-ai, low VRAM)
     pixelart-diffusion-sprites.ckpt     (4.2 GB, generate pixel art sprite sheets from four different angles, medium VRAM)

    Painting Styles
     van-gogh-diffusion-v2.ckpt          (2.1 GB, high-quality Van Gogh paintings, Loving Vincent, low VRAM)
     watercolor-diffusion-v2.pt          (0.4 GB, watercolor art by KaliYuga-ai, low VRAM)

    Craft Style
     popup-book.ckpt                     (4.2 GB, pop-up book illustrations, medium VRAM)

    OpenAI Style
     openai-256x256-diffusion.pt         (2.1 GB, trained on 256x256 images, low VRAM)
     openai-512x512-diffusion.pt         (2.1 GB, trained on 512x512 images, low VRAM)
"""


def AstroArgs():
    # Model Settings
    model_checkpoint = "anime-anything-v3.ckpt"    # one of "custom", a model checkpoint listed above. if you have no clue, use "sd-v1-5.ckpt" for starters
    check_sha256 = False                 # whether to check the sha256 hash of the checkpoint file. set to True if you have issues with model downloads
    custom_config_path = ""              # if model_checkpoint "custom", path to a custom model config yaml file. else ""
    custom_checkpoint_path = ""          # if model_checkpoint "custom", path to custom checkpoint file. else ""

    # Image Settings
    W = 832                              # image width
    H = 960                              # image height
    W, H = map(lambda x: x - x % 64,     # ensure that shape is divisable by 64
               (W, H))

    # Sampling Settings
    seed = -1                            # random seed
    sampler = "ddim"                     # one of "klms", "dpm2", "dpm2_ancestral", "euler", "euler_ancestral", "ddim", "heun"
    steps = 50                           # number of steps to run
    scale = 12                           # classifier-free guidance scale, which determines how much prompts are followed in image generation
    ddim_eta = 0.0                       # amount of ddim to use (0.0: no ddim, 1.0: full ddim)
    dynamic_threshold = None             # adaptive threshold for ddim. None: no adaptive threshold, 0.0: adaptive threshold, 0.0 < x < 1.0: adaptive threshold with x as initial threshold
    static_threshold = None              # static threshold for ddim. None: no static threshold, 0.0: static threshold, 0.0 < x < 1.0: static threshold with x as initial threshold

    # Save & Display Settings
    save_samples = True                  # whether to save samples to disk
    save_settings = False                # whether to save settings to a file
    display_samples = True               # whether to display samples in Colab
    save_sample_per_step = False         # whether to save samples per step or only the last one (only for ddim)
    show_sample_per_step = False         # whether to show samples for each step

    # Prompt Settings
    prompt_weighting = False             # one of False, "linear", "exponential"
    normalize_prompt_weights = True      # whether to normalize prompt weights to sum to 1
    log_weighted_subprompts = False      # whether to log the weighted subprompts

    # Batch Settings
    n_batch = 6                          # number of samples to generate per prompt
    output_path = "./"                   # folder path to save images to
    batch_name = "AnimFun"               # subfolder name to save images to
    seed_behavior = "iter"               # one of "iter", "fixed", "random"
    make_grid = False                    # whether to make a grid of images
    grid_rows = 2                        # number of rows in grid
    filename_format = "{timestring}_{index}_{prompt}.png"
    outdir = get_output_folder(output_path, batch_name)

    # Init Settings
    use_init = False                     # whether to use an init image
    strength = 0                         # a float between 0 and 1. 1 means the image is initialized to the prompt, 0 means the image is initialized to noise
    strength_0_no_init = True            # if True, strength becomes 0 when init is not used
    init_image = ""                      # URL or local path to image
    use_mask = False                     # whether to use a mask. whiter pixels are masked out
    use_alpha_as_mask = False            # use the alpha channel of the image as a mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" 
    invert_mask = False                  # whether to invert the mask
    mask_brightness_adjust = 1.0         # 0.0 is black, 1.0 is no change, 2.0 is white
    mask_contrast_adjust = 1.0           # 0.0 is min contrast, 1.0 is no change, 2.0 is max contrast
    overlay_mask = True                  # masked image overlay at the end of the generation so it does not get degraded by encoding and decoding
    mask_overlay_blur = 5                # bluring edges of final overlay mask. 0 is no blur, 1 is a little blur, 5 is a lot of blur

    # Noise Settings
    n_samples = 1                        # number of samples to generate
    precision = 'autocast'               # one of 'autocast', 'fp32', 'fp16'
    C = 4                                # number of channels
    f = 8                                # number of features

    timestring = ""                      # time string for output file name
    init_latent = None                   # if not None, use this latent as the starting point for generation
    init_sample = None                   # if not None, use this image as the initial sample
    init_c = None                        # if not None, use this c vector as the starting point for generation

    return locals()


args = SimpleNamespace(**AstroArgs())
args.time = str(round(time.time()))
args.strength = max(0.0, min(1.0, args.strength))

if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
if not args.use_init:
    args.init_image = None
if args.sampler != 'ddim':
    args.ddim_eta = 0

# clean up unused memory
gc.collect()
torch.cuda.empty_cache()


"""

    Example Args & Prompts:

     1. model_checkpoint = "anime-trinart.ckpt"
        # I recommend starting an image gen with 640x640 or 704x704. Then use the generated image as next init_image and scale up to 896x896
        W = 640
        H = 640
        steps = 100
        scale = 8
        sampler = "klms"
        from util import readLines
        prompts = readLines("./prompt-examples/anime_boys.txt")

     2. model_checkpoint = "anime-cyberpunk.ckpt"
        # I recommend generating 704x704 or 768x768 image(s) first. 768x768 images are either more detailed & accurate or more random & chaotic.
        W = 768
        H = 768
        steps = 60
        scale = 8
        sampler = "euler"
        prompts = readLines("./prompt-examples/anime_cyberpunk.txt")

     3. model_checkpoint = "van-gogh-diffusion-v2.ckpt"
        W = 512
        H = 512
        scale = 6.5
        steps = 25
        sampler = "euler"
        people = ['Armand Roulin', 'Vincent Van Gogh', 'Adeline Ravoux', 'Bruce Wayne', 'Steve Rogers', 'Gendarme Rigaumon', 'Louise Chevalier']
        scenes = ['catholic church', 'lake', 'mountain', 'ocean', 'river in between grass fields', 'road', 'sky', 'tree', 'waterfall', 'windmill', 'winter', 'woodland']
        prompts = people + scenes

     4. # for "robo-diffusion-v1.ckpt", use "nousr robot" near the beginning of your prompt

     5. # for "pixelart-diffusion-sprites.ckpt", use one of: PixelartFSS, PixelartRSS, PixelartBSS, or PixelartLSS to signal the direction the sprite should be facing

     6. # for "popup-book.ckpt", include "popupBook" in your prompt to get a popup book effect


    Example Negative Prompt:
     1. nprompts = ['oversaturated, ugly, 3d, render, cartoon, grain low-res, kitsch']   # good for photos

"""


def return_image_gen(prompts):
    return render_image_batch(args, prompts, upscale_ratio=1, save_image=False)


# See prompt examples in the /prompt-examples and /art-examples folder
# Use commas (,), pipes (|), or double colons (::) as hard separators
# ❗Uncomment one of the two sections below to generate image(s) from the prompts

# prompts = ['1boy, medium hair, blonde hair, blue eyes, bishounen, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden, highres']
# nprompts = []  # don't change this unless the output image is of poor quality
# render_image_batch(args, prompts, nprompts, upscale_ratio=1, save_image=True)

# from util import random_anime_tags
# prompts = random_anime_tags()
# nprompts = []
# render_image_batch(args, prompts, nprompts, upscale_ratio=1, save_image=True)