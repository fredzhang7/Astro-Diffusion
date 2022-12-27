# If you don't have a GPU, set cpu_only to True
# Uncomment the following two lines to automatically install the required libraries
# from art_generation.setup import setup_environment
# setup_environment(cpu_only=False)


from types import SimpleNamespace
import os, time
from art_generation.image_generation import render_image_batch


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path, time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


"""
    
    Stable Diffusion Style
     sd-v2-1.ckpt                        (5.2 GB, latest, family-friendly, highest resolution, uses negative prompts, general artwork, medium RAM)
     sd-v1-5.ckpt                        (4.2 GB, higher resolution, general artwork, medium RAM)
     sd-v1-1.ckpt                        (4.2 GB, lowest accuracy, general artwork, medium RAM)

    Animated Style
     anime-anything-v3.ckpt              (3.8 GB, highest resolution, anime-style characters, food, animals, sceneries, etc., medium RAM)
     anime-trinart.ckpt                  (2.1 GB, detailed, high res, anime-style characters, low RAM)
     anime-cyberpunk.ckpt                (2.1 GB, accurate, clean-cut, high res, cyberpunk anime drawings, low RAM)
     spiderverse-diffusion.ckpt          (2.1 GB, Sony's Into the Spider-Verse, draw in spiderverse style, low RAM)
     disney-diffusion.ckpt               (2.1 GB, high res Disney-style characters, animals, cars, & landscapes, low RAM)
     pony-diffusion-v2.ckpt              (3.7 GB, high res pony characters, medium RAM)
     pokemon-diffusion.ckpt              (4.2 GB, new pokedex pokemons, medium RAM)

    Robo Style
     robo-diffusion-v1.ckpt              (4.2 GB, high-quality robot, cyborg, android drawings, medium RAM)

    Drawing Styles
     chinese-sd.ckpt                     (4.2 GB, trained on 20 million Chinese text and image pairs, understands English, medium RAM)
     vector-art.ckpt                     (2.1 GB, high-quality vector art, medium RAM)
     voxel-art.ckpt                      (2.1 GB, accurate, sd-v1-5.ckpt fine tuned on high res voxel art, medium RAM)
     nitro-diffusion.ckpt                (2.1 GB, mix of archer, arcane or modern disney styles, medium RAM)
     pixelart-diffusion-sprites.ckpt     (4.2 GB, generate pixel art sprite sheets from four different angles, medium RAM)
     sjh-artist.ckpt                     (2.1 GB, sd-v1-5.ckpt but in Sin Jong Hun style, high quality drawings for sceneries and landscapes, low RAM)
     comic-diffusion.ckpt                (2.1 GB, unique but consistent comic styles, low RAM)
     redshift-diffusion.ckpt             (2.1 GB, high res 3D artwork, low RAM)
     inkpunk-diffusion.ckpt              (2.1 GB, line drawing and sketches of people, low RAM)

    Painting Styles
     midjourney-v2.ckpt                  (2.1 GB, highest-quality art I've ever seen, Modern Midjourney, low RAM)
     midjourney-v1.ckpt                  (2.1 GB, a mix of sd-v1-5 and Midjourney, Painting Midjourney, low RAM)
     van-gogh-diffusion-v2.ckpt          (2.1 GB, high-quality Van Gogh paintings, Loving Vincent, low RAM)

    Photography Styles
     analog-diffusion.ckpt               (2.1 GB, high-quality analog photography, low RAM)
     linkedin-diffusion.ckpt             (2.1 GB, trained on linkedin profile pictures, low RAM)

    Craft Style
     popup-book.ckpt                     (4.2 GB, pop-up book illustrations, medium RAM)
     papercut-diffusion.ckpt             (4.2 GB, sd-v1-5.ckpt but finetuned on paper cut images, medium RAM)
     chroma-v5.ckpt                      (2.6 GB, mixes stable diffusion v1.5 and 2.0, generates images in 3D, low RAM)

"""


def AstroArgs():
    # Model Settings
    model_checkpoint = "midjourney-v2.ckpt"    # one of "custom", a model checkpoint listed above. if you have no clue, use "sd-v1-5.ckpt" for starters
    check_sha256 = False                 # whether to check the sha256 hash of the checkpoint file. set to True if you have issues with model downloads
    custom_config_path = ""              # if model_checkpoint "custom", path to a custom model config yaml file. else ""
    custom_checkpoint_path = ""          # if model_checkpoint "custom", path to custom checkpoint file. else ""

    # Image Settings
    W = 768                              # image width
    H = 768                              # image height
    W, H = map(lambda x: x - x % 64,     # ensure that shape is divisable by 64
               (W, H))

    # Sampling Settings
    seed = -1                            # -1 means to use a random seed, a positive integer means use the number as the initial seed
    sampler = "ddim"                     # one of "klms", "dpm2", "dpm2_ancestral", "euler", "euler_ancestral", "ddim", "heun"
    steps = 50                           # number of steps to run
    scale = 7.5                          # classifier-free guidance scale, which determines how much prompts are followed in image generation
    ddim_eta = 0.0                       # mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. setting this to 1.0 favors step counts 250 and up
    dynamic_threshold = None             # adaptive threshold for ddim. None: no adaptive threshold, 0.0: adaptive threshold, 0.0 < x < 1.0: adaptive threshold with x as initial threshold
    static_threshold = None              # static threshold for ddim. rest is similar to dynamic_threshold

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
    filename_format = "{timestring}_{seed}_{prompt}.png"
    outdir = get_output_folder(output_path, batch_name)

    # Img2Img Settings
    use_init = False                     # whether to use an init image
    strength = 0                         # a float between 0 and 1. 1 means the image is initialized to the prompt, 0 means the image is initialized to noise
    strength_0_no_init = True            # if True, strength becomes 0 when init is not used
    init_image = ""                      # URL or local path to image
    
    # Inpainting/Outpainting Settings
    use_mask = False                     # whether to use a mask. whiter pixels are masked out
    use_alpha_as_mask = False            # use the alpha channel of the image as a mask
    mask_file = ""                       # URL or local path to mask
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
    init_latent = None                   # if not None, use this latent as the starting point for generation
    init_sample = None                   # if not None, use this image as the initial sample
    init_c = None                        # if not None, use this c vector as the starting point for generation

    return locals()

args = SimpleNamespace(**AstroArgs())


"""

    Example Args & Prompts:

     1. model_checkpoint = "anime-anything-v3.ckpt"
        W = 832
        H = 960
        steps = 50
        sampler = "ddim"
        scale = 12

     2. model_checkpoint = "anime-cyberpunk.ckpt"
        W = 768
        H = 768
        steps = 60
        scale = 8
        sampler = "euler"
        from art_generation.util import readLines
        prompts = readLines("./prompt-examples/cyberpunk.txt")

     3. model_checkpoint = "van-gogh-diffusion-v2.ckpt"
        W = 512
        H = 512
        scale = 6.5
        steps = 25
        sampler = "euler"
        people = ['Armand Roulin', 'Vincent Van Gogh', 'Adeline Ravoux', 'Bruce Wayne', 'Steve Rogers', 'Gendarme Rigaumon', 'Louise Chevalier']
        scenes = ['catholic church', 'lake', 'mountain', 'ocean', 'river in between grass fields', 'road', 'sky', 'tree', 'waterfall', 'windmill', 'winter', 'woodland']
        prompts = people + scenes

     4. model_checkpoint = "midjourney-v2.ckpt"
        # for a portrait photo, use a 2:3 (e.g. 512x768) or a 9:16 aspect ratio. for a landscape photo, use a 3:2 (e.g. 768x512) or a 16:9 aspect ratio
        # add "dreamlikeart" to the prompt if the artstyle is too weak
        W = 640
        H = 640
        scale = 7.5
        steps = 30
        sampler = "ddim

     5. # "robo-diffusion-v1.ckpt" - use "nousr robot" near the beginning of your prompt
        # "pixelart-diffusion-sprites.ckpt" - use one of "PixelartFSS", "PixelartRSS", "PixelartBSS", or "PixelartLSS" to signal the direction the sprite should be facing
        # "popup-book.ckpt" - include "popupBook" in your prompt to get a popup book effect
        # "spiderverse-diffusion.ckpt" - use "spiderverse style" in your prompt to get a spider-verse effect
        # "sjh-artist.ckpt" - add ", sjh style" to the end of your prompt to draw sceneries and landscapes in the style of sjh
        # "papercut-diffusion.ckpt" - include "PaperCut" in your prompt
        # "voxel-art.ckpt" - include "VoxelArt" in prompts
        # "vector-art.ckpt" - use "vectorartz" in prompts to generate beautiful vector illustrations
        # "comic-diffusion.ckpt" - include one of "charliebo artstyle", "holliemengert artstyle", "marioalberti artstyle", "pepelarraz artstyle", "andreasrocha artstyle", "jamesdaly artstyle"
        # "redshift-diffusion.ckpt" - use "redshift style", 512x704 for portraits, 704x512 for scenes and cars, steps: 20 or 40, sampler: dpm2, scale: 7
        # "nitro-diffusion.ckpt" - use "archer style", "arcane style", and/or "modern disney style"
        # "chroma-v5.ckpt" - use "ChromaV5" at the start of the prompt
        # "analog-diffusion.ckpt" - use "analog style", sample using euler_ancestral or dpm2
        # "inkpunk-diffusion.ckpt" - use "nvinkpunk"


    Example Negative Prompt:
     1. nprompts = ['oversaturated, ugly, 3d, render, cartoon, grain low-res, kitsch']   # good for stable diffusion photos

     
    DistilGPT2 Stable Diffusion V2 Usage:
     1. from art_generation.util import magic_prompt
        prompts = magic_prompt('a beautiful city', temperature=0.9, num_return_sequences=10)

"""


def return_image_gen(prompts):
    return render_image_batch(args, prompts, save_image=False, check_safety=True)


# See prompt examples in the /prompt_examples and /art_examples folder
# Use commas (,), pipes (|), or double colons (::) as hard separators

prompts = ['dreamlikeart, a rainforest, high resolution wallpaper']
nprompts = ['fog blurry lowres']
render_image_batch(args, prompts, nprompts, save_image=True, check_safety=True)