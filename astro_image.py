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
     sd-v1-5-full-ema.ckpt               (7.2 GB, latest, highest resolution, general artwork, high VRAM)
     sd-v1-5.ckpt                        (4.0 GB, latest, higher resolution, general artwork, medium VRAM)
     sd-v1-1-full-ema.ckpt               (7.2 GB, lower resolution, general artwork, high VRAM)
     sd-v1-1.ckpt                        (4.0 GB, lowest accuracy, general artwork, medium VRAM)

    Animated Style
     anime-sd.ckpt                       (2.1 GB, high-quality anime-style characters, low VRAM)
     anime-trinart.ckpt                  (2.1 GB, most accurate, super high res anime-style characters, low VRAM)
     disney-diffusion.ckpt               (2.1 GB, high res Disney-style characters, animals, cars, & landscapes, low VRAM)
     pony-diffusion-v2.ckpt              (3.7 GB, high res pony characters, medium VRAM)

    Robo Style
     robo-diffusion-v1.ckpt              (4.0 GB, high-quality robot, android, mecha images, medium VRAM)

    Drawing Styles
     van-gogh-diffusion-v2.ckpt          (2.1 GB, high-quality Van Gogh paintings, Loving Vincent, low VRAM)
     scifipulp-diffusion.pt              (0.4 GB, sci-fi & pulp art, low VRAM)
     watercolor-diffusion-v2.pt          (0.4 GB, watercolor art, low VRAM)
     portrait-diffusion.pt               (0.5 GB, portraits generator, low VRAM)
     pixelart-diffusion-expanded.pt      (0.4 GB, pixel art scribbles by KaliYuga-ai, low VRAM)
     pixelart-diffusion-sprites.ckpt     (4.0 GB, generate pixel art sprite sheets from four different angles, medium VRAM)

    OpenAI Style
     openai-256x256-diffusion.pt         (2.1 GB, trained on 256x256 images, low VRAM)
     openai-512x512-diffusion.pt         (2.1 GB, trained on 512x512 images, low VRAM)
"""


def AstroArgs():
    # Model Settings
    model_checkpoint = "anime-trinart.ckpt"    # one of "custom", a model checkpoint listed above. if have no clue, use "sd-v1-5.ckpt"
    check_sha256 = False                 # whether to check the sha256 hash of the checkpoint file. set to True if you have issues with model downloads
    custom_config_path = ""              # if model_checkpoint "custom", path to a custom model config yaml file. else ""
    custom_checkpoint_path = ""          # if model_checkpoint "custom", path to custom checkpoint file. else ""

    # Image Settings
    W = 640                              # image width
    H = 640                              # image height
    W, H = map(lambda x: x - x % 64,     # ensure that shape is divisable by 64
               (W, H))

    # Sampling Settings
    seed = -1                            # random seed
    sampler = "klms"                     # one of "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral", "ddim"
    steps = 100                          # number of steps to run
    scale = 8                            # scale (0: 4x4, 1: 8x8, ..., 7: 512x512, 8: 1024x1024)
    ddim_eta = 0.0                       # amount of ddim to use (0.0: no ddim, 1.0: full ddim)
    dynamic_threshold = None             # adaptive threshold for dpm2
    static_threshold = None              # static threshold for dpm2

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
    n_batch = 4                          # number of samples to generate in parallel
    output_path = "./"                   # folder path to save images to
    batch_name = "AnimFun"              # subfolder name to save images to
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
    If a theme doesn't appear in the list, pass "default" to get_optimized_prompts(). Lowercase works too.

    Themes:
     Default
     Nature                              (nature, landscape, scenery)
     Disney                              (disney princess, disney character, disney villain, disney animal, animated car, animated landscape)
     Space                               (universe, supernova, black hole, planet, galaxy, nebula, star, astronaut, rocket, spaceship, alien, night sky)
     Robot                               (robot, android, cyborg, mecha)
     Low Res                             (low resolution, pixel art)
     Anime Girl                          (anime girl, waifu)
     Anime Boy                           (anime boy, husbando)

    Examples:
     1. model_checkpoint = "anime-trinart.ckpt"
        # I recommend starting an image gen with 640x640 or 704x704. then use this as init_image and scale up to 896x896
        W = 640
        H = 640
        steps = 100
        scale = 8
        sampler = "klms"
        from util import readLines
        prompts = readLines("./prompt-examples/anime_boys.txt")
     2. model_checkpoint = "sd-v1-5.ckpt"
        W = 1024
        H = 1024
        steps = 50
        scale = 8
        sampler = "klms"
        from util import get_optimized_prompts
        prompts = get_optimized_prompts(prompt_source='./prompt-examples/nature.txt', theme='nature')
     3. model_checkpoint = "van-gogh-diffusion-v2.ckpt"
        W = 512
        H = 512
        scale = 6.5
        steps = 25
        sampler = "euler"
        people = ['Armand Roulin', 'Vincent Van Gogh', 'Adeline Ravoux', 'Bruce Wayne', 'Steve Rogers', 'Gendarme Rigaumon', 'Louise Chevalier']
        scenes = ['catholic church', 'lake', 'mountain', 'ocean', 'river in between grass fields', 'road', 'sky', 'tree', 'waterfall', 'windmill', 'winter', 'woodland']
        prompts = []
        for person in people:
            prompts.append(f'lvngvncnt, {person}, highly detailed')
        for scene in scenes:
            prompts.append(f'lvngvncnt, {scene}, highly detailed')
     4. model_checkpoint = "robot-diffusion-v1.ckpt"
        # use "nousr robot" near the beginning of your prompt
     5. model_checkpoint = "pixelart-diffusion-sprites.ckpt"
        # use one of: PixelartFSS, PixelartRSS, PixelartBSS, or PixelartLSS to signal the direction the sprite should be facing

"""


def return_image_gen(prompts):
    return render_image_batch(args, prompts, upscale_ratio=1, save_image=False)


# See examples of args and prompts in `examples.py` and the `/art-examples` folder
# Uncomment the lines below to generate an image from the prompts
# prompts = ['Hotarou Oreki']
# render_image_batch(args, prompts, upscale_ratio=1, save_image=True)