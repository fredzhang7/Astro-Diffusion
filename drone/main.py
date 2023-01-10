class DroneArgs():
    # Init Settings
    use_init = False
    init_image = ""

    # Generation Params
    model_checkpoint = "paint_journey_v1.ckpt [4748ee4c]"
    prompt = "a beautiful galaxy, high resolution, uhd, 4 k wallpaper"
    negative_prompt = "low-res blurry fog signature watermark"
    sampler = "Euler"
    cfg_scale = 7.5
    steps = 40
    seed = -1
    seed_behavior = "fixed"
    default_w = 1152
    default_h = 768
    portrait_w = 512
    portrait_h = 768
    landscape_w = 1152
    landscape_h = 768
    inpaint_full_res_padding = 40

    # Noise Settings
    denoising_strength = 1.0
    mask_blur = 4
    
    # Graphics Settings
    zoom_factor = "1.01"
    zoom_mode = 'center'
    fly_mode = 'over'
    fly_skip = 32
    bbox_thres = 0.9
    area_thres = 10000
    fps = 16
    superres_seconds = 1
    superres_method = "Swin2SR 4x"
    frames = 400



import warnings
warnings.filterwarnings("ignore")
from drone_view import generate_drone_video

args = DroneArgs()
generate_drone_video(args)



# convert GIF into MP4
# ffmpeg -i "input.gif" -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4