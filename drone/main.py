class DroneArgs():
    # Init Settings
    use_init = False
    init_image = ""

    # Generation Params
    model_checkpoint = "paint_journey_v2.ckpt [26fd2f6f]"
    prompt = "((oil painting)), ((clouds)), ((mist)), a lush valley with mountains in the background, high resolution, uhd, 4 k wallpaper"
    negative_prompt = "((low resolution)) ((blurry)) ((fog)) ((messy)) signature watermark"
    sampler = "Euler"
    cfg_scale = 9
    steps = 40
    seed = -1
    blend_strength = 40
    seed_behavior = "fixed"
    img_width = 1280
    img_height = 768
    inpaint_full_res_padding = 60

    # Noise Settings
    denoising_strength = 0.85
    mask_blur = 4
    
    # Video Settings
    cfg_increase = 2
    blend_strength = 40
    zoom_factor = "1.01"         
    zoom_mode = 'center'
    fly_mode = 'over'
    fly_skip = 32
    bbox_thres = 0.90
    area_thres = 8000
    fps = 16
    superres_seconds = 1
    superres_method = "R-ESRGAN General WDN 4x V3"
    frames = 400



from drone_view import generate_drone_video
args = DroneArgs()
generate_drone_video(args)



# convert GIF into MP4
# ffmpeg -i "input.gif" -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "output.mp4"

# convert MP4 into thumbnail PNG
# ffmpeg -i "input.mp4" -vframes 1 "output.png"