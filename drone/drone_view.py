from PIL import Image, ImageDraw, ImageFilter
from transformers import DetrImageProcessor, DetrForObjectDetection
from previous.super_resolution.upscale_methods import upscale_one
from torch import tensor, Tensor, cuda
import os, cv2, numpy as np
from time import strftime
from requests import post
from io import BytesIO
from base64 import b64decode, b64encode
from random import randint, choice
from math import ceil
import warnings
warnings.filterwarnings("ignore")

detrProcessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detrModel = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# function to upscale image to a certain resolution
def superres_image(image: Image.Image, resolution: tuple[int, int], superres_method: str) -> Image.Image:
    image = upscale_one(image, model_name=superres_method, return_image=True)
    
    while image.size[0] < resolution[0] or image.size[1] < resolution[1]:
        image = upscale_one(image, model_name=superres_method, return_image=True)
    
    if image.size[0] > resolution[0] or image.size[1] > resolution[1]:
        image = image.resize(resolution, resample=Image.Resampling.LANCZOS)

    return image

# function to zoom in on the image
def zoom(image: Image.Image, zoom_factor: float, zoom_mode: str = 'center', superres: bool = False, superres_method: str = "R-ESRGAN General WDN 4x V3") -> Image.Image:
    width = image.width
    height = image.height
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # zooming in means the edges of the image expand outside the image dimensions
    image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # crop the image to the original width and height
    if zoom_mode == 'random':
        zoom_modes = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'mid-left', 'mid-right', 'top-mid', 'bottom-mid']
        return zoom(image, zoom_factor, zoom_modes[randint(0, len(zoom_modes) - 1)], superres, superres_method)
    elif zoom_mode == 'center':
        x = (new_width - width) // 2
        y = (new_height - height) // 2
        image = image.crop((x, y, x + width, y + height))
    elif zoom_mode == 'top-left':
        image = image.crop((0, 0, width, height))
    elif zoom_mode == 'top-right':
        image = image.crop((new_width - width, 0, new_width, height))
    elif zoom_mode == 'bottom-left':
        image = image.crop((0, new_height - height, width, new_height))
    elif zoom_mode == 'bottom-right':
        image = image.crop((new_width - width, new_height - height, new_width, new_height))
    elif zoom_mode == 'mid-left':
        x = (new_width - width) // 2
        image = image.crop((0, x, width, x + height))
    elif zoom_mode == 'mid-right':
        x = (new_width - width) // 2
        image = image.crop((new_width - width, x, new_width, x + height))
    elif zoom_mode == 'top-mid':
        y = (new_height - height) // 2
        x = (new_width - width) // 2
        image = image.crop((x, 0, x + width, height))
    elif zoom_mode == 'bottom-mid':
        y = (new_height - height) // 2
        x = (new_width - width) // 2
        image = image.crop((x, new_height - height, x + width, new_height))
    else:
        raise ValueError(f'Invalid zoom_mode: {zoom_mode}')
    
    if superres:
        image = superres_image(image, (image.width, image.height), superres_method)
    return image

# function to get the largest bounding box in the image
def largest_bbox(image: Image.Image, threshold: float = 0.9) -> Tensor:
    inputs = detrProcessor(images=image, return_tensors="pt")
    outputs = detrModel(**inputs)
    target_sizes = tensor([image.size[::-1]])
    results = detrProcessor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    results = results['boxes']
    if len(results) == 0:
        return tensor([0, 0, 0, 0])
    largest_bbox = max(results, key=lambda x: x[2] * x[3])
    return largest_bbox

# function to fill in missing pixels in the image
def fill(image: Image.Image, args: dict) -> tuple[Image.Image, dict]:
    encoded = b64encode(open("./prefill.png", "rb").read())
    encodedStr = str(encoded, encoding='utf-8')
    goodEncodedImg ='data:image/png;base64,' + encodedStr
    encoded = b64encode(open("./mask.png", "rb").read())
    encodedStr = str(encoded, encoding='utf-8')
    goodEncodedMask ='data:image/png;base64,' + encodedStr
    img2img_url = "http://127.0.0.1:7860/sdapi/v1/img2img"
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.w,
        "height": args.h,
        "steps": args.steps,
        "init_images": [goodEncodedImg],
        "mask": goodEncodedMask,
        "mask_blur": args.mask_blur,
        "cfg_scale": args.cfg_scale,
        "sampler_index": args.sampler,
        "denoising_strength": args.denoising_strength,
        "inpaint_full_res_padding": args.inpaint_full_res_padding,
    }
    resp = post(img2img_url, json=payload).json()
    for i in resp['images']:
        image = Image.open(BytesIO(b64decode(i)))
        image.save('./postfill.png')
    return image, args

# function to generate the next seed
def next_seed(args: dict) -> int:
    if args.seed == -1:
        return randint(0, 2**32 - 1)
    if args.seed_behavior == 'iter':
        args.seed += 1
    return args.seed

# function to check if one row of the image is the same color
def check_row_color(image: Image.Image) -> None:
    img = image.copy()
    for y in range(img.height):
        row_color = img.getpixel((0, y))
        for x in range(img.width):
            if img.getpixel((x, y)) != row_color:
                break
        else:
            raise Exception("Row {} is the same color: {}".format(y, row_color))

# function to check if one column of the image is the same color
def check_col_color(image: Image.Image) -> None:
    img = image.copy()
    for x in range(img.width):
        col_color = img.getpixel((x, 0))
        for y in range(img.height):
            if img.getpixel((x, y)) != col_color:
                break
        else:
            raise Exception("Column {} is the same color: {}".format(x, col_color))

# function to round a number to the nearest multiple of 8
def round_to_multiple_of_64(x: float) -> int:
    return int(ceil(x / 64) * 64)

# function to blend regions in the image, adapted from https://github.com/Mohdyusuf786/image-blending
def blend_image(image: Image.Image, split_pos: int, vertical: bool) -> Image.Image:
    # convert input image to Mat format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # split the image into two halves
    if vertical:
        img1 = image[:split_pos, :]
        img2 = image[split_pos:, :]
    else:
        img1 = image[:, :split_pos]
        img2 = image[:, split_pos:]

    # generate gaussian pyramid for img1
    gp_img1 = [img1]
    for i in range(6):
        img1 = cv2.pyrDown(img1)
        gp_img1.append(img1)

    # generate gaussian pyramid for img2
    gp_img2 = [img2]
    for i in range(6):
        img2 = cv2.pyrDown(img2)
        gp_img2.append(img2)

    # generate laplacian pyramid for img1
    lp_img1 = [gp_img1[5]]
    for i in range(5, 0, -1):
        size = (gp_img1[i - 1].shape[1], gp_img1[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gp_img1[i], dstsize=size)
        laplacian = cv2.subtract(gp_img1[i - 1], gaussian_expanded)
        lp_img1.append(laplacian)

    # generate laplacian pyramid for img2
    lp_img2 = [gp_img2[5]]
    for i in range(5, 0, -1):
        size = (gp_img2[i - 1].shape[1], gp_img2[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gp_img2[i], dstsize=size)
        laplacian = cv2.subtract(gp_img2[i - 1], gaussian_expanded)
        lp_img2.append(laplacian)

    # add left and right halves of images in each level
    ls = []
    for img1_lap, img2_lap in zip(lp_img1, lp_img2):
        rows, cols, dpt = img1_lap.shape
        laplacian = np.hstack((img1_lap[:, :int(cols / 2)], img2_lap[:, int(cols / 2):]))
        ls.append(laplacian)

    # reconstruct image from laplacian pyramid
    ls_ = ls[0]
    for i in range(1, 6):
        size = (ls[i].shape[1], ls[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, ls[i])

    # convert to PIL image
    blended = Image.fromarray(cv2.cvtColor(ls_, cv2.COLOR_BGR2RGB))
    return blended

def crop_from_center(img: Image.Image, distance: int, vertical: bool) -> Image.Image:
    if vertical:
        top = img.height // 2 - distance
        botttom = img.height // 2 + distance
        return img.crop((0, top, img.width, botttom))
    else:
        left = img.width // 2 - distance
        right = img.width // 2 + distance
        return img.crop((left, 0, right, img.height))

def draw_high_lo_mask(mask: Image.Image, max_y: int) -> Image.Image:
    gap = mask.height // 4 * 3
    draw = ImageDraw.Draw(mask)
    for x in range(0, mask.width, gap - 16):
        draw.ellipse((x, max_y - 6, x + gap, max_y + 6), fill='white')
    return mask

def draw_left_right_mask(mask: Image.Image, max_x: int) -> Image.Image:
    gap = mask.width // 2
    draw = ImageDraw.Draw(mask)
    for y in range(0, mask.height, gap - 16):
        draw.ellipse((max_x - 6, y, max_x + 6, y + gap), fill='white')
    return mask

def gradient(img: Image.Image, args: dict, l: int, t: int, r: int, b: int, whiteout_l: int, whiteout_r: int, vertical: bool, iter: int = 2) -> tuple[Image.Image, dict]:
    # first pass
    blend = img.copy()
    blend = blend.crop((l, t, r, b))
    mask = Image.new('RGB', img.size, (0,0,0))
    mask.paste(blend, (l, t))
    distance = round(args.mask_blur * 3)
    edge = crop_from_center(blend, distance=distance, vertical=vertical)
    whiteout = Image.new('RGB', (edge.width, edge.height), (255,255,255))
    mask.paste(whiteout, (whiteout_l, whiteout_r))
    img.save('prefill.png')
    mask.save('mask.png')
    img, args = fill(img, args)

    # second pass
    for i in range(iter):
        blend = img.copy()
        blend = blend_image(blend, split_pos=args.w // 2, vertical=vertical) if vertical else blend_image(blend, split_pos=args.h // 2, vertical=vertical)
        blend = blend.crop((l, t, r, b))
        blend = blend.filter(ImageFilter.GaussianBlur(radius=2))
        mask = Image.new('RGB', img.size, (0,0,0))
        mask.paste(blend, (l, t))

        img.save('prefill.png')
        mask.save('mask.png')
        img, args = fill(img, args)
    
    return img, args


# function to generate each frame of the video
def step(image: Image.Image, args: dict, superres: bool = False) -> tuple[Image.Image | list, dict]:
    if args.fly_mode == 'random':
        args.fly_mode = choice(['over', 'left', 'right'])
        return step(image, args, superres)
    
    # deal with MemoryError
    import gc
    gc.collect()
    cuda.empty_cache()

    bbox = largest_bbox(image, args.bbox_thres)
    bbox = [bbox[i].item() for i in range(len(bbox))]
    print(f"\033[33m{bbox}\033[0m")
    zoom_factor = eval(args.zoom_factor)
    args.w = image.width
    args.h = image.height

    if bbox[2] * bbox[3] > args.area_thres:
        # large object detected
        if args.fly_mode == 'over':
            image = image.crop((0, 0, args.w, args.h // zoom_factor))

            expand_h = args.h
            for skip in range(args.fly_skip):
                expand_h = expand_h * zoom_factor
            expand_h = round_to_multiple_of_64(expand_h)
            img = Image.new('RGB', (args.w, expand_h), (255,255,255))
            mask = Image.new('RGB', (args.w, expand_h), (0,0,0))
            zoom_mode = 'top-mid'

            # project the colors of the top row to the black pixel region
            max_y = expand_h - image.height
            img.paste(image, (0, max_y))
            for i in range(args.w):
                color = img.getpixel((i, max_y))
                for j in range(max_y):
                    img.putpixel((i, j), color)
                    mask.putpixel((i, j), (255,255,255))
            
            mask = draw_high_lo_mask(mask, max_y)

        elif args.fly_mode == 'under':
            image = image.crop((0, 0, args.w, args.h // zoom_factor))

            expand_h = args.h
            for skip in range(args.fly_skip):
                expand_h = expand_h * zoom_factor
            expand_h = round_to_multiple_of_64(expand_h)
            img = Image.new('RGB', (args.w, expand_h), (255,255,255))
            mask = Image.new('RGB', (args.w, expand_h), (0,0,0))
            zoom_mode = 'bottom-mid'

            max_y = image.height
            img.paste(image, (0, 0))
            for i in range(args.w):
                color = img.getpixel((i, max_y))
                for j in range(max_y, expand_h):
                    img.putpixel((i, j), color)
                    mask.putpixel((i, j), (255,255,255))

            mask = draw_high_lo_mask(mask, max_y)
                
        elif args.fly_mode == 'left':
            image = image.crop((0, 0, args.w // zoom_factor, args.h))

            expand_w = args.w
            for skip in range(args.fly_skip):
                expand_w = expand_w * zoom_factor
            expand_w = round_to_multiple_of_64(expand_w)
            img = Image.new('RGB', (expand_w, args.h), (255,255,255))
            mask = Image.new('RGB', (expand_w, args.h), (0,0,0))
            zoom_mode = 'mid-left'

            max_x = expand_w - image.width
            img.paste(image, (max_x, 0))
            for i in range(args.h):
                color = img.getpixel((max_x, i))
                for j in range(max_x):
                    img.putpixel((j, i), color)
                    mask.putpixel((j, i), (255,255,255))

            mask = draw_left_right_mask(mask, max_x)

        elif args.fly_mode == 'right':
            image = image.crop((args.w // zoom_factor, 0, args.w, args.h))

            expand_w = args.w
            for skip in range(args.fly_skip):
                expand_w = expand_w * zoom_factor
            expand_w = round_to_multiple_of_64(expand_w)
            img = Image.new('RGB', (expand_w, args.h), (255,255,255))
            mask = Image.new('RGB', (expand_w, args.h), (0,0,0))
            zoom_mode = 'mid-right'

            max_x = image.width
            img.paste(image, (0, 0))
            for i in range(args.h):
                color = img.getpixel((max_x, i))
                for j in range(max_x):
                    img.putpixel((j, i), color)
                    mask.putpixel((j, i), (255,255,255))

            mask = draw_left_right_mask(mask, max_x)

        else:
            raise ValueError(f'Invalid fly_mode: {args.fly_mode}')
        
        temp_w = args.w
        temp_h = args.h
        args.w = img.width
        args.h = img.height
        args.seed = next_seed(args)
        mask.save('mask.png')
        img.save('prefill.png')
        args.cfg_scale += args.cfg_increase
        img, args = fill(img, args)
        distance = round(args.mask_blur * 3)

        # apply gradient
        if args.fly_mode == "over" or args.fly_mode == "under":
            top = max_y - args.blend_strength // 2
            bottom = max_y + args.blend_strength // 2 + args.mask_blur
            
            whiteout_l, whiteout_r = 0, max_y - distance + args.mask_blur // 3
            l, t, r, b = 0, top, args.w, bottom
            img, args = gradient(img, args, l, t, r, b, whiteout_l, whiteout_r, vertical=True)

        elif args.fly_mode == "left" or args.fly_mode == "right":
            left = max_x - args.blend_strength // 2
            right = max_x + args.blend_strength // 2 + args.mask_blur

            whiteout_l, whiteout_r = max_x - distance + args.mask_blur // 3, 0
            l, t, r, b = left, 0, right, args.h
            img, args = gradient(img, args, l, t, r, b, whiteout_l, whiteout_r, vertical=False)
        
        args.cfg_scale -= args.cfg_increase
        args.w = temp_w
        args.h = temp_h

        # convert to image frames
        image = img.copy()
        for i in range(args.fly_skip):
            superres = True if (args.t + i) % (args.superres_seconds * args.fps) == 0 else False
            if zoom_mode == 'top-mid':
                end_h = int(args.h + (image.height - args.h) * (1 - i / args.fly_skip))
                start_h = end_h - args.h
                img = image.crop((0, start_h, args.w, end_h))
            elif zoom_mode == 'mid-left':
                end_w = int(args.w + (image.width - args.w) * (1 - i / args.fly_skip))
                start_w = end_w - args.w
                img = image.crop((start_w, 0, end_w, args.h))
            elif zoom_mode == 'mid-right':
                end_w = int(args.w + (image.width - args.w ) * (1 - i / args.fly_skip))
                start_w = end_w - args.w
                img = image.crop((start_w, 0, end_w, args.h))
            t = args.t + i
            zoom_factor *= eval(args.zoom_factor)
            img = zoom(img, zoom_factor, zoom_mode, superres, args.superres_method)
            img.save(f'{args.tfolder}/{args.t + i}.png')
            
        images = [img]
        for i in range(args.fly_skip - 1):
            images.append(0)
        return images, args
    else:
        image = zoom(image, zoom_factor, args.zoom_mode, superres, args.superres_method)
    
    return image, args

# function to sanitize the prompt for file names
def sanitize(prompt: str) -> str:
    replace_ = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in replace_:
        prompt = prompt.replace(char, '_')
    if len(prompt) > 250:
        prompt = prompt[:250]
    return prompt

# function to generate a video taken from a drone's perspective with a given prompt
def generate_drone_video(args: dict) -> None:
    option_payload = {
        "sd_model_checkpoint": args.model_checkpoint
    }
    post("http://127.0.0.1:7860/sdapi/v1/options", json=option_payload)
    tfolder = strftime("%Y%m%d-%H%M%S")
    os.mkdir(tfolder)
    args.tfolder = tfolder

    if args.use_init:
        image = Image.open(args.init_image)
        image.save(f'{tfolder}/0.png')
    else:
        from PIL import PngImagePlugin
        txt2img_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        args.seed = next_seed(args)
        payload = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "sampler_index": args.sampler,
            "seed": args.seed,
            'width': args.img_width,
            'height': args.img_height
        }
        resp = post(txt2img_url, json=payload).json()

        for i in resp['images']:
            image = Image.open(BytesIO(b64decode(i.split(",",1)[0])))
            png_payload = {
                "image": "data:image/png;base64," + i
            }
            resp = post(url=f'http://127.0.0.1:7860/sdapi/v1/png-info', json=png_payload).json()

            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", resp.get("info"))
            image.save(f'{tfolder}/0.png', pnginfo=pnginfo)

    start_at = 0
    for t in range(1, args.frames):
        if t < start_at:
            continue
        superres = False
        if t % (args.superres_seconds * args.fps) == 0:
            superres = True
        args.t = t
        image, args = step(image, args, superres)
        if isinstance(image, list):
            start_at = t + len(image)
            image = image[0]
        else:
            image.save(f'{tfolder}/{t}.png')

    # convert the images to a GIF
    import imageio
    images = []
    for t in range(args.frames):
        images.append(imageio.imread(f'{tfolder}/{t}.png'))
    imageio.mimsave(f'{tfolder}_{sanitize(args.prompt)}.gif', images, fps=args.fps)

    # delete the temporary folder
    import shutil
    shutil.rmtree(tfolder)