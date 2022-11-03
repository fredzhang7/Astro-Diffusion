from PIL import Image
from cv2 import dnn_superres
import cv2, os, random, numpy


# if a theme doesn't appear in the list, use "default"
themes = ['default', 'anime girl', 'anime boy', 'waifu', 'nature', 'space', 'robot', 'mecha', 'android', 'cyborg', 'low res']


def upscale_image(image: Image, scale: int) -> Image:
    """
    Upscale an image using OpenCV's EDSR model
    """
    sr = dnn_superres.DnnSuperResImpl_create()
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        # convert PIL image to cv2 image
        image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    path = "EDSR_x4.pb" if scale == 4 else "EDSR_x3.pb" if scale == 3 else "EDSR_x2.pb" if scale == 2 else None
    if path is None:
        raise ValueError("Scale must be 2, 3, or 4")
    sr.readModel(path)
    sr.setModel("edsr", scale)
    result = sr.upsample(image)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def upscale_video(video_path: str, scale: int) -> None:
    """
    Upscale a video, frame by frame, using OpenCV's EDSR model
    """
    if not os.path.isfile(video_path):
        raise ValueError("Video path is invalid")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video is invalid")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width * scale, height * scale))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = upscale_image(frame, scale)
        out.write(frame)
    cap.release()
    out.release()


def get_optimized_file_prompts(filepath: str, theme: str) -> list[str]:
    """
    Read a file with each prompt in a different line and return a list of prompts.\\
    Pass a theme from the themes list to get a prompt that fits the theme.
    """
    prompts = []
    if not os.path.isfile(filepath):
        raise ValueError("Given filepath is invalid")
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            if 'anime girl' in line or 'waifu' in line or theme == "anime girl" or theme == "waifu":
                line = "correct body positions, " + line.strip() + ", sharp focus, beautiful, attractive"
                if 'girls' in line:
                    prompts.append(line)
                else:
                    prompts.append("1girl, " + line + ", anime incarnation")
            elif 'anime boy' in line or theme == "anime boy":
                emotions = ["sad", "angry", "surprised", "disgusted", "afraid", "calm", "confused", "bored"]
                if not any(emotion in line for emotion in emotions):
                    line = "1boy, " + line.strip() + ", feeling very happy, excited, joyful, cheerful, "
                    if random.randint(0, 1) == 0:
                        prompts.append(line + "trending on artstation")
                    else:
                        prompts.append(line + "trending on pixiv")
            elif theme == "nature":
                line = line.strip() + ", trending on artstation, hyperrealistic, trending, 4 k, 8 k, uhd"
                if not "detailed" in line:
                    line += ", highly detailed"
                prompts.append(line)
            elif theme == "space":
                if len(line) < 45:
                    prompts.append(line.strip() + ", octane render, masterpiece, cinematic, trending on artstation, 8k ultra hd")
                elif not "lighting" in line:
                    prompts.append(line.strip() + ", cinematic lighting, 8k ultra hd")
                else:
                    prompts.append(line.strip() + ", 8k ultra hd")
            else:
                prompts.append(line.strip())
    return prompts


def get_optimized_model_choice(theme: str, lightweight: bool) -> str:
    """
    Return the model name based on the theme and the checkpoint model size (lightweight)
    """
    if 'anime' in theme or 'waifu' in theme:
        return "waifu-diffusion-v1-3.ckpt"
    if 'robot' in theme or 'mecha' in theme or 'android' in theme or 'cyborg' in theme:
        return "robo-diffusion-v1.ckpt"
    
    if lightweight:
        if 'low res' in theme:
            return "sd-v1-1.ckpt"
        return 'sd-v1-5.ckpt'
    else:
        if 'low res' in theme:
            return "sd-v1-1-full-ema.ckpt"
        return "sd-v1-5-full-ema.ckpt"
