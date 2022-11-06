from PIL import Image
from cv2 import dnn_superres
from typing import Tuple
import cv2, os, random, numpy


def upscale_image(image: Image, scale: int) -> Image:
    """
    Upscale an image using OpenCV's EDSR model
    """
    sr = dnn_superres.DnnSuperResImpl_create()
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
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


def get_optimized_prompts(prompt_source: Tuple[str, list[str]], theme: str) -> list[str]:
    """
    - Read a file with each prompt in a different line, or pass an array of prompts.
    - Pass a theme from the themes list to get a prompt that fits the theme.
    - Returns a list of optimized prompts.
    """
    prompts = []
    theme = theme.lower()
    if not os.path.isfile(prompt_source):
        prompt_source = prompt_source
    else:
        with open(prompt_source, 'r', encoding="utf8") as f:
            prompt_source = f.readlines()
    for prompt in prompt_source:
        if "anime girl" in theme or "waifu" in theme:
            prompt = "correct body positions, " + prompt.strip() + ", sharp focus, beautiful, attractive, 4k, 8k ultra hd"
            if 'girls' in prompt:
                prompts.append(prompt)
            else:
                if "waifu" in theme:
                    if random.randint(0, 1) == 1:
                        prompts.append("1girl, " + prompt + ", anime incarnation")
                    else:
                        prompts.append("1woman, " + prompt + ", anime incarnation")
                else:
                    prompts.append("1girl, " + prompt + ", anime incarnation")
        elif "anime boy" in theme or "husbando" in theme:
            emotions = ["sad", "angry", "surprised", "disgusted", "afraid", "calm", "confused", "bored"]
            if not any(emotion in prompt for emotion in emotions):
                prompt = "1boy, " + prompt.strip() + ", feeling very happy, excited, joyful, cheerful, "
                if random.randint(0, 1) == 0:
                    prompts.append(prompt + "trending on artstation")
                else:
                    prompts.append(prompt + "trending on pixiv")
        elif "nature" in theme:
            prompt = prompt.strip() + ", trending on artstation, hyperrealistic, trending, 4 k, 8 k, uhd"
            if not "detailed" in prompt:
                prompt += ", highly detailed"
            prompts.append(prompt)
        elif "space" in theme:
            if len(prompt) < 45:
                prompts.append(prompt.strip() + ", octane render, masterpiece, cinematic, trending on artstation, 8k ultra hd")
            elif not "lighting" in prompt:
                prompts.append(prompt.strip() + ", cinematic lighting, 8k ultra hd")
            else:
                prompts.append(prompt.strip() + ", 8k ultra hd")
        elif "low res" in theme:
            if not "low res" in prompt:
                prompts.append(prompt.strip() + ", low resolution")
        else:
            prompts.append(prompt.strip())
    return prompts
