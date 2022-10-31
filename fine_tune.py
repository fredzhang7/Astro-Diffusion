from typing import List
import os

# if you have no idea what theme to use, just use the default one
themes = ['default', 'anime girl', 'anime boy', 'waifu', 'robot', 'mecha', 'android', 'cyborg', 'low resolution']

def get_optimized_file_prompts(filepath: str, allow_nsfw: bool) -> List[str]:
    """
    Read a file with each prompt in a different line and return a list of prompts.\n
    It's recommended to specify the gender of the character in the prompt.
    """
    prompts = []
    if not os.path.isfile(filepath):
        raise ValueError("Filepath is invalid")
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            if 'anime girl' in line or 'waifu' in line:
                if 'girls' in line:
                    prompts.append("correct body positions, " + line.strip() + ", sharp focus, beautiful, attractive")
                else:
                    prompts.append("1girl, correct body positions, " + line.strip() + ", sharp focus, beautiful, attractive, anime incarnation")
            elif 'anime boy' in line:
                ...
    return prompts

def get_optimized_model_choice(theme: str, lightweight: bool) -> str:
    """
    Return the model name based on the theme and the checkpoint model size
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