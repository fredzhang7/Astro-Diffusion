from PIL import Image
from cv2 import dnn_superres
from typing import Tuple
import cv2, os, random, numpy


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def readLines(path):
    with open(path, "r", encoding="utf8") as f:
        return f.readlines()


def get_optimized_prompts(prompt_source: Tuple[str, list[str]], theme: str) -> list[str]:
    """
    - Either read a file with each prompt in a different line, or pass an array of prompts.
    - Pass a theme from the themes list to get a prompt that fits the theme.
    - Returns a list of optimized prompts.
    """
    prompts = []
    theme = theme.lower()
    if isinstance(prompt_source, list):
        prompt_source = prompt_source
    else:
        prompt_source = readLines(prompt_source)
    for prompt in prompt_source:
        if len(prompt) > 100:
            prompt.append(prompt)
        elif "anime girl" in theme or "waifu" in theme:
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
            if not prompt.startswith("1boy"):
                prompt = "1boy, " + prompt.strip()
            if not any(emotion in prompt for emotion in emotions):
                prompt += ", feeling happy, anime incarnation"
            prompts.append(prompt)
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


import requests
from bs4 import BeautifulSoup
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
tokenizer, model = None, None


def google_search(query, result_format):
    """
    Googles a query
    """
    query = query.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    if result_format == 'soup':
        return soup
    results = soup.find_all("div", class_="yuRUbf")
    if result_format == 'all_links':
        return [result.find("a")["href"] for result in results]
    elif result_format == 'first_link':
        return results[0].find("a")["href"]
    elif result_format == 'first_title':
        return results[0].find("a").text


def load_summarizer():
    """
    Loads the summarizer tokenizer and model
    """
    from transformers import BartTokenizer, BartForConditionalGeneration
    global tokenizer, model
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")


def character_search(full_name, incarnation=False, seek_artist=False):
    """
    Summarizes the appearance of a character
    """
    prefix = '1'
    if 'girl' in full_name:
        prefix = '1girl'
        full_name = full_name.replace('girl', '').replace('girls', '')
    elif 'female' in full_name:
        prefix = '1woman'
        full_name = full_name.replace('female', '')
    elif 'woman' in full_name:
        prefix = '1woman'
        full_name = full_name.replace('woman', '')
    elif 'boy' in full_name:
        prefix = '1boy'
        full_name = full_name.replace('boy', '').replace('boys', '')
    elif 'male' in full_name:
        prefix = '1man'
        full_name = full_name.replace('male', '')
    elif 'man' in full_name:
        prefix = '1man'
        full_name = full_name.replace('man', '')
    prefix += ', hyperrealistic human eyes, elliptical pupil, solid shapes, solid lines, 8k, uhd, hyperrealistic'
    # perfectly round iris, perfectly circular solid colored and centered pupil, pupil centered in eyes, gradient from pupil to iris, dreamy eyes, 
    if incarnation:
        prefix += ', anime incarnation'
    anime_name = None
    if " in " in full_name:
        anime_name = full_name.split(" in ")[1]
    elif " from " in full_name:
        anime_name = full_name.split(" from ")[1]
    elif " of " in full_name:
        anime_name = full_name.split(" of ")[1]
    elif seek_artist:
        try:
            # get the first title ending in "Wikipedia" and is clickable
            title = google_search(full_name + " anime name \"wikipedia\"", result_format='first_title')
            if "(" in title:
                anime_name = title.split("(")[0]
            elif "-" in title:
                anime_name = title.split("-")[0]
        except:
            pass
        if anime_name:
            soup = google_search(anime_name + ' anime artist name', result_format='soup')
            artist_name = (soup.find("div", {"data-tts": "answers"}))
            # fallback
            if artist_name == None:
                soup = google_search(anime_name + ' anime artist name', result_format='soup')
                artist_name = (soup.find("div", {"data-tts": "answer"}))
            else:
                artist_name = artist_name.get("data-tts-text")
            if isinstance(artist_name, list):
                artist_name = artist_name[0]
            if artist_name:
                prefix += ', art by ' + artist_name
    character_page = google_search(full_name + ' fandom', result_format='first_link')
    if "fandom" not in character_page:
        return f'{prefix}, {full_name}'
    r = requests.get(character_page, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    appearance = soup.find("span", id=lambda x: x and (x.startswith("Appearance") or x.startswith("Physical"))).parent
    appearance = appearance.find_next_siblings(["p", "h2"])
    descriptions = []
    length = 0
    for tag in appearance:
        text = tag.text.strip()
        length += len(text)
        if length > 4096:
            break
        if tag.name == "p":
            descriptions.append(text)
        elif tag.name == "h2" and ["Plot", "Relationships", "Trivia", "Gallery", "Site Navigation"].count(text) > 0:
            break
    appearance = " ".join(descriptions)
    if model is None or tokenizer is None:
        load_summarizer()
    inputs = tokenizer([appearance], max_length=2048, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=40, max_length=240)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    summary = summary[:-1]
    summary = summary.replace(', ', ' ').replace('\"', '').replace(' .', ',').replace(full_name.split(" ")[0] + ' ', '').replace('She ', '').replace('However ', '')
    if not 'eyes' in summary:
        findComma = summary.find(',')
        if findComma == -1:
            summary = 'beautiful eyes, ' + summary
        else:
            findComma = summary.find(',', findComma + 1)
            chosen = random.randint(0, 1) == 0
            if findComma == -1:
                summary = summary + ', beautiful eyes, perfect face' if chosen else summary + ', beautiful eyes, body portrait'
            else:
                summary = summary[:findComma] + ', beautiful eyes, perfect face' + summary[findComma:] if chosen else summary[:findComma] + ', beautiful eyes, body portrait' + summary[findComma:]
    return f'{prefix}, {full_name.strip()}, {summary.strip()}'