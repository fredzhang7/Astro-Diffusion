def readLines(path):
    with open(path, "r", encoding="utf8") as f:
        return f.readlines()


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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w * scale, h * scale))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = upscale_image(frame, scale)
        out.write(frame)
    cap.release()
    out.release()


def containEmotions(prompt):
    emotions = ["sad", "angry", "surprised", "disgusted", "afraid", "calm", "confused", "bored", "happy"]
    return any(emotion in prompt for emotion in emotions)


def optimize_prompts(prompt_source: Tuple[str, list[str]], theme: str) -> list[str]:
    """
    - Either read a file with each prompt in a different line, or pass an array of prompts.
    - Pass a theme from the themes list to get a prompt that fits the theme.
    - Returns a list of optimized prompts.
    """
    prompts = []
    theme = theme.lower()
    if not isinstance(prompt_source, list):
        prompt_source = readLines(prompt_source)
    for p in prompt_source:
        p = p.strip()
        if "anime girl" in theme or "waifu" in theme:
            p = "correct body positions, " + p + ", sharp focus, beautiful, attractive, 4k, 8k ultra hd"
            if not 'girls' in p:
                if "waifu" in theme:
                    if random.randint(0, 1) == 1:
                        p = "1girl, " + p + ", anime incarnation"
                    else:
                        p = "1woman, " + p + ", anime incarnation"
                else:
                    p = "1girl, " + p + ", anime incarnation"
        elif "anime boy" in theme or "husbando" in theme:
            if not p.startswith("1boy"):
                p = "1boy, " + p
            if not containEmotions(p):
                p += ", feeling happy, anime incarnation"
        elif "nature" in theme:
            p += ", trending on artstation, hyperrealistic, trending, 4 k, 8 k, uhd"
            if not "detailed" in p:
                p += ", highly detailed"
        elif "space" in theme:
            if len(p) < 45:
                p += ", octane render, masterpiece, cinematic, trending on artstation, 8k ultra hd"
            elif not "lighting" in p:
                p += ", cinematic lighting, 8k ultra hd"
            else:
                p += ", 8k ultra hd"
        elif "low res" in theme:
            if not "low res" in p:
                p += ", low resolution"
        elif 'pony' in theme:
            if not 'safe, ' in p:
                p = 'safe, ' + p
            if not ' oc, ' in p:
                p += ' oc, oc only'
            if not ', high res' in p:
                p += ', high res'
            if not 'artist' in p:
                artists = ['fenwaru', 'rexyseven', 'kaylemi', 'zeepheru_pone', 'rrd-artist', 'kb-gamerartist', 'fenix-artist', 'gloriaartist', 'vensual99', 'stormcloud']
                p += f', artist:{random.choice(artists)}'
            if not containEmotions(p):
                p += ', happy'
        prompts.append(p)
    return prompts


import requests
from bs4 import BeautifulSoup
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
tokenizer, model = None, None


def google_search(query, result_format):
    """
    Googles a query. Result format is one of 'all_links', 'first_link', or 'first_title'
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


def anime_search(name, seek_artist=False) -> str:
    """
    Summarizes the appearance of an anime character
    """
    prefix = '1'
    if 'girl' in name:
        prefix = '1girl'
        name = name.replace('girl', '').replace('girls', '')
    elif 'female' in name:
        prefix = '1woman'
        name = name.replace('female', '')
    elif 'woman' in name:
        prefix = '1woman'
        name = name.replace('woman', '')
    elif 'boy' in name:
        prefix = '1boy'
        name = name.replace('boy', '').replace('boys', '')
    elif 'male' in name:
        prefix = '1man'
        name = name.replace('male', '')
    elif 'man' in name:
        prefix = '1man'
        name = name.replace('man', '')
    
    character_page = google_search(name + ' fandom', result_format='first_link')
    if "fandom" not in character_page:
        return f'{prefix}, {name}'
    try:
        r = requests.get(character_page, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        appearance = soup.find("span", id=lambda x: x and (x.startswith("Appearance") or x.startswith("Physical"))).parent
        appearance = appearance.find_next_siblings(["p", "h2"])
        if not appearance or len(appearance) == 0:
            appearance = soup.find("span", id=lambda x: x and x.startswith("Depiction")).parent
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
        summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=40, max_length=260)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        summary = summary[:-1]
        summary = summary.replace(', ', ' ').replace('\"', '').replace(' .', ',').replace(name.split(" ")[0] + ' ', '').replace('She ', '').replace('However ', '')

        prefix += ', highres, solid shapes, solid lines, 8k, uhd, hyperrealistic, hyperrealistic anime eyes, elliptical pupil, smooth brush strokes' # perfectly round iris, perfectly circular solid colored and centered pupil, pupil centered in eyes, gradient from pupil to iris, dreamy eyes 
        anime_name = None
        if " in " in name:
            anime_name = name.split(" in ")[1]
        elif " from " in name:
            anime_name = name.split(" from ")[1]
        elif " of " in name:
            anime_name = name.split(" of ")[1]
        elif seek_artist:
            try:
                # get the first title ending in "Wikipedia" and is clickable
                title = google_search(name + " anime name \"wikipedia\"", result_format='first_title')
                if "(" in title:
                    anime_name = title.split("(")[0]
                elif "-" in title:
                    anime_name = title.split("-")[0]
            except:
                pass
            if anime_name:
                soup = google_search(anime_name + ' anime artist name', result_format='soup')
                artist = (soup.find("div", {"data-tts": "answers"}))
                # fallback
                if artist == None:
                    soup = google_search(anime_name + ' anime artist name', result_format='soup')
                    artist = (soup.find("div", {"data-tts": "answer"}))
                else:
                    artist = artist.get("data-tts-text")
                if isinstance(artist, list):
                    artist = artist[0]
                if artist:
                    prefix += ', art by ' + artist
        if not 'eyes' in summary:
            c = summary.find(',')
            if c == -1:
                summary = 'beautiful eyes, ' + summary
            else:
                c = summary.find(',', c + 1)
                if c == -1:
                    summary = summary + ', beautiful eyes, perfect, body portrait'
                else:
                    summary = summary[:c] + ', beautiful eyes, perfect, body portrait' + summary[c:]

        return f'{prefix}, {name.strip()}, {summary.strip()}'
    except:
        print(bcolors.FAIL + f'Failed to summarize the appearance of {name}... Using the given prompt without prompt engineering...' + bcolors.ENDC)
        return f'{prefix}, {name}'


def parsePony(res) -> str:
    soup = BeautifulSoup(res.text, "html.parser")
    try:
        images = soup.find_all("a", href=lambda x: x and x.startswith("/images/"), title=lambda x: x and x.startswith("Size:"))
        image = random.choice(images)
        tags = image.get("title").split(" Tagged: ")[1]
    except:
        return ''
    return tags


def pony_search(name="", seek_artist=True) -> str:
    """
    Summarizes the appearance of a pony
    """
    nPony = ''
    name = name.lower()
    if 'solo' in name:
        nPony = '+solo%2C'
        name = name.replace('solo', '')
    elif 'duo' in name:
        nPony = '+duo%2C'
        name = name.replace('duo', '')
    else:
        acro = ['group', 'trio', 'quad', 'quartet', 'quintet', 'sextet', 'septet', 'octet', 'nonet', 'decet', 'hendecet', 'dodecet']
        for a in acro:
            if a in name:
                nPony = '+trio%2C'
                name = name.replace(a, '')
                break
        if len(name.split(' ')) <= 2:
            nPony = '+solo%2C'
    if name == "":
        r = requests.get('https://mlp.fandom.com/wiki/Special:Random', headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        name = soup.find("h1", class_="page-header__title").text.strip().replace(' ', '_')
    r = requests.get(f'https://derpibooru.org/search?q=safe%2C+{name}%2C+score.gte%3A100&sf=score&sd=desc', headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    resultText = soup.find("span", class_="block__header__title page__info")
    summary = name
    if not resultText:
        r = requests.get(f'https://derpibooru.org/search?page={random.randint(1, 1000)}&sd=desc&sf=upvotes&q=safe%2C{nPony}+score.gte%3A100', headers=headers)
        tags = parsePony(r)
        summary = tags if name == "" else name + ", " + tags
    else:
        resultText = resultText.text.strip()
        import math
        total = int(resultText.split("of ")[1].split(" ")[0])
        pages = math.ceil(total / 15)
        if pages > 10000:
            from numpy import log as ln
            pages = ln(pages) * 120
        elif pages > 1000:
            pages = 1000
        r = requests.get(f'https://derpibooru.org/search?page={random.randint(1, pages)}&sd=desc&sf=upvotes&q=safe%2C{nPony}+{name}%2C+score.gte%3A100', headers=headers)
        summary = parsePony(r)
    if ' gif, ' in summary or summary.endswith('gif') or 'animated' in summary or 'vulgar' in summary:
        return pony_search(name)
    if seek_artist:
        summary = optimize_prompts([summary], theme='pony')[0]
    return nPony.replace('+', '').replace('%2C', '') + ', ' + summary.replace(' 3d,', ' hyperrealistic,')