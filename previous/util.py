def readLines(path):
    with open(path, "r", encoding="utf8") as f:
        return f.readlines()

from typing import Tuple
import random

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


def containEmotions(prompt):
    emotions = ["sad", "angry", "surprised", "disgusted", "afraid", "calm", "confused", "bored", "happy"]
    return any(emotion in prompt for emotion in emotions)


def optimize_prompts(prompt_source: Tuple[str, list[str]], theme: str) -> list[str]:
    """
    - Either read a file with each prompt in a different line, or pass an array of prompts.
    - Theme is one of 'anime girl', 'anime boy', 'nature', 'space', 'low res', 'pony'
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
                    p = "1girl, " + p + ", anime incarnation" if random.randint(0, 1) == 1 else "1woman, " + p + ", anime incarnation"
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
                p += ", low res, blurry, pixelated, low quality"
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
tokenizer, distilbart = None, None


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
    global tokenizer, distilbart
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    distilbart = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")


def reset_summarizer():
    global tokenizer, distilbart
    tokenizer, distilbart = None, None


def character_search(prompt, seek_artist=False, latest=False) -> str:
    """
    Summarizes the appearance of a character
    """
    prefix = '1'
    name = prompt
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
        if distilbart is None or tokenizer is None:
            load_summarizer()
        inputs = tokenizer([appearance], max_length=2048, return_tensors="pt", truncation=True)
        summary_ids = distilbart.generate(inputs["input_ids"], num_beams=2, min_length=40, max_length=240)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        summary = summary[:-1]
        summary = summary.replace(', ', ' ').replace('\"', '').replace(' .', ',').replace('When ', ' ').replace(' He ', ' ').replace(name.split(" ")[0] + ' ', '').replace(name.split(" ")[1] + ' ', '').replace('She ', '').replace('However ', '')

        if not latest:
            prefix += ', highres, solid shapes, solid lines, 8k, uhd, hyperrealistic, hyperrealistic anime eyes, elliptical pupil, smooth brush strokes'
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
                summary = f'{summary}, beautiful eyes, perfect, body portrait' if c == -1 else f'{summary[:c]}, beautiful eyes, perfect, body portrait{summary[c:]}'
        return f'{prefix}, {name.strip()}, {summary.strip()}'
    except:
        return prompt


def parse_anime_prompts(prompts=[], is_anything=False) -> list[str]:
    if distilbart is None or tokenizer is None:
        load_summarizer()
    for i, p in enumerate(prompts):
        initial_len = len(p)
        p = danbooru_search(p) if len(p) < 75 else p
        prompts[i] = character_search(p, is_anything) if initial_len == len(p) else p
    reset_summarizer()
    return prompts


def random_pony_tags(res) -> str:
    soup = BeautifulSoup(res.text, "html.parser")
    try:
        images = soup.find_all("a", href=lambda x: x and x.startswith("/images/"), title=lambda x: x and x.startswith("Size:"))
        image = random.choice(images)
        tags = image.get("title").split(" Tagged: ")[1]
    except:
        return ''
    return tags


def pony_search(prompt="", seek_artist=True) -> str:
    """
    Summarizes the appearance of a pony
    """
    if len(prompt) > 60:
        return prompt
    nPony = ''
    name = prompt.lower()
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
    if name.replace(' ', '') == '' or name == 'random':
        r = requests.get('https://mlp.fandom.com/wiki/Special:Random', headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        name = soup.find("h1", class_="page-header__title").text.strip().replace(' ', '_')
    try:
        r = requests.get(f'https://derpibooru.org/search?q=safe%2C+{name}%2C+score.gte%3A100&sf=score&sd=desc', headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        resultText = soup.find("span", class_="block__header__title page__info")
        summary = name
        if not resultText:
            r = requests.get(f'https://derpibooru.org/search?page={random.randint(1, 1000)}&sd=desc&sf=upvotes&q=safe%2C{nPony}+score.gte%3A100', headers=headers)
            tags = random_pony_tags(r)
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
            summary = random_pony_tags(r)
        if ' gif, ' in summary or summary.endswith('gif') or 'animated' in summary or 'vulgar' in summary:
            return pony_search(name)
        if seek_artist:
            summary = optimize_prompts([summary], theme='pony')[0]
        return nPony.replace('+', '').replace('%2C', '') + ', ' + summary.replace(' 3d,', ' hyperrealistic,')
    except:
        return prompt


def parse_pony_prompts(prompts=[], seek_artist=True) -> list[str]:
    if len(prompts) == 0:
        return []
    prompts = [pony_search(prompt, seek_artist) for prompt in prompts]
    return prompts


def chinese_to_english(prompts=[]) -> list[str]:
    """
    Accurately translates prompts from Chinese to English
    """
    try:
        import sentencepiece
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
        return chinese_to_english(prompts)
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    translated = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    return translated


def translate(text="", from_lang='auto', to_lang='en') -> str:
    """
    Translates a prompt from one language to another using Google Translate
    """
    if from_lang == to_lang or len(text.replace(' ', '')) == 0:
        return text
    try:
        import googletrans
    except:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "googletrans"])
        return translate(text, from_lang, to_lang)

    from googletrans import Translator
    translator = Translator()
    return translator.translate(text, src=from_lang, dest=to_lang).text


def distilgpt2_prompt(prefix="", temperature=0.9, top_k=8, max_length=80, repitition_penalty=1.2, num_return_sequences=10) -> list[str]:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained('FredZhang7/distilgpt2-stable-diffusion-v2')

    input_ids = tokenizer(prefix, return_tensors='pt').input_ids
    output = model.generate(input_ids, do_sample=True, temperature=temperature, top_k=top_k, max_length=max_length, num_return_sequences=num_return_sequences, repetition_penalty=repitition_penalty, penalty_alpha=0.6, no_repeat_ngram_size=1, early_stopping=True)

    return [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(output))]


def random_value(obj: object) -> str:
    if isinstance(obj, dict):
        key = random.choice(random.choice(list(obj.values())))
        return random.choice(obj[key]) if key in obj else key
    raise TypeError(f'Expected dict, got {type(obj)}')


def parse_categories(name: str, string: str, data, l='[', r=']') -> str:
    if isinstance(data, list):
        return string.replace(l + name + r, random.choice(data))
    elif isinstance(data, dict):
        return string.replace(l + name + r, random_value(data))
    else:
        raise TypeError(f'Expected list or dict, got {type(data)}')


def parse_subcategories(string: str, obj: object) -> str:
    for key, value in obj.items():
        string = parse_categories(key, string, value, '<', '>')
    return string


def danbooru_search(tags="") -> str:
    if (tags.find('[') == -1 or tags.find(']') == -1) and (tags.find('<') == -1 or tags.find('>') == -1):
        return tags
    
    if not 'highres' in tags:
        tags += ', highres'
    isGirl = 'girl' in tags or 'female' in tags
    isBoy = 'boy' in tags or '1male' in tags or ' male' in tags

    from anime_webui.customizations import sections
    for category, section in sections.items():
        if category == 'male_fashion' and not isBoy:
            continue
        elif category == 'female_fashion' and not isGirl:
            continue
        tags = parse_categories(category, tags, category)
        for subcategory in section.items():
            tags = parse_subcategories(tags, subcategory)

    return tags

def download_safesearch(path):
    print("Downloading google_safesearch_mini.bin...")
    import urllib.request
    url = "https://huggingface.co/FredZhang7/google-safesearch-mini/resolve/main/pytorch_model.bin"
    urllib.request.urlretrieve(url, path)

from PIL import Image
inceptionv3 = None
def safesearch_filter(image: Image.Image) -> Image.Image:
    import torch, os
    model_path = 'google_safesearch_mini.bin'
    if not os.path.exists(model_path):
        model_path = 'art_generation/' + model_path
    if not os.path.exists(model_path):
        download_safesearch(model_path)
    model = torch.jit.load(model_path)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(image)
    img = img.unsqueeze(0)
    img = img.cpu()
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        out, _ = model(img)
        _, predicted = torch.max(out.data, 1)
        if predicted[0] != 2 and abs(out[0][2] - out[0][predicted[0]]) > 0.20:
            img = Image.new('RGB', image.size, color = (0, 255, 255))
            print("\033[93m" + "Image blocked by Google SafeSearch (Mini)" + "\033[0m")
            return img

    return image