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
    path = './super-resolution/' + path
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


def anime_search(prompt, seek_artist=False, latest=False) -> str:
    """
    Summarizes the appearance of an anime character
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
        if model is None or tokenizer is None:
            load_summarizer()
        inputs = tokenizer([appearance], max_length=2048, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=40, max_length=240)
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
        # print(bcolors.FAIL + f'Failed to summarize the appearance of \"{prompt}\"... Using the given prompt without prompt engineering...' + bcolors.ENDC)
        return prompt


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


def generate_prompts(ins="", max_tokens=80, samples=10) -> list[str]:
    if len(ins).replace(' ', '') == 0:
        return []
    
    import os
    if not os.path.exists('./distil-sd-gpt2.pt'):
        import urllib.request
        print('Downloading DistilGPT2 Stable Diffusion model...')
        urllib.request.urlretrieve('https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion/resolve/main/distil-sd-gpt2.pt', './distil-sd-gpt2.pt')
        print('Model downloaded.')
    
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.max_len = 512

    import torch
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.load_state_dict(torch.load('distil-sd-gpt2.pt'))

    from transformers import pipeline
    nlp = pipeline('text-generation', model=model, tokenizer=tokenizer)
    ins = "a beautiful city"
    outs = nlp(ins, max_length=max_tokens, num_return_sequences=samples)

    for i in range(len(outs)):
        outs[i] = str(outs[i]['generated_text']).replace('  ', '')

    return outs


def random_value(obj: object) -> str:
    if isinstance(obj, dict):
        key = random.choice(random.choice(list(obj.values())))
        return random.choice(obj[key]) if key in obj else key
    raise TypeError(f'Expected dict, got {type(obj)}')


def generate_tags(name: str, string: str, data, l='[', r=']') -> str:
    if isinstance(data, list):
        return string.replace(l + name + r, random.choice(data))
    elif isinstance(data, dict):
        return string.replace(l + name + r, random_value(data))
    else:
        raise TypeError(f'Expected list or dict, got {type(data)}')


def subgroup_tags(string: str, obj: object) -> str:
    for key, value in obj.items():
        string = generate_tags(key, string, value, '<', '>')
    return string


def danbooru_search(tags="") -> str:
    if (tags.find('[') == -1 or tags.find(']') == -1) and (tags.find('<') == -1 or tags.find('>') == -1):
        return tags
    
    neutral = {
        'simple background': ["simple background","starry background","transparent background","zoom layer","mosaic background","paneled background","photo background","backlighting","blurry background","card background","chibi inset","drama layer","fiery background","flag background","floral background","fruit background","heart background","argyle background","checkered background","food-themed background","halftone background","honeycomb background","paw print background","plaid background","polka dot background","simple background","snowflake background","spiral background","strawberry background","striped background","sunburst background","gradient background","multicolored background","rainbow background","heaven condition","two-tone background","aqua background","beige background","black background","blue background","brown background","green background","grey background","lavender background","orange background","pink background","purple background","red background","sepia background","tan background","white background","yellow background"],
        'emotion': ["angry","anger vein","annoyed","clenched teeth","scowl","annoyed","blush","blush stickers","embarrassed","full-face blush","nose blush","bored","closed eyes","confused","crazy","determined","disappointed","disdain","disgust","despair","drunk","envy","expressionless","evil","facepalm","flustered","frustrated","furrowed brow","grimace","guilt","happy","kubrick stare","lonely","nervous","one eye closed","open mouth","parted lips","pain","pout","raised eyebrow","rape face","rolling eyes","sad","depressed","frown","gloom (expression)","tears","scared","panicking","worried","serious","sigh","sleepy","tired","sulking","surprised","thinking","pensive","v-shaped eyebrows","wince","upset","^^^","color drain","depressed","despair","gloom (expression)","horrified","screaming","sobbing","turn pale","trembling","wavy mouth"],
        'emote': [";)",":d",";d","xd","d:",":3",";3","x3","3:","uwu",":p",";p",":q",";q",">:)",">:(",":t",":i",":/",":|",":x",":c","c:",":<",";<",":<>",":>",":>=",":o",";o","o3o","(-3-)",">3<","o_o","0_0","|_|","._.","solid circle eyes","heart-shaped eyes","^_^","^o^","\(^o^)/","^q^",">_<","xd","x3",">o<","@_@",">_@","+_+","+_-","=_=","=^=","=v=","<o>_<o>","<|>_<|>"],
        'smile': ["crazy smile","evil smile","fingersmile","forced smile","glasgow smile","grin","light smile","sad smile","seductive smile","stifled laugh","doyagao","smirk","smug"],
        'pupil': ["slit pupils", "symbol-shaped pupils", "heart-shaped pupils", "flower-shaped pupils", "star-shaped_pupils", "diamond-shaped pupils", "bunny-shaped pupils"],
        'eye color': ["aqua eyes","black eyes","blue eyes","brown eyes","green eyes","grey eyes","orange eyes","purple eyes","pink eyes","red eyes","yellow eyes","amber eyes","heterochromia","multicolored eyes","@ @","dashed eyes","Pac-man eyes","ringed eyes"],
        'eye style': ["print eyepatch","bags under eyes","aegyo sal","bruised eye","flaming eyes","glowing eyes","glowing eye","mako eyes","blinking","closed eyes","wince","one eye closed",";<",";>",";p","> <","X3","XD","DX","O o","0 0","3 3","6 9","@ @","^ ^","^o^","|_|","||_||","= =","+ +",". .","<o> <o>","covering eyes","hair over eyes","hair over one eye","bandage over one eye","blindfold","hat over eyes","eyepatch","eyelashes","colored eyelashes","fake eyelashes","eyes visible through hair","glasses","makeup","eyeliner","eyeshadow","mascara"],
        'eye gaze': ["eye contact","looking afar","looking at another","looking at breasts","looking at hand","looking at mirror","looking at phone","looking at viewer","looking away","looking back","looking down","looking outside","looking over eyewear","looking through legs","looking to the side","looking up"],
        'pattern': ["yagasuri","houndstooth","goshoguruma","horizontal stripes","tiger stripes","shippou (pattern)","kikumon","multicolored stripes","igeta (pattern)","invasion stripes","camouflage","diagonal stripes","sakuramon","argyle","double vertical stripe","gingham","striped","asa no ha (pattern)","kojitsunagi (pattern)","polka dot","patterned clothing","patterned background","patterned","uroko (pattern)","pinstripe pattern","colored stripes","uchiwa design","honeycomb","shima (pattern)","patterned hair","karakusa (pattern)","kanoko (pattern)","vertical stripes","seigaiha","sayagata","checkered","egasumi","kikkoumon","kagome (pattern)","plaid","genjiguruma"],
        'print': ["animal print","bat print","bear print","bird print","cow print","leopard print","tiger print","snake print","zebra print","flag print","floral print","cherry blossom print","game controller print","moon print","crescent print","hand print","leaf print","musical note print","piano print","watermelon print","print umbrella"],
        'coloring': ["inverted colors","colorized","black theme","spot color","grey theme","green theme","color drain","neon palette","greyscale with colored background","color connection","high contrast","ff gradient","orange theme","cel shading","flat color","sepia","yellow theme","brown theme","white theme","purple theme","pink theme","colorful","aqua theme","rainbow","colored with greyscale background","gradient","greyscale","multiple monochrome","red theme","pale color","muted color","rainbow order","pastel colors","blue theme","anime coloring","limited palette","monochrome"],
        'art style': ["pinup","style parody","ukiyo-e","friday night funkin'","faux traditional media","realistic","flame painter","fine art parody","nihonga","2000s","1980s","unfinished","sketch","abstract","1970s","traditional media","sumi-e","pokemon rgby","1920s","animification","bikkuriman","1940s","granblue fantasy","*_(medium)","photorealistic","ligne claire","retro artstyle","1960s","minimalism","1950s","impressionism","art nouveau","art deco","1990s","cartoonized","surreal","toon","1930s","western comics"],
        'hair style': ["whipping hair","hair slicked back","messy hair","afro","hair rings","front braid","buzz cut","sidecut","hime cut","heart ahoge","curly hair","asymmetrical bangs","sidelocks","ahoge","wavy hair","bowl cut","pixie cut","heart hair bun","hair intakes","pompadour","huge ahoge","parted bangs","split ponytail","side ponytail","shouten pegasus mix mori","two side up","swept bangs","quad tails","low twintails","hair between eyes","asymmetrical hair","multiple braids","uneven twintails","crew cut","half updo","double bun","dreadlocks","bow-shaped hair","low-braided long hair","blunt ends","braided bun","quiff","huge afro","undercut","single hair ring","folded ponytail","short ponytail","widow's peak","braided bangs","blunt bangs","mullet","topknot","multi-tied hair","hair scarf","quad braids","low-tied long hair","chonmage","triple bun","bangs","nihongami","crown braid","hair up","ponytail","single hair intake","drill hair","flattop","spiked hair","quin tails","side braid","beehive hairdo","single braid","one side up","ringlets","hair flaps","single hair bun","tri braids","hair over one eye","lone nape hair","tri tails","hair down","comb over","mizura","mohawk","cornrows","front ponytail","antenna hair","twin braids","short twintails","french braid","twintails","okappa","braid","hair over eyes","pointy hair","flipped hair","high ponytail","hair over shoulder","low twin braids","hair bun","oseledets","bob cut","twin drills","cone hair bun","hair pulled back","alternate hairstyle"],
        'hair color': ["streaked hair","gradient hair","brown hair","rainbow hair","light purple hair","two-tone hair","red hair","aqua hair","split-color hair","colored inner hair","light blue hair","purple hair","blue hair","black hair","dark green hair","light brown hair","blonde hair","green hair","multicolored hair","orange hair","dark blue hair","colored tips","grey hair","white hair","light green hair","pink hair"],
        'ear style': ["pig ears","tiger ears","horse ears","ear protection","bat ears","pikachu ears","bear ears","fox ears","behind ear","monkey ears","wolf ears","animal ears","cat ears","lion ears","squirrel ears","panda ears","dog ears","raccoon ears","fake animal ears","sheep ears","cow ears","kemonomimi mode","ferret ears","robot ears","deer ears","hair ears","rabbit ears","mouse ears","pointy ears","ear piercing","goat ears"],
        'headphones': ["headphones","headphones for animal ears","earpiece","bunny headphones","animal ear headphones","cat ear headphones","behind-the-head headphones","headset"],
        'earring': ["crescent earrings","jack-o'-lantern earrings","strawberry earrings","cross earrings","pineapple earrings","tassel earrings","cherry earrings","orange-shaped earrings","spade earrings","heart earrings","crystal earrings","yin yang earrings","skull earrings","hoop earrings","pom pom earrings","magatama earrings","planet earrings","shell earrings","bell earrings","potara earrings","single earring","multiple earrings","star earrings","pill earrings","food-themed earrings","stud earrings","flower earrings","snowflake earrings"],
        'earring states': ["adjusting earrings"],
        'glasses': ["heart-shaped eyewear","star-shaped eyewear","flight goggles","looking for glasses","opera glasses","eyewear hang","opaque glasses","lorgnette","x-ray glasses","bespectacled","pince-nez","rimless eyewear","semi-rimless eyewear","eyewear removed","nodoka glasses","fancy glasses","eyewear strap","coke-bottle glasses","simon shades","ski goggles","funny glasses","safety glasses","shooting glasses","3d glasses","scouter","aviator sunglasses","teardrop-framed glasses","diving mask","kamina shades"],
        'hat': ["mini hat","checkered headwear","police hat","straw hat","military hat","multicolored headwear","school hat","mini top hat","penguin hat","mob cap","deviruchi hat","dixie cup hat","wizard hat","tokin hat","bucket hat","songkok","nightcap","mao cap","pelt","jingasa","dog hat","fox hat","tam o' shanter","fascinator","pillow hat","rogatywka","dunce cap","cat hat","party hat","wolf hat","bashlik","mortarboard","mitre","sun hat","bear hat","pirate hat","bicorne","beret","kippah","peaked cap","mini witch hat","cowboy hat","kepi","campaign hat","tate eboshi","budenovka","rice hat","deerstalker","sailor hat","mini santa hat","eggshell hat","shako cap","bunny hat","papakha","chef hat","sajkaca","sombrero","pith helmet","garrison cap","qing guanmao","pumpkin hat","witch hat","fez hat","bearskin cap","toque blanche","shampoo hat","bowler hat","gat","cabbie hat","beanie","tsunokakushi","tricorne","nurse cap","hat with ears","straw hat","porkpie hat","boater hat","top hat","flat top chef hat","baseball cap","mian guan","ushanka","fur hat","fedora","cloche hat","santa hat","flat cap","animal hat","ajirogasa"],
        'hat states': ["putting on headwear","hands on headwear","hand on headwear","no headwear","backwards hat","sideways hat","tilted headwear","torn hat","hat tip","holding hat","hat over one eye","adjusting headwear","hat removed"],
        'sleeve states': ["sleeve grab","pinching sleeves","arm out of sleeve","hands in opposite sleeves"],
        'helmet': ["tank helmet","hardhat","stahlhelm","dragoon helmet","kabuto","kettle helm","horned helmet","bicycle helmet","helm","headlamp","sallet","baseball helmet","pickelhaube","brodie helmet","winged helmet","adrian helmet","motorcycle helmet","pith helmet","american football helmet","diving helmet"],
        'neckwear': ["lace-trimmed choker","red choker","black necktie","pink choker","pink bowtie","fur choker","neck ruff","white neckerchief","aqua bowtie","bow choker","headphones around neck","lanyard","red bowtie","amulet","necktie grab","aqua choker","grey choker","blue necktie","aqua necktie","grey necktie","green neckerchief","friendship charm","neckerchief","o-ring choker","v-neck","frilled choker","green choker","pink neckerchief","ribbon choker","feather boa","open collar","lei","popped collar","orange bowtie","white choker","star choker","necktie","cross choker","black bowtie","heart choker","purple neckerchief","anchor choker","brown bowtie","magatama","yellow necktie","flower necklace","necktie removed","cross tie","fur collar","spiked choker","blue choker","sailor collar","neck ribbon","orange choker","yellow bowtie","chain necklace","pendant","yellow neckerchief","red necktie","sleeveless turtleneck","grey bowtie","ascot","red neckerchief","black choker","pet cone","necktie between breasts","collared shirt","scarf","brown necktie","brown choker","stole","white necktie","black neckerchief","locket","purple choker","wing collar","collar tug","necktie on head","turtleneck","adjusting collar","brown neckerchief","blue bowtie","bolo tie","pink necktie","grey neckerchief","necklace","gold choker","purple bowtie","purple necktie","orange necktie","aqua neckerchief","green necktie","bowtie","choker","blue neckerchief","collar grab","high collar","pentacle","pendant choker","white bowtie","green bowtie","studded choker","detached collar","goggles around neck","lace choker","yellow choker","jabot","orange neckerchief","pearl necklace"],
        'topwear': ["print shirt","shrug","overcoat","corset","off-shoulder shirt","suit jacket","tabard","stringer","striped shirt","collared shirt","sweater dress","jacket","ribbed sweater","sweater vest","bandeau","tailcoat","crop top","dress","poncho","raglan sleeves","sash","t-shirt","sleeveless shirt","bustier","criss-cross halter","vest","hoodie","shirt","peacoat","turtleneck","blouse","blazer","halterneck","tube top","camisole","cardigan vest","compression shirt","cropped jacket","fur coat","coat","stole","raincoat","tank top","shoulder sash","winter coat","duffel coat","fur-trimmed coat","sleeveless turtleneck","sweater","aran sweater","safari jacket","pullover","surcoat","cardigan","sukajan","underbust","yellow raincoat","letterman jacket","waistcoat","trench coat","long coat"],
        'pants': ["capri pants","white pants","purple pants","bell-bottoms","orange pants","black pants","harem pants","red pants","yellow pants","grey pants","tight pants","pants rolled up","plaid pants","pink pants","single pantsleg","brown pants","jeans","leather pants","blue pants","pinstripe pants","pant suit","green pants"],
        'shorts': ["print shorts","short shorts","dolphin shorts","shorts","bike shorts","petticoat","gym shorts","pants","capri pants","lowleg shorts","pants rolled up","denim shorts","jeans","lowleg pants","buruma","bell-bottoms","pelvic curtain","yoga pants","detached pants","bloomers","chaps","micro shorts","kilt","cutoff jeans"],
        'gesture': ["fist in hand","pointing up","bowing","pointing at self","bunny pose","pointing","air quotes","\||/","cupping hands","finger frame","fox shadow puppet","finger heart","air guitar","kamina pose","high five","claw pose","power fist","own hands clasped","middle w","kuji-in","salute","pinky out","index finger raised","pointing down","noogie","money gesture","fist bump","finger gun","horns pose","heart tail","shrugging","shushing","toe-point","spread fingers","palm-fist greeting","ohikaenasutte","raised fist","pinky swear","double v","beckoning","two-finger salute","shadow puppet","curtsey","heart hands duo","reaching","victory pose","thumbs down","waving","clenched hands","straight-arm salute","paw pose","\n/","clenched hand","facepalm","finger counting","heart hands","crossed fingers","middle finger","heart tail duo","vulcan salute","fig sign","heart arms","palm-fist tap","carry me","tsuki ni kawatte oshioki yo","akanbe","pointing at viewer","fidgeting","hand glasses","\m/","open hand","steepled fingers","heart hands trio","thumbs up","saturday night fever","pointing forward","ok sign","shaka sign","inward v"],
        'posture': ["airplane arms","indian style","inugami-ke no ichizoku pose","headstand","handstand","kneeling","legs apart","contrapposto","chest stand","animal pose","battoujutsu stance","sitting on person","knees together feet apart","crossed legs","arms behind head","arched back","praise the sun","v arms","bunny pose","prostration","butterfly sitting","shrugging","arm support","one knee","head tilt","letter pose","jumping","dojikko pose","reaching","reclining","lying","straddling","t-pose","symmetrical hand pose","w arms","arm behind head","crucifixion","arm behind back","claw pose","lotus position","upside-down","yoga","leaning forward","twisted torso","superhero landing","arms up","stroking own chin","body bridge","yokozuwari","knees to chest","standing","faceplant","sitting on lap","outstretched arm","stretching","jojo pose","a-pose","wariza","leaning back","top-down bottom-up","ojou-sama pose","full scorpion","all fours","hugging own legs","on stomach","seiza","spread arms","archer pose","figure four sitting","zombie pose","paw pose","standing on one leg","flexing","gendou pose","walking","horns pose","cowering","arm at side","fetal position","slouching","thigh straddling","upright straddle","on side","sitting","bent over","saboten pose","balancing","victory pose","villain pose","squatting","bras d'honneur","running","\o/","crossed arms","on back","fighting stance","arm up","outstretched hand"],
        'hand position': ["hand on own head","hand on headwear","hands on hips","hand on own forehead","hand in pocket","hands on own face","adjusting eyewear","hand on own chest","hand on own shoulder","hands in pockets","hand on own face","hand on ear","hand on own cheek"],
        'shoulder ornaments': ["bird on shoulder","weapon over shoulder","sword over shoulder","towel around neck","cat on shoulder"],
        'pov': ["portrait","profile","sideways","wide shot","from behind","cowboy shot","portrait","very wide shot","upper body","vanishing point","lower body","dutch angle","panorama","full body","close-up","atmospheric perspective","pov","upside-down","from above","fisheye","face","perspective","cut-in","rotated"],
        'wings': ["fairy wings","white wings","light hawk wings","ladybug wings","wing ribbon","plant wings","bowed wings","liquid wings","moth wings","dragon wings","heart wings","red wings","black wings","bat wings","metal wings","butterfly wings","demon wings","hair wings","feathered wings","mechanical wings","insect wings","energy wings","angel wings","fiery wings","gradient wings","glowing wings","multiple wings","ice wings"],
        'tail': ["tail ribbon","heart tail","squirrel tail","ghost tail","pig tail","wolf tail","bear tail","lion tail","scorpion tail","pikachu tail","tadpole tail","rabbit tail","multiple tails","dragon tail","demon tail","fox tail","horse tail","leopard tail","sheep tail","mouse tail","deer tail","monkey tail","ermine tail","fiery tail","dog tail","snake tail","cow tail","cat tail","tiger tail","fish tail"],
        'skin color': ["tanlines","tan","pale skin","sun tattoo","dark skin"],
        'transformation': ["no tail","alternate hairstyle","cosplay","alternate wings","contemporary","genderswap (ftm)","hair down","casual","aged down","aged up","alternate eye color","alternate universe","no wings","genderswap (mtf)","light persona","costume combination","kemonomimi mode","out of character","personality switch","dark persona","role reversal","alternate hair length","adapted costume","no horn","alternate skin color","enmaided","hair up","alternate hair color"],
        'medic': ["intravenous drip","bandaid on ear","bandaid on nose","first aid","bandaid on face","bandages","bandaid on leg","bandage on face","syringe","doctor","surgical mask","bandaid on forehead","bandaid on cheek","hospital bed","bandaid","sarashi","bandage over one eye","stethoscope","hospital","surgery","eyepatch","nurse","bandaid on knee","rubber gloves","bandaid on arm"],
        'clothing brand': ["speedo","the north face","nike","reebok","ray-ban","converse","puma","citizen","gucci","vans","adidas","chanel","doc martens","air jordan","new balance","uniqlo","tommy hilfiger","dior","asics","levi's"],
        'sports': ["olympics","bowling","kabaddi","basketball","playing sports","rhythmic gymnastics","rugby","national football league","tennis","badminton","world cup","boxing","national basketball association","soccer","volleyball","ice hockey","croquet","billiards","baseball","gymnastics","field hockey","water polo","sepak takraw","table tennis","golf","kemari","lacrosse","american football"],
        'holiday': ["cirno day","valentine","thanksgiving","mother's day","tsukimi","bunny day","pocky day","glasses day","cat day","halloween","easter","new year","bikini day","april fools","christmas","maid day","summer festival","kiss day","mid-autumn festival, tsukimi","chinese new year","father's day"],
        'commemoration': ["anniversary","birthday","congratulations","happy birthday","thank you","milestone celebration"],
        'job': ["mechanic","dj","nurse","shepherd","butler","bartender","doctor","soldier","school nurse","lumberjack","burglar","actor/actress","maid","sailor","driver","slave","bodyguard","cyclist","astronaut","salaryman","ninja","firefighter","hikikomori","police","conductor","dentist","warrior","miner","monk","wizard/witch","prostitution","scientist","lifeguard","priest","train conductor","trucker","librarian","guard","dominatrix","standard-bearer","artist painter","office lady","geisha","teacher","croupier","engineer","samurai","cashier","merchant","musician","alchemist","politician","spy","flight attendant","miko","construction worker","florist","chemist","nun","janitor","pilot","idol","prisoner","hacker","chef","train attendant","officer","athlete","farmer","waiter/waitress","judge"],
        'action': ["imagining","drinking","waving","kicking","spinning","whistling","giggling","blocking","skipping","floating","imitating","standing","bowing","watching television","cooking","singing","teaching","broom riding","throwing","waiting","concentrating","surfing","applying makeup","cleaning","shading eyes","praying","sinking","sleeping","whispering","typing","running","studying","shopping","balancing","sketching","warming","sulking","sitting","glowing","broom surfing","splashing","apologizing","hairdressing","swinging","sliding","working","walking","pitching","spilling","screaming","wading","programming code","laughing","text messaging","stroking own chin","playing games","playing instrument","smoking","unsheathing","knocking","training","summoning","thinking","talking","sneezing","resting","shouting","staring","weightlifting","juggling","flying","whisking","driving","reloading","chasing","dancing","petting","jumping","slipping","shushing","slashing","cheering","kneeing"],
        'anime': ["genshin impact", "azur lane", "fate (series)", "touhou", "kantai collection", "serafuku"],
        'video games': ["league of legends","apex legends","plants vs zombies","dead space","capcom","playstation","bandai namco","command and conquer","star wars","titanfall (series)","walkman","mass effect (series)","battlefield (series)"]
    }
    male = {
        'swim suits': ["swim briefs","legskin","swim trunks","jammers"],
        'male': ["mature male","muscular male","male child"],
        'footwear': ["monk shoes","saddle shoes","sandals","oxfords","loafers","slippers","sneakers","flip-flops"],
        'sleeves': ["pink sleeves","grey sleeves","short sleeves","sleeves past elbows","yellow sleeves","green sleeves","black sleeves","aqua sleeves","brown sleeves","polka dot sleeves","striped sleeves","orange sleeves","plaid sleeves","sleeves past wrists","purple sleeves","blue sleeves","sleeves past fingers","white sleeves","uneven sleeves","checkered sleeves","red sleeves","long sleeves"]
    }
    female = {
        'skirt': ["hoop skirt","high-low skirt","pink skirt","green skirt","overskirt","skirt set","long skirt","miniskirt","bubble skirt","beltskirt","white skirt","hakama short skirt","layered skirt","grass skirt","side-tie skirt","skirt suit","suspender skirt","microskirt","black skirt","pleated skirt","aqua skirt","grey skirt","spank skirt","purple skirt","brown skirt","kimono skirt","red skirt","plaid skirt","bow skirt","frilled skirt","blue skirt","showgirl skirt","bikini skirt","yellow skirt","checkered skirt","pencil skirt","orange skirt","denim skirt"],
        'dress': ["short dress","taut dress","latex dress","long dress","plunging neckline","denim dress","trapeze dress","vertical-striped dress","pinstripe dress","wedding dress","yellow dress","flag dress","half-dress","lace-trimmed dress","backless dress","layered dress","impossible dress","ribbed dress","strapless dress","cake dress","plaid dress","ribbon-trimmed_dress","tennis dress","sweater dress","striped dress","sepia dress","two-tone dress","frilled dress","sailor dress","collared dress","funeral dress","negligee","see-through dress","pink dress","hobble dress","flowing dress","gown","pencil dress","brown dress","coat dress","sundress","argyle dress","santa dress","print dress","checkered dress","sleeveless dress","black dress","china dress","off-shoulder dress","grey dress","fur-trimmed_dress","highleg dress","skirt basket","evening gown","tube dress","vietnamese dress","pleated dress","nightgown","white dress","green dress","polka dot dress","halter dress","purple dress","cocktail dress","side slit","pinafore dress","armored dress","mermaid dress","multicolored dress","aqua dress","dirndl","blue dress","orange dress","high-low skirt","red dress"],
        'legwear': ["pinstripe legwear","purple pantyhose","polka dot legwear","striped legwear","grey leggings","ribbon-trimmed legwear","stirrup legwear","pink pantyhose","white thighhighs","back-seamed legwear","red leggings","grey socks","black thighhighs","argyle legwear","black socks","studded legwear","fluffy legwear","ribbed legwear","cat ear legwear","o-ring legwear","purple leggings","black leggings","bat legwear","fishnet legwear","ribbon legwear","mismatched_legwear kneehighs","animal ear legwear","pantyhose","front-seamed legwear","over-kneehighs","lace-trimmed legwear","toeless legwear","print_legwear kneehighs","print legwear","blue socks","see-through legwear","red socks","pink socks","rainbow legwear","bow_legwear kneehighs","red thighhighs","checkered legwear","shiny legwear","grey pantyhose","bear band legwear","bunny ear legwear","purple socks","pleated legwear","purple thighhighs","green socks","seamed legwear","checkered_legwear kneehighs","toeless_legwear kneehighs","american flag legwear","horn band legwear","knit legwear","bow legwear","argyle_legwear kneehighs","zipper legwear","pink thighhighs","spiked legwear","polka_dot_legwear kneehighs","lace legwear","aran legwear","gingham legwear","black pantyhose","brown socks","orange socks","frilled_legwear kneehighs","plaid legwear","legwear bell","striped_socks","side-seamed legwear","frilled legwear","vertical-striped socks","red pantyhose","grey thighhighs","side-tie legwear","cross-laced legwear","white pantyhose","white leggings","lace-up legwear","bridal legwear","diagonal-striped legwear","latex legwear","white socks","pink leggings","fur-trimmed legwear","armored legwear","vertical-striped legwear","thighhighs","thighband","thigh strap","tube socks","garter belt","legwear garter","kneehighs","garter straps","ankle socks","leg warmers","tabi","leggings","toe socks","loose socks","bobby socks"],
        'footwear': ["sandals","high heels","animal slippers","winged footwear","converse","toeless footwear","zouri","thigh boots","high heel boots","mary janes","geta","spurs","lace-up boots","cross-laced footwear","waraji","saddle shoes","pumps","platform footwear","loafers","armored boots","okobo","slippers","crocs","monk shoes","flip-flops","cowboy boots","ankle boots","rubber boots","high tops","uwabaki","boots","footwear ribbon","ballet slippers","wedge heels","gladiator sandals","oxfords","sneakers","flats","pointy footwear","cross-laced sandals","clog sandals","knee boots"],
        'shoulder style': ["off shoulder","off-shoulder coat","bare shoulders","sleeveless dress","pink tube top","off-shoulder leotard","single-shoulder dress","off-shoulder dress","sleeveless coat","single-shoulder sweater","white tube top","off-shoulder sweater","broad shoulders","off-shoulder jacket","sleeveless jacket","strap slip","strapless shirt","jacket on shoulders","sleeveless sweater","strapless dress","puffy sleeves","blue tube top","shoulder blades","epaulettes","shoulder pads","single-shoulder shirt","green tube top","purple tube top","sleeveless shirt","off-shoulder shirt","shirt on shoulders","pauldrons","red tube top","black tube top","grey tube top"],
        'hair ornament': ["leaf hair ornament","hair brush","hair ornament","fiery hair","hair ribbon","hair dryer","hairband","comb","hair tie","scrunchie","wig","hair bell","hair flower","chopsticks","hairpods","hair weapon","hair tubes","shampoo","hair bow","hair bobbles","hairpin","kanzashi","crystal hair","bun cover","hairclip"],
        'headwear': ["hair bow","crown","forehead protector","tiara","maid headdress","veil","honggaitou","balaclava","headdress","hair ribbon","hachimaki","headband","hairband","mongkhon","sweatband"],
        'costumes': ["maid","japan air self-defense force","dress shirt","american football uniform","military jacket","fast food uniform","school hat","thai student uniform","soccer uniform","miko","nurse","serafuku","hooters","japan ground self-defense force","coco's","naval uniform","soviet school uniform","kindergarten uniform","police uniform","pilot uniform","kobeya uniform","republic of china school uniform","bronze parrot","basketball uniform","indonesian high school uniform","gakuran","volleyball uniform","employee uniform","school swimsuit","rugby uniform","band uniform","chef uniform","thai college uniform","shinsengumi","waitress","meiji schoolgirl uniform","sailor","yoshinoya","military hat","anna miller","tennis uniform","malaysian secondary school uniform","baseball uniform","gym uniform","track uniform","japan maritime self-defense force","meiji schoolgirl uniform","shosei","frilled shirt","cow costume","pant suit","cowboy western","tuxedo","reindeer costume","animal costume","nontraditional miko","pilot suit","dog costume","armored dress","sailor","school uniform","armor","apron","tiger costume","shoulder cape","boar costume","sweater","capelet","habit","miko","gym uniform","gakuran","santa costume","kigurumi","sweatpants","cheerleader","ghost costume","mouse costume","buruma","hazmat suit","military uniform","bear costume","cape","panda costume","overalls","serafuku","skirt suit","loincloth","costume","suit","hood","track suit","bikini armor","cassock","tutu","plugsuit","waitress","penguin costume","pig costume","harem outfit","cat costume","seal costume","monkey costume","pajamas","sailor dress","band uniform","hev suit","rabbit costume","business suit","sheep costume"],
        'swimsuits': ["print swimsuit","black one-piece swimsuit","brown one-piece swimsuit","red one-piece swimsuit","purple one-piece swimsuit","gold one-piece swimsuit","orange one-piece swimsuit","yellow one-piece swimsuit","green one-piece swimsuit","white one-piece swimsuit","grey one-piece swimsuit","blue one-piece swimsuit","aqua one-piece swimsuit","pink one-piece swimsuit","silver one-piece swimsuit","swimsuit","jumpsuit","playboy bunny","legskin","criss-cross halter","bathrobe","sports bikini","strapless leotard","romper","tunic","bodystocking","short jumpsuit","kesa","bikesuit","robe","unitard","leotard","competition swimsuit","sarong","jammers","racing suit","rash guard","swim briefs","tankini","bodysuit"],
        'tradition': ["hakama short skirt","yukata","chinese clothes","geta","china dress","changpao","korean clothes","midriff sarashi","hanbok","straw cape","happi","tasuki","short kimono","hakama skirt","budget sarashi","hakama pants","kimono skirt","vietnamese dress","fengguan","hakama","sarashi","hanfu","chest sarashi","nontraditional miko","tangzhuang","dotera","haori","chanchanko","japanese clothes","layered kimono","furisode","uchikake","fundoshi","undone sarashi","kimono","dirndl","hanten","miko","tabi","yamakasa","longpao","mino boushi"],
        'hat': ["hijab","frilled hat","okosozukin","shower cap","head scarf","jester cap","veil","keffiyeh","aviator cap","shufa guan","habit","visor cap","bandana","bonnet","balaclava","dalachi"],
        'sleeves': male["sleeves"] + ["single detached sleeve","raglan sleeves","wide sleeves","lace-up sleeves","puffy sleeves","compression sleeve","mismatched sleeves","long sleeves","short sleeves","elbow sleeve","puff and slash sleeves","layered sleeves","puffy detached sleeves","puffy short sleeves","ribbed sleeves","torn sleeves","short over long sleeves","bell sleeves","see-through sleeves","detached sleeves","puffy long sleeves"]
    }
    tech = {
        'aircraft': ["aircraft","helicopter","airplane","airship","taking off","spacecraft"],
        'ship': ["ship, pier","gondola","airship","canoe","ship, dock","ship","sailboat","rowboat","kayak","ship, harbor","aircraft carrier","boat","destroyer","battleship","cruiser","submarine"],
        'software': ["macintosh","firefox","gnu/linux","os-tan","google","windows","internet explorer"],
        'interface': ["mouse, mousepad (object)","hologram","holographic interface","holographic keyboard","holographic monitor","drawing tablet","curved_monitor","touchscreen"],
        'hardware': ["dvd (object)","server","floppy disk","usb","cd","minidisc","laptop","graphics card","external hard drive","tablet pc","sd card"],
        'tech brand': ["sony","amd","samsung","apple inc.","google","nvidia","intel","dell","microsoft","ibm"],
        'car brand': ["audi","alfa romeo","bugatti","mercedes-benz","jeep","lamborghini","subaru (brand)","land rover","mini cooper","nissan","ducati car","volkswagen","dodge","ferrari","chevrolet","mazda","toyota","bmw car","ford"],
        'motorcycle brand': ["yamaha","suzuki","kawasaki","bmw motorcycle","vespa","scooter"],
        'comms': ["microphone stand","radio antenna","radio telescope","radio","walkie-talkie","smartphone","payphone","field radio","radio tower","speaker","radio booth","microphone","cellphone"],
        'artificial life': ["cyborg","android","mecha","transformers","non-humanoid robot","cyber elves","humanoid robot","robot animal"],
        'themes': ["personification","fantasy","mechanization","cyberpunk","science fiction","steampunk"]
    }
    pet = {
        'pokemon': ["turtwig","litten","fennekin","blaziken","rowlet","charmeleon","piplup","bulbasaur","tepig","oshawott","froakie","charmander","squirtle","mudkip","charizard","popplio","snivy","torchic","dragonite","deoxys, deoxys (normal)","groudon","dialga","miraidon","cosmog","regice","azelf","meloetta","mewtwo","regigigas","tapu koko, tapu lele, tapu bulu, tapu fini","hoopa","deoxys, deoxys (speed)","yveltal","xerneas","giratina","moltres, galarian moltres","dialga, primal dialga","zekrom","registeel","unown","kyurem","celebi","articuno, galarian articuno","zygarde, zygarde (complete)","reshiram","lunala","kyogre","moltres","necrozma","tornadus, thundurus, landorus","koraidon","marshadow","lugia, shadow lugia","silvally","giratina, giratina (origin)","lugia","fly (pokemon)","latios","latias","solgaleo","mega pokemon","zamazenta","suicune","metagross","deoxys, deoxys (attack)","diancie","rayquaza","zacian","deoxys, deoxys (defense)","surf (pokemon)","zapdos","calyrex","mew","zygarde, zygarde (10%)","pikachu","arceus","cosmoem","palkia","ho-oh","articuno","shaymin","mesprit","darkrai","cresselia","zapdos, galarian zapdos","uxie"],
        'pokemon theme': ["pokemon (game)","poke ball theme","pokemon battle"],
        'bird': ["crane (animal)","heron","parrot","condor","flamingo","sparrow","finch","owl","falcon","peacock","cockatiel","bluebird","swan","pigeon","duck","chicken","crow","hawk","blue jay","swallow","toucan","pelican","dodo (bird)","eagle","cockatoo","japanese white-eye","penguin","phoenix","stork","chick","kiwi","hummingbird","goose","ostrich","magpie","parakeet"],
        'cat': ["tiger","black cat","black panther","brown cat","pink cat","lion","purple cat","orange cat","white cat","grey cat","tortoiseshell cat","calico"],
        'dog': ["puppy","pack of dogs","guide dog","wolf"],
        'aquatic': ["clam","starfish","coral","squid","flying whale","whale","angelfish","shark","saury","tuna","swordfish","manta ray","anglerfish","butterflyfish","fish","moorish_idol","clownfish","carp","catfish"],
        'terrestrial': ["alpaca","guinea pig","boar","monkey","echidna","bear","fox","lamb","gorilla","goat","chipmunk","ferret","platypus","giraffe","mouse","capybara","bull","cow","deer","panda","horse","pig","red fox","rabbit","hamster","hedgehog","piglet","weasel","squirrel","sheep"]
    }
    food = {
        'fruit': ["chinese lantern (plant)","gooseberry","pear","cherry","lemon","lime","fruit cup","pineapple","fruit bowl","banana","pomegranate","papaya","berry","kiwi","cacao fruit","peach","yuzu (fruit)","grapes","mango","grapefruit","melon","orange","persimmon","olive","apple","avocado"],
        'vege': ["jack-o'-lantern","tomato","garlic","bitter melon","coconut","lettuce","broccoli","mint","seaweed","bell pepper","chili pepper","peanut","mushroom","turnip","cucumber","cherry tomato","onion","pepper shaker","cabbage","corn","pumpkin","sweet potato","radish","eggplant","chestnut","carrot","beans","roasted sweet potato","potato"],
        'protein': ["bacon","pork","sausage","egg yolk","lobster","egg laying","egg (food)","fried chicken","softboiled egg","hardboiled egg","eggshell","cooked crab","sashimi","steak","omurice","holding egg","chicken leg","hamburger steak","yakitori","turkey (food)","chicken nuggets","ham","gyuudon","boned meat","scrambled egg","shrimp tempura","meatball","tako-san wiener","omelet","shrimp","fried egg","kebab","kamaboko","tamagoyaki","roe","easter egg"],
        'starch': ["cereal","onigiri","pancake","baguette","bread eating race","bread crust","fried rice","curry rice","rice porridge","japari bun","senbei","croissant","melon bread","toast","waffle","udon","rice paddy","pasta","rice on face","soba","ramen","scone","bread bun","biscuit","cracker","spaghetti","empanada","omurice"],
        'dessert': ["chewing gum","churro","nerunerunerune","opera cake","stollen","brownie","mooncake","chitose ame","cream","yule log","creme egg","birthday cake","imagawayaki","m&m's","marble chocolate","pudding","chocolate chip cookie","chocolate fountain","mochi","thumbprint cookie","white chocolate","pastry box","doughnut","kitkat","cake slice","charlotte cake","cinnamon roll","chocolate syrup","cotton candy","pound cake","christmas cake","muffin","anmitsu","gelatin","taiyaki","marshmallow","chocolate bar","wedding cake","sandwich cookie","takenoko no sato","crepe","tart","dough","mont blanc","red velvet cake","warabimochi","toppo","tanghulu","tootsweets","uirou (food)","cheesecake","cigarette candy","candy apple","baumkuchen","pocky","pastry","strawberry shortcake","batter","checkerboard cookie","pinata","madeleine","jelly bean","black forest cake","chocolate cake","wafer","cupcake","tiramisu","baozi","youkan (food)","kinoko no yama","shaved ice","lollipop","popsicle","chocolate marquise","gingerbread cookie","dorayaki","ice cream","fondant au chocolat","swiss roll","caramel","apollo chocolate","layer cake","mille-feuille","konpeitou","candy cane","heart-shaped chocolate","pie"],
        'meal': ["gunkanmaki","tofu","tamagokake gohan","jiaozi","zongzi","curry rice","twice cooked pork","curry","pizza delivery","stinky tofu","lunch","fish and chips","party","takuan","fondue","pizza slice","sushi","tempura","salt","mapo tofu","konnyaku","bento","shumai","dim sum","holding pizza","ribs (food)","sandwich","katsu","canned food","feast","burger","hot dog","narutomaki","dumpling","inarizushi","corn dog","sukiyaki","makizushi","shrimp tempura","croquette","nigirizushi","zouni soup","nabe","tang yuan","birthday party","oden","salt shaker","omelet","unadon (food)","flour","crumbs","french fries","soup","tea party","pizza box","miso soup","conveyor belt sushi","okosama lunch","sushi geta","katsudon","breakfast","salad","pizza","meal","baozi","dinner","okonomiyaki","megamac","takoyaki","taco","aburaage","cooking oil","french toast"]
    }
    scene = {
        'indoor': ["bedroom","dungeon","stage","dressing room","pool","living room","office","gym","library","storage room","kitchen","staff room","infirmary","conservatory","changing room","courtroom","classroom","laboratory","bathroom","workshop","cafeteria","fitting room","dining room"],
        'nature': ["garden of the sun","forest of magic","mountain","seascape","ocean bottom","island","parking lot","water","cliff","cave","desert","jungle","pond","nature","river","glacier","beach","ocean","stream","hill","canyon","park","lake","wetland","plain","waterfall","forest","meadow"],
        'man-made': ["phone booth","bridge","graveyard","highway","amusement park","city","canal","pier","jetty","railroad tracks","tunnel","market","running track","dam","stone walkway","zoo","harbor","dock","aqueduct","fountain","street","trench","well","field","soccer field","dirt road","sidewalk","pool","garden","alley","path","crosswalk"],
        'building': ["hakugyokurou","moriya shrine","makai (touhou)","shining needle castle","voile","palanquin ship","hakurei shrine","chireiden","scarlet devil mansion","kourindou","architecture","hut","restaurant","onsen","supermarket","ruins","aquarium","convention","office","skating rink","apartment","temple","military base","planetarium","industrial","mall","bowling alley","hospital","windmill","prison","barn","gazebo","airport","hotel","observatory","tower","garage","gas station","megastructure","mosque","bookstore","skyscraper","bar","rooftop","bunker","train station","arcade","museum","shack","gym","house","castle","bakery","library","nightclub","shop","flower shop","stadium","theater","treehouse","bus stop","pagoda","greenhouse","school","lighthouse","convenience store","shrine","construction site","cafe","casino","church"],
        'interior': ["cockpit","train interior","bus interior","spacecraft interior","vehicle interior","airplane interior","car interior","tank interior"],
        'space': ["space","space station","planet","asteroid","moon"],
        'sky': ["gradient sky","cloudy sky","cloud","sunset","orange sky","rainbow, cloud","horizon","day","twilight","sky, light rays","blue sky","evening","sunlight","night sky","aurora","stars (sky)","outdoors","sun rise","sun","night, moonlight"],
        'weather': ["rain", "storm, dark clouds", "rain, storm, lightning", "fog"],
        'country': ["mexico","chile","france","japan","united states","korea, south korea","malaysia","greece","netherlands","korea, north korea","russia","turkey","brazil","algeria","poland","china","canada","spain","philippines","ukraine","australia","portugal","indonesia","finland","germany","india","united kingdom","italy","israel","egypt","taiwan","vietnam","thailand","afghanistan","argentina","austria","sweden","hungary"],
        'landmarks': ["st basil's cathedral","mount fuji","statue of liberty","times square","eiffel tower","fushimi inari taisha","tokyo big sight","elizabeth tower","moai","tokyo sky tree","empire state building","tokyo city hall","tokyo tower","golden gate bridge","colosseum"],
        'restaurant': ["krispy kreme","anna miller's","matsuya foods","dairy queen","pizza hut","coco ichibanya","kfc","cold stone creamery","wendy's","lotteria","hooters","domino's pizza","papa john's","mister donut","starbucks","mcdonald's","black star burger","burger king","little caesar's","baskin-robbins","coco's","dunkin' donuts","pizza-la","subway","jollibee","sukiya","mos burger","yoshinoya","chick-fil-a"],
        'retail': ["isetan","sofmap","marui","familymart","walmart","7-eleven","lawson","mitsukoshi","amazon","circle k sunkus","fedex","mastercard","ikea"],
    }
    season = {
        'setting': ["winter","autumn leaves","falling leaves","pink flower","snow","autumn","spring (season)","snow, cold","summer","maple leaf","snowing","cherry blossoms"]
    }
    if not 'highres' in tags:
        tags += ', highres'
    tags = generate_tags('neutral', tags, neutral)
    isGirl = 'girl' in tags or 'female' in tags
    isBoy = 'boy' in tags or '1male' in tags or ' male' in tags
    tags = generate_tags('tech', tags, tech)
    tags = generate_tags('pet', tags, pet)
    tags = generate_tags('food', tags, food)
    tags = generate_tags('scene', tags, scene)
    tags = generate_tags('season', tags, season)
    tags = subgroup_tags(tags, neutral)
    if isGirl:
        tags = generate_tags('female', tags, female)
        tags = subgroup_tags(tags, female)
    elif isBoy:
        tags = generate_tags('male', tags, male)
        tags = subgroup_tags(tags, male)
    tags = subgroup_tags(tags, tech)
    tags = subgroup_tags(tags, pet)
    tags = subgroup_tags(tags, food)
    tags = subgroup_tags(tags, scene)
    tags = subgroup_tags(tags, season)
    return tags


def random_anime_tags() -> list[str]:
    gnames = ['1boy, ((slit pupils)), (white pupil:-99), medium hair, <eye color>, bishounen, <hair color>, overcoat, light smile, blue sky, [season], from behind',
              '1boy, ((slit pupils)), (white pupil:-99), <eye color>, <hair color>, short hair, bishounen, <hat>, <man-made>, sunset, [season], sideways',
              '1boy, ((diamond-shaped pupils)), (white pupil:-99), medium hair, bishounen, <eye color>, <hair color>, t-shirt, national basketball association, [season]',
              '1boy, ((diamond-shaped pupils)), <eye color>, <hair color>, short hair, bishounen, <headphones>, <car brand>, [season]',
              '1girl, ((slit pupils)), (white pupil:-99), <eye color>, <hair style>, <hair color>, [neutral], cloudy sky, <nature>, from behind',
              '1girl, ((slit pupils)), <eye color>, <hair style>, <hair color>, [neutral], [female], sunset, [season], sideways',
              '1girl, ((heart-shaped pupils)), (white pupil:-99), <eye color>, <hair style>, <hair color>, sunrise, [female], [season], <pov>',
              '1girl, ((bunny-shaped pupils)), <eye color>, <hair style>, <hair color>, chef, [female], kitchen, <pov>'
              '<bird>',
              '[food]']
    return gnames