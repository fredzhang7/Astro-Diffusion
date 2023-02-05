import gradio as gr
from gradio.components import Textbox, Button, CheckboxGroup, Image, Checkbox
import requests, time
from anime_webui.customizations import sections
from copy import deepcopy
import PIL
from io import BytesIO

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

def get_random_image() -> dict[str, str]:
    url = 'https://safebooru.donmai.us/posts/random.json'
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        json = r.json()
    else:
        time.sleep(0.25)
        return get_random_image()

    return { 'url': json['file_url'], 'tags': json['tag_string'], 'up_score': json['up_score'] }

def search_tags(tags) -> dict[str, str]:
    if tags is None or len(tags.replace(' ', '')) == 0:
        return get_random_image()
    
    print(tags)
    if tags.endswith(', '):
        tags = tags[:-2]
    if tags.count(', ') > 1:
        tags = tags.split(', ')
        tags = tags[-2] + '+' + tags[-1]
    elif ', ' in tags:
        tags = tags.replace(', ', '+')
    tags = tags.replace(' ', '_')
    url = f'https://safebooru.donmai.us/posts/random.json?tags={tags}'
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        json = r.json()
    else:
        time.sleep(0.25)
        return search_tags(tags)

    return { 'url': json['file_url'], 'tags': json['tag_string'], 'up_score': json['up_score'] }


import json
with open('./anime_webui/post_counts.json', 'r') as f:
    post_counts = json.load(f)

def number_to_abbrev(num):
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return str(num // 1000) + 'K'
    else:
        return str(num // 1000000) + 'M'

with gr.Blocks() as webui:
    with gr.Tab('Customizations'):
        character = Image(type='pil', label='Modifications')
        prompt = Textbox(type='text', label='Prompt')
        text = ''

        for section in sections:
            with gr.Accordion(' '.join([word.capitalize() for word in section.split('_')]), open=False):
                section_count = post_counts[section]
                section_tags = sections[section]
                for key in section_tags:
                    section_copy = deepcopy(section_tags[key])
                    section_tags[key] = [tag for _, tag in sorted(zip(section_count[key], section_tags[key]), reverse=True)]
                    section_count[key] = sorted(section_count[key], reverse=True)
                    section_tags[key] = [tag + '  (' + number_to_abbrev(count) + ')' for count, tag in zip(section_count[key], section_tags[key])]
                    group = CheckboxGroup(section_tags[key], label=key.capitalize(), type='value')
                    
                    def change_tag(los, section_copy):
                        global text
                        for i in range(len(los)):
                            los[i] = los[i].split('  ')[0]
                        for tag in section_copy:
                            if tag in los and tag not in text:
                                text += tag + ', '
                            elif tag not in los and tag in text:
                                text = text.replace(tag + ', ', '')
                        return text
                    group.change(lambda los, section_copy=section_copy: change_tag(los, section_copy), inputs=[group], outputs=[prompt])
        def change_image(tags):
            return search_tags(tags)['url']
        prompt.change(lambda tags: change_image(tags), inputs=[prompt], outputs=[character])

    with gr.Tab('Search Engine'):
        input = Textbox(lines=1, label='Search Term', type='text', placeholder='Search for a tag or tags')
        note = Checkbox(label='Note: Leave the search term empty to return a random output.', interactive=False)
        image = Image(type='pil', label='Sample Image')
        output = Textbox(type='text', label='Prompt')
        up_score = Textbox(type='text', label='Upvotes')
        button = Button('Search')
        def search(tags):
            result = search_tags(tags)
            image = PIL.Image.open(BytesIO(requests.get(result['url'], headers=headers).content))
            return image, result['tags'], result['up_score']
        button.click(search, inputs=[input], outputs=[image, output, up_score])
