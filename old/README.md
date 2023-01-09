### AI Art Helpers
1. Image, GIF, and video super resolution
   * R-ESRGAN General WDN 4x V3
   * R-ESRGAN 4x+ Anime6B
   * ESRGAN 4x
   * Swin2SR 4x
   * R-ESRGAN 2x+
   * PortraitSR
      * can switch between realism (0.0 uncertainty) and painting (1.0 uncertainty)
2. [DistilGPT2 Stable Diffusion v2](https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2)
   * safe for work
   * finetuned on 2,470,000+ prompts for stable diffusion v1.1 through v1.5
   * 50% faster forwardproporgation, 40% less RAM usage, 25%+ more variations, more fluent than other HuggingFace prompt generators
3. DistilBART-powered animation prompt generator
   * given a name, describes the appearance or story of a pony, video game, fiction, manga, or anime character
4. [] and <> support in anime prompts
   * randomizes what's inside: gestures, postures, scenaries, objects, etc.
5. [Google Safesearch Mini](https://huggingface.co/FredZhang7/google-safesearch-mini)
   * finetuned on 2,220,000+ images classified as safe/unsafe by Google Safesearch, Reddit, and Imgur
   * 92% less RAM usage than stable diffusion safety checker
6. [Anime/Animation WebUI](./art_examples/anime_webui_preview.md)
   * Light and dark modes
   * Safe for work
   * Tab 1 - Customizations
      * fully customize your anime characters and scenes
   * Tab 2 - Search Engine
      * search for random images (including prompts and upvotes) containing give tags
      * leave tags as blank to generate random samples
7. Bilingual
    * Chinese (Simplified and Traditional)
    * English

### Super Res Examples
1. [Video Upscale](https://www.youtube.com/playlist?list=PLCFlAfr2X8n1oFJMEcVTCuq3df5yPcZEf)
2. [GIF Upscale](https://imgur.com/a/IEdJiyY)
3. [ImageSR and PortraitSR](https://imgur.com/a/bfRMEBt)

### Text2Img Examples
1. [Anime Models](./art_examples/astro_anime.md)
2. [Pony Model](./art_examples/astro_pony.md)
3. [Van Gogh Model](./art_examples/astro_van_gogh.md)

### Usage
The use of each file is self-explanatory. Some useful functions in `util.py` are `character_search`, `pony_search`, `chinese_to_english`, `distilgpt2_prompt`, and `parse_anime_prompts`