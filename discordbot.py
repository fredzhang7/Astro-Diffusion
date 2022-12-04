import discord
from astro_image import return_image_gen
from util import anime_search
import io, os
from dotenv import load_dotenv
load_dotenv()

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'We have logged in as {client.user}')

    async def on_message(self, message):
        if message.author == client.user:
            return
        if message.content.startswith('draw '):
            prompt = message.content[5:]
            if len(prompt) < 80:
                prompt = anime_search(prompt)
            embed = discord.Embed(title="Prompt", description=prompt, color=0x00ff00)
            await message.channel.send(embed=embed)
            image = return_image_gen([prompt])
            with io.BytesIO() as image_binary:
                image.save(image_binary, 'PNG')
                image_binary.seek(0)
                await message.channel.send(file=discord.File(fp=image_binary, filename='image.png'))
 
intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
activity = discord.Activity(name="draw <prompt>", type=discord.ActivityType.watching)
client.activity = activity
client.run(os.getenv('DISCORD_TOKEN'))