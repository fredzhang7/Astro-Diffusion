import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("MAIN_BOT_TOKEN") is None:
    raise ValueError("MAIN_BOT_TOKEN is not set")

import art_generation.discordbot