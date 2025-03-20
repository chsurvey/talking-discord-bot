import discord
from discord.ext import commands
from collections import defaultdict
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import pytz
import time
import os

intents = discord.Intents.default()
intents.members = True  # Enable GUILD_MEMBERS intent
intents.presences = True  # Enable PRESENCE_UPDATE intent (if needed)
intents.message_content = True  # Enable MESSAGE_CONTENT intent (if needed)

TOKEN = os.getenv('DISCORD_BOT_TOKEN')
bot = commands.Bot(command_prefix='!', intents=intents)

target_server = "난투할사람"

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    
    guild = next((guild for guild in bot.guilds if guild.name == target_server), None)
    # id2name = {member.id: member.name for member in guild.members}
    # with open('id2name.json', 'w', encoding='utf-8') as f:
    #     json.dump(id2name, f, ensure_ascii=False, indent=4)
    DB = defaultdict(list)
    prev_date = datetime(2025, 3, 19, 12, 0, 0, tzinfo=pytz.UTC) + timedelta(days=365*100)  # Set to a far future date
    for channel in guild.text_channels:
        try:
            start = time.time()
            async for message in channel.history(limit=None):
                cur_date = message.created_at
                
                if (prev_date - cur_date).days >= 7:
                    prev_date = cur_date
                    cur_time = time.time()
                    print(f'Updated date: {cur_date}, Total message: {len(DB[channel.name])}, Took {(cur_time - start) / 60:.2f} minutes')
                
                DB[channel.name].append({
                    "user_id": message.author.id,
                    "time": message.created_at,
                    "content": message.content,
                    "message_id": message.id
                        })
        except discord.Forbidden:
            print(f'Cannot access channel: {channel.name}')
            
        print("done with", channel.name)
    
    with open(f'/home/gos/Desktop/discord_bot/data/db/db_dump.json', 'w', encoding='utf-8') as f:
        json.dump(DB, f, ensure_ascii=False, indent=4, default=str)


# Replace 'YOUR_BOT_TOKEN' with your actual bot token
bot.run(TOKEN)