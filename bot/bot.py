import discord
from discord.ext import commands
from collections import defaultdict
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import pytz
import time
import os
import torch

from discord_bot.models.speaker_predict import NextSpeakerModel

intents = discord.Intents.default()
intents.members = True  # Enable GUILD_MEMBERS intent
intents.presences = True  # Enable PRESENCE_UPDATE intent (if needed)
intents.message_content = True  # Enable MESSAGE_CONTENT intent (if needed)

TOKEN = os.getenv('DISCORD_BOT_TOKEN')
bot = commands.Bot(command_prefix='!', intents=intents)

target_server = "chess"

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = '/home/gos/Desktop/discord_bot/outputs/lstm_speaker/20253210321/best_model.pth'
model = NextSpeakerModel(768, 256, 77, 2).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.guild.name != target_server:
        return


    # Process the message
    if message.content.startswith('!predict'):
        # Extract the input data from the message
        input_data = message.content[len('!predict '):]
        
        # Convert input data to tensor (assuming input_data is a string of space-separated numbers)
        input_tensor = torch.tensor([float(x) for x in input_data.split()], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Send the prediction result back to the Discord channel
        await message.channel.send(f'Prediction: {prediction.item()}')

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
bot.run(TOKEN)