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

from transformers import BertForSequenceClassification, BertTokenizer

from discord_bot.data.seq_dataset import parse_date
from discord_bot.models.speaker_predict import NextSpeakerModel

intents = discord.Intents.default()
intents.members = True  # Enable GUILD_MEMBERS intent
intents.presences = True  # Enable PRESENCE_UPDATE intent (if needed)
intents.message_content = True  # Enable MESSAGE_CONTENT intent (if needed)

TOKEN = os.getenv('DISCORD_BOT_TOKEN')
bot = commands.Bot(command_prefix='!', intents=intents)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

bert_path = "/home/gos/Desktop/discord_bot/outputs/discord_bert/20253201843"
bert_model = BertForSequenceClassification.from_pretrained(
    bert_path,
    output_hidden_states=True  # hidden_states 반환 활성화
).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model.eval()

model_path = '/home/gos/Desktop/discord_bot/outputs/lstm_speaker/2025321545/best_model.pth'
model = NextSpeakerModel(768, 256, 98, 2).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 데이터 로드
with open('/home/gos/Desktop/discord_bot/data/db/db_dump_일반.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['일반']

data.sort(key=lambda x: datetime.strptime(x['time'][:19], "%Y-%m-%d %H:%M:%S"))

all_user_ids = sorted(list(set(item['user_id'] for item in data)))
user2idx = {uid: i for i, uid in enumerate(all_user_ids)}
idx2user = {v: k for k, v in user2idx.items()}
num_users = len(user2idx)

speaker_id = 0
hit, miss = 0, 0

lstm_hidden = [hidden.to(device) for hidden in model.init_hidden(1)]
speaker_eye = torch.eye(len(user2idx), device=device)
prev_date = None

@bot.event
async def on_ready():
    global lstm_hidden
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    guild = next((guild for guild in bot.guilds if guild.name == target_server), None)

    prev_date = None
    for channel in guild.text_channels[:1]:
        try:
            start = time.time()
            async for message in channel.history(limit=64):
                content = message.content

                encoding = tokenizer(content,
                                    add_special_tokens=True,
                                    truncation=True,
                                    max_length=128,
                                    padding='max_length',
                                    return_tensors='pt').to(device)
                
                with torch.no_grad():
                    outputs = bert_model(encoding['input_ids'], encoding['attention_mask'])
                    hidden_states = outputs.hidden_states  # 튜플 형태로 각 레이어의 hidden state 반환
                    
                    # 마지막 레이어의 [CLS] 토큰 임베딩 추출 (shape: [batch_size, hidden_size])
                    content_embedding = hidden_states[-1][:, 0, :].reshape(1, 1, -1)
                    
                    cur_date = message.created_at.timestamp()/3600
                    if prev_date == None:
                        diff = 0  
                    else:
                        diff = cur_date - prev_date
                    prev_date = cur_date
                    diff = torch.tensor(diff).to(device).reshape(1, 1, 1).float()
                    speaker = speaker_eye[user2idx[message.author.id]]
                    print(user2idx, )
                    _, _, lstm_hidden = model.inference(content_embedding, diff, speaker, lstm_hidden)
        except discord.Forbidden:
            print(f'Cannot access channel: {channel.name}')
            
        print("done with", channel.name)
    

target_server = "난투할사람"
@bot.event
async def on_message(message):
    global prev_date, lstm_hidden, hit, miss, speaker_id, speaker_eye
    if message.author == bot.user:
        return

    if message.guild.name != target_server:
        return

    if message.channel.name != "일반":
        return
    
    if message.author.id == speaker_id:
        print("hit!")
        hit+=1
    else:
        print("miss..")
        miss+=1
        
    content = message.content

    encoding = tokenizer(content,
                         add_special_tokens=True,
                         truncation=True,
                         max_length=128,
                         padding='max_length',
                         return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = bert_model(encoding['input_ids'], encoding['attention_mask'])
        hidden_states = outputs.hidden_states  # 튜플 형태로 각 레이어의 hidden state 반환
        
        # 마지막 레이어의 [CLS] 토큰 임베딩 추출 (shape: [batch_size, hidden_size])
        content_embedding = hidden_states[-1][:, 0, :].reshape(1, 1, -1)
        print(content_embedding.shape)
        
        cur_date = message.created_at.timestamp()/3600
        if prev_date == None:
            diff = 0  
        else:
            diff = cur_date - prev_date
        prev_date = cur_date
        diff = torch.tensor(diff).to(device).reshape(1, 1, 1).float()
        speaker = speaker_eye[user2idx[message.author.id]]
        logits, pred_timediff, lstm_hidden = model.inference(content_embedding, diff, speaker, lstm_hidden)
        pred = logits.argmax(dim=-1)
        
    speaker_id = idx2user[pred[0].item()]
    speaker = await bot.fetch_user(speaker_id)
    speaker_name = speaker.name
    print(f'Pred speaker: {speaker_name}')
    del encoding, outputs
    
# Replace 'YOUR_BOT_TOKEN' with your actual bot token
try:
    bot.run(TOKEN)
finally:
    print(f"{hit}/{miss} {hit/(hit+miss)*100}% hit")