import json
import datetime
import pickle

import torch
from torch.utils.data import DataLoader, random_split

from data.seq_dataset import ConversationDataset, collate_fn
from models.speaker_predict import NextSpeakerModel
from train.train_time_seq_model import train_model


# 데이터 로드
with open('data/db/db_dump_일반.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['일반']

data.sort(key=lambda x: datetime.strptime(x['time'][:19], "%Y-%m-%d %H:%M:%S"))

all_user_ids = list(set(item['user_id'] for item in data))
user2idx = {uid: i for i, uid in enumerate(all_user_ids)}
idx2user = {v: k for k, v in user2idx.items()}
num_users = len(user2idx)

with open("embeddings_data.pkl", "rb") as f:
    message_embedding_dict = pickle.load(f)

embedding_dim = 768
for item in data:
    mid = item['message_id']
    if mid not in message_embedding_dict:
        message_embedding_dict[mid] = torch.randn(embedding_dim)

sequence_length = 30
full_dataset = ConversationDataset(data, message_embedding_dict, user2idx, sequence_length=sequence_length)

# Train/Validation Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 모델 초기화
hidden_dim = 256
num_layers = 2
model = NextSpeakerModel(embedding_dim, hidden_dim, num_users, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=3)

print("Training Complete. Best model saved as 'best_model.pth'.")
