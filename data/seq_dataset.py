import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
import pickle
import random

def parse_date(date_str):
    return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")

class ConversationDataset(Dataset):
    def __init__(self, data, message_embedding_dict, user2idx, sequence_length=5):
        self.data = data
        self.msg_emb_dict = message_embedding_dict
        self.user2idx = user2idx
        self.seq_len = sequence_length
        self.samples = []
        timestamps = [parse_date(item['time']).timestamp() for item in self.data]

        for i in range(len(data) - sequence_length):
            input_indices = list(range(i, i + sequence_length))
            target_index = i + sequence_length
            
            input_msg_ids = [self.data[idx]['message_id'] for idx in input_indices]
            input_embs = [self.msg_emb_dict[m_id] for m_id in input_msg_ids]
            input_timestamps = [timestamps[idx] for idx in input_indices]

            target_user_id = self.data[target_index]['user_id']
            target_user_idx = user2idx[target_user_id]

            self.samples.append({
                'input_embs': input_embs,
                'timestamps': input_timestamps,
                'target_user_idx': target_user_idx
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_embs = torch.stack(sample['input_embs'], dim=0)
        input_timestamps = torch.tensor(sample['timestamps'], dtype=torch.float)
        target_user_idx = torch.tensor(sample['target_user_idx'], dtype=torch.long)
        
        return input_embs, input_timestamps, target_user_idx

def collate_fn(batch):
    input_embs = torch.stack([item[0] for item in batch], dim=0)
    timestamps = torch.stack([item[1] for item in batch], dim=0)
    targets = torch.stack([item[2] for item in batch], dim=0)
    return input_embs, timestamps, targets
