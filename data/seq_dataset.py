import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import numpy as np
from tqdm import tqdm

def parse_date(date_str):
    return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")

class ConversationDataset(Dataset):
    def __init__(self, data, message_embedding_dict, user2idx, sequence_length=5):
        self.data = data
        self.msg_emb_dict = message_embedding_dict
        self.user2idx = user2idx
        self.seq_len = sequence_length
        self.samples = []
        timestamps = [parse_date(item['time']).timestamp()/(3600) for item in self.data] # make into hours
        
        # for i in range(100):
        #     print(self.data[i]['user_id'], self.data[i]['content'])
        
        for i in tqdm(range(1, len(data) - sequence_length), desc="Preprocessing.."):
            input_indices = list(range(i, i + sequence_length))
            target_index = i + sequence_length
            
            input_msg_ids = [self.data[idx]['message_id'] for idx in input_indices]
            input_embs = [self.msg_emb_dict[m_id] for m_id in input_msg_ids]
            input_timestamps = [timestamps[idx] for idx in input_indices]
            input_timediff = [timestamps[idx] - timestamps[idx-1] for idx in input_indices]# Gather the user indices in a list first
            user_indices = [self.user2idx[self.data[idx]['user_id']] for idx in input_indices]
            input_speaker = np.zeros((len(input_indices), len(user2idx)), dtype=np.int32)
            input_speaker[np.arange(len(input_indices)), user_indices] = 1
    

            target_user_id = self.data[target_index]['user_id']
            target_user_idx = user2idx[target_user_id]
            target_timediff = timestamps[target_index] - timestamps[target_index-1]

            self.samples.append({
                'input_embs': input_embs,
                'timestamps': input_timestamps,
                'timediff': input_timediff,
                'speaker': input_speaker,
                'target_user_idx': target_user_idx,
                'target_timediff': target_timediff
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_embs = torch.stack(sample['input_embs'], dim=0)
        input_timestamps = torch.tensor(sample['timestamps'], dtype=torch.float)
        input_timediff = torch.tensor(sample['timediff'], dtype=torch.float)
        input_speaker = torch.tensor(sample['speaker'], dtype=torch.float)
        target_user_idx = torch.tensor(sample['target_user_idx'], dtype=torch.long)
        target_timediff = torch.tensor(sample['target_timediff'], dtype=torch.float)
        
        return input_embs, input_timestamps, input_timediff, input_speaker, target_user_idx, target_timediff

def collate_fn(batch):
    input_embs = torch.stack([item[0] for item in batch], dim=0)
    timestamps = torch.stack([item[1] for item in batch], dim=0)
    timediff = torch.stack([item[2] for item in batch], dim=0)
    speaker = torch.stack([item[3] for item in batch], dim=0)
    targets = torch.stack([item[4] for item in batch], dim=0)
    target_timediff = torch.stack([item[5] for item in batch], dim=0)
    return input_embs, timestamps, timediff, speaker, targets, target_timediff
