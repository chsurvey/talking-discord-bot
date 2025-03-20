import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MessageDataset(Dataset):
    def __init__(self, json_file, tokenizer_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.data = self.data["일반"]
            
        # 데이터는 각 항목이 "user_id", "time", "content", "message_id" 키를 갖습니다.
        # 고유한 user_id를 모아 숫자 레이블로 매핑합니다.
        self.user_ids = list({item['user_id'] for item in self.data})
        self.user2label = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        
        # 각 데이터 항목에 레이블 추가
        for item in self.data:
            item['label'] = self.user2label[item['user_id']]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        encoding = self.tokenizer(
            content,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # 배치 차원 제거
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = item['label']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# ------------------------------------------------
# 1. PyTorch Dataset class for the Preprocessed Data
# ------------------------------------------------
class PreprocessedMessageDataset(Dataset):
    def __init__(self, data_file):
        """
        :param data_file: Path to the .pt file containing preprocessed data.
        """
        data = torch.load(data_file)
        
        self.input_ids = data["input_ids"]           # shape: [N, max_length]
        self.attention_mask = data["attention_mask"]   # shape: [N, max_length]
        self.labels = data["labels"]                   # shape: [N]
        self.user2label = data["user2label"]
        
    def __len__(self):
        return self.input_ids.size(0)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }