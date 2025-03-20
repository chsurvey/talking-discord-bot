import os
import json
import re
import html
import torch
from transformers import BertTokenizer
from tqdm import tqdm

# ------------------------------------------------
# 1. Text Preprocessing Function
# ------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Example text preprocessing pipeline.
    Modify as needed.
    """
    # 1. Decode HTML entities
    text = html.unescape(text)
    
    # 2. Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # 4. Remove punctuation/special chars (keep letters, digits, underscore, whitespace)
    text = re.sub(r"[^\w\s]", "", text)
    
    # 5. Lowercase
    text = text.lower()
    
    # 6. Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_and_save(json_file: str,
                        output_file: str,
                        tokenizer_name: str = "bert-base-uncased",
                        max_length: int = 128):
    """
    Reads JSON data, preprocesses and tokenizes, then saves to disk.
    
    :param json_file: Path to the input JSON file.
    :param output_file: Path to save the processed tensors (e.g., 'preprocessed_data.pt').
    :param tokenizer_name: Pretrained tokenizer to use (e.g., 'bert-base-uncased').
    :param max_length: Max sequence length for truncation/padding.
    """
    
    # Load JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Assuming your file structure is { "일반": [ { "user_id":..., "time":..., "content":..., "message_id":...}, ... ] }
    data = data["일반"]
    
    # Build a mapping from user_id to a label integer
    message_ids = list({item['message_id'] for item in data})
    user_ids = list({item['user_id'] for item in data})
    user2label = {uid: idx for idx, uid in enumerate(user_ids)}
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Lists to store the processed samples
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    for item in tqdm(data):
        raw_text = item["content"]
        # Preprocess text
        text = preprocess_text(raw_text)
        
        # Tokenize
        encoding = tokenizer(text,
                             add_special_tokens=True,
                             truncation=True,
                             max_length=max_length,
                             padding='max_length')
        
        # Convert to PyTorch tensors
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        label = torch.tensor(user2label[item['user_id']], dtype=torch.long)
        
        # Append to lists
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(label)
    
    # Stack all samples into tensors
    all_input_ids = torch.stack(all_input_ids)       # Shape: [dataset_size, max_length]
    all_attention_masks = torch.stack(all_attention_masks)
    all_labels = torch.stack(all_labels)
    
    # Save everything to disk
    torch.save({
        "user_ids": torch.tensor(user_ids),
        "message_ids": torch.tensor(message_ids),
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
        "user2label": user2label,
    }, output_file)
    
    print(f"Saved preprocessed data to {output_file}")


if __name__ == "__main__":
    json_file = "db_dump_일반.json"          # Update with your file path
    output_file = "preprocessed_data.pt"    # Desired output path
    preprocess_and_save(json_file, output_file)
