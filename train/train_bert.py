import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from datetime import datetime

from discord_bot.data.custom_dataset import PreprocessedMessageDataset

# ------------------------------------------------
# 2. Training & Validation Functions with Mixed Precision
# ------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler):
    """
    One training epoch with mixed precision.
    Returns the average loss across the entire epoch.
    """
    model.train()
    total_loss = 0.0
    train_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training", leave=False)
    
    for step, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        # Scales loss, backward pass, and optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        train_iterator.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss


def validate_epoch(model, dataloader, device, epoch):
    """
    One validation epoch with mixed precision.
    Returns the average loss and accuracy across the entire validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    num_samples = 0
    val_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1} - Validation", leave=False)
    
    for step, batch in enumerate(val_iterator):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            num_samples += labels.size(0)
            
            running_avg_loss = total_loss / (step + 1)
            running_acc = correct / num_samples if num_samples > 0 else 0.0
            val_iterator.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "acc": f"{running_acc:.4f}"
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    return avg_loss, accuracy


# ------------------------------------------------
# 3. Main Training Script with Mixed Precision
# ------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load preprocessed data
    data_file = "/home/gos/Desktop/discord_bot/data/preprocessed_data.pt"
    dataset = PreprocessedMessageDataset(data_file)
    
    # Train/Validation Split
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Setup
    num_labels = len(dataset.user2label)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    
    model.to(device)
    
    # Optimizer and Scheduler
    learning_rate = 2e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_epochs = 10
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler()
    
    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, scaler)
        val_loss, val_acc = validate_epoch(model, val_dataloader, device, epoch)
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}\n")
    
    # Saving the Model
    today = datetime.now()
    save_dir = f"./models/{today.year}{today.month}{today.day}{today.hour}{today.minute}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
