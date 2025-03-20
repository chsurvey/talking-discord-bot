import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime
today = datetime.now()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device='cpu', epochs=3):
    model.to(device)
    best_val_acc = 0.0
    mse = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        
        for input_embs, timestamps, timediff, input_speaker, targets, target_timediff in train_pbar:
            input_embs, timestamps, timediff, input_speaker, targets, target_timediff = input_embs.to(device), timestamps.to(device), timediff.to(device), \
                                                                       input_speaker.to(device), targets.to(device), target_timediff.to(device)
            optimizer.zero_grad()

            batch_size, seq_len = input_embs.shape[:2]
            logits, pred_timediff = model(input_embs, timediff, input_speaker)
            
            loss = criterion(logits, targets) + mse(pred_timediff.reshape(batch_size), target_timediff)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            train_pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}, Train Acc = {train_acc:.4f}")
        
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = f"./outputs/lstm_speaker/{today.year}{today.month}{today.day}{today.hour}{today.minute}/"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_dir+"best_model.pth")
            # torch.save(model.state_dict(), "/home/gos/Desktop/discord_bot/outputs/lstm_speaker/best_model.pth")

def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        
        for input_embs, timestamps, timediff, input_speaker, targets, target_timediff in val_pbar:
            input_embs, timestamps, timediff, input_speaker, targets, target_timediff = input_embs.to(device), timestamps.to(device), timediff.to(device), \
                                                                       input_speaker.to(device), targets.to(device), target_timediff.to(device)

            logits, pred_timediff = model(input_embs, timediff, input_speaker)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    return correct / total
