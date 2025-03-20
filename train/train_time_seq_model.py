import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=3):
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        
        for input_embs, timestamps, targets in train_pbar:
            input_embs, timestamps, targets = input_embs.to(device), timestamps.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(input_embs, timestamps)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            train_pbar.set_postfix(loss=loss.item())
        
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}, Train Acc = {train_acc:.4f}")
        
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Evaluating")
        
        for input_embs, timestamps, targets in val_pbar:
            input_embs, timestamps, targets = input_embs.to(device), timestamps.to(device), targets.to(device)
            logits = model(input_embs, timestamps)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    return correct / total
