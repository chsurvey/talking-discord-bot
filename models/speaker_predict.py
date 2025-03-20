import torch
import torch.nn as nn

class NextSpeakerModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_users, num_layers=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, embedding_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.classifier = nn.Linear(hidden_dim, num_users)
        self.predict_timediff = nn.Linear(hidden_dim, 1)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim))
    
    def forward(self, input_embs, timediff):
        batch_size, seq_len, emb_dim = input_embs.shape
        
        t = timediff.reshape(batch_size * seq_len, 1)
        time_embs = self.time_embedding(t)
        time_embs = time_embs.view(batch_size, seq_len, emb_dim)
        
        x = input_embs + time_embs
        
        outputs, (h_n, _) = self.lstm(x)
        
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        pred_timediff = self.predict_timediff(last_hidden) 
        return logits, pred_timediff

    def inference(self, input_embs, timediff, prev_hidden):
        batch_size, seq_len, emb_dim = input_embs.shape
        
        t = timediff.reshape(batch_size * seq_len, 1)
        time_embs = self.time_embedding(t)
        time_embs = time_embs.view(batch_size, seq_len, emb_dim)
        
        x = input_embs + time_embs
        
        outputs, hidden_state = self.lstm(x, prev_hidden)
        
        last_hidden = hidden_state[0][-1]
        logits = self.classifier(last_hidden)
        pred_timediff = self.predict_timediff(last_hidden) 
        return logits, pred_timediff, hidden_state
