import torch
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
import pickle

# 1. 모델과 토크나이저 로드
model_path = "/home/gos/Desktop/discord_bot/models/2025320139"
model = BertForSequenceClassification.from_pretrained(
    model_path,
    output_hidden_states=True  # hidden_states 반환 활성화
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

# GPU 사용 가능 시 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 미리 전처리된 데이터 로드 (preprocessed_data.pt)
data = torch.load("preprocessed_data.pt")
message_ids = data['message_ids']
input_ids = data["input_ids"]             # 텐서 크기: [dataset_size, max_length]
attention_masks = data["attention_mask"]    # 텐서 크기: [dataset_size, max_length]
labels = data["labels"]                     # 각 데이터에 해당하는 라벨

# 3. 배치 처리를 통한 문서 임베딩 생성
batch_size = 32  # 배치 크기 (필요에 따라 조정)
embeddings_list = []
with torch.no_grad():
    for i in tqdm(range(0, len(input_ids), batch_size), desc="Embedding batches"):
        # 배치 단위로 데이터 슬라이싱 및 GPU 전송
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_attention_masks = attention_masks[i:i+batch_size].to(device)
        
        # 모델 실행
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        hidden_states = outputs.hidden_states  # 튜플 형태로 각 레이어의 hidden state 반환
        
        # 마지막 레이어의 [CLS] 토큰 임베딩 추출 (shape: [batch_size, hidden_size])
        cls_embeddings = hidden_states[-1][:, 0, :]
        embeddings_list.append(cls_embeddings.cpu())
        
# 모든 배치의 임베딩을 하나의 텐서로 합치기
embeddings = torch.cat(embeddings_list, dim=0)  # shape: [dataset_size, hidden_size]

# 4. 임베딩과 message_ids를 로컬 파일에 저장

# 저장할 데이터 구조 생성
data_to_save = {
    "message_ids": message_ids,
    "embeddings": embeddings
}

# 데이터를 pickle 파일로 저장
with open("embeddings_data.pkl", "wb") as f:
    pickle.dump(data_to_save, f)
