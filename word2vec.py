import json
import re
import torch
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from torch import nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# SnowballStemmer 및 영어 불용어 세트를 초기화
ss = SnowballStemmer('english')
sw = set(stopwords.words('english'))


# JSON 파일을 읽고 데이터를 전처리하여 토큰화된 뉴스 기사 리스트를 반환
def load_preprocessed_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        news_voca = []
        for obj in data:
            news_article = obj['news_article']
            for sentence in news_article:
                tokens = split_tokens(sentence)
                news_voca.extend(tokens)
    return news_voca


# 문장을 전처리하여 토큰화된 단어 리스트를 반환
def split_tokens(sentence):
    all_tokens = [ss.stem(token) for token in re.findall(r'\b[a-zA-Z@#]+\b', sentence.lower()) if
                  token not in sw and len(token) > 1]
    return all_tokens


# 데이터를 로드하고 전처리하는 함수를 호출하여 토큰화
news_voca = load_preprocessed_data('news_data.json')

# 단어 빈도 계산
counts = Counter(news_voca)
counts = {k: v for k, v in counts.items() if v > 10}
vocab = list(counts.keys())
n_v = len(vocab)
id2tok = dict(enumerate(vocab))
tok2id = {token: id for id, token in id2tok.items()}


# 빈도가 낮은 토큰을 제거
def remove_rare_tokens(row):
    row = [t for t in row if t in vocab]
    return row


dataset = remove_rare_tokens(news_voca)


# 윈도우 생성
def windowizer(row, wsize=3):
    doc = row
    out = []
    for i, wd in enumerate(doc):
        target = tok2id[wd]
        window = [i + j for j in range(-wsize, wsize + 1)
                  if (i + j >= 0) & (i + j < len(doc)) & (j != 0)]
        out += [(target, tok2id[doc[w]]) for w in window]
    return out


window = windowizer(dataset, wsize=1)


# Word2Vec 데이터셋 생성
class Word2VecDataset(Dataset):
    def __init__(self, vocab_size, window):
        self.vocab_size = vocab_size
        self.data = window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 데이터 로더 초기화
BATCH_SIZE = 2 ** 14
N_LOADER_PROCS = 10
dataloader = DataLoader(
    Word2VecDataset(vocab_size=n_v, window=window),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_LOADER_PROCS
)


# Word2Vec 모델 클래스
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input):
        hidden = self.embed(input)
        logits = self.expand(hidden)
        return logits


from scipy.spatial import distance
import numpy as np


def get_distance_matrix(wordvecs, metric):
    dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
    return dist_matrix


def get_k_similar_words(word, dist_matrix, k=10):
    idx = tok2id[word]
    dists = dist_matrix[idx]
    ind = np.argpartition(dists, k)[:k + 1]
    ind = ind[np.argsort(dists[ind])][1:]
    out = [(i, id2tok[i], dists[i]) for i in ind]
    return out


# 모델 초기화
EMBED_SIZE = 100
model = Word2Vec(n_v, EMBED_SIZE)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 훈련 파라미터
LR = 3e-4
EPOCHS = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

if __name__ == '__main__':
    progress_bar = tqdm(range(EPOCHS * len(dataloader)))
    running_loss = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            optimizer.zero_grad()
            logits = model(input=context)
            loss = loss_fn(logits, center)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
        epoch_loss /= len(dataloader)
        running_loss.append(epoch_loss)

    print("\n")
    # 학습된 임베딩
    wordvecs = model.expand.weight.cpu().detach().numpy()
    tokens = ['good', 'father', 'school', 'hate']

    dmat = get_distance_matrix(wordvecs, 'cosine')
    for word in tokens:
        print(word, [t[1] for t in get_k_similar_words(word, dmat)], "\n")
