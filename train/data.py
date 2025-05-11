import random
import pandas as pd
from typing import List
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import PreTrainedTokenizer

from .augment import multi_scale_augment
from .clean import clean_text



class MPUDataset(Dataset):
    def __init__(self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer: PreTrainedTokenizer, 
        max_seq_len: int = 512, 
        aug_min_len: int = 0,
        aug_ratio: float = 0.2, 
        is_train: bool = False
    ) -> None:
        super(MPUDataset, self).__init__()

        self.texts = texts
        self.labels = labels
        assert len(self.texts) == len(self.labels)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.aug_min_len = aug_min_len
        self.aug_ratio = aug_ratio
        self.is_train = is_train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text, label = self.texts[index], self.labels[index]

        if self.is_train and self.aug_min_len > 0:
            text = multi_scale_augment(text, self.aug_min_len, self.aug_ratio)
        
        output = self.tokenizer(text, padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt")

        return output['input_ids'].squeeze(0), output['attention_mask'].squeeze(0), label


def load_texts_tweep(data_file):
    text_labels = []
    data = pd.read_csv(data_file, sep=";")
    
    for _, row in data.iterrows():
        account_type = row["account.type"]
        if account_type in ["human", "bot"]:
            text = clean_text(row["text"])
            label = 1 if account_type == "human" else 0
            text_labels.append([text, label])
        
    random.shuffle(text_labels)
    texts, labels = [], []
    for text, label in text_labels:
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def load_datasets(
    path: str, 
    tokenizer: PreTrainedTokenizer, 
    batch_size: int, 
    aug_min_len: int = 1, 
    aug_ratio: float = 0.2, 
    max_seq_len: int = 512, 
    is_train: bool = False
):
    texts, labels = load_texts_tweep(path)
    dataset = MPUDataset(texts, labels, tokenizer, max_seq_len, aug_min_len, aug_ratio, is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset))

    return dataloader
    

    