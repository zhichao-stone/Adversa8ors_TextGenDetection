import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


import re
import numpy as np
from typing import List, Union


class SingleMPU:
    def __init__(self, model_path: str, device: torch.device, max_len: int = 512) -> None:
        self.device = device
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, batch: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.max_len
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        return 1 - probs[:, 1].detach().cpu().numpy()

    
class MPUDetector:
    def __init__(self, 
        en_model_path: str, 
        zh_model_path: str = None,
        max_len: int = 512
    ) -> None:
        self.max_len = max_len

        device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
        device2 = "cuda:1" if torch.cuda.device_count() > 1 else device1

        if en_model_path:
            self.en_model = SingleMPU(en_model_path, device1, self.max_len)
        if zh_model_path:
            self.zh_model = SingleMPU(zh_model_path, device2, self.max_len)

    def compute_score(self, text: Union[str, List[str]]) -> List[float]:
        batch = [text] if isinstance(text, str) else text
        
        en_indices, zh_indices = [], []
        en_batches, zh_batches = [], []
        for i, t in enumerate(batch):
            zh_chars = re.findall(r"[\u4e00-\u9fff！，。；“”？、]", t)
            if len(zh_chars) / len(t) >= 0.35:
                zh_indices.append(i)
                zh_batches.append(t)
            else:
                en_indices.append(i)
                en_indices.append(t)
        
        scores = np.zeros(len(batch))
        if en_batches:
            en_scores = self.en_model.predict(en_batches)
            scores[en_indices] = en_scores
        if zh_batches:
            zh_scores = self.zh_model.predict(zh_batches)
            scores[zh_indices] = zh_scores

        scores = scores.tolist()
        return scores