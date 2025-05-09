from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    temperature: float = 1.0
) -> np.ndarray:
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ppl: torch.Tensor = (F.cross_entropy(shifted_logits.transpose(1, 2), shifted_labels, reduction="none") * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)

    return ppl.cpu().float().numpy()


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    temperature: float = 1.0
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_prob = F.softmax(p_scores, dim=-1).view(-1, vocab_size)
    q_scores = q_scores.view(-1, vocab_size)
    ce = F.cross_entropy(input=q_scores, target=p_prob, reduction="none").view(-1, total_tokens_available)

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    agg_ce: torch.Tensor = (ce * padding_mask).sum(1) / padding_mask.sum(1)

    return agg_ce.cpu().float().numpy()


class LLMClient:
    def __init__(self, 
        model_path: str, 
        device: torch.device, 
        max_len: int = 512
    ) -> None:
        self.device = device
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map={"": self.device}, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def get_logits(self, batch: List[str]) -> torch.Tensor:
        encodings = self.tokenizer(
            batch, 
            return_tensors="pt", 
            padding="longest" if len(batch) > 1 else False, 
            truncation=True, 
            max_length=self.max_len
        ).to(self.device)
        logits = self.model(**encodings).logits
        return logits

class Binoculars:
    def __init__(self, 
        observer_path: str = "tiiuae/falcon-7b", 
        performer_path: str = "tiiuae/falcon-7b-instruct", 
        max_len: int = 512, 
        threshold: float = 0.85364323
    ) -> None:
        self.threshold = threshold
        self.max_len = max_len

        self.device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device2 = "cuda:1" if torch.cuda.device_count() > 1 else self.device1

        self.tokenizer = AutoTokenizer.from_pretrained(observer_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_path, 
            device_map={"": self.device1}, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_path, 
            device_map={"": self.device2}, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.observer.eval()
        self.performer.eval()

    def adjust_scores(self, scores: np.ndarray) -> List[float]:
        scores = scores - self.threshold
        scores: np.ndarray = (scores / self.threshold + 1)*0.5
        scores = np.nan_to_num(scores, nan=0, posinf=1, neginf=0)
        scores = np.clip(scores, 0, 1)
        return scores.tolist()

    def compute_score(self, text: Union[str, List[str]]) -> List[float]:
        batch = [text] if isinstance(text, str) else text

        encodings = self.tokenizer(
            batch, 
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_len,
        )

        observer_logits: torch.Tensor = self.observer(**encodings.to(self.device1)).logits
        performer_logits: torch.Tensor = self.performer(**encodings.to(self.device2)).logits
        if self.device1 != "cpu":
            torch.cuda.synchronize()

        ppl = perplexity(encodings, performer_logits)
        xppl = entropy(observer_logits.to(self.device1), performer_logits.to(self.device1), encodings.to(self.device1), self.tokenizer.pad_token_id)

        scores: np.ndarray = ppl / (xppl + 1e-4)
        scores = self.adjust_scores(scores)
        return scores