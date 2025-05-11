import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mpuloss import MPULoss



def cal_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    if list(logits.shape) == list(labels.shape) + [2]:
        preds = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        preds = (logits > 0).long().flatten()

    acc = (preds == labels).sum().cpu().item()

    return acc


def train(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    loader: DataLoader, 
    len_threshold: int, 
    mpu_loss_fn: MPULoss = None, 
    device: torch.device = torch.device("cuda")
):
    model.train()

    train_acc, train_loss = 0.0, 0.0
    total = 0
    for texts, masks, labels in loader:
        texts: torch.Tensor = texts.to(device)
        masks: torch.Tensor = masks.to(device)
        labels: torch.Tensor = labels.to(device)
        batch_size = texts.shape[0]

        optimizer.zero_grad()
        results = model(texts, attention_mask=masks, labels=labels)
        loss: torch.Tensor = results["loss"]
        logits: torch.Tensor = results["logits"]

        if mpu_loss_fn:
            pad_id = model.module.config.pad_token_id if hasattr(model, "module") else model.config.pad_token_id
            sent_length = (texts != pad_id).sum()
            pseudo_labels = (~labels.bool()).float()

            short_mask = (sent_length < len_threshold)
            u_mask = short_mask & (labels.bool()) # short AI text as unlabeled
            p_short_mask = short_mask & (~labels.bool()) # short human texts
            pseudo_labels[u_mask] = -1
            pseudo_labels[p_short_mask] = 0

            scores = F.softmax(logits, dim=-1)[..., 0]
            mpuloss = mpu_loss_fn(scores, pseudo_labels, sent_length)
            loss += mpuloss

        loss.backward()
        optimizer.step()

        batch_acc = cal_accuracy(logits, labels)
        train_acc += batch_acc
        train_loss += loss.item() * batch_size
        total += batch_size

    train_acc = train_acc / total

    return train_acc, train_loss

    
def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device=torch.device("cuda")
):
    model.eval()
    val_acc, val_loss = 0.0, 0.0
    total = 0

    with torch.no_grad():
        for texts, masks, labels in loader:
            texts: torch.Tensor = texts.to(device)
            masks: torch.Tensor = masks.to(device)
            labels: torch.Tensor = labels.to(device)
            batch_size = texts.shape[0]

            results = model(texts, attention_mask=masks, labels=labels)
            loss: torch.Tensor = results["loss"]
            logits: torch.Tensor = results["logits"]

            batch_acc = cal_accuracy(logits, labels)
            val_acc += batch_acc
            val_loss += loss.item() * batch_size
            total += batch_size

    val_acc = val_acc / total
    return val_acc, val_loss