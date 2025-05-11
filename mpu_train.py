import os
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train import MPULoss, load_datasets, train, evaluate
from utils import get_train_args


if __name__ == "__main__":
    # get training arguments
    args = get_train_args()
    train_data_path = args.train_data
    valid_data_path = args.valid_data
    model_name_or_path = args.model
    save_dir = args.save_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.wd
    lamb = args.lamb
    pi = args.pi

    max_length = args.max_length
    len_threshold = args.len_threshold
    aug_min_length = args.aug_min_len
    aug_ratio = args.aug_ratio

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model:nn.Module = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model = model.to(device)
    mpu_loss = MPULoss(pi, max_length, lamb, device)

    # load data
    train_loader = load_datasets(train_data_path, tokenizer, batch_size, aug_min_length, aug_ratio, max_length, is_train=True)
    valid_loader = load_datasets(valid_data_path, tokenizer, batch_size, aug_min_length, aug_ratio, max_length, is_train=False)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train
    best_valid_acc, best_valid_epoch = 0.0, -1
    for epoch in tqdm(range(1, max_epochs+1)):
        train_acc, train_loss = train(model, optimizer, train_loader, len_threshold, mpu_loss, device)
        valid_acc, valid_loss = evaluate(model, valid_loader, device)
        print(f"Epoch: {epoch:03d} | Train: acc={train_acc:.4f}, loss={train_loss:.4f} | Valid: acc={valid_acc:.4f}, loss={valid_loss:.4f}")
        if valid_acc > best_valid_acc:
            best_valid_acc, best_valid_epoch = valid_acc, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best-model.pt"))
