import warnings
warnings.filterwarnings("ignore")

from detector import Binoculars, MPUDetector

bino = Binoculars(
    observer_path="tiiuae/falcon-7b", 
    performer_path="tiiuae/falcon-7b-instruct"
)
mpu = MPUDetector(
    en_model_path="Your Path of Trained Model for Non-Chinese", 
    zh_model_path="Your Path of Trained Model for Chinese"
)

import os
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from typing import List, Dict, Tuple
from utils import get_dataset, split_batch
from train.augment import multi_scale_augment
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def data_augment(
    prompts: List[str], 
    texts: List[str], 
    labels: List[int], 
) -> Tuple[List[str], List[str], List[int]]:
    new_prompts, new_texts, new_labels = [], [], []
    for p, t, l in zip(prompts, texts, labels):
        ps, ts, ls = [p, p, p], [t], [l, l, l]
        # augment--sentence
        if l == 1:
            aug_t = multi_scale_augment(t, aug_mode="sentence", aug_ratio=0.1)
        else:
            aug_t = multi_scale_augment(t, aug_mode="sentence", aug_ratio=0.25)
        ts.append(aug_t)
        # augment--word
        if l == 1:
            aug_t = multi_scale_augment(t, aug_mode="word", aug_ratio=0.05)
        else:
            aug_t = multi_scale_augment(t, aug_mode="word", aug_ratio=0.15)
        if random.uniform(0, 1) < 1e-4:
            ps.append(p)
            ts.append("".join([random.choice(" -@#a7=+!?~/.,") for _ in range(random.randint(1, 15))]))
            ls.append(0)
        new_prompts += ps
        new_texts += ts
        new_labels += ls
    return new_prompts, new_texts, new_labels

def read_scores(path: str, sheet_name: str = "sheet1", column: str = "text_prediction") -> np.ndarray:
    data = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    return data[column].values

def write_results(
    results: Dict[str, List], 
    save_path: str, 
    sheet_name: str = "sheet1"
):
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        frame = pd.DataFrame(data = results)
        frame.to_excel(writer, sheet_name=sheet_name, index=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--s", type=float, default=0.0)
    parser.add_argument("--e", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    batch_size = args.bs


    data_path = "./datasets/aisad_text_detection/UCAS_AISAD_TEXT-val.csv"
    aug_data_path = "./datasets/aisad_text_detection/UCAS_AISAD_TEXT-val-aug.csv"

    if not os.path.exists(aug_data_path):
        dataset = get_dataset(data_path, ["prompt", "text", "label"])
        prompts, texts, labels = dataset["prompt"].tolist(), dataset["text"].tolist(), dataset["label"].tolist()

        prompts, texts, labels = data_augment(prompts, texts, labels)
        aug_data_frame = pd.DataFrame(data={"prompt": prompts, "text": texts, "label": labels})
        aug_data_frame.to_csv(aug_data_path, index=False)
    else:
        dataset = get_dataset(data_path, ["prompt", "text", "label"])
        prompts, texts, labels = dataset["prompt"].tolist(), dataset["text"].tolist(), dataset["label"].tolist()

    pred1_path = "./result/val-aug-pred1.xlsx"
    pred2_path = "./result/val-aug-pred2.xlsx"
    pred1_exist = os.path.exists(pred1_path)
    pred2_exist = os.path.exists(pred2_path)
    
    if not (pred1_exist and pred2_exist):
        text_batches = split_batch(texts, batch_size=batch_size)

    if not pred1_exist:
        pred1 = []
        for batch in tqdm(text_batches, desc="bino"):
            scores = bino.compute_score(batch)
            pred1 += scores
        write_results({"prompt": prompts, "text_prediction": pred1}, pred1_path, "predictions")
        pred1 = np.array(pred1)
    else:
        pred1 = read_scores(pred1_path, "predictions")
    
    if not pred2_exist:
        pred2 = []
        for batch in tqdm(text_batches, desc="bino"):
            scores = mpu.compute_score(batch)
            pred2 += scores
        write_results({"prompt": prompts, "text_prediction": pred2}, pred2_path, "predictions")
        pred2 = np.array(pred2)
    else:
        pred2 = read_scores(pred2_path, "predictions")

    beta_s: float = args.s
    beta_e: float = args.e
    step: float = args.step
    beta = beta_s

    best_score, best_score_beta = 0.0, 0.0
    best_auc, best_auc_beta = 0.0, 0.0
    best_acc, best_acc_beta = 0.0, 0.0
    best_f1, best_f1_beta = 0.0, 0.0
    while beta <= beta_e:
        pred = beta * pred1 + (1 - beta) * pred2
        binary_pred = (pred >= 0.5).astype(int)

        auc = roc_auc_score(labels, pred)
        acc = accuracy_score(labels, binary_pred)
        f1 = f1_score(labels, binary_pred)
        score = 0.6*auc + 0.3*acc + 0.1*f1

        print(f"# beta={beta:.3f} | auc={auc:.4f} , acc={acc:.4f} , f1={f1:.4f} , score={score:.4f}")

        if score > best_score:
            best_score, best_score_beta = score, beta
        if auc > best_auc:
            best_auc, best_auc_beta = auc, beta
        if acc > best_acc:
            best_acc, best_acc_beta = acc, beta
        if f1 > best_f1:
            best_f1, best_f1_beta = f1, beta


    print(f"score: best={best_score:.4f} , beta={best_score_beta:.3f}")
    print(f"auc: best={best_auc:.4f} , beta={best_auc_beta:.3f}")
    print(f"acc: best={best_acc:.4f} , beta={best_acc_beta:.3f}")
    print(f"f1: best={best_f1:.4f} , beta={best_f1_beta:.3f}")

    import json
    hyperparam_results = {
        "score": {"best": best_score, "beta": best_score_beta}, 
        "auc": {"best": best_auc, "beta": best_auc_beta}, 
        "acc": {"best": best_acc, "beta": best_acc_beta}, 
        "f1": {"best": best_f1, "beta": best_f1_beta}, 
    }
    with open("./results/best_hyper_params.json", "r", encoding="utf-8") as fw:
        json.dump(hyperparam_results, fw, ensure_ascii=False, indent=4)