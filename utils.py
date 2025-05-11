import math
import argparse
import pandas as pd
from typing import List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--your-team-name", type=str, default="Adversa8ors")
    parser.add_argument("--data", type=str, default="./datasets/aisad_text_detection/UCAS_AISAD_TEXT-val.csv")
    parser.add_argument("--result", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args
    

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="file path of data for training")
    parser.add_argument("--valid_data", type=str, help="file path of data for validation")
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--aug_min_len", type=int, default=1)
    parser.add_argument("--aug_ratio", type=float, default=0.2)
    parser.add_argument("--lamb", type=float, default=0.4)
    parser.add_argument("--pi", type=float, default=0.2)
    parser.add_argument("--len_threshold", type=int, default=55, help="length threshold for regarding text as short text.")
    parser.add_argument("--save_dir", type=str, default="save_model")
    args = parser.parse_args()
    return args


def get_dataset(args):
    print(f"Loading dataset from {args.data}...")
    data = pd.read_csv(args.data)

    # New format: prompt, text
    dataset = data[["prompt", "text"]].dropna().copy()
    
    return dataset


def split_batch(arr: List[str], batch_size: int = 128) -> List[List[str]]:
    bn = math.ceil(len(arr) / batch_size)
    batches: List[List[str]] = []
    for i in range(bn):
        batches.append(arr[i*batch_size : (i+1)*batch_size])
    return batches