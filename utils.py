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