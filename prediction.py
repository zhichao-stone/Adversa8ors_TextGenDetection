import warnings
warnings.filterwarnings("ignore")

from detector import Binoculars, MPUDetector, EnsembleDetector

bino = Binoculars(
    observer_path="tiiuae/falcon-7b", 
    performer_path="tiiuae/falcon-7b-instruct"
)
mpu = MPUDetector(
    en_model_path="Your Path of Trained Model for Non-Chinese", 
    zh_model_path="Your Path of Trained Model for Chinese"
)
detector = EnsembleDetector(module1=bino, module2=mpu)


import os
import pandas as pd
from tqdm import tqdm
from utils import *

def run_prediction(dataset: pd.DataFrame, batch_size: int = 32):
    print("Starting prediction process...")
    prompts = dataset["prompt"].tolist()
    texts = dataset["text"].tolist()
    text_batches = split_batch(texts, batch_size)

    text_predictions = []
    start_time = pd.Timestamp.now()

    print("Processing texts...")
    for batch in tqdm(text_batches, desc="text detect"):
        scores = detector.compute_score(batch)
        text_predictions += scores

    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()

    # Create results dictionary
    results = {
        "predictions_data": {
            'prompt': prompts,
            'text_prediction': text_predictions
        }, 
        "time": processing_time
    }

    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results



if __name__ == "__main__":
    args = get_args()
    dataset = get_dataset(args)
    results = run_prediction(dataset, batch_size=args.batch_size)
    
    # Save results
    os.makedirs(args.result, exist_ok=True)
    save_path = os.path.join(args.result, args.your_team_name + ".xlsx")
    writer = pd.ExcelWriter(save_path, engine='openpyxl')
    
    # Create prediction dataframe with the required columns
    prediction_frame = pd.DataFrame(
        data = results["predictions_data"]
    )
    
    # Filter out rows with None values
    prediction_frame = prediction_frame.dropna()
    print(len(prediction_frame))
    
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(prediction_frame)],
            "Time": [results["time"]],
        }
    )
    
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    
    print(f"Results saved to {save_path}")