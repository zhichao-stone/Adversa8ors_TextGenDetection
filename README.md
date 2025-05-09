# Adversa8ors_TextGenDetection
Code of Team Adversa8ors for UCAS course AISAD in the spring of 2025.



### 1. Requirements

```
torch
transformers
numpy
scikit-learn
pandas
```



### 2. How to Run

#### 2.1 compute scores

Please modify the model paths in `prediction.py`. Then run the code using following instruction. You can modify the hyper-parameter as you want.

```shell
CUDA_VISIBLE_DEVICES=0,1 python prediction.py \
	--your-team-name Adversa8ors \
	--data ./aisad_text_detection/UCAS_AISAD_TEXT-val.csv \
	--result ./results \
	--batch_size 32
```

#### 2.2 Evaluate

```shell
python evaluate.py \
	--submit-path ./results \
	--gt-path ./aisad_text_detection/UCAS_AISAD_TEXT-val.csv
```

The evaluation results will be in the `./results/LeaderBoard.xlsx`



### 3. Acknowledgement

The algorithm design of this code references:

- [[Binoculars](https://arxiv.org/abs/2401.12070)] : **Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text**  `ICML 2024`
- [[MPU](./https://arxiv.org/abs/2305.18149)] : **Multiscale Positive-Unlabeled Detection of AI-Generated Texts**  `ICLR 2024`

Parts of base functions in this code reference [https://github.com/UCASAISecAdv/TextGenAdvTrack-2025Spring](./https://github.com/UCASAISecAdv/TextGenAdvTrack-2025Spring).

Thanks for their wonderful works on LLM-generated Text Detection ! 

Thanks to all team members for their contributions to this project.
