import re


QUANJIAO = ['……','。。。','。', '，', '；', '：', '？', '！', '“', '”', "‘", "’", "（", "）", '【', '】', '、']
BANJIAO = ['...','...','.', ',', ';', ':', '?', '!', '"', '"', "'", "'", "(", ")", '[', ']', ',']
PUNCUTATIONS = [',', '.', ';', ':', '"', "'", "?", '!']

def clean_text(text: str):
    # replace quanjiao punctuation with banjiao
    for i, j in zip(QUANJIAO, BANJIAO):
        text = text.replace(i, j)
    # deal with \n and space around punctuation
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    for punc in PUNCUTATIONS:
        pattern = "\s*" + (punc if punc not in ['.', '"', "?"] else "\\" + punc) + "\s*"
        text = re.sub(pattern, punc + " ", text)
    # deal ...
    text = text.replace('. . . ', '... ')
    # deal with 's, 't
    text = text.replace("' s ", "'s ")
    text = text.replace("' t ", "'t ")
    return text.strip()
