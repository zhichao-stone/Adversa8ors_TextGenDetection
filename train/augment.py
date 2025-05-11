import random
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize, word_tokenize



def multi_scale_augment(
    text: str, 
    min_len: int = 50, 
    aug_mode: str = "sentence", 
    aug_ratio: float = 0.25, 
) -> str:
    aug_text = ""

    if len(text) < min_len:
        aug_text = text
        
    else:
        if "word" in aug_mode:
            if "zh" in aug_mode:
                word_aug_num = int(len(text)*aug_ratio)
                del_indices = random.sample(list(range(len(text))), word_aug_num)
                del_indices = [-1] + del_indices + [len(text)]
                new_words = []
                for i in range(word_aug_num + 1):
                    i1, i2 = del_indices[i]+1, del_indices[i+1]
                    new_words.append(text[i1:i2])
                aug_text = "".join(new_words)
            else:
                words = word_tokenize(text)
                word_aug_num = int(len(words)*aug_ratio)
                del_indices = random.sample(list(range(len(words))), word_aug_num)
                del_indices = [-1] + del_indices + [len(words)]
                new_words = []
                for i in range(word_aug_num + 1):
                    i1, i2 = del_indices[i]+1, del_indices[i+1]
                    new_words.append(" ".join(words[i1:i2]))
                aug_text = " ".join(new_words)
                            
        if "sentence" in aug_mode:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                aug_text = text
            else:
                new_sentences = []
                for sent in sentences:
                    if random.uniform(0, 1) <= 1 - aug_ratio:
                        new_sentences.append(sent)
                if len(new_sentences) == 0:
                    aug_text = text
                else:
                    connect_char = "" if "zh" in aug_mode else " "
                    aug_text = connect_char.join(new_sentences)

    return aug_text.strip()