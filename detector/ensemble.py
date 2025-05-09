from typing import List, Union
import numpy as np
from .binoculars import Binoculars
from .mpu import MPUDetector



class EnsembleDetector:
    def __init__(self, 
        module1: Binoculars, 
        module2: MPUDetector, 
        alpha: float = 0.912
    ) -> None:
        self.alpha = alpha
        self.module1 = module1
        self.module2 = module2

    def compute_score(self, text: Union[str, List[str]]) -> List[float]:
        score1 = self.module1.compute_score(text)
        score2 = self.module2.compute_score(text)

        final_score: np.ndarray = self.alpha * np.array(score1) + (1 - self.alpha) * np.array(score2)

        return final_score.tolist()