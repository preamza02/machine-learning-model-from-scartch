import numpy as np
from abc import ABC,abstractmethod


class _base(ABC):
    def __init__(self) -> None:
        super().__init__(self)
        self.name = self.__setname

    def __str__(self) -> str:
        return "{self.name}"
    
    @abstractmethod 
    def __setname(self) -> str:
        return 'base_model'
    

class _base_model(_base):
    def __init__(self) -> None:
        super().__init__(self)

    def __setname(self) -> str:
        return "base model"

    @abstractmethod
    def train(x,y) -> None:
        return 

class _base_loss_function(_base):
    def __init__(self) -> None:
        super().__init__(self)

    def __setname(self) -> str:
        return "base loss function"
