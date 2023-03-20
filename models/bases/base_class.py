from abc import ABC,abstractmethod
import numpy as np


class Base(ABC):
    def __init__(self) -> None:
        self.name = self._set_name()
        print(f"Created {self.name}")

    def __str__(self) -> str:
        return f"{self.name}"

    @abstractmethod
    def _set_name(self) -> str:
        return 'base_model'
    

class BaseModel(Base):
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        pass


# how can i create base class that when some class inherit i need to assiagn name and it will be print when initialize