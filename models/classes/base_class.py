from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self) -> None:
        self.name = self._set_name()
        self._init_print()

    def __str__(self) -> str:
        return f"{self.name}"

    def _init_print(self):
        print(f"Created {self.name}")

    @abstractmethod
    def _set_name(self) -> str:
        return 'base_model'
    
