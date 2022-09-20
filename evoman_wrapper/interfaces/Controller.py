import numpy as np

from abc import ABC, abstractmethod


class Controller(ABC):

    @abstractmethod
    def control(self, inputs: list, individual: np.array):
        pass
