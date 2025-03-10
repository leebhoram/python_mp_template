from multiprocessing.managers import SharedMemoryManager
import numpy as np
from .base_proc import BaseProc


class DoSomethingCore:
    """A class that is dedicated to DO SOMETHING other than dealing with multiprocessing."""

    def __init__(self, input_example: np.ndarray, output_example: np.ndarray) -> None:
        # store shape/type info from input/output dymmay samples if needed
        self.input_shape = input_example.shape
        self.input_dtype = input_example.dtype

    def update(self, input_: np.ndarray) -> np.ndarray:
        """Implement the processing you'd like to apply

        As a toy example, it mutiplies the input array by 2 and return the result.
        """
        return 2 * input_


class DoSomethingProc(BaseProc):
    """A wrapper class for DoSomethingCore that abstracts all the multiprocessing codes.

    This is a derived class of BaseProc which is a multiprocessing.Process.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        input_example: np.ndarray,
        output_example: np.ndarray,
    ) -> None:
        super().__init__(shm_manager, input_example, output_example)
        self.do_something = DoSomethingCore(input_example, output_example)

    def _update(self, input_: np.ndarray) -> np.ndarray:
        return self.do_something.update(input_)
