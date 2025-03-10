"""
This is an example demonstrating modularized multiprocessing (mp) objects.
The main process passes a numpy array input to a compuatation module which returns a processed numpy array as output.

Usage:
    python run.py

"""

from multiprocessing.managers import SharedMemoryManager
import time
import numpy as np
from mp_module import DoSomethingProc


def run():
    print("=====")
    print(" Press Ctrl+c to quit.")
    print("=====")

    data_shape = (1, 3)
    data_dtype = np.int32

    with SharedMemoryManager() as shm_manager:
        with DoSomethingProc(
            shm_manager,
            np.zeros(data_shape, dtype=data_dtype),
            np.zeros(data_shape, dtype=data_dtype),
        ) as proc:

            assert proc.is_ready

            while True:
                try:
                    arr_input = np.random.randint(
                        0, 100, size=data_shape, dtype=data_dtype
                    )
                    print(f"\nInput: {arr_input}")
                    proc.put_to_input_queue(arr_input)
                    received = False
                    while not received:
                        (n, data) = proc.get_from_output_queue()
                        if n > 0 and data is not None:
                            print(f"Output (2 x Input) : {data[-1]}")
                            received = True
                        time.sleep(0.01)
                    time.sleep(1)
                except KeyboardInterrupt:
                    break


if __name__ == "__main__":
    run()
