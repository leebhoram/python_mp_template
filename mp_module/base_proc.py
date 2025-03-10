import sys
import time
from typing import Optional, Tuple, List
from enum import IntEnum, unique
from enum import auto as enumAuto
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from queue import Empty as excEmpty
from queue import Full as excFull
import numpy as np

sys.path.append("../shm_utils")
from shm_utils import SharedMemoryQueue


@unique
class ProcCommand(IntEnum):
    STOP = enumAuto()
    # define more commands if needed


class BaseProc(mp.Process):
    """A base class dealing with numpy array input/output queues, derived from multiprocessing Process.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        input_example: np.ndarray,
        output_example: np.ndarray,
        timeout: int = 5,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        cmd_sample = {"cmd": ProcCommand.STOP.value}
        self.cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=cmd_sample, buffer_size=8
        )

        dummy_input = {"data": input_example}
        self.input_shape = input_example.shape
        self.input_dtype = input_example.dtype
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=dummy_input, buffer_size=8
        )

        dummy_output = {"data": output_example}
        self.output_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=dummy_output, buffer_size=8
        )

        self.launch_timeout = timeout
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.verbose = verbose

    # context manager
    def __enter__(self):
        self.start()
        return self

    # context manager
    def __exit__(self, exc_type, exc_val, exc_traceback) -> Optional[bool]:
        self.stop()
        if exc_type is not None:
            if exc_type is KeyboardInterrupt:
                return True
            else:
                return False

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(self.launch_timeout)
            assert self.is_alive(), f"timeout! ({self.launch_timeout} sec)"

    def stop(self, wait=True):
        stop_cmd = {"cmd": ProcCommand.STOP.value}
        self.cmd_queue.put(stop_cmd)
        if wait:
            self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def put_to_input_queue(self, data: np.ndarray) -> None:
        assert (
            data.shape == self.input_shape
        ), f"{data.shape} is not equal to {self.input_shape}"
        assert (
            data.dtype == self.input_dtype
        ), f"{data.dtype} is not equal to {self.input_dtype}"
        input_data = {"data": data}
        try:
            self.input_queue.put(input_data)
        except excFull:
            if self.verbose:
                print("input queue is full! failed to put new data.")
            pass

    def get_from_output_queue(self, k: int = -1) -> Tuple[int, List[np.ndarray]]:
        output_list = []
        output_len = 0
        try:
            if k < 1:  # get all
                output_list = self.output_queue.get_all()["data"]
            else:  # (if k >= 1)
                output_list = self.output_queue.get_k(k=k)["data"]
            output_len = len(output_list)
        except excEmpty:
            if self.verbose:
                print("output queue is empty! returning empty list.")
            pass
        finally:
            return (output_len, output_list)

    def _update(self, input_data: np.ndarray) -> np.ndarray:
        """Derived classes must implement this."""
        raise NotImplementedError

    def run(self):
        started = False
        while not self.stop_event.is_set():
            # ==========
            # handle command queue
            n_cmd = 0
            try:
                cmds = self.cmd_queue.get_all()["cmd"]
                n_cmd = len(cmds)
            except excEmpty:
                pass

            if n_cmd > 0:
                for cmd in cmds:
                    if cmd == ProcCommand.STOP.value:
                        self.stop_event.set()
                        break

            if self.stop_event.is_set():
                break

            # ==========
            # handle data queue
            n_data = 0
            try:
                data = self.input_queue.get_all()["data"]
                n_data = len(data)
            except excEmpty:
                pass

            if n_data > 0:
                # In this example, use only the latest one from the input queue
                input_ = data[n_data - 1]
                # process it
                output = self._update(input_)
                # put the processed result to the output queue
                output_data = {"data": output}
                try:
                    self.output_queue.put(output_data)
                except excFull:
                    if self.verbose:
                        print("output queue is full! failed to put new output.")

            if not started:
                self.ready_event.set()
                started = True

            time.sleep(0.01)

        if self.verbose:
            print("exiting the main loop.")
