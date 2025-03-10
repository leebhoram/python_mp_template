# Python MultiProcessing Template

A simple Python multiprocessing example with shared memory queue of numpy arrays.

The example demonstrates a derived multiprocessing.Process object that takes a numpy array as input, processes it, and return a output numpy array.


## 💡 Troubleshooting: installing `atomics` on Mac M1/M2 with arm64

If you're working on Mac with M chip, install `atomics` by the following:
```
export CFLAGS="-Wno-error=deprecated-copy"
export CXXFLAGS="-Wno-error=deprecated-copy"
pip install --no-binary=atomics atomics
```
[reference](https://github.com/doodspav/atomics/issues/5)