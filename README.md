# TensorFlow-Recommenders-demo
Getting started with TensorFlow Recommenders.

## Setup
First create a virtual environment:
```
# Create and activate it with Conda
conda create --name=tensorflow-recommenders python=3.7
conda activate tensorflow-recommenders

# Now install cuDNN, it will also install CUDA Toolkit
conda install -c anaconda cudnn=8.2.1

# Include libcudart, see https://stackoverflow.com/questions/69917132/could-not-load-dynamic-library-libcudart-so-11-0-in-conda-enviroment
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```

Then clone this repository and install the requirements:
```
git clone https://github.com/davidguzmanr/TensorFlow-Recommenders-demo.git
cd TensorFlow-Recommenders-demo
pip install -r requirements.txt
```
## References

- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders)