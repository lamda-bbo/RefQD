# Quality-Diversity with Limited Resources

The official repository of our ICML'24 paper [*Quality-Diversity with Limited Resources*](https://openreview.net/pdf?id=64I29YeQdt).

This repository contains the Python code of RefQD, a resource-efficient Quality-Diversity (QD) optimization algorithm proposed in the paper. By decomposing a neural network into representation and decision parts, sharing the representation part with all decision parts, and employing a series of strategies to address the mismatch issue between the old decision parts and the newly updated representation part, RefQD not only uses significantly fewer resources, but also achieves comparable or better performance compared to sample-efficient QD algorithms.

## Abstract

Quality-Diversity (QD) algorithms have emerged as a powerful optimization paradigm with the aim of generating a set of high-quality and diverse solutions. To achieve such a challenging goal, QD algorithms require maintaining a large archive and a large population in each iteration, which brings two main issues, sample and resource efficiency. Most advanced QD algorithms focus on improving the sample efficiency, while the resource efficiency is overlooked to some extent. Particularly, the resource overhead during the training process has not been touched yet, hindering the wider application of QD algorithms. In this paper, we highlight this important research question, i.e., how to efficiently train QD algorithms with limited resources, and propose a novel and effective method called RefQD to address it. RefQD decomposes a neural network into representation and decision parts, and shares the representation part with all decision parts in the archive to reduce the resource overhead. It also employs a series of strategies to address the mismatch issue between the old decision parts and the newly updated representation part. Experiments on different types of tasks from small to large resource consumption demonstrate the excellent performance of RefQD: it not only uses significantly fewer resources (e.g., 16% GPU memories on QDax and 3.7% on Atari) but also achieves comparable or better performance compared to sample-efficient QD algorithms. Our code is available at [https://github.com/lamda-bbo/RefQD](https://github.com/lamda-bbo/RefQD).

## Requirements

The implementation utilizes [conda](https://www.anaconda.com/) environments and builds on top of the [QDax](https://github.com/adaptive-intelligent-robotics/QDax) framework. Some wheels are built using [Bazel](https://bazel.build/) version 6.4.0, which can be installed via the [npm](https://nodejs.org/en/download/) package manager with the following command:

```bash
npm install -g @bazel/bazelisk
export USE_BAZEL_VERSION=6.4.0
bazel --version
```

Once conda and Bazel are installed, you can build the environment by running the following command:

```bash
make
```

## Running Experiments

To evaluate RefQD on Humanoid Uni task, run the following commands in the root directory of the repository:

```bash
conda activate refqd
python -m refqd task=humanoid_uni framework=ME emitter=Ref-PGA-ME seed=1
```

You can replace any of the four parameters (i.e., task, framework, emitter, and seed) to evaluate different methods on different environments with different random seeds. Please refer to [`refqd/config/`](refqd/config/) for available choices. In our paper, we use five random seeds (1-5) to evaluate the performance of the methods.

## File Structure

- [`Makefile`](Makefile): The makefile for building the environment.
- [`environment.yml`](environment.yml): The environment definition file. Please use `make` instead.
- [`.external/`](.external/): The directory containing files for building some wheels.
- [`config/config.yaml`](config/config.yaml): The root config file for hydra.
- [`refqd/`](refqd/): The directory containing source code.
    - [`config/`](refqd/config/): The directory containing config files for hydra.
    - [`containers/`](refqd/containers/): The implementations of different archives.
    - [`emitters/`](refqd/emitters/): The implementations of different variation operators.
    - [`neuroevolution/`](refqd/neuroevolution/): The implementations of different networks and loss functions.
    - [`tasks/`](refqd/tasks/): The implementations of different tasks.
    - [`treax/`](refqd/treax/): The implementations of tree-transformed Jax functions.
    - [`__main__.py`](refqd/__main__.py): The entry point.
    - [`cluster.py`](refqd/cluster.py): The Jax-based implementation of the KMeans algorithm.
    - [`extended_me.py`](refqd/extended_me.py): The implementations of the MAP-Elites algorithm.
    - [`main.py`](refqd/main.py): The main function wrapped with hydra.
    - [`manager_base.py`](refqd/manager_base.py): The base classes for experiment manager.
    - [`manager.py`](refqd/manager.py): The experiment manager.
    - [`metrics.py`](refqd/metrics.py): The implementations of QD metrics.
    - [`utils.py`](refqd/utils.py): The utility classes and functions.

## Citation

If you find this work useful in your research, please consider citing:

**Quality-diversity with limited resources**.\
[Ren-Jian Wang](https://www.lamda.nju.edu.cn/wangrj/), [Ke Xue](https://www.lamda.nju.edu.cn/xuek/), [Cong Guan](https://www.lamda.nju.edu.cn/guanc/), and [Chao Qian](https://www.lamda.nju.edu.cn/qianc/).\
In *Proceedings of the 41st International Conference on Machine Learning (ICML'24)*, Vienna, Austria, 2024, to appear.

```bibtex
@inproceedings{RefQD,
    author = {Ren-Jian Wang and Ke Xue and Cong Guan and Chao Qian},
    title = {Quality-Diversity with Limited Resources},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML'24)},
    address = {Vienna, Austria},
    year = {2024},
}
```
