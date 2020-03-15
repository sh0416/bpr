Bayesian Personalized Ranking from Implicit Feedback
====================================================
[![Build Status](https://travis-ci.com/sh0416/bpr.svg?branch=master)](https://travis-ci.com/sh0416/bpr)

The repository implement the Bayesian Personalized Ranking using pyTorch ([https://arxiv.org/pdf/1205.2618](https://arxiv.org/pdf/1205.2618))  
Other repositories also implement this model, but the evaluation takes longer time.  
So, I implement this model using pyTorch with GPU acceleration for evaluation.  
Implementation detail will be explained in the following section.  

## Environment

### Hardware

* AMD Ryzen 7 3700X 8-Core Processor
* Samsung DDR4 32GB
* NVIDIA TitanXp

### Software

#### OS

I use both Windows and Linux(Ubuntu).

#### Python package

You have to install the following packages before executing this code.

* python==3.6
* pytorch==1.3.1
* numpy==1.15.4
* pandas==0.23.4

You can install these package by executing the following command or through anaconda.

```bash
pip install -r requirements.txt
```

## Usage

### 0. Prepare data

This code support the movielens 1m data and movielens 20m data.
You can get the dataset from the following list.

* [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
* [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/).

After downloading that file, unzip it.  
We call the path for unzipped directory `$data_dir`.

### 1. Preprocess data

For basic usage, execute following command line to preprocess the data.
It **randomly** split the whole dataset into two parts, training data and test data.
```bash
python preprocess.py --dataset ml-1m --data_dir $data_dir --output_data preprocessed/ml-1m.pickle
python preprocess.py --dataset ml-20m --data_dir $data_dir --output_data preprocessed/ml-20m.pickle
python preprocess.py --dataset amazon-beauty --data_dir $data_dir --output_data preprocessed/amazon-beauty.pickle
```

If you want to split training data and test data with time order, then execute the following command line.
This code **sorts the item list for each user using time order**. After that, it splits the whole data into two parts, training data and test data.
First 80% of the item list will become the training data and the last 20% of the item list will become test data.
```bash
python preprocess.py --dataset ml-1m --output_data preprocessed/ml-1m.pickle --time_order
python preprocess.py --dataset ml-20m --output_data preprocessed/ml-20m.pickle --time_order
```

Help message will give you more detail description for arguments.

```bash
python preprocess.py --help
```

### 2. Training MF model using BPR-OPT

Now, for real show, let's train MF model using BPR-OPT loss.
You can execute the following command to train MF model using BPR-OPT.

```bash
python train.py --data preprocessed/ml-1m.pickle
python train.py --data preprocessed/ml-20m.pickle
```

Help message will give you more detail description for arguments.
You can train MF model with different hyperparameter.

```bash
python train.py --help
```

## Implementation detail

## Result

The evaluation benchmark for movielens 1m is the following table.
I think more tuning will get better result, but this value is reasonably around the
statistic.
I got very weird statistic when I train MovieLens-1M. I think I have to check my function more rigorously.

| Dataset       | Preprocess | P@1    | P@5    | P@10   | R@1    | R@5    | R@10   |
|---------------|------------|--------|--------|--------|--------|--------|--------|
| Movielens-1m  | Random     | 0.3881 | 0.2987 | 0.2683 | 0.0178 | 0.0616 | 0.1018 |
| Movielens-1m  | Time-order | 0.1588 | 0.1348 | 0.1297 | 0.0071 | 0.0294 | 0.0519 |
| Movielens-20m | Random     | 0.2359 | 0.1790 | 0.1529 | 0.0118 | 0.0395 | 0.0652 |
| Movielens-20m | Time-order | 0.1070 | 0.0887 | 0.0809 | 0.0059 | 0.0237 | 0.0431 |

### Training loss curve

#### MovieLens 1M

| Dataset | Random | Time-order |
|---------|--------|------------|
| MovieLens-1M |![](https://github.com/sh0416/bpr/blob/master/result/ml1m-loss.JPG)|![](https://github.com/sh0416/bpr/blob/master/result/ml1m-timeorder-loss.JPG)|

### Evaluation metric curve

#### MovieLens 1M

| Dataset | Random | Time-order |
|---------|--------|------------|
| MovieLens-1M |![](https://github.com/sh0416/bpr/blob/master/result/ml1m-eval.JPG)|![](https://github.com/sh0416/bpr/blob/master/result/ml1m-timeorder-eval.JPG)|

More information will get from the `result` directory.

## FAQ


## Continuous integration (Travis CI)

I use `pytest` framework to make my function reliable.
The execution code for testing is `pytest`.
You can get some useful code snippets from `tests` directory.
As I cannot find continuous integration tool which freely support gpu, I couldn't test CUDA code I implemented.
So, for stable execution, do test manually by using `pytest` to check it works well in your environment.

## Laboratory (Experimental development)

### Brand new data structure `VariableShapeList`

I am working for more elaborated approach to calculate evaluation metric.
For now, I develop `VariableShapeList` which can handle list of tensors which has different length.
Someone might said that it is equivalent with `PackedSequence` which is already implemented in pyTorch, but I can't use that data structure for evaluation metric.
Some operation is directly implemented by CPP function and will be implemented by CUDA kernel function.

#### CPP Build tools (Optional)

For Windows, Visual Studio Build tool is needed for CPP extension. Install it from [here](https://visualstudio.microsoft.com/vs/older-downloads/)

### Use IterableDataset for delivering fast data structure

I figure out that the setup time for multiprocessing `DataLoader` is major bottleneck in my training script.
Therefore, I refactor my dataset with `IterableDataset` and get 10x faster than existing implementation.
*This implementation needs to be tested.*

### Performance optimization

The large batch size and speed performance optimization boost evaluation metric.
I will updated all statistics for MovieLens-1M and MovieLens-20M.

## Contact

If you have any problem or encounter mysterious things during simulating this code, **open issue** or contact me by sending email to seonghyeon.drew@gmail.com
