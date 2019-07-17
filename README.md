Bayesian Personalized Ranking from Implicit Feedback
====================================================

The repository implement the Bayesian Personalized Ranking using pyTorch.
([https://arxiv.org/pdf/1205.2618](https://arxiv.org/pdf/1205.2618))
Other repositories also implement this model, but the evaluation takes longer time.
So, I implement this model using pyTorch with GPU acceleration for evaluation.
Implementation detail will be explained in the following section.

## Set up environment

You have to install the following package before executing this code.

* python==3.6
* pytorch==1.0.0
* numpy==1.15.4
* pandas==0.23.4

You can install these package by executing the following command or through anaconda.

```bash
pip install -r requirements.txt
```

## Usage

### 0. Prepare data

This code support only the movielens 1m data.
You can get the dataset from [here](https://grouplens.org/datasets/movielens/1m/).  
*NOTE: If your dataset is structured like movielens 1m data, which means contains `user,item,rate,time`, then it will work.*  

### 1. Preprocess data

For basic usage, execute following command line to preprocess the data.
It **randomly** split the whole dataset into two parts, training data and test data.
```bash
python preprocess.py
```

If you want to split training data and test data with time order, then execute the following command line.
This code **sorts the item list for each user using time order**. After that, it splits the whole data into two parts, training data and test data.
First 80% of the item list will become the training data and the last 20% of the item list will become test data.
```bash
python preprocess.py --time_order
```

Help message will give you more detail description for arguments.

```bash
python preprocess.py --help
```

### 2. Training MF model using BPR-OPT

Now, for real show, let's train MF model using BPR-OPT loss.
You can execute the following command to train MF model using BPR-OPT.

```bash
python train.py
```

Help message will give you more detail description for arguments.
You can train MF model with different hyperparameter.

```bash
python train.py --help
```

## Implementation detail

* I didn't use regularization coefficient for each embedding matrix. If you want
to tune each coefficient, then you need to add regularization term inside the loss
value.

## Result

The evaluation benchmark for movielens 1m is the following table.
I think more tuning will get better result, but this value is reasonably around the
statistic.

| Dataset      | Preprocess | P@1    | P@5    | P@10   | R@1    | R@5    | R@10   |
|--------------|------------|--------|--------|--------|--------|--------|--------|
| Movielens-1m | Random     | 0.2421 | 0.2058 | 0.1821 | 0.0096 | 0.0392 | 0.0674 |
| Movielens-1m | Time-order | 0.1307 | 0.1133 | 0.1034 | 0.0052 | 0.0216 | 0.0388 |

## FAQ

Q. Loss doesn't decrease, why?
A. I print the smoothing loss and the initial smoothing loss is 0, which cause
this problem. Waiting just a moment, then loss will go down.

Q. Loss converge to 0.6931.
A. Because weight decay is so strong that model cannot learn from dataset.
Decrease the weight decay factor.

## Contact

If you have any problem during simulating this code, open issue or contact me
by sending email to sh0416@postech.ac.kr
