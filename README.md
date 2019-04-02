Bayesian Personalized Ranking from Implicit Feedback
====================================================

The repository implement the Bayesian Personalized Ranking using pyTorch.
Other repository also implement this model, but the evaluation is much longer
than my thought. So, I implement this model using pyTorch with GPU acceleration.
Implementation detail will be explained in the following section.
[https://arxiv.org/pdf/1205.2618](https://arxiv.org/pdf/1205.2618)

## Set up environment

I implement using the following package.

* python==3.6
* pytorch==1.0.0
* numpy==1.15.4
* pandas==0.23.4

You can install these package by executing the following command. You can
manually install the above package by yourself using anaconda.

```bash
pip install -r requirements.txt
```

## Usage

### 0. Prepare data

Now, I use the movielens 1m data. You can get the data from [here](https://grouplens.org/datasets/movielens/1m/).
### 1. Preprocess data

So far, this repository support only movielens 1m dataset to preprocess. For
basic usage, following command line will preprocess the data.

```bash
python preprocess.py
```

If you want to split training and test data using timestep, which means that
try to use latest item for test data, then execute the following command line.

```bash
python preprocess.py --time_order
```

Help message for argument can be accessed via the following command.

```bash
python preprocess.py --help
```

### 2. Training MF model using BPR-OPT

Now, for real show, let's train MF model using BPR-OPT loss. You can execute
the following command to train MF model using BPR-OPT.

```bash
python train.py
```

Help message for argument can be accessed via the following command. You can
train MF model with different hyperparameter.

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
