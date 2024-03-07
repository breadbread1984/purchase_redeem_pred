# Introduction

this project is a solution to the [purchase redeeom prediction problem](https://tianchi.aliyun.com/competition/entrance/231573) posted at tianchi platform.

# Usage

## install prerequisites

```shell
python3 -m pip install tensorflow
```

## download dataset

download dataset from the [problem page](https://tianchi.aliyun.com/competition/entrance/231573)

## convert dataset

```shell
python3 create_dataset.py --input_csv <path/to/user_balance_table.csv> --output dataset.tfrecord
```

## train the model

```shell
python3 train.py --dataset dataset.tfrecord --ckpt ckpt
```
