# Active Federated Learning
An unofficial implementation of the paper [[Active Federated Learning](https://arxiv.org/pdf/1909.12641.pdf)] in PyTorch


### examples
1. Reddit dataset
    - binary classification
    - implemented in original paper
    - not reproducible currently
```shell
python main.py --dataset Reddit --model BLSTM --method AFL --fed_algo FedAdam \
  --client_optimizer sgd --lr_local 0.01 --lr_global 0.001 --wdecay 0 --momentum 0.9 \
  --beta1 0.9 --beta2 0.999 --epsilon 1e-8 --alpha1 0.75 --alpha2 0.01 --alpha3 0.1 \
  -E 2 -B 128 -R 20 -A 200 --maxlen 400  
```