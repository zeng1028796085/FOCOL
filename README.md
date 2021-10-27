# FOCOL

The code has been tested on an environment with Python 3.9.7, PyTorch 1.9.1, Numpy 1.21.2, and DGL 0.7.1.

The datasets are taken from the previous work [Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation (AAAI'21)](https://arxiv.org/abs/2012.06852). You can download the datasets in [this GitHub repo](https://github.com/xiaxin1998/DHCN/) and extract them to the [datasets](./datasets) folder.

To train FOCOL, run the following command:
```bash
python run.py expts/focol/focol.py --dataset-dir datasets/diginetica
```
which trains FOCOL on _diginetica_ with default hyperparameters.

You can see the detailed usage with the following command:
```bash
python run.py expts/focol/focol.py -h
```
