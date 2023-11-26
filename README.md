# FedPseudo
### The implementation of federated survival analysis using FedPseudo framework.

<p align="center">
    <a href="#Paper">Paper</a> •
    <a href="#installation">Installation</a> •    
    <a href="#required-packages">Required Packages</a> •
    <a href="#running-experiments">Running Experiments</a> • 
    <a href="#how-to-cite">How to Cite</a> •    
    <a href="#Contact">Contact</a> •
</p>


## Paper
For more details, see full paper [FedPseudo: Privacy-Preserving Pseudo Value-Based Deep Learning Models for Federated Survival Analysis](https://dl.acm.org/doi/10.1145/3580305.3599348).

## Installation
### From source
Download a local copy of FedPseudo_KDD23 and install from the directory:

	git clone https://github.com/umbc-sanjaylab/FedPseudo_KDD23.git
	cd FedPseudo_KDD23
	pip install .

### Configure the environement

	conda env create --name FedPseudo
	conda activate FedPseudo

## Required Packages
* PyTorch
* pycox
* scikit-learn
* ray
* numpy
* pandas

### Install the packages
pip install -r requirements.txt


## Running Experiments
Here is one example to run this code for centralized training:
```
for dataset in ["metabric","support", "gbsg"]: 
    for model_name in ["FedPDNN","FedPLSTM","FedPAttn"]:
        for seed in range(1):
            !python main.py --model=$model_name \
                --dataset=$dataset \
                --alg='all_in' \
                --lr=0.001 \
                --batch-size=128 \
                --epochs=1000 \
                --n_parties=1 \
                --comm_round=1 \
                --partition='centralized' \
                --optimizer='adam'\
                --device='cuda:0'\
                --datadir='./Data/'\
                --logdir='./logs/centralized/'$dataset'/' \
                --patience=100\
                --init_seed=$seed\
                --sample=1.0\
```

Here is one example to run this code for federated training (iid and non-iid):

```
for part in ["iid", "non-iid"]:
    for dataset in ["metabric","support", "gbsg"]: 
        for model_name in ["FedPDNN","FedPLSTM","FedPAttn"]:
            for seed in range(1):
                !python main.py --model=$model_name \
                    --dataset=$dataset \
                    --alg='fedavg' \
                    --lr=0.001 \
                    --batch-size=128 \
                    --epochs=20 \
                    --n_parties=5 \
                    --comm_round=20 \
                    --partition=$part \
                    --optimizer='adam'\
                    --device='cuda:0'\
                    --datadir='/home/local/AD/mrahman6/0.AAAI_2023/Data/'\
                    --logdir='/home/local/AD/mrahman6/0.AAAI_2023/FedPseudo_Github/logs/'$part'/'$dataset'/' \
                    --patience=20\
                    --sample=1.0\
                    --init_seed=$seed\
                    --sensitivity=2\
                    --epsilon=8.5\
                    # --is_DP\ #Uncomment if DP is enforced

```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `FedPDNN`, `FedLSTM`, `FedPAttn`. Default = `FedPDNN`. |
| `dataset`      | Dataset to use. Options: `metabric`, `support`, `gbsg`. Default = `gbsg`. |
| `alg` | The training algorithm. Options: `fedavg`. Default = `fedavg`. |
| `lr` | Learning rate for the local models. Default = `0.001`. |
| `batch-size` | Batch size. Default = `128`. |
| `epochs` | Number of local training epochs. Default = `20`. |
| `n_parties` | Number of parties. Default = `5`. |
| `comm_round`    | Number of communication rounds to use. Default = `20`. |
| `partition`    | The partition type. Options: `centralized`, `iid`, `non-iid`. Default = `iid` |
| `optimizer` | Specify the optimizer. Options: `adam`, `sgd`, `amsgrad`. Default = `adam`. |
| `device` | Specify the device to run the program. Default = `cuda:0`. |
| `datadir` | The path of the dataset. Default = `./Data/`. |
| `logdir` | The path to store the logs. Default = `./logs/`. |
| `patience` | Number of patience. Default = `20`. |
| `init_seed` | The initial seed. Default = `0`. |
| `sample` | Ratio of parties that participate in each communication round. Default = `1`. |
| `sensitivity` | Sensitivity parameter for differential privacy. Default = `2.0`. |
| `epsilon` | Privacy budget parameter for differential privacy. Default = `1`. |
| `is_DP` | True if differential privacy is enforced. Default = `False`. |



## How to Cite

	@inproceedings{rahman2023fedpseudo,
    title={FedPseudo: Privacy-Preserving Pseudo Value-Based Deep Learning Models for Federated Survival Analysis},
    author={Rahman, Md Mahmudur and Purushotham, Sanjay},
    booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages={1999--2009},
    year={2023}
    }
  
## Contact
* Md Mahmudur Rahman (mrahman6@umbc.edu)
* Sanjay Purushotham (psanjay@umbc.edu)
