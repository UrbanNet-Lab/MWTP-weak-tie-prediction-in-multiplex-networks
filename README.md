## MWTP

#### Prerequisites

- Python
- Pytorch

#### Getting Started

##### Installation

clone this repo.

`git clone https://github.com/UrbanNet-Lab/MWTP-weak-tie-prediction-in-multiplex-networks.git`

Please first install PyTorch, and then install other dependencies by 

`pip install -r requirements.txt`

##### Dataset

We provide the CKM sample dataset. 

##### Training

If you want to execute the MWTP-semantic version you can execute `python src/main-re.py --datasetname=ckm --epoches=200 --run_times=10 --save_checkpoint=1 --device=gpu`, if you want to execute the MWTP-logit version you can execute `python src/main-re.py --datasetname=ckm --epoches=200 --run_times=10 --save_checkpoint=1 --device=gpu`.

#### Training on your own datasets

If you want to train MWTP with your own dataset, you can refer to the way to load CKM dataset in ‘dataloder.py’ and add your dataset.

##### Cite

Please cite our paper if you find this code useful for your research: 

`@`