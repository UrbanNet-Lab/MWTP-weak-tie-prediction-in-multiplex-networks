python src/main.py --datasetname=ckm --epoches=200 --run_times=10 --save_checkpoint=1 --device=gpu


semantic:
nohup python -u src/main-re.py --datasetname=ckm --epoches=200 --run_times=10 --save_checkpoint=1 --device=gpu > log/ckm.log 2>&1 &
nohup python -u src/main-re.py --datasetname=amazon --epoches=60 --run_times=10 --save_checkpoint=1 --device=gpu > log/amazon.log 2>&1 &
nohup python -u src/main-re.py --datasetname=acm --epoches=20 --run_times=10 --save_checkpoint=1 --device=gpu > log/acm.log 2>&1 &
nohup python -u src/main-re.py --datasetname=imdb --epoches=120 --run_times=10 --save_checkpoint=1 --device=gpu > log/imdb.log 2>&1 &

logit:
nohup python -u src/main-re.py --datasetname=ckm --epoches=200 --run_times=10 --save_checkpoint=1 --device=gpu --inter_aggregation=logit > log/logit_ckm.log 2>&1 &
nohup python -u src/main-re.py --datasetname=amazon --epoches=60 --run_times=10 --save_checkpoint=1 --device=gpu --inter_aggregation=logit > log/logit_amazon.log 2>&1 &
nohup python -u src/main-re.py --datasetname=acm --epoches=20 --run_times=10 --save_checkpoint=1 --device=gpu --inter_aggregation=logit > log/logit_acm.log 2>&1 &
nohup python -u src/main-re.py --datasetname=imdb --epoches=150 --run_times=10 --save_checkpoint=1 --device=gpu --inter_aggregation=logit > log/logit_imdb.log 2>&1 &




python src/main-re.py --datasetname=ckm --epoches=1 --run_times=1 --save_checkpoint=1 --device=gpu --inter_aggregation=logit

python src/main-re.py --datasetname=ckm --epoches=1 --run_times=1 --save_checkpoint=0 --device=gpu --inter_aggregation=logit
