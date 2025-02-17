import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', type=str, default='ckm',
                        help='Dataset name.')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--batch_num', type=int, default=256,
                        help='Number of batch size. Default is .')
    parser.add_argument('--epoches', type=int, default=200,
                        help='Number of epoch. Default is 100.')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Whether to use GPU to run (e.g., gpu, cpu).')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    parser.add_argument('--save_checkpoint', type=int, default=1,
                        help='Whether to save model checkpoints (1: save, 0: do not save).')
    parser.add_argument('--run_test', type=int, default=0,
                        help='Run train or test (1: test, 0: train).')
    parser.add_argument('--run_times', type=int, default=10,
                        help='Number of times to run the experiment.')
    parser.add_argument('--inter_aggregation', type=str, default='semantic',
                        help='Whatever to use aggregation to run (e.g., semantic, logit).')
    return parser.parse_args()
