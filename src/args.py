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


    # parser.add_argument('--input', type=str, default='data/amazon',
    #                     help='Input dataset path')
    #
    # parser.add_argument('--features', type=str, default=None,
    #                     help='Input node features')
    #
    # parser.add_argument('--walk-file', type=str, default=None,
    #                     help='Input random walks')
    #
    # parser.add_argument('--epoch', type=int, default=1,
    #                     help='Number of epoch. Default is 100.')
    #
    # parser.add_argument('--batch-size', type=int, default=64,
    #                     help='Number of batch_size. Default is 64.')
    #
    # parser.add_argument('--eval-type', type=str, default='all',
    #                     help='The edge type(s) for evaluation.')
    #
    # parser.add_argument('--schema', type=str, default=None,
    #                     help='The metapath schema (e.g., U-I-U,I-U-I).')
    #
    # parser.add_argument('--dimensions', type=int, default=200,
    #                     help='Number of dimensions. Default is 200.')
    #
    # parser.add_argument('--edge-dim', type=int, default=10,
    #                     help='Number of edge embedding dimensions. Default is 10.')
    #
    # parser.add_argument('--att-dim', type=int, default=20,
    #                     help='Number of attention dimensions. Default is 20.')
    #
    # parser.add_argument('--walk-length', type=int, default=10,
    #                     help='Length of walk per source. Default is 10.')
    #
    # parser.add_argument('--num-walks', type=int, default=20,
    #                     help='Number of walks per source. Default is 20.')
    #
    # parser.add_argument('--window-size', type=int, default=5,
    #                     help='Context size for optimization. Default is 5.')
    #
    # parser.add_argument('--negative-samples', type=int, default=5,
    #                     help='Negative samples for optimization. Default is 5.')
    #
    # parser.add_argument('--neighbor-samples', type=int, default=10,
    #                     help='Neighbor samples for aggregation. Default is 10.')
    #
    # parser.add_argument('--patience', type=int, default=5,
    #                     help='Early stopping patience. Default is 5.')
    #
    # parser.add_argument('--num-workers', type=int, default=16,
    #                     help='Number of workers for generating random walks. Default is 16.')
    # parser.add_argument('--iteration_num', type=int, default=0, help='iteration_num')
    #
    # return parser.parse_args()