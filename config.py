import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch AttTreeLSTM for Complex Knowledge Base Question Answering')
    # data arguments
    parser.add_argument('--attention', default=True, action='store_true', help='use AttTreeLSTM')

    parser.add_argument('--use_group', default=False, action='store_true', help='Use group as target')

    parser.add_argument('--use_parafrase', default=False, action='store_true', help='Use group as target')

    parser.add_argument('--data', default='data/lc-quad-2-wikidata-parafrase/',
                        help='path to dataset')
    parser.add_argument('--analysis', default='analysis/',
                        help='path to save analysis')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='ATT_TREE_group_paraphase',
                        help='Name to identify experiment')
    # model arguments
    parser.add_argument('--mem_dim', default=150, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--frgeeze_embed', action='store_true',
                        help='Freeze word embeddings')
    # training arguments
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=2.25e-3, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--emblr', default=1e-2, type=float,
                        metavar='EMLR', help='initial embedding learning rate')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='sgd',
                        help='optimizer (default: sgd)')

    # miscellaneous options
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed (default: 42)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    print(args)
    return args
