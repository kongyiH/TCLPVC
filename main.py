import os
import torch
import argparse

from util import set_seed
from runmodule import RunModule
from configure import get_default_config

dataset = {
    0: "Fashion",  # this
    1: "BDGP",
    2: "HandWritten",
    3: "Reuters_dim10",
    4: "WebKB",
    5: "Caltech101-7",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='2', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='10', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='20', help='number of test times')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['Dataset']['name'] = dataset

    # set seed
    set_seed(config['training']['seed'])

    # training module
    run = RunModule(config, device)
    # run.pretrain_ae()
    # run.pretrain_cl()
    #
    # for run.cfg['training']['knn'] in [5, 8, 10, 15, 20, 25, 30, 35, 50, 60]:
    #     for run.cfg['training']['lambda_graph'] in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    #         print("knn: " + str(run.cfg['training']['knn']) + ", lambda_graph: " + str(run.cfg['training']['lambda_graph']))
    #         run.pretrain_ae()
    #         run.pretrain_cl()

    run.train1()


if __name__ == '__main__':
    main()
