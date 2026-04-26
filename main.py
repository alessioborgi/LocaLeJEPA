# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import os
import pprint
import yaml

from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='path to YAML config file',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    import logging

    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.info('called-params %s', fname)

    with open(fname, 'r', encoding='utf-8') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    logger.info('loaded params...')
    pprint.PrettyPrinter(indent=4).pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info('Running... (rank: %d/%d)', rank, world_size)

    method = params.get('method', 'ijepa')
    if method == 'lejepa':
        from src.lejepa.train import main as app_main
    elif method == 'simclr':
        from src.baselines.simclr.train import main as app_main
    elif method == 'phc_supervised':
        from src.baselines.phc_supervised import main as app_main
    else:
        from src.ijepa.train import main as app_main

    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
