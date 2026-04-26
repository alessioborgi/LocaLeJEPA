# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pprint
import sys
import yaml

import submitit

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-launch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml config file (or list of config files if --batch-launch)',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')


class Trainer:

    def __init__(self, fname='configs.yaml', load_model=None):
        self.fname = fname
        self.load_model = load_model

    def __call__(self):
        fname = self.fname
        load_model = self.load_model
        logger.info('called-params %s', fname)

        with open(fname, 'r', encoding='utf-8') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pprint.PrettyPrinter(indent=4).pprint(params)

        resume_preempt = False if load_model is None else load_model

        method = params.get('method', 'ijepa')
        if method == 'lejepa':
            from src.lejepa.train import main as app_main
        elif method == 'simclr':
            from src.baselines.simclr.train import main as app_main
        elif method == 'phc_supervised':
            from src.baselines.phc_supervised import main as app_main
        else:
            from src.ijepa.train import main as app_main

        app_main(args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.fname, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch(args):
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.folder, 'job_%j'),
        slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='55G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)

    config_fnames = [args.fname]

    jobs, trainers = [], []
    with executor.batch():
        for cf in config_fnames:
            fb_trainer = Trainer(cf)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    launch(parser.parse_args())
