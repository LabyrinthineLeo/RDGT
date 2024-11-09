# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np

from time import time
from config import Config
from utils.utils import Helper, get_random_seed
from dataset import RDGT_Dataset
import logging
from tensorboardX import SummaryWriter
import datetime
from train import training, evaluation
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="slp_math")
    parser.add_argument("--model_name", type=str, default="rdgt")
    parser.add_argument("--dir_type", type=str, default="Split_ET")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--bsz", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--use_graph", type=bool, default=True)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--dr", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_sch", type=str, default="None")
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    config = Config(args)
    get_random_seed(args.seed)

    helper = Helper()
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)
    ques_concept = helper.get_ques_concept_dict(config.ques_concept_path)

    dataset = RDGT_Dataset(config.user_dataset,
                        config.group_dataset,
                        ques_concept,
                        config.metadata['num_students'],
                        config.metadata['num_questions'],
                        config.metadata['num_concepts'])

    model = config.model(args,
                         config.metadata['num_students'],
                         config.metadata['num_questions'],
                         config.metadata['num_groups'],
                         config.metadata['num_concepts'],
                         g_m_d,
                         config.group_node_sim,
                         config.group_node_degree,
                         config.group_node_related,
                         config.group_ques_sim,
                         'attention')  # 'avg'

    # init info
    init_info = {'model_name': config.model_name,
                 'dataset': config.data_flag,
                 'test_config': config.dir_name,
                 'epoch': config.epoch,
                 'lr': config.lr,
                 'batch_size': config.batch_size,
                 'dataset_info': config.metadata}

    params_str = "_".join([str(v) for k, v in init_info.items() if k not in ['dataset', 'dataset_info']])
    time_suffix = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_path = os.path.join(config.result_path, "RDGT_"+params_str+f"_{time_suffix}")
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    logging.basicConfig(
        filename=os.path.join(result_path, f'log.txt'),
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s %(levelname)s] %(message)s',
    )

    optimizer = optim.Adam(model.parameters(), config.lr)

    if 'cosine' in args.lr_sch:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epoch, 0, -1
        )
    elif 'cycle' in args.lr_sch:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=args.n_epoch, max_lr=args.lr
        )

    print("="*20+"RDGT Training"+"="*20)
    logging.info("="*20+"RDGT Training"+"="*20)
    for i, j in init_info.items():
        print("{}: {}".format(i, j))
        logging.info("{}: {}".format(i, j))

    # train the model
    best_metrics = {'epoch': 0,
                    'acc': 0,
                    'auc': 0,
                    'rmse': np.inf,
                    'mae': 0}

    for epoch in range(config.epoch):
        model = model.to(device)
        model.train()
        if args.lr_sch != "None":
            lr_cur = scheduler.get_last_lr()[0]
        else:
            lr_cur = config.lr

        t1 = time()

        train_data, test_data = dataset.get_group_dataloader(config.batch_size)

        training(args, dataset, model, train_data, epoch, optimizer, lr_cur)
        print("user and group training time is: [%.1f s]" % (time() - t1))
        logging.info("user and group training time is: [%.1f s]" % (time() - t1))

        if args.lr_sch != 'None':
            scheduler.step()

        t2 = time()
        acc, auc, rmse_value, mae_value = evaluation(model, test_data, dataset)

        if rmse_value <= best_metrics['rmse'] and epoch >= 0:
            best_metrics['epoch'] = epoch
            best_metrics['acc'] = acc
            best_metrics['auc'] = auc
            best_metrics['rmse'] = rmse_value
            best_metrics['mae'] = mae_value

        print('Group Iteration %d [%.1f s]: acc = %.4f, auc = %.4f, rmse = %.4f, mae = %.4f, [%.1f s]' % (epoch, time() - t1, acc, auc, rmse_value, mae_value, time() - t2))
        logging.info('Group Iteration %d [%.1f s]: acc = %.4f, auc = %.4f, rmse = %.4f, mae = %.4f, [%.1f s]' % (epoch, time() - t1, acc, auc, rmse_value, mae_value, time() - t2))

    print("the best results:")
    logging.info("the best results:")
    print("="*20+"best values"+"="*20)
    logging.info("="*20+"best values"+"="*20)
    for i, j in best_metrics.items():
        print(i, j)
        logging.info("{}: {}".format(i, j))

    print("Done!")
    logging.info("Done!")