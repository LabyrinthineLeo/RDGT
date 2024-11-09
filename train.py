# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils.utils import rmse, mae
from tqdm import tqdm
import logging
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(args, dataset, model, train_loader, epoch_id, optimizer, lr):

    loss_fun_group = nn.MSELoss().to(device)

    losses = []
    for batch_id, (g_s_ids, question_ids, correctness, concepts) in tqdm(enumerate(train_loader)):
        g_s_ids, question_ids, correctness, concepts = g_s_ids.to(device), question_ids.to(device), correctness.to(device), concepts.to(device)
        correctness = correctness.float()
        concepts = concepts.float()

        pred_grp, loss_stu = model(g_s_ids, question_ids, concepts, dataset)
        pred_grp = pred_grp.view(-1)

        loss_grp = loss_fun_group(pred_grp, correctness)
        loss = loss_grp + args.gamma * loss_stu

        optimizer.zero_grad()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    print('Iteration %d, lr: %.4f, loss: %.4f' % (epoch_id, lr, np.mean(losses)))
    logging.info('Iteration %d, lr: %.4f, loss: %.4f' % (epoch_id, lr, np.mean(losses)))


def evaluation(model, test_data, dataset):
    real = []
    pred = []
    with torch.no_grad():
        model.eval()
        for g_s_ids, question_ids, correctness, concepts in test_data:
            g_s_ids, question_ids, correctness, concepts = g_s_ids.to(device), question_ids.to(device), correctness.to(device), concepts.to(device)
            correctness = correctness.float()
            concepts = concepts.float()
            output, _ = model(g_s_ids, question_ids, concepts, dataset)
            output = output.view(-1)

            pred.extend(output.tolist())
            real.extend(correctness.tolist())

        model.train()

        pred_bin = np.array([1 if i >= 0.5 else 0 for i in pred])
        real_bin = np.array([1 if i >= 0.5 else 0 for i in real])
        pred = np.array(pred)
        real = np.array(real)

        acc = np.mean(real_bin == pred_bin)
        auc = roc_auc_score(real_bin, pred_bin)
        rmse_value = rmse(pred, real)
        mae_value = mae(pred, real)

        return acc, auc, rmse_value, mae_value