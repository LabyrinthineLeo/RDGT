# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
import os, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Helper(object):
    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d_ = json.load(open(path, 'r'))
        g_m_d = {int(i): j for i, j in g_m_d_.items()}
        return g_m_d

    def get_ques_concept_dict(self, path):
        ques_concept_ = json.load(open(path, 'r'))
        ques_concept = {int(i): j for i, j in ques_concept_.items()}
        return ques_concept

    def get_group_ques_dict(self, path):
        group_ques_ = json.load(open(path, 'r'))
        group_ques = {int(i): j for i, j in group_ques_.items()}
        return group_ques


def _loss_function(pred, real):
    return (-(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()).to(device)


def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))


def mae(a, b):
    return np.mean(np.abs(a-b))


def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


