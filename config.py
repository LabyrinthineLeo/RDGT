# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import pandas as pd
from models.rdgt import RDGT


class Config(object):
    def __init__(self, args):
        model_zoo = {
            'rdgt': RDGT
        }

        self.data_flag = args.dataset_name
        self.dir_name = args.dir_type
        self.graph_flag = args.use_graph
        self.set_path_dict()
        self.path = self.path_dict['path']
        self.user_dataset = self.path + 'stu_train.csv'
        self.group_dataset = self.path + 'group_{}.csv'
        self.user_in_group_path = self.path_dict['g_m_d']
        self.group_ques_path = self.path_dict['g_q_d']
        self.ques_concept_path = self.path_dict['q_c_d']

        self.model_name = args.model_name
        self.model = model_zoo[self.model_name]

        self.metadata = json.load(open(self.path_dict['meta'], 'r'))

        if self.graph_flag:
            self.graph_matrix = np.array(pd.read_csv(self.path_dict['graph_matrix']))
            self.group_node_sim = np.array(pd.read_csv(self.path_dict['group_node_sim']))
            self.group_node_degree = np.array(pd.read_csv(self.path_dict['group_node_degree']))
            self.group_node_related = np.array(pd.read_csv(self.path_dict['group_node_related']))
            self.group_ques_sim = np.array(pd.read_csv(self.path_dict['group_ques_sim']))

        self.result_path = "./results/{}/".format(self.data_flag)
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

        self.embedding_size = self.metadata['num_concepts']
        self.epoch = args.epoch
        self.batch_size = args.bsz
        self.lr = args.lr
        self.drop_ratio = args.dropout
        self.topK = args.topk

    def set_path_dict(self):
        self.path_dict = {}
        self.path_dict['path'] = './datas/{}/{}/'.format(self.data_flag, self.dir_name)
        self.path_dict['g_m_d'] = "./datas/{}/{}/group_stu_dict.json".format(self.data_flag, self.dir_name)
        self.path_dict['g_q_d'] = "./datas/{}/{}/group_ques_dict.json".format(self.data_flag, self.dir_name)
        self.path_dict['q_c_d'] = "./datas/{}/{}/ques_concept.json".format(self.data_flag, self.dir_name)
        self.path_dict['meta'] = "./datas/{}/{}/metadata.json".format(self.data_flag, self.dir_name)
        self.path_dict['graph_matrix'] = "./datas/{}/{}/graph_matrix.csv".format(self.data_flag, self.dir_name)
        self.path_dict['group_node_sim'] = "./datas/{}/{}/group_node_sim.csv".format(self.data_flag, self.dir_name)
        self.path_dict['group_node_degree'] = "./datas/{}/{}/group_node_degree.csv".format(self.data_flag, self.dir_name)
        self.path_dict['group_node_related'] = "./datas/{}/{}/group_node_related.csv".format(self.data_flag, self.dir_name)
        self.path_dict['group_ques_sim'] = "./datas/{}/{}/group_ques_sim.csv".format(self.data_flag, self.dir_name)

