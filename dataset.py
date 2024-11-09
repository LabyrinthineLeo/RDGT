# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class _Dataset(object):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        self._raw_data = data
        self._concept_map = concept_map
        # reorganize datasets
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(int(sid), {})
            self._data[sid].setdefault(int(qid), {})
            self._data[sid][qid] = correct

        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts
        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)
        concept_ids = set(sum(concept_map.values(), []))

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def num_concepts(self):
        return self.n_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map


class LogDataset(_Dataset, Dataset):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):

        super().__init__(data, concept_map,
                         num_students, num_questions, num_concepts)

    def __getitem__(self, item):
        sid, qid, score = self._raw_data[item]
        sid = int(sid)
        qid = int(qid)
        concepts = np.array([0.] * self.n_concepts)
        concepts[self.concept_map[qid]] = 1.
        return sid, qid, score, concepts

    def __len__(self):
        return len(self._raw_data)


class MyDataset(object):

    def __init__(self, user_path, group_path, ques_concept, num_g_u, num_ques, num_concept, flag=False, h_flag=False):
        self.test_flag = flag
        self.hetero_flag = h_flag

        # read datas from csv files
        group_train = np.array(pd.read_csv(group_path.format('train')))
        group_test = np.array(pd.read_csv(group_path.format('test')))
        stu_data = np.array(pd.read_csv(user_path))
        self.stu_all_matrix = stu_data

        self.group_train = LogDataset(group_train, ques_concept, num_g_u, num_ques, num_concept)
        self.group_test = LogDataset(group_test, ques_concept, num_g_u, num_ques, num_concept)
        self.stu_data = LogDataset(stu_data, ques_concept, num_g_u, num_ques, num_concept)

    def get_group_dataloader(self, batch_size):
        train_data_loader = DataLoader(self.group_train,
                                       batch_size=batch_size,
                                       shuffle=True)

        test_data_loader = DataLoader(self.group_test,
                                       batch_size=batch_size,
                                       shuffle=True)

        return train_data_loader, test_data_loader

    def get_user_dataloader(self, batch_size):
        if self.hetero_flag == 'all_matrix':
            return self.stu_all_matrix
        else:
            data_loader = DataLoader(self.stu_data,
                                     batch_size=batch_size,
                                     shuffle=True)

            return data_loader


class RDGT_Dataset(object):
    def __init__(self, user_path, group_path, ques_concept, num_g_u, num_ques, num_concept):
        self.ques_concept = ques_concept
        self.num_g_u = num_g_u
        self.num_ques = num_ques
        self.num_concept = num_concept

        # read datas from csv files
        group_train = np.array(pd.read_csv(group_path.format('train')))
        group_test = np.array(pd.read_csv(group_path.format('test')))
        stu_data = np.array(pd.read_csv(user_path))
        self.stu_data = stu_data

        self.group_train = LogDataset(group_train, ques_concept, num_g_u, num_ques, num_concept)
        self.group_test = LogDataset(group_test, ques_concept, num_g_u, num_ques, num_concept)

    def get_group_dataloader(self, batch_size):
        train_data_loader = DataLoader(self.group_train,
                                       batch_size=batch_size,
                                       shuffle=True)

        test_data_loader = DataLoader(self.group_test,
                                       batch_size=batch_size,
                                       shuffle=True)

        return train_data_loader, test_data_loader

    def get_user_dataloader(self, stu_list, batch_size=256):
        stu_data_each_group = self.stu_data[np.isin(self.stu_data[:, 0], stu_list)]
        if stu_data_each_group.shape[0] < 1:
            return False
        else:
            self.batch_stu_data = LogDataset(stu_data_each_group, self.ques_concept, self.num_g_u, self.num_ques, self.num_concept)  # stu 训练集
            data_loader = DataLoader(self.batch_stu_data,
                                     batch_size=batch_size,
                                     shuffle=True)

            return data_loader
