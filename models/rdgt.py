# -*- coding: utf-8 -*- 
# @Time : 2022/12/20
# @Desc : RDGT: Enhancing Group Cognitive Diagnosis with Relation-Guided Dual-Side Graph Transformer

import math
from sklearn.preprocessing import LabelEncoder
from utils.utils import _loss_function
import torch
import torch.nn as nn
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RDGT(nn.Module):
    def __init__(self, args, num_users, num_items, num_groups, num_concepts, group_member_dict, group_node_sim, group_node_degree, group_node_related, group_ques_sim, mode='avg', drop_ratio=0.2):
        super(RDGT, self).__init__()

        self.concept_dim = num_concepts
        # Embedding Layers
        self.userembeds = UserEmbeddingLayer(num_users, num_concepts)
        self.itemembeds = ItemEmbeddingLayer(num_items, num_concepts)
        self.groupembeds = GroupEmbeddingLayer(num_groups, num_concepts)

        # Output Layer
        self.predictlayer = PredictLayer(num_concepts, drop_ratio)
        self.linear = nn.Linear(num_concepts, 1)

        self.stu_loss = _loss_function

        self.graphormer_stu = GTNet(n_layers=args.n_layers,
                                    num_heads=args.n_heads,
                                    hidden_dim=num_concepts,
                                    dropout_rate=args.dr,
                                    intput_dropout_rate=args.dr,
                                    ffn_dim=num_concepts,
                                    edge_type='multi_hop',
                                    attention_dropout_rate=args.dr)

        self.graphormer_exe = GTNet(n_layers=args.n_layers,
                                    num_heads=args.n_heads,
                                    hidden_dim=num_concepts + 1,
                                    dropout_rate=args.dr,
                                    intput_dropout_rate=args.dr,
                                    ffn_dim=num_concepts + 1,
                                    edge_type='multi_hop',
                                    attention_dropout_rate=args.dr)

        self.group_member_dict = group_member_dict
        self.group_node_sim = group_node_sim
        self.group_node_degree = group_node_degree
        self.group_node_related = group_node_related
        self.group_ques_sim = group_ques_sim
        self.num_users = num_users
        self.num_groups = num_groups
        self.mode = mode

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, item_inputs, concept_inputs, dataset):
        # train group
        out, loss_stu = self.grp_forward(group_inputs, item_inputs, concept_inputs, dataset)

        return out, loss_stu

    def grp_forward(self, group_inputs, item_inputs, concept_inputs, dataset):
        group_embeds = torch.Tensor().to(device)
        k_diff = torch.Tensor().to(device)
        e_disc = torch.Tensor().to(device)
        batch_stu_loss = torch.tensor(0.).to(device)
        bsz = group_inputs.shape[0]

        for m, n in zip(group_inputs, item_inputs):

            # ====================group learning====================
            members = self.group_member_dict[m.item()]
            group_size = len(members)
            members_ori2now = {i: j for i, j in zip(members, LabelEncoder().fit_transform(members))}

            group_node_info_related = self.group_node_related[self.group_node_related[:, 0] == m.item()]
            exter_stu_ = list(set(group_node_info_related[:, 2]))
            exter_stu = [int(i) for i in exter_stu_]
            exter_size = len(exter_stu)
            extstu_ori2now = {j: i+group_size for i, j in enumerate(exter_stu)}

            group_node_info_sim = self.group_node_sim[self.group_node_sim[:, 0] == m.item()]

            x = torch.LongTensor(sorted(list(members_ori2now.values())+list(extstu_ori2now.values()))).to(device).reshape(1, -1, 1)

            source_nodes = sum([[i] * (group_size - 1) for i in range(group_size)], [])
            target_nodes = [j for i in range(group_size) for j in range(group_size) if i != j]
            for row in group_node_info_related:
                i_id = members_ori2now[int(row[1])]
                j_id = extstu_ori2now[int(row[2])]
                source_nodes.append(i_id)
                target_nodes.append(j_id)
                source_nodes.append(j_id)
                target_nodes.append(i_id)
            edge_index = torch.LongTensor([source_nodes, target_nodes]).to(device)

            spatial_pos = torch.eye(group_size+exter_size, dtype=torch.float)
            group_node_info_sim = torch.from_numpy(group_node_info_sim).float()
            group_node_info_related = torch.from_numpy(group_node_info_related).float()
            for node_pair in group_node_info_sim:
                node_pair[1] = members_ori2now[int(node_pair[1])]
                node_pair[2] = members_ori2now[int(node_pair[2])]
            for node_pair in group_node_info_related:
                node_pair[1] = members_ori2now[int(node_pair[1])]
                node_pair[2] = extstu_ori2now[int(node_pair[2])]

            spatial_pos[group_node_info_sim[:, 1].long(), group_node_info_sim[:, 2].long()] = group_node_info_sim[:, 3]
            spatial_pos[group_node_info_related[:, 1].long(), group_node_info_related[:, 2].long()] = group_node_info_related[:, 3]
            spatial_pos = spatial_pos.to(device)

            attn_bias = torch.zeros([1, group_size+exter_size+1, group_size+exter_size+1], dtype=torch.float).to(device)

            group_data = Data(x=x, group_size=group_size,
                              edge_index=edge_index, spatial_pos=spatial_pos, attn_bias=attn_bias)

            g_embeds = self.graphormer_stu(group_data)  # [1, n_dim]
            g_embeds = torch.sigmoid(g_embeds)
            group_embeds = torch.cat((group_embeds, g_embeds))  # [bsz, n_dim]

            # ====================exercise learning====================
            group_ques_info_sim = self.group_ques_sim[self.group_ques_sim[:, 0] == m.item()]
            questions = list(set(group_ques_info_sim[:, 1]))
            ques_size = len(questions)
            questions_ori2now = {i: j for i, j in zip(questions, LabelEncoder().fit_transform(questions))}

            x_q = torch.LongTensor(sorted(list(questions_ori2now.values()))).to(device).reshape(1, -1, 1)

            source_nodes_q = sum([[i] * (ques_size - 1) for i in range(ques_size)], [])
            target_nodes_q = [j for i in range(ques_size) for j in range(ques_size) if i != j]
            edge_index_q = torch.LongTensor([source_nodes_q, target_nodes_q]).to(device)

            spatial_pos_q = torch.eye(ques_size, dtype=torch.float)
            group_ques_info_sim = torch.from_numpy(group_ques_info_sim).float()
            for node_pair in group_ques_info_sim:
                node_pair[1] = questions_ori2now[int(node_pair[1])]
                node_pair[2] = questions_ori2now[int(node_pair[2])]
            spatial_pos_q[group_ques_info_sim[:, 1].long(), group_ques_info_sim[:, 2].long()] = group_ques_info_sim[:, 3]
            spatial_pos_q = spatial_pos_q.to(device)

            attn_bias_q = torch.zeros([1, ques_size+1, ques_size+1], dtype=torch.float).to(device)
            ques_id = questions_ori2now[n.item()]
            ques_data = Data(x=x_q, ques_id=ques_id,
                              edge_index=edge_index_q, spatial_pos=spatial_pos_q, attn_bias=attn_bias_q)

            q_embeds = self.graphormer_exe(ques_data)  # [1, n_dim+1]
            q_embeds = torch.sigmoid(q_embeds)

            k_diff_emb = q_embeds[:, :self.concept_dim]
            e_disc_emb = q_embeds[:, self.concept_dim:]*10

            k_diff = torch.cat((k_diff, k_diff_emb))  # [bsz, n_dim]
            e_disc = torch.cat((e_disc, e_disc_emb))  # [bsz, 1]

            # ====================Multi-task Data \mathcal{U}====================
            stu_dataset = dataset.get_user_dataloader(list(members), 256)
            stu_pred, stu_label = torch.Tensor().to(device), torch.Tensor().to(device)
            if stu_dataset:
                for batch_id, (sids, qids, scores, cpts) in enumerate(stu_dataset):
                    sids, qids, scores, cpts = sids.to(device), qids.to(device), scores.to(device), cpts.to(device)
                    scores = scores.float()
                    cpts = cpts.float()
                    y_stu = self.usr_forward(sids, qids, cpts)
                    stu_pred = torch.cat((stu_pred, y_stu))
                    stu_label = torch.cat((stu_label, scores))

                batch_stu_loss += self.stu_loss(stu_pred, stu_label)

        # output
        input = e_disc * (group_embeds - k_diff) * concept_inputs
        y_grp = torch.sigmoid(self.predictlayer(input))
        # y = torch.sigmoid(self.linear(input))
        stu_loss = batch_stu_loss / bsz

        return y_grp, stu_loss

    def usr_forward(self, user_inputs, item_inputs, concept_inputs):
        user_embeds = torch.sigmoid(self.userembeds(user_inputs))
        k_diff, e_disc = self.itemembeds(item_inputs)
        k_diff, e_disc = torch.sigmoid(k_diff), torch.sigmoid(e_disc)*10
        input = e_disc * (user_embeds - k_diff) * concept_inputs
        y = torch.sigmoid(self.predictlayer(input))
        return y

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.predictlayer.apply(clipper)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GTNet(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        edge_type,
        attention_dropout_rate
    ):
        super().__init__()

        self.num_heads = num_heads
        self.node_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.edge_type = edge_type

        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(128 * num_heads * num_heads, 1)

        self.spatial_pos_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.Linear(128, num_heads)
        )

        self.positive_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        self.negative_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads) for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.downstream_out_proj = AttentionLayer(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.hidden_dim = hidden_dim
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x

        # if 'group_size' in batched_data.keys():
        if 'group_size' in batched_data.keys:
            group_size = batched_data.group_size

        # if 'ques_id' in batched_data.keys():
        if 'ques_id' in batched_data.keys:
            ques_id = batched_data.ques_id

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()

        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos.view(n_graph, n_node, n_node, -1)).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:]
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        node_feature = self.node_encoder(x).sum(dim=-2)
        if perturb is not None:
            node_feature += perturb

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # if 'group_size' in batched_data.keys():
        if 'group_size' in batched_data.keys:
            output = self.downstream_out_proj(output[:, 1:group_size+1, :]).squeeze(1)

        # elif 'ques_id' in batched_data.keys():
        elif 'ques_id' in batched_data.keys:
            output = self.downstream_out_proj(output[:, ques_id, :])

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        # Q matrix
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        # K matrix
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        # V matrix
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        # dropout layer
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        # output layer
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)  # [b, q_len, h_dim]

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        # Normalize & Add
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.k_difficulty = nn.Embedding(num_items, embedding_dim)
        self.e_discrimination = nn.Embedding(num_items, 1)

    def forward(self, item_inputs):
        k_diff, e_disc = self.k_difficulty(item_inputs), self.e_discrimination(item_inputs)
        return k_diff, e_disc


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.5):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return torch.matmul(weight, x)


class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=-1, keepdim=True)
        return x


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.5):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.Sigmoid(),
            nn.Dropout(drop_ratio),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
