import torch
import torch.nn as nn
import numpy as np

from model.pgcfc_seq import PGCFC_Decoder, PGCFC_Encoder

class PGCFC(nn.Module):
    def __init__(self, args):
        super(PGCFC, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.cl_decay_steps = args.lr_decay_step

        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.time_dim))

        self.encoder = PGCFC_Encoder(
            args.num_nodes, args.input_dim, args.rnn_units,
            args.cheb_k, args.embed_dim, args.time_dim, args.num_layers
        )
        self.decoder = PGCFC_Decoder(
            args.num_nodes, args.input_dim, args.rnn_units,
            args.cheb_k, args.embed_dim, args.time_dim, args.num_layers
        )

        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        nn.init.xavier_uniform_(self.node_embeddings1)
        nn.init.xavier_uniform_(self.T_i_D_emb1)
        nn.init.xavier_uniform_(self.D_i_W_emb1)
        nn.init.xavier_uniform_(self.T_i_D_emb2)
        nn.init.xavier_uniform_(self.D_i_W_emb2)

    def forward(self, source, target=None, batches_seen=None):
        if target is None and self.training:
            raise ValueError("target must be provided during training")

        t_i_d_data1 = source[..., 0, -2]
        t_i_d_data2 = target[..., 0, -2] if target is not None else torch.zeros_like(t_i_d_data1)

        T_i_D_emb1_en = self.T_i_D_emb1[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb2_en = self.T_i_D_emb2[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb1_de = self.T_i_D_emb1[(t_i_d_data2 * 288).type(torch.LongTensor)]
        T_i_D_emb2_de = self.T_i_D_emb2[(t_i_d_data2 * 288).type(torch.LongTensor)]

        if self.use_W:
            d_i_w_data1 = source[..., 0, -1]
            d_i_w_data2 = target[..., 0, -1] if target is not None else torch.zeros_like(d_i_w_data1)

            D_i_W_emb1_en = self.D_i_W_emb1[(d_i_w_data1).type(torch.LongTensor)]
            D_i_W_emb2_en = self.D_i_W_emb2[(d_i_w_data1).type(torch.LongTensor)]
            D_i_W_emb1_de = self.D_i_W_emb1[(d_i_w_data2).type(torch.LongTensor)]
            D_i_W_emb2_de = self.D_i_W_emb2[(d_i_w_data2).type(torch.LongTensor)]

            node_embedding_en1 = torch.mul(T_i_D_emb1_en, D_i_W_emb1_en)
            node_embedding_en2 = torch.mul(T_i_D_emb2_en, D_i_W_emb2_en)
            node_embedding_de1 = torch.mul(T_i_D_emb1_de, D_i_W_emb1_de)
            node_embedding_de2 = torch.mul(T_i_D_emb2_de, D_i_W_emb2_de)
        else:
            node_embedding_en1 = T_i_D_emb1_en
            node_embedding_en2 = T_i_D_emb2_en
            node_embedding_de1 = T_i_D_emb1_de
            node_embedding_de2 = T_i_D_emb2_de

        en_node_embeddings = [node_embedding_en1, node_embedding_en2, self.node_embeddings1]
        source = source[..., 0].unsqueeze(-1)

        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)
        state_all_steps, final_states_list = self.encoder(source, init_state, en_node_embeddings)
        global_context = torch.mean(state_all_steps, dim=1)

        ht_list = []
        for i in range(self.num_layers):
            layer_init_state = 0.7 * final_states_list[i] + 0.3 * global_context
            ht_list.append(layer_init_state)

        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(
                go, ht_list,
                [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1]
            )
            go = self.proj(state)
            out.append(go)

            if self.training and batches_seen is not None:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    go = target[:, t, :, 0].unsqueeze(-1)

        output = torch.stack(out, dim=1)
        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x
