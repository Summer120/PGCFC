import torch
import torch.nn as nn

from model.pgcfc_layers import PGCFCCell

class PGCFC_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PGCFC_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PGCFC_cells = nn.ModuleList()
        self.PGCFC_cells.append(PGCFCCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.PGCFC_cells.append(PGCFCCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, x, init_state, node_embeddings):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.PGCFC_cells[i](
                    current_inputs[:, t, :, :],
                    state,
                    [node_embeddings[0][:, t, :], node_embeddings[1][:, t, :], node_embeddings[2]]
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PGCFC_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)

class PGCFC_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PGCFC_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PGCFC_cells = nn.ModuleList()
        self.PGCFC_cells.append(PGCFCCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.PGCFC_cells.append(PGCFCCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, xt, init_state, node_embeddings):
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PGCFC_cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1], node_embeddings[2]])
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden
