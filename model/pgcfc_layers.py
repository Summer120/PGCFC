import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from collections import OrderedDict

class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FC, self).__init__()
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.mlp = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(self.middle_dim, dim_out))
            ])
        )

    def forward(self, x):
        ho = self.mlp(x)
        return ho

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("bnm,bmc->bnc", A, x)
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self, k=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.k = k

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.k + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.stack(out, dim=1)
        return h

class PDGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(PDGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k*2+1, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))       
        self.spatial_weight = nn.Parameter(torch.FloatTensor(cheb_k * 2 + 1))

        self.embed_dim = embed_dim
        self.gcn = gcn(cheb_k)

        nn.init.xavier_normal_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
        nn.init.constant_(self.spatial_weight, 1.0)

    def forward(self, x, adj, node_embedding):
        x_g = self.gcn(x, adj)

        weights = torch.einsum('nd,dkio->nkio', node_embedding, self.weights_pool)
        bias = torch.matmul(node_embedding, self.bias_pool)

        x_g = x_g.permute(0, 2, 1, 3)

        spatial_attn = torch.softmax(self.spatial_weight, dim=0)
        x_g = x_g * spatial_attn.view(1, 1, -1, 1)

        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias

        return x_gconv

class LecunTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh(0.666 * x)

class CfcCell(nn.Module):
    def __init__(self, hidden_dim):
        super(CfcCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.backbone_activation = LecunTanh()

        self.backbone = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            self.backbone_activation,
            nn.Dropout(0.1)
        )

        self.ff1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            LecunTanh()
        )
        self.ff2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            LecunTanh()
        )
        self.time_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.time_b = nn.Linear(self.hidden_dim, self.hidden_dim)

        self._apply_weight_decay()

    def _apply_weight_decay(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs, hidden_state, elapsed_time=None):
        assert inputs.shape[-1] == 2 * self.hidden_dim, \
            f"CfcCell输入维度错误：期望{2*self.hidden_dim}，实际{inputs.shape[-1]}"
        assert hidden_state.shape[-1] == self.hidden_dim, \
            f"CfcCell隐藏状态维度错误：期望{self.hidden_dim}，实际{hidden_state.shape[-1]}"

        x = self.backbone(inputs)

        ff1 = self.ff1(x)
        ff2 = self.ff2(x)

        t = torch.tensor(5.0, device=inputs.device) if elapsed_time is None else elapsed_time
        t = t.expand(inputs.shape[0], 1)

        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = torch.sigmoid(-t_a * t + t_b)
        new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden

class GRUCFCCell(nn.Module):
    def __init__(self, hidden_dim):
        super(GRUCFCCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.cfc = CfcCell(hidden_dim)

        self.input_kernel = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.recurrent_kernel = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)
        nn.init.zeros_(self.recurrent_kernel.bias)

    def forward(self, inputs, states, elapsed_time=None):

        batch_size, num_nodes, _ = inputs.shape
        inputs_flat = inputs.reshape(-1, self.hidden_dim)
        hidden_state_flat = states.reshape(-1, self.hidden_dim)

        z = self.input_kernel(inputs_flat) + self.recurrent_kernel(hidden_state_flat)
        r, z, n = torch.split(z, self.hidden_dim, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)

        cfc_input = torch.cat([inputs_flat, r * hidden_state_flat], dim=-1)
        new_cfc_state_flat = self.cfc(cfc_input, hidden_state_flat, elapsed_time)

        new_hidden_flat = (1 - z) * hidden_state_flat + z * new_cfc_state_flat
        new_hidden = new_hidden_flat.reshape(batch_size, num_nodes, self.hidden_dim)

        return new_hidden

class PGCFCCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(PGCFCCell , self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = PDGCN(dim_in + self.hidden_dim, 2 * self.hidden_dim, cheb_k, embed_dim, time_dim)
        self.update = PDGCN(dim_in + self.hidden_dim, self.hidden_dim, cheb_k, embed_dim, time_dim)
        self.fc1 = FC(dim_in + self.hidden_dim, time_dim)
        self.fc2 = FC(dim_in + self.hidden_dim, time_dim)

        self.cfc = GRUCFCCell(self.hidden_dim)

    def forward(self, x, state, node_embeddings):
        state = state.to(x.device)

        input_and_state = torch.cat((x, state), dim=-1)
        filter1 = self.fc1(input_and_state)
        filter2 = self.fc2(input_and_state)

        nodevec1 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[0], filter1))
        nodevec2 = torch.tanh(torch.einsum('bd,bnd->bnd', node_embeddings[1], filter2))

        adj = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1))

        adj1 = self.preprocessing(F.relu(adj))
        adj2 = self.preprocessing(F.relu(-adj.transpose(-2, -1)))
        adj = [adj1, adj2]

        z_r = torch.sigmoid(self.gate(input_and_state, adj, node_embeddings[2]))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, adj, node_embeddings[2]))

        elapsed_time = torch.tensor(5.0, device=x.device)
        new_state = self.cfc(hc, state, elapsed_time)

        h = r * state + (1 - r) * new_state

        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    @staticmethod
    def preprocessing(adj):
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes).to(adj.device)
        x = torch.unsqueeze(adj.sum(-1), -1)
        adj = adj / x
        return adj
