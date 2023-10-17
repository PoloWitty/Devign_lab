from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import global_add_pool
import pdb

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

class Readout(nn.Module):
    def __init__(self, emb_size, hidden_dim):
        super(Readout, self).__init__()
        self.hidden_dim = hidden_dim # the out_channel from GatedGraphConv
        self.emb_size = emb_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim+emb_size,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
        )
        self.out_fn = nn.Linear(64,1)
        self.act_fn = nn.Sigmoid()

    def forward(self, h, x, batch):
        feat = torch.concat([h,x],dim=-1)
        feat = self.mlp(feat)
        global_feat = global_add_pool(feat,batch)
        global_feat = self.act_fn(self.out_fn(global_feat).squeeze())
        return global_feat

class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, max_nodes, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device) 
        self.emb_size=emb_size
        self.readout = Readout(emb_size,gated_graph_conv_args['out_channels']).to(device) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.ggc(x, edge_index)
        x = self.readout(x, data.x, data.batch)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
