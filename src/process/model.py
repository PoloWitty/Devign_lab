from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import GCNConv, Sequential, GATConv
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

class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)

        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


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
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, max_nodes, device):
        super(Net, self).__init__()
        # self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device) 
        # self.gcn = Sequential(
        #     'x, edge_index',[
        #     (GCNConv(emb_size, 256), 'x, edge_index -> x'),
        #     nn.ReLU(inplace=True),
        #     (GCNConv(256, gated_graph_conv_args['out_channels']), 'x, edge_index -> x'),
        #     nn.ReLU(inplace=True)
        #     ]).to(device)
        self.gat = Sequential(
            'x, edge_index',[
            (GATConv(emb_size, 256), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(256, gated_graph_conv_args['out_channels']), 'x, edge_index -> x'),
            nn.ReLU(inplace=True)
            ]).to(device)
        
        self.emb_size=emb_size
        self.readout = Readout(emb_size,gated_graph_conv_args['out_channels']).to(device) 
        self.conv = Conv(**conv_args,
                    fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                    fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Process:
        # method 1)
        # x = self.ggc(x, edge_index)
        # method 2)
        # x = self.gcn(x,edge_index)
        # method 3)
        x = self.gat(x,edge_index)

        # Readout:
        # method 1)
        x = self.conv(x, data.x)
        # method 2)
        # x = self.readout(x, data.x, data.batch)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
