import torch
import torch.nn as nn

class MLP(nn.Module):

    search_space = []
    scaler = None

    def __init__(self, n_i, n_1, n_2, n_3, n_o, search_space):
        super(MLP, self).__init__()
        self.linear_in = nn.Linear(n_i, n_1)
        self.linear_hidden1 = nn.Linear(n_1, n_2)
        self.linear_hidden2 = nn.Linear(n_2, n_3)
        #self.linear_hidden3 = nn.Linear(n_h, n_h)
        self.linear_out = nn.Linear(n_3, n_o)
        self.search_space = search_space
 
    def forward(self, x):
        x = self.linear_in(x).relu()
        x = self.linear_hidden1(x).relu()
        x = self.linear_hidden2(x).relu()
        x = self.linear_out(x).sigmoid()
        return x

polynomial_n = 4