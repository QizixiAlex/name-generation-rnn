import torch
import torch.nn.functional as F
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, category_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2o = nn.Linear(category_size+input_size+hidden_size, output_size)
        self.i2h = nn.Linear(category_size+input_size+hidden_size, hidden_size)
        self.o2o = nn.Linear(output_size+hidden_size, output_size)

    def forward(self, category, input, hidden):
        output = self.i2o(torch.cat((category, input, hidden), 1))
        output = self.o2o(torch.cat((output, hidden), 1))
        hidden = self.i2h(torch.cat((category, input, hidden), 1))
        output = F.log_softmax(F.dropout(output, p=0.1), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

