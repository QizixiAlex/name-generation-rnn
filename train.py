import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import init_data, load_data, NameDataset, all_categories, all_letters
from model import RNN


# parameters
n_hidden = 128
epochs = 100
plot_every = 1000


# init data
init_data()
# create model
rnn = RNN(len(all_categories), len(all_letters), n_hidden, len(all_letters))
# setup data
train_data = load_data()
train_dataset = NameDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# setup optimizer and criterion
optimizer = optim.Adam(rnn.parameters(), lr=0.0005)
criterion = nn.NLLLoss()
# train
all_loss = []
for epoch in range(epochs):
    rnn = rnn.train()
    current_loss = 0
    for idx, (category, name) in enumerate(train_dataloader):
        category, name = category[0], name[0]
        category, name = Variable(category), Variable(name)
        hidden = rnn.initHidden()
        for i in range(name.size()[0]-1):
            optimizer.zero_grad()
            output, hidden = rnn(category, name[i], hidden)
            loss = criterion(output, torch.argmax(name[i+1], dim=1).long())
            loss.backward(retain_graph=True)
            optimizer.step()
            current_loss += loss
        if idx >= plot_every and idx % plot_every == 0:
            all_loss.append(float(current_loss)/plot_every)
            current_loss = 0
    # save model
    torch.save(rnn, 'saved_model/char-rnn-generation.pt')
    print("epoch: ", str(epoch))

# plot all loss
plt.figure(0)
plt.title("loss")
plt.plot(all_loss)
plt.show()