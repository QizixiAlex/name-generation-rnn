import sys
import random
import torch
from torch.autograd import Variable
from data import eof, all_letters, all_categories, name_to_tensor, tensor_to_letter, init_data, category_to_tensor

init_data()
rnn = torch.load('saved_model/char-rnn-generation.pt')
rnn = rnn.eval()


def generate(category, name_start):
    name = name_start
    if category not in all_categories:
        category = random.choice(all_categories)
    category_tensor = category_to_tensor(category)
    name_tensor = Variable(name_to_tensor(name_start))
    hidden = rnn.initHidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(category_tensor, name_tensor[i], hidden)
    eof_tensor = name_to_tensor(eof)[0]
    max_name_len = 10
    name_len = len(name_start)
    while torch.argmax(output) != torch.argmax(eof_tensor) and name_len <= max_name_len:
        name += tensor_to_letter(output)
        name_len += 1
        prev_output = output
        output = torch.zeros(1, len(all_letters))
        output[0][torch.argmax(output, dim=1)] = 1
        output, hidden = rnn(category_tensor, output, hidden)
        if torch.argmax(output) == torch.argmax(prev_output):
            break
    return name.title()


if __name__ == '__main__':
    generated_name = generate('names\\'+sys.argv[1], sys.argv[2].lower())
    print(generated_name)