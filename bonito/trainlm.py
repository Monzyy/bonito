from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import re

alphabet = 'ACGT'

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


class RNN(nn.Module):
    def __init__(self, tokens, n_hidden, n_layers=1):
        super(RNN, self).__init__()
        self.n_layers = n_layers

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.chars = tokens

        self.gru = nn.GRU(len(self.chars), n_hidden, n_layers)
        self.out = nn.Linear(n_hidden, len(self.chars))
        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = output.contiguous().view(-1, self.n_hidden)
        output = self.out(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        print("inside init_hidden")
        try:
            test = next(self.parameters()).data
        except Exception as e:
            print(e)
        weight = next(self.parameters()).data
        
        #if (train_on_gpu):
        #    hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
        #          weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        #else:
        #    hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
        #              weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        if (train_on_gpu):
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
        else:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
        return hidden



def encode_ref(ref):
    return [alphabet.index(c) for c in ref]


def ref_to_tensor(ref):
    tensor = torch.zeros(len(ref), 1, len(alphabet))
    for idx, char in enumerate(ref):
        tensor[idx][0][alphabet.index(char)] = 1
    return tensor


def ref_to_target_tensor(ref):
    tensor = torch.zeros(len(ref), len(alphabet), dtype=torch.long)
    for idx, char in enumerate(ref):
        tensor[idx][alphabet.index(char)] = 1
    return tensor

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


def get_samples(references, batch_size, seq_length):
    batch_size_total = batch_size * seq_length

    arr = []
    
    xs = []
    ys = []
    for ref in references:
        for i in range(0, len(ref) - seq_length, seq_length):
            xs.append(ref[i:i + seq_length])
            j = i + 1
            ys.append(ref[j:j + seq_length])
    xs = np.array(xs)
    ys = np.array(ys)
    for n in range(0, xs.shape[0] - batch_size, batch_size):
        yield xs[n:n + batch_size, :], ys[n:n + batch_size, :]

def train(net, data, epochs=10, batch_size=10, seq_length=200, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if(train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)

    for e in range(epochs):

        h = net.init_hidden(batch_size)

        for x, y in get_samples(data, batch_size, seq_length):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            inputs = inputs.transpose(1, 0)
            targets = targets.transpose(1, 0)
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            #h = tuple([each.data for each in h])
            #h = net.init_hidden(batch_size)
            h = h.detach()

            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),clip)
            opt.step()

            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_samples(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    #val_h = tuple([each.data for each in val_h])
                    #val_h = net.init_hidden(batch_size)
                    val_h = val_h.detach()
                    
                    inputs, targets = x, y
                    inputs = inputs.transpose(1, 0)
                    targets = targets.transpose(1, 0)
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length))
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
                      
def is_actg(ref):
    pattern = r'[^\.ACTG]'
    return not bool(re.search(pattern, ref))


def main(args):
    source = open(args.source, 'r')
    references = []
    ref = ''
    for line in source:
        if line.startswith('>'):
            if ref:
                if is_actg(ref):
                    references.append(ref)
                ref = ''
            continue
        ref += line.strip()
    if is_actg(ref):
        references.append(ref)
    source.close()


    encoded = np.array([encode_ref(ref) for ref in references])

    n_letters = len(alphabet)
    n_hidden = 256
    net = RNN(alphabet, n_hidden, 1)
    print(net)
    
    batch_size = 2
    seq_length = 500 #max length verses
    n_epochs = 100 # start smaller if you are just testing initial behavior

    # train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)
    

    checkpoint = {'n_hidden': net.n_hidden,
                'n_layers': net.n_layers,
                'state_dict': net.state_dict(),
                'tokens': net.chars}

    with open(args.dest, 'wb') as f:
        torch.save(checkpoint, f)
    


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("source")
    parser.add_argument("dest")

    return parser
