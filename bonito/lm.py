import torch
from bonito.trainlm import RNN
import numpy as np
from bonito.trainlm import one_hot_encode
import torch.nn.functional as F


class RNNLanguageModel:
    def __init__(self, lm_net, device='cuda'):
        self.device = device
        self.net = lm_net
        self.clear()

    def init_search_points(self, nodes, parents, tip_labels, prefix_lens):
        try:
            inputs = []
            hiddens = []
            beam_probs = np.zeros((len(nodes), len(self.net.chars)))
            for beam_idx, node in enumerate(nodes):
                if node == -1:  # Root node
                    beam_probs[beam_idx] = self.suffix_tree[-1]['rnn_probs']
                elif node in self.suffix_tree:  # Nodes with hidden vector and rnn_probs already computed
                    beam_probs[beam_idx] = self.suffix_tree[parents[beam_idx]]['rnn_probs']
                else:
                    hiddens.append(self.suffix_tree[parents[beam_idx]]['hidden'])
                    x = one_hot_encode(np.array([[tip_labels[beam_idx]]]), len(self.net.chars))
                    inputs.append(torch.tensor(x, device=self.device))

            # Purge nodes that can no longer be reached
            shortest_prefix = min(prefix_lens)
            to_remove = [node for node, val in self.suffix_tree.items() if val['prefix_len'] < shortest_prefix and node not in parents]
            for node in to_remove:
                self.suffix_tree.pop(node)

            if inputs:
                # stack them and compute them
                p, h = self.forward(inputs, hiddens)

            p_idx = 0
            for beam_idx, node in enumerate(nodes):
                if node not in self.suffix_tree:
                    beam_probs[beam_idx] = p[p_idx, :]
                    self.suffix_tree[node] = {'hidden': h[:, p_idx, :],
                                              'rnn_probs': p[p_idx, :],
                                              'prefix_len': prefix_lens[beam_idx]}

            return beam_probs
        except Exception as e:
            print(e)

    def forward(self, inputs, hiddens):
        inputs = torch.cat(inputs, dim=1)
        hiddens = torch.stack(hiddens, dim=1)
        with torch.no_grad():
            out, h = self.net(inputs, hiddens)
            p = F.softmax(out, dim=1).detach()
            p = p.cpu().numpy()
        return p, h

    def clear(self):
        hidden_zero = self.net.init_hidden(1, self.device)[:, 0, :]
        input_zero = torch.zeros((1, 1, len(self.net.chars)), device=self.device)
        first_probs, first_hidden = self.forward([input_zero], [hidden_zero])
        self.suffix_tree = {-1: {'hidden': first_hidden[:, 0, :],
                                 'rnn_probs': first_probs[0, :],
                                 'prefix_len': 0}}



def load_rnn_lm(path, device):
    with open(path, 'rb') as f:
        checkpoint = torch.load(f)
    loaded = RNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])
    loaded.to(device)
    loaded.eval()
    return loaded
