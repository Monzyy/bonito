"""
Bonito Decoding functions
"""

from itertools import groupby
import re
import numpy as np
from fast_ctc_decode import beam_search
from collections import defaultdict, Counter
from string import ascii_uppercase

def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def greedy_ctc_decode(predictions, labels):
    """
    Greedy argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def decode(predictions, alphabet, beam_size=5, threshold=0.1):
    """
    Decode model posteriors to sequence
    """
    print(predictions[0])
    alphabet = ''.join(alphabet)
    if beam_size == 1:
        return greedy_ctc_decode(predictions, alphabet)
    return beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)

def prefix_beam_search(ctc_predictions, lm_path, prune=0.2, alpha=0.5, beta=0.05, k=10):
    lm = {}
    test_alphabet = ['A', 'C', 'G', 'T']
    with open(lm_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(';')
            lm[key] = value
            print("Key. " + key + " value: " + lm[key])

    temp_dict = {}
    for i in test_alphabet:
        for j in test_alphabet:
            for m in test_alphabet:
                temp_dict[i+j+m] = lm[i+j+m]

            sum = 0
            for values in temp_dict.values():
                sum += float(values)

            for n in test_alphabet:
                lm[i+j+n] = temp_dict[i+j+n] / float(sum)

            temp_dict = {}

    W = lambda l: re.findall(r'\w+[\s|>]', l)
    alphabet = ['A', 'C', 'G', 'T'] + [' ', '>', '%'] #list(ascii_uppercase) + [' ', '>', '%']
    F = ctc_predictions.shape[1]
    ctc = np.vstack((np.zeros(F), ctc_predictions))  # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]

        for l in A_prev:
            if len(l) > 0 and l[-1] == '>':
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                if c == '%':
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3

                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and c in (' ', '>'):

                        #Find all probabilities in lm for key and use highest probability
                        lm_values = []

                        for letter in ['A', 'C', 'G', 'T']:
                            print(l_plus.strip(' >')+letter)
                            lm_values.append(lm[l_plus.strip(' >')+letter])

                        lm_prob = float(max(lm_values)) ** float(alpha)
                        #lm_prob = lm[l_plus.strip(' >')] ** alpha
                        print("Language_model Probability: ")
                        print(lm_prob)
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        print("hej")
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]

                # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
    # END: STEP 7
    print("A_prev:")
    print(A_prev.__len__())
    return A_prev[0].strip('>')
