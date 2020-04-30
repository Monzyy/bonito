"""
Bonito Decoding functions
"""

from itertools import groupby
import re
import numpy as np
from fast_ctc_decode import beam_search
from collections import defaultdict, Counter
import time
from string import ascii_lowercase
import csv
import sys
from joblib import Parallel, delayed

OOV_SCORE = 0 # ~ log(1)


class LanguageModel:
    def __init__(self, lm_path, is_character_based=True):
        self.log_cond_probs = {}
        self.setup(lm_path)
        self._is_character_based = is_character_based
        self.n_gram_length = len(list(self.log_cond_probs.keys())[0])  # Unsafe
        self.vocab_miss = 0
        self.vocab_hit = 0

    def setup(self, lm_path):
        word_idx = 0
        log_prob_idx = 1

        with open(lm_path, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                self.log_cond_probs[line[word_idx]] = float(line[log_prob_idx])

    def get_log_cond_prob(self, word):
        if len(word) > self.n_gram_length:
            n_gram = word[-self.n_gram_length:]
        else:
            n_gram = word
        if self.log_cond_probs.get(n_gram) is not None:
            self.vocab_hit += 1
            return self.log_cond_probs[n_gram]
        else:
            self.vocab_miss += 1
            return OOV_SCORE

    def is_character_based(self):
        return self._is_character_based


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
    alphabet = ''.join(alphabet)
    if beam_size == 1:
        return greedy_ctc_decode(predictions, alphabet)
    return beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)


def prefix_beam_search(ctc, alphabet, beam_size=25, threshold=0.1, lm=None, alpha=1, beta=1):
    """
    Performs prefix beam search on the output of a CTC network.
    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        lm (LanguageModel): Language model. lm.get_log_cond_prob(word)
        should return the log conditional probability for the word
        beam_size (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        threshold (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
    Returns:
        string: The decoded CTC output.
    """

    lm = LanguageModel(lm) if lm else None
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    blank_idx = 0  # The blank character is the first character
    F = ctc.shape[1]
    ctc = np.log(np.vstack((np.zeros(F), ctc)))  # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]
    log_threshold = np.log(threshold)

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 0.0  # log(1)
    Pnb[0][O] = float('-inf')  # log(0)
    A_prev = [O]
    # END: STEP 1

    t_time = time.time()
    # STEP 2: Iterations and pruning
    for t in range(1, T):

        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > log_threshold)[0]]
        for l in A_prev:

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)

                # Extending with a blank
                if c == 'N':
                    Pb[t][l] = log_sum_exp(
                        Pb[t][l],
                        ctc[t][blank_idx] + log_sum_exp(
                            Pb[t - 1][l],
                            Pnb[t - 1][l]
                        )
                    )
                    #Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])

                else:
                    l_plus = l + c
                    # Extending with a repeated character
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] = log_sum_exp(
                            Pnb[t][l_plus],
                            ctc[t][c_ix] + Pb[t - 1][l]
                        )
                        #Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]

                        Pnb[t][l] = log_sum_exp(
                            Pnb[t][l],
                            ctc[t][c_ix] + Pnb[t - 1][l]
                        )
                        #Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]

                    # Extend with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and (c in (' ', '>') or (lm and lm.is_character_based())):
                        lm_log_prob = lm.get_log_cond_prob(l_plus.strip(' >')) * alpha
                        Pnb[t][l_plus] = log_sum_exp(
                            Pnb[t][l_plus],
                            lm_log_prob + ctc[t][c_ix] + log_sum_exp(
                                Pb[t - 1][l],
                                Pnb[t - 1][l]
                            )
                        )
                        #Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        # Will be called in a word based language model for normal characters
                        #if t == 190:
                        #    print()
                        Pnb[t][l_plus] = log_sum_exp(
                            Pnb[t][l_plus],
                            ctc[t][c_ix] + log_sum_exp(
                                Pb[t - 1][l],
                                Pnb[t - 1][l]
                            )
                        )
                        #Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])

                    # Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pnb[t - 1][l] = log_sum_exp(
                            Pnb[t - 1][l],
                            ctc[t][blank_idx] + log_sum_exp(
                                Pb[t - 1][l_plus],
                                Pnb[t - 1][l_plus]
                            )
                        )
                        #Pnb[t - 1][l] += ctc[t][blank_idx] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])

                        Pnb[t][l_plus] = log_sum_exp(
                            Pnb[t][l_plus],
                            ctc[t][c_ix] + Pnb[t - 1][l_plus]
                        )
                        #Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]

        # Select most probable prefixes
        A_next = Pb[t] + Pnb[t]

        sorter = lambda l: A_next[l] + (len(l) + 1) * beta
        #sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:beam_size]
        if t >= 2:
            Pb.pop(t-2)
            Pnb.pop(t-2)

        sys.stderr.write(f'\r{1/(time.time()-t_time)}tps')
        t_time = time.time()
        sys.stderr.flush()

    if lm:
        sys.stderr.write(f'Vocabulary hits: {lm.vocab_hit}  Vocabulary misses: {lm.vocab_miss} '
                         f'Hit percentage: {(100 * (lm.vocab_hit/(lm.vocab_hit + lm.vocab_miss)))}\%')
    return A_prev[0].strip('>')


def log_sum_exp(*log_probs):
    max = np.max(log_probs)
    ds = log_probs - max
    sum_of_exp = np.exp(ds).sum()
    return max + np.log(sum_of_exp)


def extend_beam(pruned_alphabet, l, Pb_prev, Pnb_prev, ctc_t, A_prev, lm, blank_idx, alphabet, alpha):
    Pb_t = defaultdict(lambda: float('-inf'))
    Pnb_t = defaultdict(lambda: float('-inf'))
    for c in pruned_alphabet:
        c_ix = alphabet.index(c)

        # Extending with a blank
        if c == 'N':
            Pb_t[l] = log_sum_exp(
                Pb_t[l],
                ctc_t[blank_idx] + log_sum_exp(
                    Pb_prev[l],
                    Pnb_prev[l]
                )
            )

        else:
            l_plus = l + c
            # Extending with a repeated character
            if len(l) > 0 and c == l[-1]:
                Pnb_t[l_plus] = log_sum_exp(
                    Pnb_t[l_plus],
                    ctc_t[c_ix] + Pb_prev[l]
                )

                Pnb_t[l] = log_sum_exp(
                    Pnb_t[l],
                    ctc_t[c_ix] + Pnb_prev[l]
                )

            # Extend with any other non-blank character and LM constraints
            elif len(l.replace(' ', '')) > 0 and (c in (' ', '>') or (lm and lm.is_character_based())):
                lm_log_prob = lm.get_log_cond_prob(l_plus.strip(' >')) * alpha
                Pnb_t[l_plus] = log_sum_exp(
                    Pnb_t[l_plus],
                    lm_log_prob + ctc_t[c_ix] + log_sum_exp(
                        Pb_prev[l],
                        Pnb_prev[l]
                    )
                )
            else:
                Pnb_t[l_plus] = log_sum_exp(
                    Pnb_t[l_plus],
                    ctc_t[c_ix] + log_sum_exp(
                        Pb_prev[l],
                        Pnb_prev[l]
                    )
                )

            # Make use of discarded prefixes
            if l_plus not in A_prev:
                Pnb_prev[l] = log_sum_exp(
                    Pnb_prev[l],
                    ctc_t[blank_idx] + log_sum_exp(
                        Pb_prev[l_plus],
                        Pnb_prev[l_plus]
                    )
                )

                Pnb_t[l_plus] = log_sum_exp(
                    Pnb_t[l_plus],
                    ctc_t[c_ix] + Pnb_prev[l_plus]
                )
    return dict(Pb_t), dict(Pnb_t)


def prefix_beam_search_parallel(ctc, alphabet, beam_size=25, threshold=0.1, lm=None, alpha=2.0, beta=1.5, n_jobs=8):
    lm = LanguageModel(lm) if lm else None
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    blank_idx = 0  # The blank character is the first character
    F = ctc.shape[1]
    ctc = np.log(np.vstack((np.zeros(F), ctc)))  # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]
    log_threshold = np.log(threshold)

    # STEP 1: Initiliazation
    O = ''
    default_log_zero = lambda: float('-inf')
    Pb_prev, Pnb_prev = defaultdict(default_log_zero), defaultdict(default_log_zero)
    Counter()
    Pb_prev[O] = np.log(1)
    A_prev = [O]

    with Parallel(n_jobs) as parallel:
        t_time = time.time()
        for t in range(1, T):
            pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > log_threshold)[0]]
            results = parallel(delayed(extend_beam)(pruned_alphabet, l, Pb_prev, Pnb_prev, ctc[t], A_prev, lm, blank_idx, alphabet, alpha) for l in A_prev)
            Pb_prev.clear()
            Pb_prev.update(merge_dict_list([r[0] for r in results]))
            Pnb_prev.clear()
            Pnb_prev.update(merge_dict_list([r[1] for r in results]))

            # Select most probable prefixes
            A_next = {**Pb_prev, **Pnb_prev}

            sorter = lambda l: A_next[l] + (len(l) + 1) * beta
            # sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
            A_prev = sorted(A_next, key=sorter, reverse=True)[:beam_size]

            sys.stderr.write(f'\r{1 / (time.time() - t_time)}tps')
            t_time = time.time()
            sys.stderr.flush()

def merge_dict_list(dicts):
    res = {}
    for d in dicts:
        res = {**res, **d}
    return res
