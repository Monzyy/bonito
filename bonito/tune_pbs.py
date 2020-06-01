"""
Tune Prefix beam-search parameters
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.multiprocessing import Queue, set_start_method

from bonito.util import load_model
from bonito.bonito_io import HDF5Reader, TunerProcess
from bonito.decode import LanguageModel
from bonito.lm import load_rnn_lm
from hyperopt import hp, fmin, tpe
import queue

import torch
import numpy as np


def main(args):
    if args.lmdevice == 'cuda':
        torch.backends.cudnn.enabled = True
    set_start_method('spawn')
    space = [hp.uniform('alpha', 0, 2), hp.uniform('beta', 0, 2), args]

    sys.stderr.write("> loading model\n")

    best = fmin(lambda x: -1 * objective(x), space, algo=tpe.suggest, max_evals=25)
    print(best)


def objective(args):
    alpha, beta, args = args
    print(f'basecalling with alpha: {alpha} beta: {beta}')
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)
    if args.lm and args.decoder == 'lm_rnn_pbs':
        lm = load_rnn_lm(args.lm, args.lmdevice)
    else:
        lm = LanguageModel(args.lm)
    if args.lmdevice == 'cuda':
        torch.backends.cudnn.enabled = True
    samples = 0
    num_reads = 0
    max_read_size = 1e9
    dtype = np.float16 if args.half else np.float32
    reader = HDF5Reader(args.hdf5)
    n_decoder_processes = args.nprocs
    processes = []
    posteriors_queue = Queue()
    output_queue = Queue()

    t0 = time.perf_counter()
    sys.stderr.write("> calling\n")

    for i in range(n_decoder_processes):
        p = TunerProcess(posteriors_queue, output_queue, model, args.beamsize,
                         decoder=args.decoder, lm=lm, alpha=alpha, beta=beta, device=args.lmdevice)
        processes.append(p)
        p.start()
    print('Decoder processes started')

    with reader, torch.no_grad():

        while True:

            read = reader.queue.get()
            if read is None:
                # Add n_decoder_processes END messages such that each process can read an END message
                for i in range(n_decoder_processes):
                    posteriors_queue.put('END')
                break

            read_id, raw_data, reference = read

            if len(raw_data) > max_read_size:
                sys.stderr.write("> skipping %s: %s too long\n" % (len(raw_data), read_id))
                pass

            num_reads += 1
            samples += len(raw_data)

            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data = torch.tensor(raw_data).to(args.device)
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()

            posteriors_queue.put((read_id, posteriors, reference))
    print('all posteriors done')

    accuracies = []
    for i in range(n_decoder_processes):
        while True:
            try:
                accuracy = output_queue.get()
            except queue.Empty:
                continue
            if accuracy == 'END':
                break
            accuracies.append(accuracy)

    for p in processes:
        p.join()

    avg_acc = sum(accuracies) / len(accuracies)
    print(f'average accuracy: {avg_acc}')
    sys.stderr.write(f"> time elapsed: {time.perf_counter() - t0} seconds\n")
    return avg_acc


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("hdf5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--lm", type=str)
    parser.add_argument("--decoder", type=str)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--lmdevice", default="cuda")
    return parser
