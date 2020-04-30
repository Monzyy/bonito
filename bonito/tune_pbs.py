"""
Tune Prefix beam-search parameters
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model
from bonito.bonito_io import HDF5Reader, TunerProcess
from sklearn.model_selection import GridSearchCV

import torch
import numpy as np


def main(args):

    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)

    samples = 0
    num_reads = 0
    max_read_size = 1e9
    dtype = np.float16 if args.half else np.float32
    reader = HDF5Reader(args.hdf5)
    tuner = TunerProcess(model.alphabet, args.beamsize, decoder='pbs',
                           lm=args.lm)

    t0 = time.perf_counter()
    sys.stderr.write("> calling\n")

    with tuner, reader, torch.no_grad():

        while True:

            read = reader.queue.get()
            if read is None:
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

            tuner.queue.put((read_id, posteriors, reference))
    avg_acc = sum(tuner.accuracies) / len(tuner.accuracies)

    duration = time.perf_counter() - t0

    sys.stderr.write("> completed reads: %s\n" % num_reads)
    sys.stderr.write("> samples per second %.1E\n" % (samples / duration))
    sys.stderr.write(f"> time elapsed: {duration} seconds\n")
    sys.stderr.write("> done\n")


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
    return parser