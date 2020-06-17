"""
Bonito Basecaller
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model
from bonito.bonito_io import DecoderWriter, PreprocessReader
from bonito.decode import LanguageModel

import torch
import numpy as np
from torch.multiprocessing import Queue, set_start_method
from bonito.lm import load_rnn_lm


def main(args):
    #if args.lmdevice == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_start_method('spawn')
    sys.stderr.write("> loading model\n")
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)

    samples = 0
    num_reads = 0
    max_read_size = 4e6
    dtype = np.float16 if args.half else np.float32
    reader = PreprocessReader(args.reads_directory)
    if args.lm and args.decoder == 'lm_rnn_pbs':
        lm = load_rnn_lm(args.lm, args.lmdevice)
    elif args.lm and args.decoder in ('r_pbs', 'py_pbs'):
        lm = LanguageModel(args.lm)
    else:
        lm = None
    posteriors_queue = Queue()

    t0 = time.perf_counter()

    processes = []
    for i in range(args.nprocs):
        p = DecoderWriter(posteriors_queue, model, beamsize=args.beamsize, fastq=args.fastq,
                          decoder=args.decoder, lm=lm, alpha=args.alpha, beta=args.beta, device=args.lmdevice,
                          analysis=args.analysis)
        processes.append(p)
        p.start()
    sys.stderr.write("> calling\n")

    with reader, torch.no_grad():

        while True:

            read = reader.queue.get()
            if read is None:
                for i in range(args.nprocs):
                    posteriors_queue.put('END')
                break

            read_id, raw_data = read

            if len(raw_data) > max_read_size:
                sys.stderr.write("> skipping long read %s (%s samples)\n" % (read_id, len(raw_data)))
                continue

            num_reads += 1
            samples += len(raw_data)

            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data = torch.tensor(raw_data).to(args.device)
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()

            posteriors_queue.put((read_id, posteriors.astype(np.float32)))

    for p in processes:
        p.join()

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
    parser.add_argument("reads_directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--decoder", type=str)
    parser.add_argument("--lm", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--fastq", action="store_true", default=False)
    parser.add_argument("--nprocs", default=4, type=int)
    parser.add_argument("--lmdevice", default="cuda")
    parser.add_argument("--analysis", action="store_true", help="output analysis csv")
    return parser
