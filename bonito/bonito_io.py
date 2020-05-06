"""
Bonito Input/Output
"""

import os
import sys
from glob import glob
from textwrap import wrap
from multiprocessing import Process, Queue

from tqdm import tqdm
import mappy
from bonito.decode import decode, prefix_beam_search, prefix_beam_search_parallel, decode_ref
from bonito.util import get_raw_data, get_raw_hdf5_data, accuracy, alt_calc_read_length_accuracy


class PreprocessReader(Process):
    """
    Reader Processor that reads and processes fast5 files
    """
    def __init__(self, directory, maxsize=5):
        super().__init__()
        self.directory = directory
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        for fast5 in tqdm(glob("%s/*fast5" % self.directory), ascii=True, ncols=100):
            for read_id, raw_data in get_raw_data(fast5):
                self.queue.put((read_id, raw_data))
        self.queue.put(None)

    def stop(self):
        self.join()


class DecoderWriter(Process):
    """
    Decoder Process that writes fasta records to stdout
    """
    def __init__(self, alphabet, beamsize=5, wrap=100, decoder=None, lm=None, alpha=None, beta=None):
        super().__init__()
        self.queue = Queue()
        self.wrap = wrap
        self.beamsize = beamsize
        self.alphabet = ''.join(alphabet)
        if decoder == 'pbs':
            self.decode = prefix_beam_search
        elif decoder == 'pbsp':
            self.decode = prefix_beam_search_parallel
        else:
            self.decode = decode
        self.kwargs = {}
        for k, v in (('lm', lm), ('alpha', alpha), ('beta', beta)):
            if v is not None:
                self.kwargs[k] = v

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions = job
            sequence = self.decode(predictions, self.alphabet, self.beamsize, **self.kwargs)

            sys.stdout.write(">%s\n" % read_id)
            sys.stdout.write("%s\n" % os.linesep.join(wrap(sequence, self.wrap)))
            sys.stdout.flush()

    def stop(self):
        self.queue.put(None)
        self.join()


class HDF5Reader(Process):
    """
    Reader Processor that reads and processes hdf5 files
    """
    def __init__(self, hdf5, maxsize=5):
        super().__init__()
        self.hdf5 = hdf5
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        for read_id, raw_data, reference in get_raw_hdf5_data(self.hdf5):
            self.queue.put((read_id, raw_data, reference))
        self.queue.put(None)

    def stop(self):
        self.join()


class TunerProcess(Process):
    """
    Decoder Process that writes fasta records to stdout
    """

    def __init__(self, alphabet, beamsize=5, wrap=100, decoder=None, lm=None, alpha=None, beta=None):
        super().__init__()
        self.queue = Queue()
        self.wrap = wrap
        self.beamsize = beamsize
        self.alphabet = ''.join(alphabet)
        if decoder == 'pbs':
            self.decode = prefix_beam_search
        elif decoder == 'pbsp':
            self.decode = prefix_beam_search_parallel
        else:
            self.decode = decode
        self.kwargs = {}
        for k, v in (('lm', lm), ('alpha', alpha), ('beta', beta)):
            if v is not None:
                self.kwargs[k] = v
        self.accuracies = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions, reference = job
            sequence = self.decode(predictions, self.alphabet, self.beamsize, **self.kwargs)
            # measure basecalling accuracy here
            if 4000 <= len(sequence) <= 5000:
                acc = alt_calc_read_length_accuracy(read_id, (decode_ref(reference, self.alphabet), sequence))
                self.accuracies.append(acc)
            sys.stdout.write(">%s\n" % read_id)
            sys.stdout.write("%s\n" % os.linesep.join(wrap(sequence, self.wrap)))
            sys.stdout.flush()

    def stop(self):
        self.queue.put(None)
        self.join()