"""
Bonito Input/Output
"""

import os
import sys
from glob import glob
from textwrap import wrap
from logging import getLogger
from multiprocessing import Process, Queue
import queue

from tqdm import tqdm

from bonito.decode import decode_sequence
from bonito.util import get_raw_data, get_raw_hdf5_data, accuracy


logger = getLogger('bonito')


def write_fasta(header, sequence, fd=sys.stdout, maxlen=100):
    """
    Write a fasta record to a file descriptor.
    """
    fd.write(">%s\n" % header)
    fd.write("%s\n" % os.linesep.join(wrap(sequence, maxlen)))
    fd.flush()


def write_fastq(header, sequence, qstring, fd=sys.stdout):
    """
    Write a fastq record to a file descriptor.
    """
    fd.write("@%s\n" % header)
    fd.write("%s\n" % sequence)
    fd.write("+\n")
    fd.write("%s\n" % qstring)
    fd.flush()

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
    def __init__(self,  model, fastq=False, beamsize=5, wrap=100, decoder=None, lm=None, alpha=None, beta=None):
        super().__init__()
        self.queue = Queue()
        self.model = model
        self.wrap = wrap
        self.fastq = fastq
        self.beamsize = beamsize
        self.decoder = decoder
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
            read_id, predictions = job
            sequence, path = self.model.decode(
                predictions, beamsize=self.beamsize, qscores=self.fastq, return_path=True,
                decoder=self.decoder, **self.kwargs
            )
            if sequence:
                if self.fastq:
                    write_fastq(read_id, sequence[:len(path)], sequence[len(path):])
                else:
                    write_fasta(read_id, sequence, maxlen=self.wrap)
            else:
                logger.warn("> skipping empty sequence %s", read_id)

    def stop(self):
        self.queue.put(None)
        self.join()


class HDF5Reader(Process):
    """
    Reader Processor that reads and processes hdf5 files
    """
    def __init__(self, hdf5, maxsize=14):
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
    def __init__(self, posterior_queue, output_queue,
                 model, fastq=False, beamsize=5, wrap=100, decoder=None, lm=None, alpha=None, beta=None):
        super().__init__()
        self.queue = posterior_queue
        self.output_queue = output_queue
        self.model = model
        self.wrap = wrap
        self.fastq = fastq
        self.beamsize = beamsize
        self.decoder = decoder
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
            try:
                job = self.queue.get_nowait()
            except queue.Empty:
                continue
            else:
                if job == 'END':
                    self.output_queue.put('END')
                    return
                read_id, predictions, reference = job
                sequence, path = self.model.decode(
                    predictions, beamsize=self.beamsize, qscores=self.fastq, return_path=True,
                    decoder=self.decoder, **self.kwargs
                )
                # filter away too long or short
                if len(sequence) < 4000 or len(sequence) > 5000:
                    continue

                acc = accuracy(decode_sequence(reference, self.model.alphabet[1:]), sequence)
                self.output_queue.put(acc)

    def stop(self):
        self.join()
