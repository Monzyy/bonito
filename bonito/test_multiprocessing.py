from multiprocessing import Process, Queue, current_process
from itertools import product
import queue
import random


class Reader(Process):
    """
    Reader Processor that reads and processes hdf5 files
    """
    def __init__(self, maxsize, tasks):
        super().__init__()
        self.queue = Queue(maxsize)
        self.tasks = tasks
        self.start_gram = 5

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        for gram in range(self.start_gram, self.start_gram+self.tasks):
            print(f'{gram} added to queue')
            self.queue.put(gram)
        self.queue.put(None)

    def stop(self):
        self.join()
        print('reader stopped')


class Writer(Process):
    """
    Decoder Process that writes fasta records to stdout
    """

    def __init__(self, posterior_queue, output_queue):
        super().__init__()
        self.queue = posterior_queue
        self.output_queue = output_queue

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
            if job == 'END':
                self.output_queue.put('END')
                return
            gram = job
            print(f'{gram} popped from queue by {current_process().name}')
            for n_gram in (''.join(x) for x in product('ACTG', repeat=12)):
                continue
            print(f'{gram} done by {current_process().name}')
            self.output_queue.put(random.randrange(80, 90))

    def stop(self):
        self.join()
        print(f'{current_process().name} stopped')


if __name__ == '__main__':
    n_tasks = 10
    num_processes = 4
    reader = Reader(maxsize=2, tasks=n_tasks)
    posteriors_queue = Queue()
    output_queue = Queue()
    processes = []

    for w in range(num_processes):
        p = Writer(posteriors_queue, output_queue)
        processes.append(p)
        p.start()

    with reader:
        while True:
            read = reader.queue.get()
            if read is None:
                for i in range(num_processes):
                    posteriors_queue.put('END')
                break

            # Calculate posteriors

            posteriors_queue.put(read)

    accuracies = []

    for i in range(num_processes):
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
    print(avg_acc)

