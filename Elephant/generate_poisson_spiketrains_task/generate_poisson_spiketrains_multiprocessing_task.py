#! /usr/bin/env python

import elephant as el
from quantities import Hz, ms
from neo import io
# from active_worker.task import task
import multiprocessing as mp
# import logging


# many child-processes are running as initiated by number_of_cpucores
def worker(q, rate, time):
    # print "Process worker = ", mp.current_process().name
    result = el.spike_train_generation.homogeneous_poisson_process(rate*Hz, t_stop=time*1000.0*ms)
    q.put(result)
    # print "Result is being pushed into Queue = ", result
    print "Queue size worker = ", q.qsize()
    return result


def start_process():
    print "Starting ", mp.current_process().name


def generate_poisson_spiketrains_task(rate, number_of_spiketrains, time):

    # Output-hdf5_file
    output = 'generate_poisson_spiketrains.h5'

    f = io.NeoHdf5IO(output)

    # Init manager and Queue
    manager = mp.Manager()
    q = manager.Queue()

    # Use all available CPU_Cores
    pool_size = mp.cpu_count()
    print "pool_size = ", pool_size
    pool = mp.Pool(pool_size, initializer=start_process,
                   maxtasksperchild=2)

    # init workers: calculate function and put result into queue
    for x in range(number_of_spiketrains):
        # pool.apply_async returns a proxy object immediately
        result = pool.apply_async(worker, (q, rate, time))
        # proxy.get() waits for task completion and returns the result
        print "Result = ", result.get()
        # Dequeue and write content to HDF5 concurrently using main_process
        f.write(result.get())
        # res.append(result)

    print "Queue size = ", q.qsize()

    f.close()

    print " -------------------- BEGIN TERMINATED ------------------- "
    print "Main process = ", mp.current_process().name
    pool.close()
    pool.join()
    print " -------------------- TERMINATED ------------------- "


if __name__ == '__main__':
    # ------ Input rate in Hz (integer), i.e. rate = 4 spikes/ s
    rate = 100

    # ------ Input number of spiketrains
    number_of_spiketrains = 1000

    # ------ Input time in (s), convert it into t_stop in 1000ms
    time = 1000
    generate_poisson_spiketrains_task(rate, number_of_spiketrains, time)
