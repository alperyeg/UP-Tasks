#! /usr/bin/env python
__author__ = "Long Phan, INM-6, FZJ"

import elephant as el
from quantities import Hz, ms
from neo import io
# from active_worker.task import task
import multiprocessing as mp
import threading
import logging
# import time
# from multiprocessing import Pool
# from Queue import Empty


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-2s) %(message)s')
output = 'generate_poisson_spiketrains.h5'

# Use all available CPU_Cores
pool_size = mp.cpu_count() * 2
print "pool_size = ", pool_size

f = io.NeoHdf5IO(output)


# class PoolActive(object):
#
#     # def __init__(self):
#     #     self.active = []   # show status of thread
#     #     self.lock = threading.Lock()
#     #     self.write_active = []
#     #     self.write_lock = threading.Lock()
#
#     def makeActive(self, name):
#         with self.lock:
#             self.active.append(name)
#             logging.debug('makeActive is running: %s ', self.active)
#
#     def makeInactive(self, name):
#         with self.lock:
#             self.active.remove(name)
#             logging.debug('thread %s is removed, ActiveList %s', name,
#                           self.active)
#
#     # def makeActive_write(self, name):
#     #     with self.write_lock:
#     #         self.write_active.append(name)
#     #         logging.debug('Active_write is running: %s ', self.write_active)
#
#     # def makeInactive_write(self, name):
#     #     with self.write_lock:
#     #         self.write_active.remove(name)
#     #         logging.debug('Inactive_write is running: %s ', self.write_active)


def _write_hdf5(local_data):
    # print local_data.value
    f.write(local_data.value)


# many child-processes are running as initiated by number_of_cpucores
def worker(s, local_data, write_lock, rate, time):
    logging.debug('Waiting for lock')
    with s:
        # name = threading.currentThread().getName()
        # pool.makeActive(name)
        result = el.spike_train_generation.homogeneous_poisson_process(rate*Hz, t_stop=time*1000.0*ms)
        # pool.makeInactive(name)
        # print "Result = ", result
        # pool.makeActive_write(name)
        write_lock.acquire()
        try:
            print "Write result to HDF5 ...."
            # print "type of result = ", type(result)
            local_data.value = result
            # _write_hdf5(local_data)
            print local_data.value
            print "type of local_data = ", type(local_data)
            print "Lock acquired ..."
            f.write(local_data.value)
        finally:
            print "Release lock ..."
            write_lock.release()
        # pool.makeInactive_write(name)
        # f.write(result)

    # return result


def generate_poisson_spiketrains_task(rate, number_of_spiketrains, time):

    # Init result and output-hdf5_file
    # res = []

    # Init number of Semaphore
    s = threading.Semaphore(pool_size)

    local_data = threading.local()

    write_lock = threading.Lock()

    # pool = PoolActive()
    for i in range(number_of_spiketrains):
        t = threading.Thread(target=worker, name=str(i), args=(s, local_data,
                                                               write_lock,
                                                               rate,
                                                               time))
        t.start()

    # ----------- Joining , wait for all thread done ---------
    logging.debug('Waiting for worker threads')
    main_thread = threading.currentThread()
    print "Threading enumerate ", threading.enumerate()
    for t in threading.enumerate():
        if t is not main_thread:
            print "t = ", t.getName()
            t.join()
    logging.debug('TASK DONE!')

    f.close()


if __name__ == '__main__':
    # ------ Input rate in Hz (integer), i.e. rate = 4 spikes/ s
    rate = 100

    # ------ Input number of spiketrains
    number_of_spiketrains = 1500

    # ------ Input time in (s), convert it into t_stop in 1000ms
    time = 1000
    generate_poisson_spiketrains_task(rate, number_of_spiketrains, time)
