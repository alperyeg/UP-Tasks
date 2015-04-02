#! /usr/bin/env python

import elephant as el
from quantities import Hz, ms
from neo import io
from active_worker.task import task


@task
def generate_poisson_spiketrains_task(rate, number_of_spiketrains, time):
    '''
        Task Manifest Version: 1
        Full Name: generate_poisson_spiketrains_task
        Caption: generate_poisson_spiketrains
        Author: Elephant_Developers
        Description: |
            Returns a spike train whose spikes are a realization of a Poisson
            process with the given rate, starting at time `t_start` and
            stopping time `t_stop`.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            rate: long
            number_of_spiketrains: long
            time: long

        Returns:
            res: application/unknown
    '''

    # Init result
    res = []

    for x in range(number_of_spiketrains):
        result = el.spike_train_generation.homogeneous_poisson_process(rate*Hz, t_stop=time*1000.0*ms)
        print "........", type(result)
        res.append(result)

    print res
    # print type(res[0])

    # Output result to HDF5
    output_file = 'generate_poisson_spiketrains.h5'
    iom = io.NeoHdf5IO(output_file)
    iom.write(res)

    return generate_poisson_spiketrains_task.task.uri.save_file(mime_type='application/x-hdf', src_path=output_file, dst_path=output_file)


if __name__ == '__main__':
    # ------ Input rate in Hz (integer), i.e. rate = 4 spikes/ s
    rate = 10

    # ------ Input number of spiketrains
    number_of_spiketrains = 100

    # ------ Input time in (s), convert it into t_stop in 1000ms
    time = 1000
    generate_poisson_spiketrains_task(rate, number_of_spiketrains, time)
