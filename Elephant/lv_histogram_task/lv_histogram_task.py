#! /usr/bin/env python
__author__ = "Long Phan, INM-6, FZJ"

import elephant as el
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neo import io
from active_worker.task import task
from task_types import TaskTypes as tt


@task
def lv_histogram_task(input_data):
    '''
        Task Manifest Version: 1
        Full Name: lv_histogram_task
        Caption: lv_histogram
        Author: Elephant_Developers
        Description: |
            Calculate the measure of local variation LV for a sequence of time
            intervals between events.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            input_data: application/unknown

        Returns:
            res: image/png
    '''

    # stage the input file
    original_path = lv_histogram_task.task.uri.get_file(input_data)

    # open file from path using Neo
    ion = io.NeoHdf5IO(filename=original_path)
    number_of_SpikeTrains = ion.get_info()['SpikeTrain']

    # Init result
    res = []

    # Query data from hdf5-file
    for k in range(number_of_SpikeTrains):
        poisson_data = ion.get("/"+"SpikeTrain_"+str(k))

        # Calculate value for lv
        lv_data = el.statistics.isi(poisson_data)
        lv = el.statistics.lv(lv_data)
        print "LV = ", lv

        # limiting floats to three decimal points
        sub_res = round(lv, 3)

        # save result
        res.append(sub_res)

    # Close remaining file h5
    ion.close()

    print res

    # Create plot
    plt.title("LV Histogram")
    plt.xlabel("LV")
    plt.ylabel("")

    # Plotting an histogram
    plt.grid(True)
    plt.hist(res, bins=100, normed=1, histtype='bar', rwidth=1)
    # plt.show()

    output_file = 'result_lv_histogram_task.png'
    with open(output_file, 'w') as output:
        plt.savefig(output)
    return lv_histogram_task.task.uri.save_file(mime_type='image/png',
                                                src_path=output_file,
                                                dst_path=output_file)


if __name__ == '__main__':

    # Run local test
    filename = tt.URI('application/unknown', 'generate_poisson_spiketrains.h5')
    lv_histogram_task(filename)
