#! /usr/bin/env python

import elephant as el
import matplotlib.pyplot as plt
from neo import io
from active_worker.task import task


@task
def cv_histogram_task(input_data):
    '''
        Task Manifest Version: 1
        Full Name: cv_histogram_task
        Caption: cv_histogram
        Author: Elephant_Developers
        Description: |
            Computes the coefficient of variation, the ratio of the biased
            standard deviation to the mean.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            input_data: application/unknown

        Returns:
            res: image/png
    '''

    print input_data
    ion = io.NeoHdf5IO(input_data)
    number_of_SpikeTrains = ion.get_info()['SpikeTrain']

    # Init result
    res = []

    # Query data from hdf5-file
    for k in range(number_of_SpikeTrains):
        poisson_data = ion.get("/"+"SpikeTrain_"+str(k))

        # Calculate value for cv
        cv_data = el.statistics.isi(poisson_data)
        cv = el.statistics.cv(cv_data)
        print "CV = ", cv

        # limiting floats to three decimal points
        sub_res = round(cv, 3)

        # save result
        res.append(sub_res)

    # Close remaining file h5
    ion.close()

    print res

    # Create plot
    plt.title("CV Histogram")
    plt.xlabel("CV")
    plt.ylabel("")

    # Plotting an histogram
    plt.grid(True)
    plt.hist(res, bins=100, normed=1, histtype='bar', rwidth=1)
    # plt.show()

    output_file = 'result_cv_histogram_task.png'
    with open(output_file, 'w') as output:
        plt.savefig(output)
    return cv_histogram_task.task.uri.save_file(mime_type='image/png',
                                                src_path=output_file,
                                                dst_path=output_file)


if __name__ == '__main__':

    # Run local test
    filename = 'generate_poisson_spiketrains.h5'
    cv_histogram_task(filename)
