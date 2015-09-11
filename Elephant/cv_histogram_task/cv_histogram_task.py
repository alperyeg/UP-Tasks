#! /usr/bin/env python
__author__ = "Long Phan, Alper Yegenoglu. INM-6, FZJ"

import numpy as np
import elephant as el
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neo import io
from active_worker.task import task
from task_types import TaskTypes as tt


@task
def cv_histogram_task(input_data):
    """
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
            res: application/vnd.juelich.bundle.elephant.data
    """

    # stage the input file
    original_path = cv_histogram_task.task.uri.get_file(input_data)

    # open file from path using Neo
    ion = io.NeoHdf5IO(filename=original_path)
    number_of_SpikeTrains = ion.get_info()['SpikeTrain']

    my_bundle_mimetype = "application/vnd.juelich.bundle.elephant.data"
    bundle = cv_histogram_task.task.uri.build_bundle(my_bundle_mimetype)

    # Init result
    res = []

    # Query data from hdf5-file
    for k in range(number_of_SpikeTrains):
        poisson_data = ion.get("/"+"SpikeTrain_"+str(k))

        if poisson_data.size > 0:
            # Calculate value for cv
            cv_data = el.statistics.isi(poisson_data)
            cv = el.statistics.cv(cv_data)
            print "CV = ", cv

            if not np.isnan(cv):
                # limiting floats to three decimal points
                sub_res = round(cv, 3)

                # save result
                res.append(sub_res)

    # Close remaining file h5
    ion.close()

    # print res

    # Create plot
    plt.title("CV Histogram")
    plt.xlabel("CV")
    plt.ylabel("")

    # Plotting an histogram
    plt.grid(True)
    if len(res) > 0:
        plt.hist(res, bins=100, normed=1, histtype='bar', rwidth=1)
    else:
        print 'Warning: could not calculate CV for input(s).'

    output_file_img = 'result_cv_histogram_task.png'
    output_file = output_file_img.split('.')[0]+str('.json')
    with open(output_file_img, 'w') as output:
        plt.savefig(output)

    # Add image meta data to bundle
    bundle.add_file(mime_type='image/png',
                    src_path=output_file_img,
                    bundle_path=output_file_img,
                    dst_path=output_file_img)

    # Write a file containing the results
    with open(output_file, 'w') as f:
        json.dump(res, f)
    f.close()

    # Add json meta data to bundle
    bundle.add_file(mime_type='application/unknown',
                    src_path=output_file,
                    bundle_path=output_file,
                    dst_path=output_file)
    return bundle.save('cv_histogram_task_bundle')


if __name__ == '__main__':

    # Run local test
    filename = tt.URI('application/unknown', 'generate_poisson_spiketrains.h5')
    cv_histogram_task(filename)
