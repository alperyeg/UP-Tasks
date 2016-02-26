#!/usr/bin/env python

# =============================================================================
# Initialization
# =============================================================================

from active_worker.task import task
from task_types import TaskTypes as tt
from elephant.conversion import BinnedSpikeTrain
from elephant.statistics import mean_firing_rate
import quantities as pq
import neo
import matplotlib.pyplot as plt
import os
import h5py


@task
def mean_firingrate_task(inputdata):
    '''
        Task Manifest Version: 1
        Full Name: mean_firingrate_task
        Caption: mean_firingrate
        Author: Elephant-Developers
        Description: |
            This task calculates mean firing rate of each spike trains
            results in a numpy 2d-array.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz']
        Accepts:
            inputdata:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file.
        Returns:
            res: application/unknown
    '''
    # =========================================================================
    # Load data
    # =========================================================================
    # stage the input file
    original_path = mean_firingrate_task.task.uri.get_file(inputdata)
    bundle = mean_firingrate_task.task.uri.build_bundle("application/unkown")

    session = neo.NeoHdf5IO(filename=original_path)
    block = session.read_block()

    # select spike trains
    spiketrains = block.list_children_by_class(neo.SpikeTrain)


    # =========================================================================
    # Pairwise_correlation_histogram
    # =========================================================================

    mean_rates = mean_firing_rate(spiketrains, t_start=None, t_stop=None, axis=None)
    filename = 'mean_firingrate_result'

    # Store to hdf5
    f = h5py.File(os.path.splitext(filename)[0] + '.h5', 'w')
    dataset = f.create_dataset("mean_rates", data=mean_rates)
    f.close()

    output_filename = os.path.splitext(filename)[0] + '.png'

    # Plotting
    plt.hist(mean_rates, 100)
    plt.xlabel('firing rate')
    plt.ylabel('PDF')
    with open(output_filename, 'w') as result_pth:
        plt.savefig(result_pth, dpi=100)

    dst_name = os.path.basename(filename)
    session.close()

    bundle.add_file(src_path=filename,
                    dst_path=dst_name,
                    bundle_path=dst_name,
                    mime_type="image/png")
    bundle.add_file(src_path=filename,
                    dst_path=dst_name,
                    bundle_path=dst_name,
                    mime_type="application/unkown")
    return bundle.save("bundle_" + filename)

if __name__ == '__main__':
    inputdata = tt.URI('application/unknown', 'experiment.h5')
    mean_firingrate_task(inputdata)
