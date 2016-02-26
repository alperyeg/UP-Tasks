#!/usr/bin/env python

# =============================================================================
# Initialization
# =============================================================================

from active_worker.task import task
from task_types import TaskTypes as tt
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
import quantities as pq
import numpy as np
import neo
import matplotlib.pyplot as plt
import os
import h5py


@task
def pairwise_correlation_histogram_task(inputdata, binsize):
    '''
        Task Manifest Version: 1
        Full Name: pairwise_correlation_histogram_task
        Caption: pairwiae_correlation_histogram
        Author: Elephant-Developers
        Description: |
            This task calculates all pair-wise correlations between all
            combinations of spike trains in the input file.
            The output is a bundle containing an png file showing the
            correlation result as well as an h5 file with the cross-correlation
            results in a numpy 2d-array.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz']
        Accepts:
            inputdata:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file.
            binsize:
                type: int
                description: Bin width to bin the spike trains. Uses ms resolution.
        Returns:
            res: application/unknown
    '''
    # =========================================================================
    # Load data
    # =========================================================================
    # stage the input file
    original_path = pairwise_correlation_histogram_task.task.uri.get_file(inputdata)
    print original_path
    bundle = pairwise_correlation_histogram_task.task.uri.build_bundle("application/unkown")

    session = neo.NeoHdf5IO(filename=original_path)
    block = session.read_block()

    # select spike trains
    sts = block.list_children_by_class(neo.SpikeTrain)[:100]

    # =========================================================================
    # Pairwise_correlation_histogram
    # =========================================================================
    cc_matrix = corrcoef(BinnedSpikeTrain(sts, binsize=binsize * pq.ms))
    filename = 'pairwise_correlations_result'

    # Store to hdf5
    f = h5py.File(os.path.splitext(filename)[0] + '.h5', 'w')
    dataset = f.create_dataset("cc_matrix", data=cc_matrix)
    f.close()

    output_filename = os.path.splitext(filename)[0] + '.png'

    # Plotting
    plt.hist(cc_matrix, 100)
    plt.xlabel('$\\rho$')
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
    inputdata = tt.URI('application/unknown', 'spikes_L5E.h5')
    binsize = 5
    pairwise_correlation_histogram_task(inputdata, binsize)
