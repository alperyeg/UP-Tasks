#!/usr/bin/env python

# =============================================================================
# Initialization
# =============================================================================

from active_worker.task import task
from task_types import TaskTypes as tt
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
import quantities as pq
import neo
import matplotlib.pyplot as plt
import os


@task
def pairwise_correlation_task(inputdata, binsize):
    '''
        Task Manifest Version: 1
        Full Name: pairwise_correlation_histogram_task
        Caption: pairwiae_correlation_histogram
        Author: Elephant-Developers
        Description: |
            This task calculates all pair-wise correlations between all
            combinations of spike trains in the input file.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz']
        Accepts:
            inputdata: application/unknown
            binsize: int

        Returns:
            res: application/unknown
    '''
    # =========================================================================
    # Load data
    # =========================================================================
    # stage the input file
    original_path = pairwise_correlation_task.task.uri.get_file(inputdata)

    session = neo.NeoHdf5IO(filename=original_path)
    block = session.read_block()

    # select spike trains
    sts = block.filter(use_st=True)

    # =========================================================================
    # Pairwise_correlation_histogram
    # =========================================================================
    cc_matrix = corrcoef(BinnedSpikeTrain(sts, binsize=binsize * pq.ms))
    output_filename = 'pairwise_correlations_result.h5'

    plt.hist(cc_matrix, 100)
    plt.xlabel('$\\rho$')
    plt.ylabel('PDF')

    output_filename = os.path.splitext(original_path)[0] + '.png'
    with open(output_filename, 'w') as result_pth:
        plt.savefig(result_pth, dpi=100)
    dst_name = os.path.basename(output_filename)
    return pairwise_correlation_task.task.uri.save_file(mime_type='image/png',
                                                        src_path=output_filename,
                                                        dst_path=dst_name)

if __name__ == '__main__':
    inputdata = tt.URI('application/unknown', 'experiment.h5')
    binsize = 5
    pairwise_correlation_task(inputdata, binsize)
