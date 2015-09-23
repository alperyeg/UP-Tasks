import elephant
import neo
import os
from elephant import cubic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from active_worker.task import task
from task_types import TaskTypes as tt


@task
def cubic_task(h5_file, binsize, alpha):
    """
        Task Manifest Version: 1
        Full Name: cubic_task
        Caption: cubic
        Author: Elephant_Developers
        Description: |
            Analyses the correlation of parallel recorded spike trains
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            h5_file:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file.
            binsize:
                type: double
                description: Bin width used to compute the PSTH in ms.
            alpha:
                type: double
                description: The significance level of the test.

        Returns:
            res: image/png
    """
    h5_path = cubic_task.task.uri.get_file(h5_file)
    ion = neo.io.NeoHdf5IO(filename=h5_path)
    number_of_spike_trains = ion.get_info()['SpikeTrain']

    spiketrains = []

    for k in range(number_of_spike_trains):
        spiketrains.append(ion.get("/" + "SpikeTrain_" + str(k)))

    ion.close()

    psth_as = elephant.statistics.time_histogram(spiketrains,
                                                 binsize=binsize * pq.ms)

    result = cubic.cubic(psth_as, alpha=alpha)

    # Plot
    plt.bar(np.arange(0.75, len(result[1]) + .25, 1), result[1], width=.5)
    plt.axhline(alpha, ls='--', color='r')
    plt.xlabel('$\\xi$')
    plt.ylabel('P value')
    plt.title('$\hat\\xi$=' + str(result[0]))

    output_filename = os.path.splitext(h5_path)[0] + '.png'
    with open(output_filename, 'w') as result_pth:
        plt.savefig(result_pth, dpi=100)
    dst_name = output_filename.split('/')[-1]
    return cubic_task.task.uri.save_file(mime_type='image/png',
                                         src_path=output_filename,
                                         dst_path=dst_name)


if __name__ == '__main__':
    input_filename = tt.URI('application/unknown',
                            'spikes_L5I-44930-0.h5')
    alpha = 0.05
    binsize = 1
    cubic_task(input_filename, binsize, alpha)
