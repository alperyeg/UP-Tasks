#! /usr/bin/env python
import quantities as pq
import gdfio
import neo
import glob
import os
from active_worker.task import task
from task_types import TaskTypes as tt


@task
def nestBundle2NeoH5_task(nest_bundle_file, t_start, t_stop):
    """
        Task Manifest Version: 1
        Full Name: nestBundle2NeoH5_task
        Caption: Convert NEST bundle to NeoHDF5 bundle
        Author: NEST and Elephant Developers
        Description: |
            Takes a bundle of NEST (http://nest-simulator.org/) output files,
            extracts the GDF files which contain spike data, converts them to
            NeoHDF5 files which can be processed by
            Elephant (http://neuralensemble.org/elephant).
            It returns a bundle containing the HDF5 files.
        Categories:
            - NEST
            - FDAT
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            nest_bundle_file:
                type: application/vnd.juelich.bundle.nest.data
                description: Input bundle file of NEST output containing
                    spike data in GDF format.
            t_start:
                type: double
                description: Start time in ms of spike train recording.
            t_stop:
                type: double
                description: Stop time in ms of spike train recording.
        Returns:
            res: application/unknown
    """

    input_path = nestBundle2NeoH5_task.task.uri.get_bundle(nest_bundle_file)
    input_files = glob.glob(input_path + '/' + '*.gdf')

    # no h5 specific mime type available
    h5_bundle_mime_type = "application/unknown"
    bundle = nestBundle2NeoH5_task.task.uri.build_bundle(h5_bundle_mime_type)

    for gdf in input_files:
        input_file = gdfio.GdfIO(gdf)
        # in a bundle all neurons that spike at least once are registered
        seg = input_file.read_segment(gdf_id_list=[],
                                      t_start=t_start * pq.ms,
                                      t_stop=t_stop * pq.ms)
        output_filename = os.path.splitext(gdf)[0] + '.h5'
        output_file = neo.io.NeoHdf5IO(output_filename)
        output_file.write(seg.spiketrains)
        output_file.close()

        output_dst = os.path.basename(output_filename)

        bundle.add_file(src_path=output_filename,
                        dst_path=output_dst,
                        bundle_path=output_dst,
                        mime_type=h5_bundle_mime_type)

    return bundle.save("neo_h5_bundle")


if __name__ == '__main__':
    input_filename = tt.URI('application/unknown',
                            'bundle_example/microcircuit_model_bundle')
    t_start = 0.
    t_stop = 300.
    nestBundle2NeoH5_task(input_filename, t_start, t_stop)
