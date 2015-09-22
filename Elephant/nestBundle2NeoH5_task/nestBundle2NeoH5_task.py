#! /usr/bin/env python
import quantities as pq
import gdfio
import neo
from active_worker.task import task
from task_types import TaskTypes as tt
import glob
import os


@task
def nestBundle2NeoH5_task(nest_bundle_file, t_start, t_stop):
    """
        Task Manifest Version: 1
        Full Name: nestBundle2NeoH5_task
        Caption: nestBundle2NeoH5
        Author: NEST and Elephant developers
        Description: |
            Takes a bundle of NEST output files
            (application/vnd.juelich.bundle.nest.data), extracts the GDF files
            which contain spike train data, converts them to Neo HDF5 files
            and returns a corresponding bundle file.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            nest_bundle_file:
                type: application/vnd.juelich.bundle.nest.data
                description: Input bundle file of NEST output containing
                    spike train data in GDF format.
            t_start:
                type: double
                description: Start time of spike train recording.
            t_stop:
                type: double
                description: Stop time of spike train recording.
        Returns:
            res: application/unknown
    """

    input_path = nestBundle2NeoH5_task.task.uri.get_bundle(nest_bundle_file)
    input_files = glob.glob(os.path.split(input_path)[0] +'/'+ '*.gdf')
    print input_files
    # h5_bundle_mime_type = "application/vnd.juelich.bundle.nest.data"
    h5_bundle_mime_type = "application/unknown"
    bundle = nestBundle2NeoH5_task.task.uri.build_bundle(h5_bundle_mime_type)
    for input_fn in input_files:
        gdf = nestBundle2NeoH5_task.task.uri.get_file(tt.URI(
            h5_bundle_mime_type, input_fn))
        input_file = gdfio.GdfIO(gdf)
        # In a bundle all Neurons that spike at least once are registered
        seg = input_file.read_segment(gdf_id_list=[],
                                      t_start=t_start * pq.ms,
                                      t_stop=t_stop * pq.ms)
        output_filename = os.path.splitext(gdf)[0] + '.h5'
        output_file = neo.io.NeoHdf5IO(output_filename)
        output_file.write(seg.spiketrains)
        output_file.close()
        bundle.add_file(src_path=output_filename,
                        dst_path=output_filename,
                        bundle_path=output_filename,
                        mime_type=h5_bundle_mime_type)
    return bundle.save("neo_h5_bundle")


if __name__ == '__main__':
    input_filename = tt.URI('application/unknown',
                            'bundle_example/microcircuit_model_bundle')
    t_start = 0.
    t_stop = 300.
    nestBundle2NeoH5_task(input_filename, t_start, t_stop)