#!/usr/bin/env python
from __future__ import print_function # python 2 & 3 compatible
import os
import glob
import yaml
from active_worker.task import task
from task_types import TaskTypes as tt

@task
def microcircuit_task(configuration_file,
                      simulation_duration,
                      thalamic_input,
                      threads):
    '''
        Task Manifest Version: 1
        Full Name: microcircuit_task
        Caption: Cortical microcircuit simulation
        Author: NEST Developers
        Description: |
            Multi-layer microcircuit model of early sensory cortex
            (Potjans, T. C., & Diesmann, M. (2014)
            Cerebral Cortex 24(3):785-806, code available at
            http://www.opensourcebrain.org/projects/potjansdiesmann2014),
            originally implemented in NEST (http://nest-simulator.org).
            PyNN version modified to run as task in the Collaboratory.
            Simulation parameters are defined in microcircuit.yaml, which needs
            to be passed as a configuration file. A template can be downloaded
            from https://github.com/INM-6/UP-Tasks.
            It is possible to provide an empty or partial configuration file.
            For the missing parameters, default values will be used.
            After uploading the YAML file, its content type needs to be changed
            to 'application/vnd.juelich.simulation.config'. Parameters defined
            in the WUI overwrite values defined in the configuration file.
            For running the full model, 8 CPU cores and 15360 MB memory should
            be requested.
        Categories:
            - NEST
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            configuration_file:
                type: application/vnd.juelich.simulation.config
                description: YAML file, specifying parameters of the simulation.
                    Point to an empty file to use default parameters.
            simulation_duration:
                type: double
                description: Simulation duration in ms [default=1000].
            thalamic_input:
                type: bool
                description: If True, a transient thalamic input is applied to
                    the network [default=False].
            threads:
                type: long
                description: Number of threads NEST uses [default=1].
                    Needs to be set to the same value as 'CPU cores'.
        Returns:
            res: application/vnd.juelich.bundle.nest.data
    '''

    # load config file provided by user
    user_cfile = microcircuit_task.task.uri.get_file(configuration_file)
    with open(user_cfile, 'r') as f:
        user_conf = yaml.load(f)

    # load default config file
    default_cfile = 'microcircuit.yaml'
    yaml_path = os.path.join(os.path.dirname(__file__), default_cfile)
    with open(yaml_path) as f:
        default_conf = yaml.load(f)

    # create config by merging user and default dicts
    conf = default_conf.copy()
    if user_conf is not None:
        conf.update(user_conf)

    # update dict with parameters given in webinterface; these take
    # precedence over those in the configuration file
    conf['simulator_params']['nest']['sim_duration'] = simulation_duration
    conf['simulator_params']['nest']['threads'] = threads
    conf['thalamic_input'] = thalamic_input

    # create bundle & export bundle, mime type for nest simulation output
    my_bundle_mimetype = "application/vnd.juelich.bundle.nest.data"
    bundle = microcircuit_task.task.uri.build_bundle(my_bundle_mimetype)

    bundle_files = _run_microcircuit(conf)
    print('files in bundle: \n', bundle_files)

    for file_name, file_mimetype in bundle_files:
        bundle.add_file(src_path=file_name,
                        dst_path=file_name,
                        bundle_path=file_name,
                        mime_type=file_mimetype)

    my_bundle_name = 'microcircuit_model_bundle'
    return bundle.save(my_bundle_name)


def _run_microcircuit(conf):

    # run microcircuit simulation with given parameters
    import microcircuit
    microcircuit.run_microcircuit(conf)

    # list of tuples for all output files that shall be returned in a bundle:
    # (file name, file type)
    bundle_files = []
    # go through all created output files and assign file types
    filenames  = conf['system_params']['output_path'] + '*'
    for f in glob.glob(filenames):
        fname, extension = os.path.splitext(f)
        if extension == '.gdf':
            filetype = 'application/vnd.juelich.nest.spike_times'
        elif extension == '.png':
            filetype = 'image/png'
        elif extension == '.dat':
            if 'voltages' in fname:
                filetype = 'application/vnd.juelich.nest.analogue_signal'
            elif 'covariances' in fname:
                filetype = 'text/plain'
        elif extension == '.yaml':
            filetype = 'text/plain'
        else:
            filetype = 'application/unknown'
        res = (f, filetype)
        bundle_files.append(res)

    return bundle_files


if __name__ == '__main__':
    configuration_file = 'user_config.yaml'
    simulation_duration = 1000.
    thalamic_input = False
    threads = 4
    filename = tt.URI(
        'application/vnd.juelich.simulation.config', configuration_file)
    task_result = microcircuit_task(
        filename, simulation_duration, thalamic_input, threads)
    print('returned by task: \n', task_result)
