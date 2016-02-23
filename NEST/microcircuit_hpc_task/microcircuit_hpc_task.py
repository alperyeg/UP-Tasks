#!/usr/bin/env python
import os
import yaml
from active_worker.task import task
from task_types import TaskTypes as tt
import unicore_client

@task
def microcircuit_task(configuration_file,
                      simulation_duration,
                      thalamic_input,
                      threads,
                      nodes):
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
            The simulation is run on a HPC machine (currently JUQUEEN) via
            UNICORE.
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
            nodes:
                type: long
                description: Number of compute nodes [default=1].
        Returns:
            res: application/vnd.juelich.bundle.nest.data
    '''

    # load config file provided by user
    user_cfile = microcircuit_task.task.uri.get_file(configuration_file)
    with open(user_cfile, 'r') as f:
        config_file_data=f.readlines()

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

    if not hpc_url.startswith("https://"):
        # lookup the correct base URL
        hpc_url = unicore_client.get_site(hpc_url)
    if hpc_url is None:
        raise RuntimeError("No valid HPC site")

    results = _run_microcircuit(hpc_url, nodes, conf)

    # return bundle
    for file_name, file_mimetype in results:
        bundle.add_file(src_path=file_name,
                        dst_path=file_name,
                        bundle_path=file_name,
                        mime_type=file_mimetype)

    my_bundle_name = 'microcircuit_model_bundle'
    return bundle.save(my_bundle_name)


def _run_microcircuit(hpc_url, nodes, conf):
    """ outsources simulation to HPC via UNICORE 
        config file is uploaded
        TODO upload python code too
        TODO 'nodes' parameter shall be passed to juqueen
    """

    # Auth header for REST calls - this will only work ion the collab
    auth = unicore_client.get_oidc_auth()

    job = {}
    job['ApplicationName'] = "NEST"
    job['Parameters'] = {}
    job['Parameters']['NESTCODE'] = 'microcircuit.py'
    job['Parameters']['PARAMETERS'] = 'config.yaml'

    # files to upload : yaml config and microcircuit code
    config = {'To': 'config.yaml', 'Data': yaml.dump(conf) }
    # TODO: load .py file
    code   = {'To': 'microcircuit.py', 'Data': 'pass' }
    inputs = [config, code]

    # submit the job to the selected site
    job_url = unicore_client.submit(base_url + '/jobs', job, headers, inputs)

    # wait for finish (ie. success or fail) - this can take a while!
    unicore_client.wait_for_completion(job_url, headers)

    # check if it was successful

    results = []
    # TODO collect output files
    return results


if __name__ == '__main__':
    configuration_file = 'user_config.yaml' #'microcircuit.yaml'
    simulation_duration = 1000.
    thalamic_input = True
    threads = 4
    nodes = 1
    filename = tt.URI(
        'application/vnd.juelich.simulation.config', configuration_file)
    result = microcircuit_hpc_task(
        filename, simulation_duration, thalamic_input, threads, nodes)
    print result
