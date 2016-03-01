#!/usr/bin/env python
import os
import yaml
from active_worker.task import task
from task_types import TaskTypes as tt
import unicore_client

def load_local_file(name):
    """ loads local file and returns contents as string """
    file_path = os.path.join(os.path.dirname(__file__), name)
    with open(file_path) as f:
        return f.read()
    

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
                description: Number of compute nodes [default=32].
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

    if not hpc_url.startswith("https://"):
        # lookup the correct base URL
        hpc_url = unicore_client.get_site(hpc_url)
    if hpc_url is None:
        raise RuntimeError("No valid HPC site")

    results = _run_microcircuit(hpc_url, nodes, conf)

    bundle_files = _run_microcircuit(conf)
    print('files in bundle: \n', bundle_files)

    for file_name, file_mimetype in bundle_files:
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
    oauth_token = microcircuit_task.task.uri.get_oauth_token()
    auth = unicore_client.get_oidc_auth(oauth_token)

    # setup UNICORE job
    job = {}
    job['ApplicationName'] = "NEST"
    # TODO probably there is no need yet to distinguish NEST versions
    #job['ApplicationVersion'] = "2.6.1"
    job['Parameters'] = {}
    job['Parameters']['NESTCODE'] = 'microcircuit.py'
    job['Parameters']['PARAMETERS'] = 'config.yaml'
    # TODO further arguments
    #job['Arguments']= [ "arg1", "arg2", "arg3" ]

    # TODO resource requests - nodes, runtime etc
    #job['Resources'] = { 'Nodes': '32' }
    
    # files to upload : yaml config, microcircuit code, helper files
    config = {'To': 'config.yaml', 'Data': yaml.dump(conf) }
    code   = {'To': 'microcircuit.py', 'Data': load_local_file('microcircuit.py' }
    connectivity = {'To': 'connectivity.py',
                'Data': load_local_file('connectivity.py')}
    network = {'To': 'network.py', 
               'Data': load_local_file('network.py')}
    helper = {'To': 'helper_functions.py', 
              'Data': load_local_file('helper_functions.py')}
    plotting = {'To': 'plotting.py', 
                'Data': load_local_file('plotting.py')}
    scaling = {'To': 'scaling.py', 
               'Data': load_local_file('scaling.py')}
    inputs = [config, code, connectivity, network, helper, plotting, scaling]

    # submit the job to the selected site
    job_url = unicore_client.submit(hpc_url + '/jobs', job, auth, inputs)

    # wait for finish (ie. success or fail) - this can take a while!
    unicore_client.wait_for_completion(job_url, auth)

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
    configuration_file = 'user_config.yaml' #'microcircuit.yaml'
    simulation_duration = 1000.
    thalamic_input = False
    threads = 4
    nodes = 32
    filename = tt.URI(
        'application/vnd.juelich.simulation.config', configuration_file)
    result = microcircuit_hpc_task(
        filename, simulation_duration, thalamic_input, threads, nodes)
    print result
