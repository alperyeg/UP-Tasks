#!/usr/bin/env python
from active_worker.task import task
import yaml
import matplotlib
matplotlib.use('Agg')
import time
import os
import glob
import numpy as np
from task_types import TaskTypes as tt
import helper_functions

# The simulation runs with values defined in config-file microcircuit.yaml
#
# NOTE: according to an incompatibility with Python2.6, NEST2.6.0 and PyNN0.7.5
# , connections to spike detectors and voltmeters are here established with
# PyNEST instead of PyNN. respective code is marked with "PYTHON2.6:"
# and can be changed back in future versions.


@task
def microcircuit_task(configuration_file, simulation_duration, thalamic_input, threads):
    '''
        Task Manifest Version: 1
        Full Name: microcircuit_task
        Caption: Cortical microcircuit simulation
        Author: Johanna Senk, Jakob Jordan, Sacha van Albada
        Description: |
            Multi-layer microcircuit model of early sensory cortex
            (Potjans, T. C., & Diesmann, M. (2014) Cerebral Cortex 24(3):785-806).
            PyNN version modified to run as task in the Collaboratory.
            Simulation paramters are defined in microcircuit.yaml, which needs
            to be passed as a configuration file. A template can be downloaded from
            https://github.com/INM-6/UP-Tasks/blob/master/NEST/microcircuit_task/microcircuit.yaml.
            It is possible to provide an empty or partial configuration file. For the missing
            parameters, default values will be used. After uploading the YAML file,
            its content type needs to be changed to 'application/vnd.juelich.simulation.config'.
            For running the full model, 4 CPU cores and 15360 MB memory should be requested.
        Categories:
            - NEST
        Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
        Accepts:
            configuration_file:
                type: application/vnd.juelich.simulation.config
                description: YAML file, specifying parameters of the simulation. Point to an empty file to use default parameters.
            simulation_duration:
                type: double
                description: Simulation duration in ms [default=1000]. Overrides value in configuration file.
            thalamic_input:
                type: bool
                description: If True, a transient thalamic input is applied to the network [default=False].
            threads:
                type: long
                description: Number of threads NEST should use for the simulation [default=1]. Needs to be set to the same value as 'CPU cores'.
        Returns:
            res: application/vnd.juelich.bundle.nest.data
    '''

    # load config file provided by user
    user_cfile = microcircuit_task.task.uri.get_file(configuration_file)
    with open(user_cfile, 'r') as f:
        user_conf = yaml.load(f)

    # load default config file
    default_cfile = 'microcircuit.yaml'
    with open('./' +  default_cfile, 'r') as f: # datapath necessary
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


    plot_filename = 'spiking_activity.png'

    # create bundle & export bundle, mime type for nest simulation output
    my_bundle_mimetype = "application/vnd.juelich.bundle.nest.data"
    bundle = microcircuit_task.task.uri.build_bundle(my_bundle_mimetype)

    results = _run_microcircuit(plot_filename, conf)

    # print and return bundle
    print "results = ", results

    for file_name, file_mimetype in results:
        bundle.add_file(src_path=file_name,
                        dst_path=file_name,
                        bundle_path=file_name,
                        mime_type=file_mimetype)

    my_bundle_name = 'microcircuit_model_bundle'
    return bundle.save(my_bundle_name)


def _run_microcircuit(plot_filename, conf):
    import plotting
    import logging

    simulator = conf['simulator']
    # we here only need nest as simulator, simulator = 'nest'
    import pyNN.nest as sim

    # prepare simulation
    logging.basicConfig()

    # extract parameters from config file
    master_seed = conf['params_dict']['nest']['master_seed']
    layers = conf['layers']
    pops = conf['pops']
    plot_spiking_activity = conf['plot_spiking_activity']
    raster_t_min = conf['raster_t_min']
    raster_t_max = conf['raster_t_max']
    frac_to_plot = conf['frac_to_plot']
    record_corr = conf['params_dict']['nest']['record_corr']
    tau_max = conf['tau_max']

    # Numbers of neurons from which to record spikes
    n_rec = helper_functions.get_n_rec(conf)

    sim.setup(**conf['simulator_params'][simulator])

    if simulator == 'nest':
        n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
        if sim.rank() == 0:
            print 'n_vp: ', n_vp
            print 'master_seed: ', master_seed
        sim.nest.SetKernelStatus({'print_time': False,
                                  'dict_miss_is_error': False,
                                  'grng_seed': master_seed,
                                  'rng_seeds': range(master_seed + 1,
                                                     master_seed + n_vp + 1),
                                  # PYTHON2.6: FOR WRITING OUTPUT FROM
                                  # RECORDING DEVICES WITH PYNEST FUNCTIONS,
                                  # THE OUTPUT PATH IS NOT AUTOMATICALLY THE
                                  # CWD BUT HAS TO BE SET MANUALLY
                                  'data_path': conf['system_params']['output_path']})

    import network

    # result of export-files
    results = []

    # create network
    start_netw = time.time()
    n = network.Network(sim)

    # PYTHON2.6: device_list CONTAINS THE GIDs OF THE SPIKE DETECTORS AND VOLTMETERS
    # NEEDED FOR RETRIEVING FILENAMES LATER
    device_list = n.setup(sim, conf)

    end_netw = time.time()
    if sim.rank() == 0:
        print 'Creating the network took ', end_netw - start_netw, ' s'

    # simulate
    if sim.rank() == 0:
        print "Simulating..."
    start_sim = time.time()
    sim.run(conf['simulator_params'][simulator]['sim_duration'])
    end_sim = time.time()
    if sim.rank() == 0:
        print 'Simulation took ', end_sim - start_sim, ' s'

    # extract filename from device_list (spikedetector/voltmeter),
    # gid of neuron and thread. merge outputs from all threads
    # into a single file which is then added to the task output.
    # PYTHON2.6: NEEDS TO BE ADAPTED IF NOT RECORDED VIA PYNEST
    for dev in device_list:
        label = sim.nest.GetStatus(dev)[0]['label']
        gid = sim.nest.GetStatus(dev)[0]['global_id']
        # use the file extension to distinguish between spike and voltage output
        extension = sim.nest.GetStatus(dev)[0]['file_extension']
        if extension == 'gdf': # spikes
            data = np.empty((0, 2))
        elif extension == 'dat': # voltages
            data = np.empty((0, 3))
        for thread in xrange(conf['simulator_params']['nest']['threads']):
            filenames = glob.glob(conf['system_params']['output_path'] \
                                  + '%s-*%d-%d.%s' % (label, gid, thread, extension))
            assert(len(filenames) == 1), 'Multiple input files found. Use a clean output directory.'
            data = np.vstack([data, np.loadtxt(filenames[0])])
            # delete original files
            os.remove(filenames[0])
        order = np.argsort(data[:, 1])
        data = data[order]
        outputfile_name = 'collected_%s-%d.%s' % (label, gid, extension)
        outputfile = open(outputfile_name, 'w')
        # the outputfile should have same format as output from NEST.
        # i.e., [int, float] for spikes and [int, float, float] for voltages,
        # hence we write it line by line and assign the corresponding filetype
        if extension == 'gdf':  # spikes
            for line in data:
                outputfile.write('%d\t%.3f\n' % (line[0], line[1]))
            outputfile.close()
            filetype = 'application/vnd.juelich.nest.spike_times'

        elif extension == 'dat':  # voltages
            for line in data:
                outputfile.write('%d\t%.3f\t%.3f\n' % (line[0], line[1], line[2]))
            outputfile.close()
            filetype = 'application/vnd.juelich.nest.analogue_signal'

        res = (outputfile_name, filetype)
        results.append(res)

    # start_writing = time.time()

    # PYTHON2.6: SPIKE AND VOLTAGE FILES ARE CURRENTLY WRITTEN WHEN A SPIKE
    # DETECTOR OR A VOLTMETER IS CONNECTED WITH 'to_file': True

    # for layer in layers:
    #     for pop in pops:
    #         # filename = conf['system_params']['output_path'] + '/spikes_' + layer + pop + '.dat'
    #         filename = conf['system_params']['output_path'] + 'spikes_' + layer + pop + '.dat'
    #         n.pops[layer][pop].printSpikes(filename, gather=False)

    #         # add filename and filepath into results
    #         subres = (filename, 'application/vnd.juelich.bundle.nest.data')
    #         results.append(subres)

    # if record_v:
    #     for layer in layers:
    #         for pop in pops:
    #             filename = conf['system_params']['output_path'] + '/voltages_' + layer + pop + '.dat'
    #             n.pops[layer][pop].print_v(filename, gather=False)

    if record_corr and simulator == 'nest':
        start_corr = time.time()
        if sim.nest.GetStatus(n.corr_detector, 'local')[0]:
            print 'getting count_covariance on rank ', sim.rank()
            cov_all = sim.nest.GetStatus(n.corr_detector, 'count_covariance')[0]
            delta_tau = sim.nest.GetStatus(n.corr_detector, 'delta_tau')[0]

            cov = {}
            for target_layer in np.sort(layers.keys()):
                for target_pop in pops:
                    target_index = conf['structure'][target_layer][target_pop]
                    cov[target_index] = {}
                    for source_layer in np.sort(layers.keys()):
                        for source_pop in pops:
                            source_index = conf['structure'][source_layer][source_pop]
                            cov[target_index][source_index] = np.array(list(cov_all[target_index][source_index][::-1])
                                                                       + list(cov_all[source_index][target_index][1:]))

            f = open(conf['system_params']['output_path'] + '/covariances.dat', 'w')
            print >>f, 'tau_max: ', tau_max
            print >>f, 'delta_tau: ', delta_tau
            print >>f, 'simtime: ', conf['simulator_params'][simulator]['sim_duration'], '\n'

            for target_layer in np.sort(layers.keys()):
                for target_pop in pops:
                    target_index = conf['structure'][target_layer][target_pop]
                    for source_layer in np.sort(layers.keys()):
                        for source_pop in pops:
                            source_index = conf['structure'][source_layer][source_pop]
                            print >>f, target_layer, target_pop, '-', source_layer, source_pop
                            print >>f, 'n_events_target: ', sim.nest.GetStatus(n.corr_detector, 'n_events')[0][target_index]
                            print >>f, 'n_events_source: ', sim.nest.GetStatus(n.corr_detector, 'n_events')[0][source_index]
                            for i in xrange(len(cov[target_index][source_index])):
                                print >>f, cov[target_index][source_index][i]
                            print >>f, ''
            f.close()

            # add file covariances.dat into bundle
            res_cov = ('covariances.dat',
                       'text/plain')
            results.append(res_cov)

        end_corr = time.time()
        print "Writing covariances took ", end_corr - start_corr, " s"

    # end_writing = time.time()
    # print "Writing data took ", end_writing - start_writing, " s"

    if plot_spiking_activity and sim.rank() == 0:
        plotting.plot_raster_bars(raster_t_min, raster_t_max, n_rec,
                                  frac_to_plot, n.pops,
                                  conf['system_params']['output_path'],
                                  plot_filename, conf)
        res_plot = (plot_filename, 'image/png')
        results.append(res_plot)

    sim.end()

    return results

if __name__ == '__main__':
    configuration_file = 'microcircuit.yaml' #user_config.yaml'
    simulation_duration = 1000.
    thalamic_input = True
    threads = 4
    filename = tt.URI('application/vnd.juelich.simulation.config', configuration_file)
    microcircuit_task(filename, simulation_duration, thalamic_input, threads)
