#!/usr/bin/env python
def run_microcircuit(conf):
    '''Cortical microcircuit simulation.
    conf: path to yaml config file.
    '''

    import os
    import time
    import glob
    import logging
    import numpy as np
    import pyNN.nest as sim # only nest as simulator (simulator = 'nest')
    from mpi4py import MPI
    import helper_functions
    import plotting
    import network

    COMM = MPI.COMM_WORLD

    # extract parameters from config dictionary
    master_seed = conf['params_dict']['nest']['master_seed']
    layers = conf['layers']
    pops = conf['pops']
    plot_spiking_activity = conf['plot_spiking_activity']
    raster_t_min = conf['raster_t_min']
    raster_t_max = conf['raster_t_max']
    frac_to_plot = conf['frac_to_plot']
    record_corr = conf['params_dict']['nest']['record_corr']
    tau_max = conf['tau_max']
    simulator = conf['simulator'] # = 'nest'
    sim_duration = conf['simulator_params'][simulator]['sim_duration']

    # Numbers of neurons from which to record spikes
    n_rec = helper_functions.get_n_rec(conf)


    # prepare simulation
    logging.basicConfig()
    sim.setup(**conf['simulator_params'][simulator])

    # nest setup
    if simulator == 'nest':
        n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
        if sim.rank() == 0:
            print('n_vp: ', n_vp)
            print('master_seed: ', master_seed)
        sim.nest.SetKernelStatus({'print_time': False,
                                  'dict_miss_is_error': False,
                                  'grng_seed': master_seed,
                                  'rng_seeds': range(master_seed + 1,
                                                     master_seed + n_vp + 1),
                                  'data_path': conf['system_params'] \
                                                   ['output_path']})
        sim.nest.set_verbosity(30) # M_ERROR: do inform about severe errors only
        if sim.num_processes() > 1: # MPI
            sim.nest.SetNumRecProcesses(1) # use global spike detector


    # create network
    start_netw = time.time()
    n = network.Network(sim)
    # setup returns the GIDs of the spike detectors and voltmeters needed for
    # retrieving filenames later
    device_list = n.setup(sim, conf)
    end_netw = time.time()
    if sim.rank() == 0:
        print('Creating the network took ', end_netw - start_netw, ' s')


    # simulate
    if sim.rank() == 0:
        print("Simulating...")
    start_sim = time.time()
    sim.run(sim_duration)
    end_sim = time.time()
    if sim.rank() == 0:
        print('Simulation took ', end_sim - start_sim, ' s')
    sim.end()


    # filenames and filetypes to be returned
    results = []

    # extract filename from device_list (spikedetector/voltmeter),
    # gid of neuron and thread. merge outputs from all threads
    # into a single file which is then added to the task output.
    for dev in device_list:
        if sim.nest.GetStatus(dev, 'local')[0]:
            label = sim.nest.GetStatus(dev)[0]['label']
            gid = sim.nest.GetStatus(dev)[0]['global_id']
            # use the file extension to distinguish between spike and voltage
            # output
            extension = sim.nest.GetStatus(dev)[0]['file_extension']
            if extension == 'gdf':  # spikes
                data = np.empty((0, 2))
                filetype = 'application/vnd.juelich.nest.spike_times'
            elif extension == 'dat':  # voltages
                data = np.empty((0, 3))
                filetype = 'application/vnd.juelich.nest.analogue_signal'
            outputfile_name = 'collected_%s.%s' % (label, extension)

            if sim.num_processes() == 1: # threads only: merge files
                for thread in range(conf['simulator_params']['nest']['threads']):
                    filenames = glob.glob(conf['system_params']['output_path']
                                          + '%s-*%d-%d.%s' % (label, gid, thread, extension))
                    assert(
                        len(filenames) == 1), 'Multiple input files found. Use a clean output directory.'
                    data = np.vstack([data, np.loadtxt(filenames[0])])
                    # delete original files
                    os.remove(filenames[0])
                order = np.argsort(data[:, 1])
                data = data[order]
                outputfile = open(outputfile_name, 'w')
                # the outputfile should have same format as output from NEST.
                # i.e., [int, float] for spikes and [int, float, float] for voltages,
                # hence we write it line by line and assign the corresponding filetype
                if extension == 'gdf':  # spikes
                    for line in data:
                        outputfile.write('%d\t%.3f\n' % (line[0], line[1]))
                    outputfile.close()

                elif extension == 'dat':  # voltages
                    for line in data:
                        outputfile.write(
                            '%d\t%.3f\t%.3f\n' % (line[0], line[1], line[2]))
                    outputfile.close()

            else: # global spike detector: just rename files
                filenames = glob.glob(conf['system_params']['output_path']
                                          + '%s-*%d-*.%s' % (label, gid, extension))
                assert(
                    len(filenames) == 1), 'Multiple input files found. Use a clean output directory.'
                os.rename(filenames[0], outputfile_name)

            res = (outputfile_name, filetype)
            results.append(res)


    if record_corr and simulator == 'nest':
        start_corr = time.time()
        if sim.nest.GetStatus(n.corr_detector, 'local')[0]:
            print('getting count_covariance on rank ', sim.rank())
            cov_all = sim.nest.GetStatus(
                n.corr_detector, 'count_covariance')[0]
            delta_tau = sim.nest.GetStatus(n.corr_detector, 'delta_tau')[0]

            cov = {}
            for target_layer in np.sort(layers.keys()):
                for target_pop in pops:
                    target_index = conf['structure'][target_layer][target_pop]
                    cov[target_index] = {}
                    for source_layer in np.sort(layers.keys()):
                        for source_pop in pops:
                            source_index = conf['structure'][
                                source_layer][source_pop]
                            cov[target_index][source_index] = \
                                np.array(list(
                                    cov_all[target_index][source_index][::-1])
                                + list(cov_all[source_index][target_index][1:]))

            f = open(conf['system_params'][
                     'output_path'] + '/covariances.dat', 'w')
            print('tau_max: ' + tau_max, end="", file=f)
            print('delta_tau: ' + delta_tau, end="", file=f)
            s = 'simtime: ', conf['simulator_params'][
                simulator]['sim_duration'], '\n'
            print(s, end="", file=f)

            for target_layer in np.sort(layers.keys()):
                for target_pop in pops:
                    target_index = conf['structure'][target_layer][target_pop]
                    for source_layer in np.sort(layers.keys()):
                        for source_pop in pops:
                            source_index = conf['structure'][
                                source_layer][source_pop]
                            s = target_layer, target_pop, '-', source_layer, source_pop
                            print(s, end="", file=f)
                            s = 'n_events_target: ', sim.nest.GetStatus(
                                n.corr_detector, 'n_events')[0][target_index]
                            print(s, end="", file=f)
                            s = 'n_events_source: ', sim.nest.GetStatus(
                                n.corr_detector, 'n_events')[0][source_index]
                            print(s, end="", file=f)
                            for i in xrange(len(cov[target_index][source_index])):
                                s = cov[target_index][source_index][i]
                                print(s, end="", file=f)
                            print('', end="", file=f)
            f.close()

            # add file covariances.dat into bundle
            res_cov = ('covariances.dat',
                       'text/plain')
            results.append(res_cov)

        end_corr = time.time()
        print("Writing covariances took ", end_corr - start_corr, " s")
    COMM.Barrier()

    if plot_spiking_activity and sim.rank() == 0:
        print('Plotting')
        plot_filename = 'spiking_activity.png'
        plotting.plot_raster_bars(raster_t_min, raster_t_max, n_rec,
                                  frac_to_plot, n.pops,
                                  conf['system_params']['output_path'],
                                  plot_filename, conf)
        res_plot = (plot_filename, 'image/png')
        results.append(res_plot)
    COMM.Barrier()

    # TODO: DO STH WITH RESULTS
    return


if __name__ == '__main__':
    import yaml
    import sys
    import readline # needed for testing with anaconda3

    # provide file name of yaml file with configuration parameters
    # on the commandline
    if len(sys.argv) == 2:
        fname = sys.argv[-1]
        with open(fname, 'r') as f:
            conf = yaml.load(f)
    else:
        import os
        # input parameters as in GUI
        user_cfile = 'user_config.yaml' #'microcircuit.yaml'
        simulation_duration = 1000.
        thalamic_input = False
        threads = 16

        # load config file provided by user
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

    run_microcircuit(conf)
