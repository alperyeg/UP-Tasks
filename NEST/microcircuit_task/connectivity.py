class Connectivity:

    @staticmethod
    def FixedTotalNumberConnect(sim, pop1, pop2, K, w_mean, w_sd, d_mean, d_sd,
                                conf):
        """
            Function connecting two populations with multapses and a fixed
            total number of synapses using new NEST implementation of Connect
        """

        simulator = conf['simulator']
        save_connections = conf['params_dict']['nest']['save_connections']
        if not K:
            return

        source_neurons = list(pop1.all_cells)
        target_neurons = list(pop2.all_cells)
        n_syn = int(round(K*len(target_neurons)))

        conn_dict = {'rule': 'fixed_total_number', 'N': n_syn}

        syn_dict = {'model': 'static_synapse',
                    'weight': {'distribution': 'normal_clipped',
                               'mu': 1000. * w_mean,
                               'sigma': 1000. * w_sd},
                    'delay': {'distribution': 'normal_clipped',
                              'low': conf['simulator_params'][simulator]['min_delay'],
                              'mu': d_mean,
                              'sigma': d_sd}}
        if w_mean > 0:
            syn_dict['weight']['low'] = 0.0
        if w_mean < 0:
            syn_dict['weight']['high'] = 0.0

        sim.nest.Connect(source_neurons, target_neurons, conn_dict, syn_dict)


        if save_connections:
            print "save_connections was set to True, but connections are \
                currently not saved because the output exceeds the user's \
                default capacity on the UP."
            pass

            # - weights are in pA
            # - no header lines
            # - one file for each MPI process
            # - GIDs

            # get connections to target on this MPI process
            # conn = sim.nest.GetConnections(source=source_neurons,
            #                                target=target_neurons)
            # conns = sim.nest.GetStatus(conn, ['source', 'target', 'weight',
            #                                   'delay'])
            # if not os.path.exists(conf['system_params']['output_path']):
            #     try:
            #         os.makedirs(conf['system_params']['conn_dir'])
            #     except OSError, e:
            #         if e.errno != 17:
            #             raise
            #         pass
            # f = open(conf['system_params']['output_path'] + '/' + 'conn_' + pop1.label +
            #          '_' + pop2.label + '_' + str(sim.rank())'.dat' , 'w')
            # for c in conns:
            #     print >> f, str(c).replace('(', '').replace(')', '').replace(', ', '\t')
            # f.close()

        return
