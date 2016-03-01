import scaling
import connectivity
import helper_functions
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np


class Network:

    def __init__(self, sim):
        return None

    def setup(self, sim, conf):

        # extract parameters
        pyseed = conf['params_dict']['nest']['pyseed']
        parallel_safe = conf['params_dict']['nest']['parallel_safe']
        input_type = conf['params_dict']['nest']['input_type']
        layers = conf['layers']
        pops = conf['pops']
        bg_rate = conf['bg_rate']
        w_mean = conf['w_mean']
        K_scaling = conf['params_dict']['nest']['K_scaling']
        N_scaling = conf['params_dict']['nest']['N_scaling']
        n_record = conf['params_dict']['nest']['n_record']
        neuron_model = conf['neuron_model']
        tau_max = conf['tau_max']
        record_corr = conf['params_dict']['nest']['record_corr']
        n_layers = conf['n_layers']
        n_pops_per_layer = conf['n_pops_per_layer']
        V0_mean = conf['V0_mean']
        n_record_v = conf['params_dict']['nest']['n_record_v']
        record_v = conf['params_dict']['nest']['record_v']
        record_fraction = conf['params_dict']['nest']['record_fraction']
        thalamic_input = conf['thalamic_input']
        w_rel = conf['w_rel']
        w_rel_234 = conf['w_rel_234']
        simulator = conf['simulator']
        N_full = conf['N_full']
        K_ext = conf['K_ext']
        tau_syn_E = conf['neuron_params']['tau_syn_E']
        v_thresh = conf['neuron_params']['v_thresh']
        v_rest = conf['neuron_params']['v_rest']
        neuron_params = conf['neuron_params']
        thal_params = conf['thal_params']
        structure = conf['structure']
        d_mean = conf['d_mean']
        d_sd = conf['d_sd']
        frac_record_v = conf['params_dict']['nest']['frac_record_v']
        n_rec = helper_functions.get_n_rec(conf)

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        script_rng = NumpyRNG(seed=pyseed,
                              parallel_safe=parallel_safe)

        # Compute DC input before scaling
        if input_type == 'DC':
            self.DC_amp = {}
            for target_layer in sorted(layers):
                print(target_layer)
                self.DC_amp[target_layer] = {}
                for target_pop in sorted(pops):
                    self.DC_amp[target_layer][target_pop] = bg_rate * \
                        K_ext[target_layer][target_pop] * \
                        w_mean * tau_syn_E / 1000.
        else:
            self.DC_amp = {'L23': {'E': 0., 'I': 0.},
                           'L4': {'E': 0., 'I': 0.},
                           'L5': {'E': 0., 'I': 0.},
                           'L6': {'E': 0., 'I': 0.}}

        # In-degrees of the full-scale and scaled models
        K_full = scaling.get_indegrees(conf)
        self.K = K_scaling * K_full

        self.K_ext = {}
        for layer in sorted(layers):
            self.K_ext[layer] = {}
            for pop in sorted(pops):
                self.K_ext[layer][pop] = K_scaling * K_ext[layer][pop]

        self.w = helper_functions.create_weight_matrix(conf)
        # Network scaling
        if K_scaling != 1:
            self.w, self.w_ext, self.DC_amp = scaling.adjust_w_and_ext_to_K(
                K_full, K_scaling, self.w, self.DC_amp, conf)
        else:
            self.w_ext = w_mean

        Vthresh = {}
        for layer in sorted(layers):
            Vthresh[layer] = {}
            for pop in sorted(pops):
                Vthresh[layer][pop] = v_thresh

        # Initial membrane potential distributions
        # The original study used V0_mean = -58 mV, V0_sd = 5 mV.
        # This is adjusted here to any changes in v_rest and scaling of V.
        V0_mean = {}
        V0_sd = {}
        for layer in sorted(layers):
            V0_mean[layer] = {}
            V0_sd[layer] = {}
            for pop in sorted(pops):
                V0_mean[layer][pop] = (v_rest + Vthresh[layer][pop]) / 2.
                V0_sd[layer][pop] = (Vthresh[layer][pop] -
                                     v_rest) / 3.

        V_dist = {}
        for layer in sorted(layers):
            V_dist[layer] = {}
            for pop in sorted(pops):
                try: # pyNN 0.8.0
                    V_dist[layer][pop] = RandomDistribution('normal',
                                                            mu=V0_mean[layer][pop],
                                                            sigma= V0_sd[layer][pop],
                                                            rng=script_rng)
                except TypeError: # pyNN 0.7.5
                    V_dist[layer][pop] = RandomDistribution('normal',
                                                            [V0_mean[layer][pop],
                                                             V0_sd[layer][pop]],
                                                            rng=script_rng)


        model = getattr(sim, neuron_model)

        if record_corr and simulator == 'nest':
            # Create correlation recording device
            sim.nest.SetDefaults('correlomatrix_detector', {'delta_tau': 0.5})
            self.corr_detector = sim.nest.Create('correlomatrix_detector')
            sim.nest.SetStatus(
                self.corr_detector, {'N_channels': n_layers * n_pops_per_layer,
                                     'tau_max': tau_max,
                                     'Tstart': tau_max,
                                     })

        if sim.rank() == 0:
            print('neuron_params:', conf['neuron_params'])
            print('K: ', self.K)
            print('K_ext: ', self.K_ext)
            print('w: ', self.w)
            print('w_ext: ', self.w_ext)
            print('DC_amp: ', self.DC_amp)
            print('V0_mean: ')
            for layer in layers:
                for pop in pops:
                    print(layer, pop, V0_mean[layer][pop])
            print('n_rec:')
            for layer in sorted(layers):
                for pop in sorted(pops):
                    print(layer, pop, n_rec[layer][pop])
                    if not record_fraction and n_record > \
                       int(round(N_full[layer][pop] * N_scaling)):
                        print('Note that requested number of neurons to record')
                        print('exceeds ', layer, pop, ' population size')

        # Create cortical populations
        self.pops = {}
        global_neuron_id = 1
        self.base_neuron_ids = {}
        # list containing the GIDs of recording devices, needed for output
        # bundle
        device_list = []
        for layer in sorted(layers):
            self.pops[layer] = {}
            for pop in sorted(pops):
                cellparams = neuron_params
                self.pops[layer][pop] = sim.Population(
                    int(round(N_full[layer][pop] * N_scaling)),
                    model,
                    cellparams=cellparams,
                    label=layer + pop)
                this_pop = self.pops[layer][pop]

                # Provide DC input in the current-based case
                # DC input is assumed to be absent in the conductance-based
                # case
                try: # pyNN 0.8.0
                    this_pop.set(i_offset = self.DC_amp[layer][pop])
                except TypeError: # pyNN 0.7.5
                    this_pop.set('i_offset', self.DC_amp[layer][pop])

                self.base_neuron_ids[this_pop] = global_neuron_id
                global_neuron_id += len(this_pop) + 2

                try: # pyNN 0.8.0
                    this_pop.initialize(v = V_dist[layer][pop])
                except TypeError: # pyNN 0.7.5
                    this_pop.initialize('v', V_dist[layer][pop])

                # Spike recording
                sd = sim.nest.Create('spike_detector',
                                     params={
                                         'label': 'spikes_{0}{1}'.format(layer, pop),
                                         'withtime': True,
                                         'withgid': True,
                                         'to_file': True})
                device_list.append(sd)
                sim.nest.Connect(
                    list(this_pop[0:n_rec[layer][pop]].all_cells), sd)

                # Membrane potential recording
                if record_v:
                    if record_fraction:
                        n_rec_v = round(this_pop.size * frac_record_v)
                    else:
                        n_rec_v = n_record_v
                    vm = sim.nest.Create('voltmeter',
                                         params={
                                             'label': 'voltages_{0}{1}'.format(layer, pop),
                                             'withtime': True,
                                             'withgid': True,
                                             'to_file': True})

                    device_list.append(vm)
                    sim.nest.Connect(vm, list(this_pop[0:n_rec_v]))

                # Correlation recording
                if record_corr and simulator == 'nest':
                    index = structure[layer][pop]
                    sim.nest.SetDefaults(
                        'static_synapse', {'receptor_type': index})
                    sim.nest.Connect(list(this_pop.all_cells),
                                     self.corr_detector)
                    sim.nest.SetDefaults(
                        'static_synapse', {'receptor_type': 0})

        if thalamic_input:
            self.thalamic_population = sim.nest.Create('parrot_neuron',
                                                       thal_params['n_thal'])
            # create and connect a poisson generator for stimulating the
            # thalamic population
            thal_pg = sim.nest.Create('poisson_generator',
                                      params={'rate': thal_params['rate'],
                                              'start': thal_params['start'],
                                              'stop': thal_params['start'] \
                                              + thal_params['duration']})
            sim.nest.Connect(thal_pg, self.thalamic_population)

        possible_targets_curr = ['inhibitory', 'excitatory']

        # Connect
        for target_layer in sorted(layers):
            for target_pop in sorted(pops):
                target_index = structure[target_layer][target_pop]
                this_target_pop = self.pops[target_layer][target_pop]
                w_ext = self.w_ext
                # External inputs
                if input_type == 'poisson':
                    rate = bg_rate * self.K_ext[target_layer][target_pop]
                    if simulator == 'nest':
                        # create only a single Poisson generator for each
                        # population, since the native NEST implementation sends
                        # independent spike trains to all targets
                        if sim.rank() == 0:
                            print('Connecting Poisson generator to',
                                  target_layer, target_pop)

                        pg = sim.nest.Create(
                            'poisson_generator', params={'rate': rate})

                        conn_dict = {'rule': 'all_to_all'}
                        syn_dict = {'model': 'static_synapse',
                                    'weight': 1000. * w_ext,
                                    'delay': d_mean['E']}
                        sim.nest.Connect(
                            pg, list(this_target_pop.all_cells), conn_dict,
                                                                 syn_dict)

                if thalamic_input:
                    if sim.rank() == 0:
                        print('Creating thalamic connections to ', target_layer,
                              target_pop)
                    C_thal = thal_params['C'][target_layer][target_pop]
                    n_target = N_full[target_layer][target_pop]
                    K_thal = round(np.log(1 - C_thal) / \
                                   np.log(
                                       (n_target * thal_params['n_thal'] - 1.) /
                                       (n_target * thal_params['n_thal']))) / \
                             n_target * K_scaling

                    target_neurons = list(this_target_pop.all_cells)
                    n_syn = int(round(K_thal * len(target_neurons)))
                    conn_dict = {'rule': 'fixed_total_number', 'N': n_syn}
                    syn_dict = {'model': 'static_synapse',
                                'weight': {'distribution': 'normal_clipped',
                                           'mu': 1000. * w_ext,
                                           'sigma': 1000. * w_rel * w_ext},
                                'delay': {'distribution': 'normal_clipped',
                                          'low': conf['simulator_params'] \
                                                     [simulator]['min_delay'],
                                          'mu': d_mean['E'],
                                          'sigma': d_sd['E']}}
                    sim.nest.Connect(self.thalamic_population, target_neurons,
                                     conn_dict, syn_dict)

                # Recurrent inputs
                for source_layer in sorted(layers):
                    for source_pop in sorted(pops):
                        source_index = structure[source_layer][source_pop]
                        this_source_pop = self.pops[source_layer][source_pop]
                        weight = self.w[target_index][source_index]

                        possible_targets_curr[int((np.sign(weight) + 1) / 2)]

                        if sim.rank() == 0:
                            print('Creating connections from ', source_layer + \
                                source_pop + ' to ' + target_layer + target_pop)

                        if source_pop == 'E' and source_layer == 'L4' and \
                           target_layer == 'L23' and target_pop == 'E':
                            w_sd = weight * w_rel_234
                        else:
                            w_sd = abs(weight * w_rel)
                        connectivity.FixedTotalNumberConnect(
                            sim,
                            this_source_pop,
                            this_target_pop,
                            self.K[target_index][
                                source_index],
                            weight, w_sd,
                            d_mean[source_pop], d_sd[source_pop], conf)

        return device_list
