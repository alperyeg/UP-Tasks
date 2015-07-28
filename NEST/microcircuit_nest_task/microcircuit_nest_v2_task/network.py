from scaling import Scaling
from connectivity import Connectivity
from helper_functions import Help_func
# import pyNN
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np
# import sys


class Network:

    def __init__(self, sim):
        return None

    def setup(self, sim, conf):
        from Init_microcircuit import Init_microcircuit
        mc = Init_microcircuit(conf)

        pyseed = mc.properties['params_dict']['nest']['pyseed']
        parallel_safe = mc.properties['params_dict']['nest']['parallel_safe']
        input_type = mc.properties['params_dict']['nest']['input_type']
        layers = mc.properties['layers']
        pops = mc.properties['pops']
        bg_rate = mc.properties['bg_rate']
        w_mean = mc.properties['w_mean']
        K_scaling = mc.properties['params_dict']['nest']['K_scaling']
        N_scaling = mc.properties['params_dict']['nest']['N_scaling']
        n_record = mc.properties['params_dict']['nest']['n_record']
        neuron_model = mc.properties['neuron_model']
        tau_max = mc.properties['tau_max']
        record_corr = mc.properties['params_dict']['nest']['record_corr']
        n_layers = mc.properties['n_layers']
        n_pops_per_layer = mc.properties['n_pops_per_layer']
        V0_mean = mc.properties['V0_mean']
        n_record_v = mc.properties['params_dict']['nest']['n_record_v']
        record_v = mc.properties['params_dict']['nest']['record_v']
        record_fraction = mc.properties['params_dict']['nest']['record_fraction']
        thalamic_input = mc.properties['thalamic_input']
        w_rel = mc.properties['w_rel']
        w_rel_234 = mc.properties['w_rel_234']
        simulator = mc.properties['simulator']
        N_full = mc.properties['N_full']
        K_ext = mc.properties['K_ext']
        tau_syn_E = mc.properties['neuron_params']['tau_syn_E']
        v_thresh = mc.properties['neuron_params']['v_thresh']
        v_rest = mc.properties['neuron_params']['v_rest']
        neuron_params = mc.properties['neuron_params']
        thal_params = mc.properties['thal_params']
        structure = mc.properties['structure']
        d_mean = mc.properties['d_mean']
        d_sd = mc.properties['d_sd']
        n_rec = mc.get_n_rec()

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        script_rng = NumpyRNG(seed=pyseed,
                              parallel_safe=parallel_safe)

        # Compute DC input before scaling
        if input_type == 'DC':
            self.DC_amp = {}
            for target_layer in layers:
                self.DC_amp[target_layer] = {}
                for target_pop in pops:
                    self.DC_amp[target_layer][target_pop] = bg_rate * K_ext[target_layer][target_pop] * w_mean * tau_syn_E / 1000.
        else:
            self.DC_amp = {'L23': {'E': 0., 'I': 0.},
                           'L4': {'E': 0., 'I': 0.},
                           'L5': {'E': 0., 'I': 0.},
                           'L6': {'E': 0., 'I': 0.}}

        # In-degrees of the full-scale and scaled models
        K_full = Scaling.get_indegrees(conf)
        self.K = K_scaling * K_full

        self.K_ext = {}
        for layer in layers:
            self.K_ext[layer] = {}
            for pop in pops:
                self.K_ext[layer][pop] = K_scaling * K_ext[layer][pop]

        self.w = Help_func.create_weight_matrix(neuron_model, conf)
        # Network scaling
        if K_scaling != 1:
            self.w, self.w_ext, self.DC_amp = Scaling.adjust_w_and_ext_to_K(K_full, K_scaling, self.w, self.DC_amp, conf)
        else:
            self.w_ext = w_mean

        Vthresh = {}
        for layer in layers:
            Vthresh[layer] = {}
            for pop in pops:
                Vthresh[layer][pop] = v_thresh

        # Initial membrane potential distributions
        # The original study used V0_mean = -58 mV, V0_sd = 5 mV.
        # This is adjusted here to any changes in v_rest and scaling of V.
        V0_mean = {}
        V0_sd = {}
        for layer in layers:
            V0_mean[layer] = {}
            V0_sd[layer] = {}
            for pop in pops:
                V0_mean[layer][pop] = (v_rest + Vthresh[layer][pop]) / 2.
                V0_sd[layer][pop] = (Vthresh[layer][pop] -
                                     v_rest) / 3.

        V_dist = {}
        for layer in layers:
            V_dist[layer] = {}
            for pop in pops:
                V_dist[layer][pop] = RandomDistribution('normal',
                                                        [V0_mean[layer][pop],
                                                         V0_sd[layer][pop]],
                                                        rng=script_rng)

        model = getattr(sim, neuron_model)

        if record_corr and simulator == 'nest':
            # Create correlation recording device
            sim.nest.SetDefaults('correlomatrix_detector', {'delta_tau': 0.5})
            self.corr_detector = sim.nest.Create('correlomatrix_detector')
            sim.nest.SetStatus(self.corr_detector, {'N_channels': n_layers*n_pops_per_layer,
                                                    'tau_max': tau_max,
                                                    'Tstart': tau_max,
                                                    })

        if sim.rank() == 0:
            print 'neuron_params:', mc.properties['neuron_params']
            print 'K: ', self.K
            print 'K_ext: ', self.K_ext
            print 'w: ', self.w
            print 'w_ext: ', self.w_ext
            print 'DC_amp: ', self.DC_amp
            print 'V0_mean: '
            for layer in layers:
                for pop in pops:
                    print layer, pop, V0_mean[layer][pop]
            print 'n_rec:'
            for layer in layers:
                for pop in pops:
                    print layer, pop, n_rec[layer][pop]
                    if not record_fraction and n_record > int(round(N_full[layer][pop] * N_scaling)):
                        print 'Note that requested number of neurons to record exceeds ', \
                               layer, pop, ' population size'

        # Create cortical populations
        self.pops = {}
        global_neuron_id = 1
        self.base_neuron_ids = {}
        device_list = []
        for layer in layers:
            self.pops[layer] = {}
            for pop in pops:
                cellparams = neuron_params

                self.pops[layer][pop] = sim.Population(int(round(N_full[layer][pop] * N_scaling)),
                                                       model,
                                                       cellparams=cellparams,
                                                       label=layer+pop)
                this_pop = self.pops[layer][pop]

                # Provide DC input in the current-based case
                # DC input is assumed to be absent in the conductance-based case
                this_pop.set('i_offset', self.DC_amp[layer][pop])

                self.base_neuron_ids[this_pop] = global_neuron_id
                global_neuron_id += len(this_pop) + 2

                this_pop.initialize('v', V_dist[layer][pop])

                # Spike recording
                # PYTHON2.6: SINCE SPIKES CANNOT BE RECORDED AT THE MOMENT
                # USING PYNN'S record(), WE CREATE AND CONNECT SPIKE DETECTORS
                # WITH PYNEST
                # this_pop[0:n_rec[layer][pop]].record()
                sd = sim.nest.Create('spike_detector',
                                     params={'label': 'spikes_{0}{1}'.format(layer, pop),
                                             'withtime': True,
                                             'withgid': True,
                                             'to_file': True})
                device_list.append(sd)
                sim.nest.Connect(list(this_pop[0:n_rec[layer][pop]].all_cells), sd)

                # Membrane potential recording
                if record_v:
                    if record_fraction:
                        n_rec_v = round(this_pop.size * Network.frac_record_v)
                    else :
                        n_rec_v = n_record_v
                    # PYTHON2.6: SINCE VOLTAGES CANNOT BE RECORDED AT THE MOMENT
                    # USING PYNN'S record_v(), WE CREATE AND CONNECT VOLTMETERS
                    # WITH PYNEST
                    # this_pop[0 : n_rec_v].record_v()
                    vm = sim.nest.Create('voltmeter',
                                         params={'label': 'voltages_{0}{1}'.format(layer, pop),
                                                 'withtime': True,
                                                 'withgid': True,
                                                 'to_file': True})
                    device_list.append(vm)
                    sim.nest.Connect(vm, list(this_pop[0 : n_rec_v]))

                # Correlation recording
                if record_corr and simulator == 'nest':
                    index = Network.structure[layer][pop]
                    sim.nest.SetDefaults('static_synapse', {'receptor_type': index})
                    sim.nest.Connect(list(this_pop.all_cells), self.corr_detector)


        if record_corr and simulator == 'nest':
            # reset receptor_type
            sim.nest.SetDefaults('static_synapse', {'receptor_type': 0})

        if thalamic_input:
        # Create thalamic population
            self.thalamic_population = sim.Population(thal_params['n_thal'],
                                                      sim.SpikeSourcePoisson,
                                                      {'rate': thal_params['rate'],
                                                       'start': thal_params['start'],
                                                       'duration': thal_params['duration']},
                                                      label='thalamic_population')
            self.base_neuron_ids[self.thalamic_population] = global_neuron_id
            global_neuron_id += len(self.thalamic_population) + 2

        possible_targets_curr = ['inhibitory', 'excitatory']

        # Connect

        for target_layer in layers:
            for target_pop in pops:
                target_index = structure[target_layer][target_pop]
                this_target_pop = self.pops[target_layer][target_pop]
                w_ext = self.w_ext
                # External inputs
                if input_type == 'poisson':
                    rate = bg_rate * self.K_ext[target_layer][target_pop]
                    if simulator == 'nest':
                    # create only a single Poisson generator for each population,
                    # since the native NEST implementation sends independent spike trains to all targets
                        if sim.rank() == 0:
                            print 'connecting Poisson generator to', target_layer, target_pop, ' via SLI'
                        sim.nest.sli_run('/poisson_generator Create /poisson_generator_e Set poisson_generator_e << /rate ' \
                            + str(rate) + ' >> SetStatus')

                        sim.nest.sli_run("[ poisson_generator_e ] " + str(list(this_target_pop.all_cells)).replace(',', '') \
                            + " /all_to_all << /model /static_synapse /weight " + str(1000 * w_ext) \
                            + " /delay " + str(d_mean['E']) + " >> Connect")
                    else : # simulators other than NEST
                        if sim.rank() == 0:
                            print 'connecting Poisson generators to', target_layer, target_pop
                        poisson_generator = sim.Population(this_target_pop.size, \
                            sim.SpikeSourcePoisson, {'rate': rate})
                        conn = sim.OneToOneConnector(weights = w_ext)
                        sim.Projection(poisson_generator, this_target_pop, conn, target = 'excitatory')

                if thalamic_input:
                    # Thalamic inputs
                    if sim.rank() == 0:
                        print 'creating thalamic connections to ' + target_layer + target_pop
                    C_thal = thal_params['C'][target_layer][target_pop]
                    n_target = N_full[target_layer][target_pop]
                    K_thal = round(np.log(1 - C_thal) / np.log((n_target * thal_params['n_thal'] - 1.)/ \
                             (n_target * thal_params['n_thal']))) / n_target * K_scaling
                    Connectivity.FixedTotalNumberConnect(sim, self.thalamic_population,
                                                         this_target_pop,
                                                         K_thal, w_ext,
                                                         w_rel * w_ext,
                                                         d_mean['E'],
                                                         d_sd['E'], conf)

                # Recurrent inputs
                for source_layer in layers:
                    for source_pop in pops:
                        source_index = structure[source_layer][source_pop]
                        this_source_pop = self.pops[source_layer][source_pop]
                        weight = self.w[target_index][source_index]

                        possible_targets_curr[int((np.sign(weight)+1)/2)]

                        if sim.rank() == 0:
                            print 'creating connections from ' + source_layer + \
                            source_pop + ' to ' + target_layer + target_pop

                        if source_pop == 'E' and source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                            w_sd = weight * w_rel_234
                        else:
                            w_sd = abs(weight * w_rel)

                        Connectivity.FixedTotalNumberConnect(sim,
                                                             this_source_pop,
                                                             this_target_pop,
                                                             self.K[target_index][source_index],
                                                             weight, w_sd,
                                                             d_mean[source_pop]
                                                             , d_sd[source_pop]
                                                             , conf)

        return device_list
