# edited: Long Phan

# from network_params import *
# from sim_params import *
from scaling import Scaling
from connectivity import Connectivity
from helper_functions import Help_func
# import pyNN
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np
# import sys


# ------------------------------ Additional Code ----------------------------
# move these additional code into class


# ---------------------------------------------------------------------------
class Network:
    import yaml
    from Init_microcircuit import Init_microcircuit
    with open('microcircuit.yaml', 'r') as f:
        conf = yaml.load(f)
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
    n_rec = mc.get_n_rec()

    def __init__(self, sim):
        return None

    def setup(self,sim):

        # if parallel_safe=False, PyNN offsets the seeds by 1 for each rank
        script_rng = NumpyRNG(seed=Network.pyseed,
                              parallel_safe=Network.parallel_safe)

        # Compute DC input before scaling
        if Network.input_type == 'DC':
            self.DC_amp = {}
            for target_layer in Network.layers:
                self.DC_amp[target_layer] = {}
                for target_pop in Network.pops:
                    self.DC_amp[target_layer][target_pop] = Network.bg_rate * Network.mc.properties['K_ext'][target_layer][target_pop] * Network.w_mean * Network.mc.properties['neuron_params']['tau_syn_E'] / 1000.
        else:
            self.DC_amp = {'L23': {'E':0., 'I':0.},
                           'L4' : {'E':0., 'I':0.},
                           'L5' : {'E':0., 'I':0.},
                           'L6' : {'E':0., 'I':0.}}

        # In-degrees of the full-scale and scaled models
        # sc = Scaling()
        K_full = Scaling.get_indegrees()
        self.K = Network.K_scaling * K_full

        self.K_ext = {}
        for layer in Network.layers:
            self.K_ext[layer] = {}
            for pop in Network.pops:
                self.K_ext[layer][pop] = Network.K_scaling * Network.mc.properties['K_ext'][layer][pop]

        self.w = Help_func.create_weight_matrix(Network.neuron_model)
        # Network scaling
        if Network.K_scaling != 1:
            self.w, self.w_ext, self.DC_amp = Scaling.adjust_w_and_ext_to_K(K_full, Network.K_scaling, self.w, self.DC_amp)

        Vthresh = {}
        for layer in Network.layers:
            Vthresh[layer] = {}
            for pop in Network.pops:
                Vthresh[layer][pop] = Network.mc.properties['neuron_params']['v_thresh']

        # Initial membrane potential distributions
        # The original study used V0_mean = -58 mV, V0_sd = 5 mV.
        # This is adjusted here to any changes in v_rest and scaling of V.
        V0_mean = {}
        V0_sd = {}
        for layer in Network.layers:
            V0_mean[layer] = {}
            V0_sd[layer] = {}
            for pop in Network.pops:
                V0_mean[layer][pop] = (Network.mc.properties['neuron_params']['v_rest'] + Vthresh[layer][pop]) / 2.
                V0_sd[layer][pop] = (Vthresh[layer][pop] -
                                     Network.mc.properties['neuron_params']['v_rest']) / 3.

        V_dist = {}
        for layer in Network.layers:
            V_dist[layer] = {}
            for pop in Network.pops:
                V_dist[layer][pop] = RandomDistribution( \
                    'normal', [V0_mean[layer][pop], V0_sd[layer][pop]], rng=script_rng)

        model = getattr(sim, Network.neuron_model)

        if Network.record_corr and Network.simulator == 'nest':
            # Create correlation recording device
            sim.nest.SetDefaults('correlomatrix_detector', {'delta_tau': 0.5})
            self.corr_detector = sim.nest.Create('correlomatrix_detector')
            sim.nest.SetStatus(self.corr_detector, {'N_channels': Network.n_layers*Network.n_pops_per_layer, \
                                                    'tau_max': Network.tau_max, 'Tstart': Network.tau_max})


        if sim.rank() == 0:
            print 'neuron_params:', Network.mc.properties['neuron_params']
            print 'K: ', self.K
            print 'K_ext: ', self.K_ext
            print 'w: ', self.w
            print 'w_ext: ', self.w_ext
            print 'DC_amp: ', self.DC_amp
            print 'V0_mean: '
            for layer in Network.layers:
                for pop in Network.pops:
                    print layer, pop, V0_mean[layer][pop]
            print 'n_rec:'
            for layer in Network.layers:
                for pop in Network.pops:
                    print layer, pop, Network.n_rec[layer][pop]
                    if not Network.record_fraction and Network.n_record > int(round(Network.N_full[layer][pop] * Network.N_scaling)):
                        print 'Note that requested number of neurons to record exceeds ', \
                               layer, pop, ' population size'


        # Create cortical populations
        self.pops = {}
        global_neuron_id = 1
        self.base_neuron_ids = {}
        for layer in Network.layers:
            self.pops[layer] = {}
            for pop in Network.pops:
                cellparams = Network.mc.properties['neuron_params']

                self.pops[layer][pop] = sim.Population( \
                    int(round(Network.mc.properties['N_full'][layer][pop] * Network.N_scaling)), \
                    model, cellparams=cellparams, label=layer+pop)
                this_pop = self.pops[layer][pop]

                # Provide DC input in the current-based case
                # DC input is assumed to be absent in the conductance-based case
                this_pop.set('i_offset', self.DC_amp[layer][pop])

                self.base_neuron_ids[this_pop] = global_neuron_id
                global_neuron_id += len(this_pop) + 2

                this_pop.initialize('v', V_dist[layer][pop])

                # Spike recording
                this_pop[0:Network.n_rec[layer][pop]].record()

                # Membrane potential recording
                if Network.record_v:
                    if Network.record_fraction:
                        n_rec_v = round(this_pop.size * Network.frac_record_v)
                    else :
                        n_rec_v = Network.n_record_v
                    this_pop[0 : n_rec_v].record_v()

                # Correlation recording
                if Network.record_corr and Network.simulator == 'nest':
                    index = Network.structure[layer][pop]
                    sim.nest.SetDefaults('static_synapse', {'receptor_type': index})
                    sim.nest.Connect(list(this_pop.all_cells), self.corr_detector)


        if Network.record_corr and Network.simulator == 'nest':
            # reset receptor_type
            sim.nest.SetDefaults('static_synapse', {'receptor_type': 0})

        if Network.thalamic_input:
        # Create thalamic population
            self.thalamic_population = sim.Population( \
                Network.thal_params['n_thal'], sim.SpikeSourcePoisson, {'rate': Network.thal_params['rate'], \
                'start': Network.thal_params['start'], 'duration': Network.thal_params['duration']}, \
                label='thalamic_population')
            self.base_neuron_ids[self.thalamic_population] = global_neuron_id
            global_neuron_id += len(self.thalamic_population) + 2

        possible_targets_curr = ['inhibitory', 'excitatory']

        # Connect

        for target_layer in Network.layers:
            for target_pop in Network.pops:
                target_index = Network.mc.properties['structure'][target_layer][target_pop]
                this_target_pop = self.pops[target_layer][target_pop]
                w_ext = self.w_ext
                # External inputs
                if Network.input_type == 'poisson':
                    rate = Network.bg_rate * self.K_ext[target_layer][target_pop]
                    if Network.simulator == 'nest':
                    # create only a single Poisson generator for each population,
                    # since the native NEST implementation sends independent spike trains to all targets
                        if sim.rank() == 0:
                            print 'connecting Poisson generator to', target_layer, target_pop, ' via SLI'
                        sim.nest.sli_run('/poisson_generator Create /poisson_generator_e Set poisson_generator_e << /rate ' \
                            + str(rate) + ' >> SetStatus')

                        sim.nest.sli_run("[ poisson_generator_e ] " + str(list(this_target_pop.all_cells)).replace(',', '') \
                            + " /all_to_all << /model /static_synapse /weight " + str(1000 * w_ext) \
                            + " /delay " + str(Network.mc.properties['d_mean']['E']) + " >> Connect")
                    else : # simulators other than NEST
                        if sim.rank() == 0:
                            print 'connecting Poisson generators to', target_layer, target_pop
                        poisson_generator = sim.Population(this_target_pop.size, \
                            sim.SpikeSourcePoisson, {'rate': rate})
                        conn = sim.OneToOneConnector(weights = w_ext)
                        sim.Projection(poisson_generator, this_target_pop, conn, target = 'excitatory')

                if Network.thalamic_input:
                    # Thalamic inputs
                    if sim.rank() == 0:
                        print 'creating thalamic connections to ' + target_layer + target_pop
                    C_thal = Network.thal_params['C'][target_layer][target_pop]
                    n_target = Network.N_full[target_layer][target_pop]
                    K_thal = round(np.log(1 - C_thal) / np.log((n_target * Network.thal_params['n_thal'] - 1.)/ \
                             (Network.n_target * Network.thal_params['n_thal']))) / Network.n_target * Network.K_scaling
                    Connectivity.FixedTotalNumberConnect(sim, self.thalamic_population, \
                        this_target_pop, K_thal, w_ext, Network.w_rel * w_ext, \
                        Network.d_mean['E'], Network.d_sd['E'])

                # Recurrent inputs
                for source_layer in Network.layers:
                    for source_pop in Network.pops:
                        source_index = Network.mc.properties['structure'][source_layer][source_pop]
                        this_source_pop = self.pops[source_layer][source_pop]
                        weight = self.w[target_index][source_index]

                        conn_type = possible_targets_curr[int((np.sign(weight)+1)/2)]

                        if sim.rank() == 0:
                            print 'creating connections from ' + source_layer + \
                            source_pop + ' to ' + target_layer + target_pop

                        if source_pop == 'E' and source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                            w_sd = weight * Network.w_rel_234
                        else:
                            w_sd = abs(weight * Network.w_rel)

                        Connectivity.FixedTotalNumberConnect( \
                            sim, this_source_pop, this_target_pop, \
                            self.K[target_index][source_index], weight, w_sd, \
                            Network.mc.properties['d_mean'][source_pop],
                                                             Network.mc.properties['d_sd'][source_pop])
