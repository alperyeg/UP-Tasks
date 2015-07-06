#############################################################################
### Functions for computing and adjusting connection and input parameters ###
#############################################################################


import numpy as np


class Scaling:

    # ---------------------- Additional Code ------------------------------------
    import yaml
    with open('microcircuit.yaml', 'r') as f:
        conf = yaml.load(f)
    from Init_microcircuit import Init_microcircuit
    mc = Init_microcircuit(conf)
    n_layers = mc.properties['n_layers']
    n_pops_per_layer = mc.properties['n_pops_per_layer']
    layers = mc.properties['layers']
    pops = mc.properties['pops']
    input_type = mc.properties['params_dict']['nest']['input_type']
    w_mean = mc.properties['w_mean']
    bg_rate = mc.properties['bg_rate']
    # ---------------------------------------------------------------------------

    @staticmethod
    def get_indegrees():
        '''Get in-degrees for each connection for the full-scale (1 mm^2) model'''
        K = np.zeros([Scaling.n_layers*Scaling.n_pops_per_layer,
                      Scaling.n_layers*Scaling.n_pops_per_layer])
        for target_layer in Scaling.layers:
            for target_pop in Scaling.pops:
                for source_layer in Scaling.layers:
                    for source_pop in Scaling.pops:
                        target_index = Scaling.mc.properties['structure'][target_layer][target_pop]
                        source_index = Scaling.mc.properties['structure'][source_layer][source_pop]
                        n_target = Scaling.mc.properties['N_full'][target_layer][target_pop]
                        n_source = Scaling.mc.properties['N_full'][source_layer][source_pop]
                        K[target_index][source_index] = round(np.log(1. - \
                            Scaling.mc.properties['conn_probs'][target_index][source_index]) / np.log( \
                            (n_target * n_source - 1.) / (n_target * n_source))) / n_target
        return K

    @staticmethod
    def adjust_w_and_ext_to_K(K_full, K_scaling, w, DC):
        '''Adjust synaptic weights and external drive to the in-degrees
         to preserve mean and variance of inputs in the diffusion approximation'''

        w_new = w / np.sqrt(K_scaling)
        I_ext = {}
        for target_layer in Scaling.layers:
            I_ext[target_layer] = {}
            for target_pop in Scaling.pops:
                target_index = Scaling.mc.properties['structure'][target_layer][target_pop]
                x1 = 0
                for source_layer in Scaling.layers:
                    for source_pop in Scaling.pops:
                        source_index = Scaling.mc.properties['structure'][source_layer][source_pop]
                        x1 += w[target_index][source_index] * K_full[target_index][source_index] * \
                            Scaling.mc.properties['full_mean_rates'][source_layer][source_pop]

                if Scaling.input_type == 'poisson':
                    x1_ext = Scaling.w_mean * Scaling.mc.properties['K_ext'][target_layer][target_pop] * Scaling.bg_rate
                    w_ext_new = Scaling.w_mean / np.sqrt(K_scaling)
                    I_ext[target_layer][target_pop] = 0.001 * Scaling.mc.properties['neuron_params']['tau_syn_E'] * \
                        ((1. - np.sqrt(K_scaling)) * x1 + \
                        (1. - np.sqrt(K_scaling)) * x1_ext) + DC[target_layer][target_pop]
                else :
                    w_ext_new = np.nan
                    I_ext[target_layer][target_pop] = 0.001 * Scaling.mc.properties['neuron_params']['tau_syn_E'] * \
                        ((1. - np.sqrt(Scaling.K_scaling)) * x1) + DC[target_layer][target_pop]

        return w_new, w_ext_new, I_ext
