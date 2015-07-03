# edited: Long Phan

# from network_params import *
import numpy as np


class Help_func:
    # ----------------------- Additional Code ---------------------------------
    import yaml
    with open('microcircuit.yaml', 'r') as f:
        conf = yaml.load(f)
    from Init_microcircuit import Init_microcircuit
    mc = Init_microcircuit(conf)
    n_layers = mc.properties['n_layers']
    n_pops_per_layer = mc.properties['n_pops_per_layer']
    layers = mc.properties['layers']
    pops = mc.properties['pops']
    w_234 = mc.properties['w_234']
    w_mean = mc.properties['w_mean']
    g = mc.properties['g']
    simulator = mc.properties['simulator']

    # -------------------------------------------------------------------------

    @staticmethod
    def create_weight_matrix(neuron_model, **kwargs):
        w = np.zeros([Help_func.n_layers * Help_func.n_pops_per_layer, Help_func.n_layers * Help_func.n_pops_per_layer])
        if neuron_model == 'IF_curr_exp':
            for target_layer in Help_func.layers:
                for target_pop in Help_func.pops:
                    target_index = Help_func.mc.properties['structure'][target_layer][target_pop]
                    for source_layer in Help_func.layers:
                        for source_pop in Help_func.pops:
                            source_index = Help_func.mc.properties['structure'][source_layer][source_pop]
                            if source_pop == 'E':
                                if source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                                    w[target_index][source_index] = Help_func.w_234
                                else:
                                    w[target_index][source_index] = Help_func.w_mean
                            else:
                                w[target_index][source_index] = Help_func.g * Help_func.w_mean
        elif neuron_model == 'IF_cond_exp':
            g_e = kwargs.get('g_e', None)
            g_i = kwargs.get('g_i', None)
            # ----- add temporary code ----------
            conn_routine = 'fixed_total_number'
            # -----------------------------------

            if conn_routine == 'fixed_total_number' and Help_func.simulator == 'nest':
            # inhibitory conductances get a minus sign
                for target_layer in Help_func.layers:
                    for target_pop in Help_func.pops:
                        target_index = Help_func.mc.properties['structure'][target_layer][target_pop]
                        w[target_index] = Help_func.n_layers * [g_e[target_layer][target_pop], -g_i[target_layer][target_pop]]
            else :
                for target_layer in Help_func.layers:
                    for target_pop in Help_func.pops:
                        target_index = Help_func.mc.properties['structure'][target_layer][target_pop]
                        w[target_index] = Help_func.n_layers * [g_e[target_layer][target_pop], g_i[target_layer][target_pop]]
        else:
            pass

        return w
