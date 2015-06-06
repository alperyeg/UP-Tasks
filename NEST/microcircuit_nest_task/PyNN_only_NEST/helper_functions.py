# edited: Long Phan

# from network_params import *
import numpy as np

# ----------------------- Additional Code ---------------------------------
import yaml
with open('microcircuit.yaml' , 'r') as f:
    conf = yaml.load(f)

n_layers = conf['n_layers']
n_pops_per_layer = conf['n_pops_per_layer']
layers = conf['layers']
pops = conf['pops']
w_234 = conf['w_234']
w_mean = conf['w_mean']
g = conf['g']
simulator = conf['simulator']

# -------------------------------------------------------------------------


def create_weight_matrix(neuron_model, **kwargs):
    w = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
    if neuron_model == 'IF_curr_exp':
        for target_layer in layers:
            for target_pop in pops:
                target_index = conf['structure'][target_layer][target_pop]
                for source_layer in layers:
                    for source_pop in pops:
                        source_index = conf['structure'][source_layer][source_pop]
                        if source_pop == 'E':
                            if source_layer == 'L4' and target_layer == 'L23' and target_pop == 'E':
                                w[target_index][source_index] = w_234
                            else:
                                w[target_index][source_index] = w_mean
                        else:
                            w[target_index][source_index] = g * w_mean
    elif neuron_model == 'IF_cond_exp':
        g_e = kwargs.get('g_e', None)
        g_i = kwargs.get('g_i', None)
        # ----- add temporary code ----------
        conn_routine = 'fixed_total_number'
        # -----------------------------------

        if conn_routine == 'fixed_total_number' and simulator == 'nest':
        # inhibitory conductances get a minus sign
            for target_layer in layers:
                for target_pop in pops:
                    target_index = conf['structure'][target_layer][target_pop]
                    w[target_index] = n_layers * [g_e[target_layer][target_pop], -g_i[target_layer][target_pop]]
        else :
            for target_layer in layers:
                for target_pop in pops:
                    target_index = conf['structure'][target_layer][target_pop]
                    w[target_index] = n_layers * [g_e[target_layer][target_pop], g_i[target_layer][target_pop]]
    else:
        pass

    return w
