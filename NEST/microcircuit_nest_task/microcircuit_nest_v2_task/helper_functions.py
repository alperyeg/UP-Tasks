import numpy as np


class Help_func:

    @staticmethod
    def create_weight_matrix(neuron_model, conf, **kwargs):
        n_layers = conf['n_layers']
        n_pops_per_layer = conf['n_pops_per_layer']
        layers = conf['layers']
        pops = conf['pops']
        w_234 = conf['w_234']
        w_mean = conf['w_mean']
        g = conf['g']
        # simulator = conf['simulator']

        w = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
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

        return w
