import numpy as np


class Help_func:

    @staticmethod
    def create_weight_matrix(conf):
        n_layers = conf['n_layers']
        n_pops_per_layer = conf['n_pops_per_layer']
        layers = conf['layers']
        pops = conf['pops']
        w_234 = conf['w_234']
        w_mean = conf['w_mean']
        g = conf['g']

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


    @staticmethod
    def get_n_rec(conf):
        '''
            compute the number of neurons from which to record spikes
        '''
        N_full = conf['N_full']
        N_scaling = conf['params_dict']['nest']['N_scaling']
        frac_record_spikes = conf['params_dict']['nest']['frac_record_spikes']
        layers = conf['layers']
        pops = conf['pops']
        record_fraction = conf['params_dict']['nest']['record_fraction']
        n_record = conf['params_dict']['nest']['n_record']

        n_rec = {}
        for layer in layers:
            n_rec[layer] = {}
            for pop in pops:
                if record_fraction:
                    n_rec[layer][pop] = min(int(round(N_full[layer][pop] *
                                                      N_scaling *
                                                      frac_record_spikes)),
                                            int(round(N_full[layer][pop] *
                                                      N_scaling)))
                else:
                    n_rec[layer][pop] = min(n_record,
                                            int(round(N_full[layer][pop] *
                                                      N_scaling)))
        return n_rec
