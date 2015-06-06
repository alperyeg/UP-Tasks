# edited: Long Phan

# from network_params import *
# import random
# from scipy import stats
import os
# import math
# import pyNN
# from pyNN.random import RandomDistribution
# from sim_params import *

# --------------------------- Additional Code ---------------------
import yaml
with open('microcircuit.yaml', 'r') as f:
    conf = yaml.load(f)

simulator = conf['simulator']
save_connections = conf['params_dict']['nest']['save_connections']
# -----------------------------------------------------------------


def FixedTotalNumberConnect(sim, pop1, pop2, K, w_mean, w_sd, d_mean, d_sd):
    """
        Function connecting two populations with multapses and a fixed total
        number of synapses Using new NEST implementation of Connect
    """

    if not K:
        return

    source_neurons = list(pop1.all_cells)
    target_neurons = list(pop2.all_cells)
    n_syn = int(round(K*len(target_neurons)))
    # weights are multiplied by 1000 because NEST uses pA whereas PyNN uses nA
    # RandomPopulationConnectD is called on each process with the full sets of
    # source and target neurons, and internally only connects the target
    # neurons on the current process.

    conn_dict = {'rule' : 'fixed_total_number',
                 'N'    : n_syn}

    syn_dict = {'model' : 'static_synapse',
                'weight': {'distribution': 'normal_clipped',
                           'mu': 1000. * w_mean,
                           'sigma': 1000. * w_sd},
                'delay' : {'distribution': 'normal_clipped',
                           'low': conf['simulator_params'][simulator]['min_delay'],
                           'mu': d_mean,
                           'sigma': d_sd}}
    if w_mean > 0:
       syn_dict['weight']['low'] = 0.0
    if w_mean < 0:
       syn_dict['weight']['high'] = 0.0

    sim.nest.sli_push(source_neurons)
    sim.nest.sli_push(target_neurons)
    sim.nest.sli_push(conn_dict)
    sim.nest.sli_push(syn_dict)
    sim.nest.sli_run("Connect")

    if save_connections:
        # - weights are in pA
        # - no header lines
        # - one file for each MPI process
        # - GIDs

        # get connections to target on this MPI process
        conn = sim.nest.GetConnections(source=source_neurons, target=target_neurons)
        conns = sim.nest.GetStatus(conn, ['source', 'target', 'weight', 'delay'])
        if not os.path.exists(conf['system_params']['conn_dir']):
            try:
                os.makedirs(conf['system_params']['conn_dir'])
            except OSError, e:
                if e.errno != 17:
                    raise
                pass
        f = open(conf['system_params']['conn_dir'] +  '/' + pop1.label + "_" + \
                 pop2.label + '.conn' + str(sim.rank()), 'w')
        for c in conns:
            print >> f, str(c).replace('(','').replace(')','').replace(', ', '\t')
        f.close()
