###################################################
###     	Main script			###
###################################################

import sys
# from sim_params import *
import yaml

# ---------------------------------------------------------------
with open('microcircuit.yaml', 'r') as f:
    conf = yaml.load(f)

sys.path.append(conf['system_params']['backend_path'])
sys.path.append(conf['system_params']['pyNN_path'])

simulator = conf['simulator']

# from network_params import *
# import pyNN
import time
import plotting
import numpy as np
import logging

# prepare simulation
logging.basicConfig()
# simulator = 'nest'
# exec('import pyNN.%s as sim' %simulator)
import pyNN.nest as sim
master_seed = conf['params_dict']['nest']['master_seed']
layers = conf['layers']
pops = conf['pops']
record_v = conf['params_dict']['nest']['record_v']
record_corr = conf['params_dict']['nest']['record_corr']
tau_max = conf['tau_max']
plot_spiking_activity = conf['plot_spiking_activity']
raster_t_min = conf['raster_t_min']
raster_t_max = conf['raster_t_max']
frac_to_plot = conf['frac_to_plot']
record_fraction = conf['params_dict']['nest']['record_fraction']
N_scaling = conf['params_dict']['nest']['N_scaling']
frac_record_spikes = conf['params_dict']['nest']['frac_record_spikes']
n_record = conf['params_dict']['nest']['n_record']

# Numbers of neurons from which to record spikes
n_rec = {}
for layer in layers:
    n_rec[layer] = {}
    for pop in pops:
        if record_fraction:
            n_rec[layer][pop] = min(int(round(conf['N_full'][layer][pop] * N_scaling * frac_record_spikes)), \
                                    int(round(conf['N_full'][layer][pop] * N_scaling)))
        else:
            n_rec[layer][pop] = min(n_record, int(round(conf['N_full'][layer][pop] * N_scaling)))


# ---------------------------------------------------------------

sim.setup(**conf['simulator_params'][simulator])

if simulator == 'nest':
    n_vp = sim.nest.GetKernelStatus('total_num_virtual_procs')
    if sim.rank() == 0:
        print 'n_vp: ', n_vp
        print 'master_seed: ', master_seed
    sim.nest.SetKernelStatus({'print_time' : False,
                              'dict_miss_is_error': False,
                              'grng_seed': master_seed,
                              'rng_seeds': range(master_seed + 1, master_seed + n_vp + 1)})

import network

# create network
start_netw = time.time()
n = network.Network(sim)
n.setup(sim)
end_netw = time.time()
if sim.rank() == 0:
    print 'Creating the network took ', end_netw - start_netw, ' s'

# simulate
if sim.rank() == 0:
    print "Simulating..."
start_sim = time.time()
t = sim.run(conf['simulator_params'][simulator]['sim_duration'])
end_sim = time.time()
if sim.rank() == 0:
    print 'Simulation took ', end_sim - start_sim, ' s'

start_writing = time.time()
for layer in layers:
   for pop in pops:
      filename = conf['system_params']['output_path'] + '/spikes_' + layer + pop + '.dat'
      n.pops[layer][pop].printSpikes(filename, gather=False)

if record_v:
    for layer in layers:
        for pop in pops:
            filename = conf['system_params']['output_path'] + '/voltages_' + layer + pop + '.dat'
            n.pops[layer][pop].print_v(filename, gather=False)

if record_corr and simulator == 'nest':
    if sim.nest.GetStatus(n.corr_detector, 'local')[0]:
        print 'getting count_covariance on rank ', sim.rank()
        cov_all = sim.nest.GetStatus(n.corr_detector, 'count_covariance')[0]
        delta_tau = sim.nest.GetStatus(n.corr_detector, 'delta_tau')[0]

        cov = {}
        for target_layer in np.sort(layers.keys()):
            for target_pop in pops:
                target_index = conf['structure'][target_layer][target_pop]
                cov[target_index] = {}
                for source_layer in np.sort(layers.keys()):
                    for source_pop in pops:
                        source_index = conf['structure'][source_layer][source_pop]
                        cov[target_index][source_index] = np.array(list(cov_all[target_index][source_index][::-1]) \
                        + list(cov_all[source_index][target_index][1:]))

        f = open(conf['system_params']['output_path'] + '/covariances.dat', 'w')
        print >>f, 'tau_max: ', tau_max
        print >>f, 'delta_tau: ', delta_tau
        print >>f, 'simtime: ', conf['simulator_params'][simulator]['sim_duration'], '\n'

        for target_layer in np.sort(layers.keys()):
            for target_pop in pops:
                target_index = conf['structure'][target_layer][target_pop]
                for source_layer in np.sort(layers.keys()):
                    for source_pop in pops:
                        source_index = conf['structure'][source_layer][source_pop]
                        print >>f, target_layer, target_pop, '-', source_layer, source_pop
                        print >>f, 'n_events_target: ', sim.nest.GetStatus(n.corr_detector, 'n_events')[0][target_index]
                        print >>f, 'n_events_source: ', sim.nest.GetStatus(n.corr_detector, 'n_events')[0][source_index]
                        for i in xrange(len(cov[target_index][source_index])):
                            print >>f, cov[target_index][source_index][i]
                        print >>f, ''
        f.close()


end_writing = time.time()
print "Writing data took ", end_writing - start_writing, " s"

if plot_spiking_activity and sim.rank()==0:
    plotting.plot_raster_bars( \
        raster_t_min, raster_t_max, n_rec, frac_to_plot, \
        conf['system_params']['output_path'])

sim.end()
