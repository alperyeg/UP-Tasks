import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_raster_bars(n_rec, network_pops, conf):

    layers = conf['layers']
    pops = conf['pops']
    n_layers = conf['n_layers']
    n_pops_per_layer = conf['n_pops_per_layer']
    t_start = conf['raster_t_min']
    raster_t_max = conf['raster_t_max']
    frac_to_plot = conf['frac_to_plot']
    output_path = conf['system_params']['output_path']
    sim_duration = conf['simulator_params']['nest']['sim_duration']

    if raster_t_max == 'sim_duration' or raster_t_max > sim_duration:
        t_stop = sim_duration
    else:
        t_stop = raster_t_max

    # dictionary of spike arrays, one entry for each population
    spikes = {}

    # read out spikes for each population
    for layer in sorted(layers)[::-1]:
        spikes[layer] = {}
        for pop in sorted(pops)[::-1]:
            fname = output_path + 'spikes_' + layer + pop + '.gdf'
            try:
                spike_array = np.loadtxt(fname)
                spike_array[:, [0, 1]] = spike_array[:, [1, 0]]
                first_id = network_pops[layer][pop].first_id
                spike_array[:, 1] = [gdf_id - first_id
                                     for gdf_id in
                                     spike_array[:, 1]]

            except IOError:
                print('reading spike data from ', fname, ' failed')
                pass
            spikes[layer][pop] = spike_array

    # Plot spike times in raster plot and bar plot with the average firing
    # rates of each population
    pop_labels = ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E',
                  'L6I']
    color = {'E': '#595289', 'I': '#af143c'}
    color_list = ['#595289', '#af143c']
    fig = plt.figure()
    axarr = []
    axarr.append(fig.add_subplot(121))
    axarr.append(fig.add_subplot(122))

    # Plot raster plot
    id_count = 0
    print("Mean rates")
    rates = {}
    for layer in sorted(layers)[::-1]:
        rates[layer] = {}
        for pop in sorted(pops)[::-1]:
            rate = 0.0
            t_spikes = spikes[layer][pop][:, 0]
            ids = spikes[layer][pop][:, 1] + (id_count + 1)
            filtered_times_indices = [
                np.where((t_spikes > t_start) & (t_spikes < t_stop))][0]
            t_spikes = t_spikes[filtered_times_indices]
            ids = ids[filtered_times_indices]

            # Compute rates with all neurons
            rate = 1000 * \
                len(t_spikes) / (t_stop - t_start) / n_rec[layer][pop]
            rates[layer][pop] = rate
            print(layer +  pop +': ' + ' {:.2f}'.format(rate))
            # Reduce data for raster plot
            num_neurons = frac_to_plot * n_rec[layer][pop]
            t_spikes = t_spikes[np.where(ids < num_neurons + id_count + 1)[0]]
            ids = ids[np.where(ids < num_neurons + id_count + 1)[0]]
            axarr[0].plot(t_spikes, ids, '.', color=color[pop])
            id_count += num_neurons

    rate_list = np.zeros(n_layers * n_pops_per_layer)
    for i, layer in enumerate(sorted(layers)):
        for j, pop in enumerate(sorted(pops)):
            rate_list[i * n_pops_per_layer + j] = rates[layer][pop]

    # Plot bar plot
    axarr[1].barh(np.arange(0, 8, 1) + 0.1, rate_list[::-1],
                  color=color_list[::-1] * 4)

    # Set labels
    axarr[0].set_ylim((0.0, id_count))
    axarr[0].set_yticklabels([])
    axarr[0].set_xlabel('time (ms)')
    axarr[1].set_ylim((0.0, 8.5))
    axarr[1].set_yticks(np.arange(0.5, 8.5, 1.0))
    axarr[1].set_yticklabels(pop_labels[::-1])
    axarr[1].set_xlabel('rate (spikes/s)')

    plt.savefig(output_path + 'spiking_activity.png')
