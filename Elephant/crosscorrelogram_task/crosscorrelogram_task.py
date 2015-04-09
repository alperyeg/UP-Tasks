#!/usr/bin/env python

# =============================================================================
# Initialization
# =============================================================================

from active_worker.task import task
from task_types import TaskTypes as tt
import numpy as np
import h5py

num_surrs = 10


@task
def crosscorrelogram_task(inputdata, number_of_jobs, job_id):
    '''
        Task Manifest Version: 1
        Full Name: crosscorrelogram_task
        Caption: cross-correlogram
        Author: Elephant-Developers
        Description: |
            This task calculates all pair-wise cross-correlograms between all
            combinations of spike trains in the input file. Significance of
            the correlation is evaluated based on spike-dither surrogates.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz']
        Accepts:
            inputdata: application/unknown
            number_of_jobs: long
            job_id: long

        Returns:
            res: application/unknown
    '''
    import quantities as pq
    import neo
    import elephant

    if job_id > number_of_jobs:
        print "Input data is invalid, exiting"
        return
    # =========================================================================
    # Load data
    # =========================================================================

    # stage the input file
    original_path = crosscorrelogram_task.task.uri.get_file(inputdata)

    session = neo.NeoHdf5IO(filename=original_path)
    block = session.read_block()

    # select spike trains
    sts = block.filter(use_st=True)

    # print("Number of spike trains: " + str(len(sts)))

    # =========================================================================
    # Cross-correlograms
    # =========================================================================

    max_lag_bins = 200
    lag_res = 1 * pq.ms
    max_lag = max_lag_bins * lag_res
    smoothing = 10 * pq.ms

    num_neurons = len(sts)

    cc = {}
    cc['unit_i'] = {}
    cc['unit_j'] = {}
    cc['times_ms'] = {}
    cc['original'] = {}
    cc['surr'] = {}
    cc['original_measure'] = {}
    cc['surr_measure'] = {}
    cc['pvalue'] = {}

    # create all combinations of tasks
    num_total_pairs = 0
    all_combos_unit_i = []
    all_combos_unit_j = []
    for ni in range(num_neurons):
        for nj in range(ni, num_neurons):
            all_combos_unit_i.append(ni)
            all_combos_unit_j.append(nj)
            num_total_pairs += 1

    # calculate indices in cc['unit_i'] list which to calculate for each task
    idx = np.linspace(0, num_total_pairs, number_of_jobs + 1, dtype=int)
    task_starts_idx = idx[:-1]
    task_stop_idx = idx[1:]

    # Loop over all pairs of neurons
    for calc_i in range(task_starts_idx[job_id], task_stop_idx[job_id]):
        # save neuron i,j index
        ni = all_combos_unit_i[calc_i]
        nj = all_combos_unit_j[calc_i]

        cc['unit_i'][calc_i] = ni
        cc['unit_j'][calc_i] = nj

        print("Cross-correlating %i and %i" % (ni, nj))

        # original CCH
        cco = elephant.spikecorr.cch(
            sts[ni], sts[nj], w=lag_res, lag=max_lag, smooth=smoothing)
        cc['original'][calc_i] = cco.magnitude
        cc['times_ms'][calc_i] = cco.times.rescale(pq.ms).magnitude

        # extract measure
        ind = np.argmin(np.abs(cco.times))
        ccom = cch_measure(cco, ind)
        cc['original_measure'][calc_i] = ccom

        surr_i = elephant.surrogates.spike_dithering(
            sts[ni], dither=50. * pq.ms, n=num_surrs)
        surr_j = elephant.surrogates.spike_dithering(
            sts[nj], dither=50. * pq.ms, n=num_surrs)

        ccs = []
        ccsm = []

        # cross-correlogram of each surrogate pair
        for surrogate in range(num_surrs):
            scc = elephant.spikecorr.cch(
                surr_i[surrogate], surr_j[surrogate],
                w=lag_res, lag=max_lag, smooth=smoothing)
            ccs.append(scc.magnitude)
            ccsm.append(cch_measure(scc, ind))
        cc['surr'][calc_i] = np.array(ccs)
        cc['surr_measure'][calc_i] = np.sort(ccsm)
        cc['pvalue'][calc_i] = np.count_nonzero(np.array(ccsm) >= ccom)

    # save result to hdf5
    outputname = 'cc_result'+str(number_of_jobs)+'_'+str(job_id)+'.h5'
    export_hdf5(cc, outputname)
    return crosscorrelogram_task.task.uri.save_file(mime_type='\
                                                    application/unknown',
                                                    src_path=outputname,
                                                    dst_path=outputname)
    # write parameters to disk
    # import h5py_wrapper.wrapper
    # h5py_wrapper.wrapper.add_to_h5(
    #     'correlation_output_' + filename + '_' + str(job_id) + '.h5',
    #     cc, write_mode='w', overwrite_dataset=True)


def cch_measure(cch, ind):
    return np.sum(cch[ind - 5:ind + 5].magnitude)


def export_hdf5(cc, outputname):
    # cc has type dict with 8-keys
    file = h5py.File(outputname, 'w')

    get_hdf5_surr_measure(cc, file)
    get_hdf5_original_measure(cc, file)
    get_hdf5_pvalue(cc, file)
    get_hdf5_unit_i(cc, file)
    get_hdf5_unit_j(cc, file)
    get_hdf5_times_ms(cc, file)
    get_hdf5_surr(cc, file)
    get_hdf5_original(cc, file)
    file.close()
    return file


def get_hdf5_surr_measure(cc, file):
    # --------------------- cc['surr_measure']
    # numsurr = 10 -> create_dataset(...(l_surr_measure, 10), ...)
    l_surr_measure = len(cc['surr_measure'])
    dataset_surr_measure = file.create_dataset("/cc_group/surr_measure",
                                               (l_surr_measure, num_surrs),
                                               dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_surr_measure, num_surrs))
    for i in range(l_surr_measure):
        for j in range(len(cc['surr_measure'][i])):
            data[i][j] = cc['surr_measure'][i][j]

    dataset_surr_measure[...] = data


def get_hdf5_original_measure(cc, file):
    # --------------------- cc['original_measure'] dict
    l_orig_measure = len(cc['original_measure'])
    dataset_orig_measure = file.create_dataset("/cc_group/original_measure",
                                               (l_orig_measure, 1),
                                               dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_orig_measure, 1))
    for i in range(l_orig_measure):
        data[i] = cc['original_measure'].items()[i][1]
    dataset_orig_measure[...] = data


def get_hdf5_pvalue(cc, file):
    # --------------------- cc['pvalue'] dict
    l_pvalue = len(cc['pvalue'])
    dataset_pvalue = file.create_dataset("/cc_group/pvalue",
                                         (l_pvalue, 1),
                                         dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_pvalue, 1))
    for i in range(l_pvalue):
        data[i] = cc['pvalue'].items()[i][1]
    dataset_pvalue[...] = data


def get_hdf5_unit_i(cc, file):
    # --------------------- cc['unit_i'] dict
    l_unit_i = len(cc['unit_i'])
    dataset_unit_i = file.create_dataset("/cc_group/unit_i",
                                         (l_unit_i, 1),
                                         dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_unit_i, 1))
    for i in range(l_unit_i):
        data[i] = cc['unit_i'].items()[i][1]
    dataset_unit_i[...] = data


def get_hdf5_unit_j(cc, file):
    # --------------------- cc['unit_j'] dict
    l_unit_j = len(cc['unit_j'])
    dataset_unit_j = file.create_dataset("/cc_group/unit_j",
                                         (l_unit_j, 1),
                                         dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_unit_j, 1))
    for i in range(l_unit_j):
        data[i] = cc['unit_j'].items()[i][1]
    dataset_unit_j[...] = data


def get_hdf5_times_ms(cc, file):
    # --------------------- cc['times_ms'] dict array
    l_times_ms = len(cc['times_ms'])
    dataset_times_ms = file.create_dataset("/cc_group/times_ms", (l_times_ms,
                                           cc['times_ms'][0].size),
                                           dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_times_ms, cc['times_ms'][0].size))
    for i in range(l_times_ms):
        for j in range(cc['times_ms'][i].size):
            data[i][j] = cc['times_ms'][i].item(j)
    dataset_times_ms[...] = data


def get_hdf5_surr(cc, file):
    # --------------------- cc['surr'] dict numpy.ndarray
    l_surr = len(cc['surr'])
    dataset_surr = file.create_dataset("/cc_group/surr",
                                       (l_surr, cc['surr'][0].size),
                                       dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_surr, cc['surr'][0].size))
    for i in range(l_surr):
        for j in range(cc['surr'][i].size):
            data[i][j] = cc['surr'][i].item(j)
    dataset_surr[...] = data


def get_hdf5_original(cc, file):
    # --------------------- cc['original']
    l_original = len(cc['original'])
    dataset_orig = file.create_dataset("/cc_group/original",
                                       (l_original, cc['original'][0].size),
                                       dtype=h5py.h5t.NATIVE_FLOAT)
    data = np.zeros((l_original, cc['original'][0].size))
    for i in range(l_original):
        for j in range(cc['original'][i].size):
            data[i][j] = cc['original'][i].item(j)
    dataset_orig[...] = data


if __name__ == '__main__':
    # this number relates to the "-t" parameter:
    #   -t 0-X => number_of_jobs=X+1
    # INPUT-second parameter
    # number_of_jobs is (0, 200]
    number_of_jobs = 1

    # INPUT-third parameter
    # job parameter: a number between 0 and number_of_jobs-1
    import os
    PBS_value = os.getenv('PBS_ARRAYID')
    if PBS_value is not None:
        job_id = int(PBS_value)
    else:
        job_id = 0

    # INPUT-first parameter
    inputdata = tt.URI('application/unknown', 'data/experiment.h5')
    crosscorrelogram_task(inputdata, number_of_jobs, job_id)
