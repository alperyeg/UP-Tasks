#!/usr/bin/env python

# =============================================================================
# Initialization
# =============================================================================

from active_worker.task import task
from task_types import TaskTypes as tt
import unicore_client
import os


def load_local_file(name):
    """ loads local file and returns contents as string """
    file_path = os.path.join(os.path.dirname(__file__), name)
    with open(file_path) as f:
        return f.read()


@task
def cch_vista_submit_task(inputdata_spinnaker, inputdata_nest, script):
    '''
        Task Manifest Version: 1
        Full Name: cch_vista_submit_task
        Caption: cch_vista_submit_task
        Author: Elephant-Developers
        Description: |
            This task submitts a script to an HPC, calculates the pairwise 
            correlation and stores an p-value significance matrix, which can be 
            read in by the `vista` visualization framework.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz']
        Accepts:
            inputdata_spinnaker:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file generated from spinnaker simulation.
            inputdata_spinnaker:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file from nest simulation.
            script:
                type: application/unkown
                description: Script which will be executed on an HPC.

        Returns:
            res: application/unknown
    '''
    spinnaker_data_path = cch_vista_submit_task.task.uri.get_file(inputdata_spinnaker)
    nest_data_path = cch_vista_submit_task.task.uri.get_file(inputdata_nest)
    submit_script_path = cch_vista_submit_task.task.uri.get_file(script)
    code = {'To': '{}'.format(submit_script_path), 'Data': load_local_file('{}'.format(submit_script_path))}
    spinnaker_data = {'To': '{}'.format(spinnaker_data_path), 'Data': load_local_file('{}'.format(spinnaker_data_path))}
    nest_data = {'To': '{}'.format(nest_data_path), 'Data': load_local_file('{}'.format(nest_data_path))}
    inputs = [code, spinnaker_data, nest_data]

    oauth_token = cch_vista_submit_task.task.uri.get_oauth_token()
    auth = unicore_client.get_oidc_auth(oauth_token)
    # TODO: define job params for this task
    job = {}
    job['ApplicationName'] = 'Elephant'
    job['Parameters'] = {'ELEPHANT': submit_script_path, 'PARAMETERS': inputs}

    base_url = unicore_client.get_sites()['JURECA']['url']
    job_url = unicore_client.submit(os.path.join(base_url, 'jobs'), job, auth, inputs)
    unicore_client.wait_for_completion(job_url)
    print "Submitting to {}".format(base_url)


if __name__ == '__main__':
    inputdata_spinnaker = tt.URI('application/unknown', 'spikes_L5E.h5')
    inputdata_nest = tt.URI('application/unknown', 'spikes_L5E.h5')
    script = ''
    cch_vista_submit_task(inputdata_spinnaker, inputdata_nest, script)
