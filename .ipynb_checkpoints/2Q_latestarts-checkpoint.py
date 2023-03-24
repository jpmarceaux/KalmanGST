import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import numpy as np
import time
from kalman_gst import *  
from pygsti.modelpacks import smq2Q_XYCNOT as std

import pygsti
from pygsti.baseobjs.resourceallocation import ResourceAllocation

# setup the FOGI model
mdl_datagen = std.target_model('H+s')
basis1q = pygsti.baseobjs.Basis.cast('pp', 4)
gauge_basis = pygsti.baseobjs.CompleteElementaryErrorgenBasis(
                        basis1q, mdl_datagen.state_space, elementary_errorgen_types='HS')
mdl_datagen.setup_fogi(gauge_basis, None, None, reparameterize=True,
                     dependent_fogi_action='drop', include_spam=True)
target_model = mdl_datagen.copy()

fogi_labels = mdl_datagen.fogi_errorgen_component_labels()

hidxs_1qb = []
hidxs_2qb = []
sidxs_1qb = []
sidxs_2qb = []
for idx, lbl in enumerate(fogi_labels):
    if 'H' in lbl and 'ga' not in lbl and 'Gcnot' in lbl:
        hidxs_2qb.append(idx)
    elif 'H' in lbl and 'ga' not in lbl:
        hidxs_1qb.append(idx)
    if 'S' in lbl and 'ga' not in lbl and 'Gcnot' in lbl:
        sidxs_2qb.append(idx)
    elif 'S' in lbl and 'ga' not in lbl:
        sidxs_1qb.append(idx)
        
STARTING_CIRC = 512
        
if rank == 0:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

num_iterations = 10
for iteration in range(num_iterations):
    # add noise to the models
    SEED = None
    SAMPLES = 256
    np.random.seed(SEED)

    max_stochastic_error_rate_1qb = 0.005
    max_stochastic_error_rate_2qb = 0.003

    hamiltonian_error_var_1qb = 0.01
    hamiltonian_error_var_2qb = 0.05

    ar = np.zeros(len(mdl_datagen.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=True)))

    # add hamiltonian noise
    ar[hidxs_1qb] = np.random.normal(0, hamiltonian_error_var_1qb, len(hidxs_1qb))
    ar[hidxs_2qb] = np.random.normal(0, hamiltonian_error_var_2qb, len(hidxs_2qb))

    mdl_datagen.set_fogi_errorgen_components_array(ar, include_fogv=False, normalized_elem_gens=True)
    print('hamiltonian-only MSE with target', mserror(mdl_datagen, target_model))
    print('hamiltonian-only agsi with target', avg_gs_infidelity(mdl_datagen, target_model))

    # add stochastic noise
    ar[sidxs_1qb] = max_stochastic_error_rate_1qb*np.random.rand(len(sidxs_1qb))
    ar[sidxs_2qb] = max_stochastic_error_rate_2qb*np.random.rand(len(sidxs_2qb))
    mdl_datagen.set_fogi_errorgen_components_array(ar, include_fogv=False, normalized_elem_gens=True)

    print('MSE with target', mserror(mdl_datagen, target_model))
    print('agsi with target', avg_gs_infidelity(mdl_datagen, target_model))
    print('model is cptp', model_is_cptp(mdl_datagen))
    print('two qubit gate infidelity', pygsti.report.entanglement_infidelity(mdl_datagen[('Gcnot', 0, 1)].to_dense(), target_model[('Gcnot', 0, 1)].to_dense(), 'pp'))
    
    #make a GST edesign and simulate the data
    maxLengths = [1, 2, 4]
    edesign = pygsti.protocols.StandardGSTDesign(target_model, std.prep_fiducials(), std.meas_fiducials(),
                                                    std.germs(), maxLengths)
    dataset = pygsti.data.simulate_data(mdl_datagen, edesign, SAMPLES, seed=SEED) #, sample_error='none')



    # Kalman estimation
    circuits = edesign.circuit_lists[-1][STARTING_CIRC::]

    filter_model = target_model.copy()
    covar_strength = mserror(mdl_datagen, target_model)/filter_model.num_params
    P = covar_strength*np.eye(filter_model.num_params)

    param_hist = [filter_model.to_vector()]
    covar_hist = [P]

    for circ in tqdm(circuits): 
        J = filter_model.sim.dprobs(circ, ResourceAllocation(comm=comm, mem_limit=1*(1024**3))) # Parallelize with MPI with 1 Gb/thread 
        if rank == 0:
            hilbert_dims = 2**circ.width
            probs = filter_model.probabilities(circ)
            p_model = vector_from_outcomes(probs, hilbert_dims)
            count_vec = vector_from_outcomes(dataset[circ].counts, hilbert_dims)
            observation = count_vec/sum(count_vec)
            mean_frequency = ( count_vec+np.ones(len(count_vec)) )/( sum(count_vec)+len(count_vec) )
            R = (1/(sum(count_vec)+len(count_vec)+1))*categorical_covar(mean_frequency)
            jacob = matrix_from_jacob(J, hilbert_dims)
            kgain = P@jacob.T@np.linalg.pinv(jacob@P@jacob.T + R, 1e-15)

            # Kalman update
            innovation = observation - p_model
            post_state = filter_model.to_vector() + kgain@innovation
            filter_model.from_vector(post_state)
            P = (np.eye(filter_model.num_params) - kgain@jacob)@P

            param_hist.append(post_state)
            covar_hist.append(P)
            
    

    # GST fits
    proto = pygsti.protocols.GST(target_model, gaugeopt_suite=None, verbosity=1)
    estimates = []
    for l in maxLengths:
        edesign = std.create_gst_experiment_design(l)
        data = pygsti.protocols.ProtocolData(edesign, dataset)
        results_after = proto.run(data)
        fit = results_after.estimates['GateSetTomography'].models['final iteration estimate']
        estimates.append(fit)

        

    if rank == 0:
        germ_length_ranges = {
            0: [0, 731], 
            1: [731, 1509], 
            2: [1509, 2999], 
        }
        cmap = plt.cm.get_cmap('PiYG', num_iterations)    # 11 discrete colors

        

        axs[0].set_title('Total Estimate Error')
        axs[1].set_title('Uncertainty')

        true_params = mdl_datagen.to_vector()
        ekf_error = []
        ekf_uncert = []
        for i in range(len(param_hist)):
            ekf_error.append( np.log10((param_hist[i]-true_params)@(param_hist[i]-true_params)) )
            ekf_uncert.append( np.log10((np.trace(covar_hist[i]))) )
        axs[0].plot(range(STARTING_CIRC, STARTING_CIRC+len(param_hist)), ekf_error, c=cmap(iteration))
        axs[1].plot(range(STARTING_CIRC, STARTING_CIRC+len(param_hist)), ekf_uncert, c=cmap(iteration))

        for i, mdl in enumerate(estimates):
            mle_error = np.log10(mserror(mdl, mdl_datagen))
            mle_line, = axs[0].plot(germ_length_ranges[i], (mle_error, mle_error), c=cmap(iteration), label='MLE Estimate')
            mle_line, = axs[1].plot(germ_length_ranges[i], (mle_error, mle_error), c=cmap(iteration), label='MLE Estimate')



        
axs[0].set_xlabel('Circuit Index')
axs[1].set_xlabel('Circuit Index')
axs[0].set_ylabel('MSE [log]')
plt.show()
fig.savefig('Figures/MSE_evo_2Q.png', dpi=350, format="png", bbox_inches='tight')