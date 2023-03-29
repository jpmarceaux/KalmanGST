import numpy as np
import pygsti
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def process_is_cptp(process_mat):
    """
    returns true if the given process matrix has a 
        positive, semi-definite choi representation
    else returns false
    """
    choi_mat = pygsti.tools.jamiolkowski_iso(process_mat)
    eigs = np.linalg.eigvals(choi_mat)
    if np.any(eigs < -1e-6):
        return False
    else:
        return True
    
def model_is_cptp(model_in):
    """
    returns true if all error processes in the model are CPTP
    """
    members_not_cptp = 0
    for key in model_in.operations.keys():
        if not process_is_cptp(model_in[key].to_dense()):
            members_not_cptp += 1
    for key in model_in.preps.keys():
        if not process_is_cptp(model_in[key].error_map.to_dense()):
            members_not_cptp += 1
    for key in model_in.povms.keys():
        if not process_is_cptp(model_in[key].error_map.to_dense()):
            members_not_cptp += 1
    if members_not_cptp == 0:
        return True
    else:
        return False
    
from scipy.optimize import NonlinearConstraint, minimize 

def choi_eigs(param_vec, ref_model):
    """
    return a stacked vector of eigenvalues of all model choi matrices
    """
    ref_model.from_vector(param_vec)
    cmats = []
    for g in ref_model.operations.keys():
        cmats.append(pygsti.tools.jamiolkowski.jamiolkowski_iso(ref_model[g].to_dense()))
    cmats.append(pygsti.tools.jamiolkowski.jamiolkowski_iso(ref_model['Mdefault'].error_map.to_dense()))
    cmats.append(pygsti.tools.jamiolkowski.jamiolkowski_iso(ref_model['rho0'].error_map.to_dense()))
    return np.hstack([np.linalg.eigvals(c).real for c in cmats])

def MSE(x1, x2):
    return sum((x1[i] - x2[i])**2 for i in range(len(x1)))

def project_params_to_cp(model):
    """
    find the "nearest" cp model to the given model
    """
    constraints = NonlinearConstraint(lambda x: choi_eigs(x, model), 0, np.inf)
    res = minimize(MSE, model.to_vector(), args=model.to_vector(), constraints=[constraints])  
    return res.x
    
from pygsti.report import reportables as rptbl

def avg_gs_infidelity(model, noise_model, qubits=1):
    aei = 0
    for op in list(model.operations.keys()):
        aei += pygsti.report.entanglement_infidelity(model[op].to_dense(), noise_model[op].to_dense(), 'pp')
    aei += pygsti.report.entanglement_infidelity(model[('Mdefault')].error_map.to_dense(), noise_model[('Mdefault')].error_map.to_dense(), 'pp')
    aei += pygsti.report.entanglement_infidelity(model[('rho0')].error_map.to_dense(), noise_model[('rho0')].error_map.to_dense(), 'pp')
    return aei/(len(list(model.operations.keys()))+2)


def vector_from_outcomes(outcomes, num_outcomes):
    """
    returns a vector from pygsti probability outcomes
    
    --- Arguments ---
    outcomes: dictionary returned by pygsti.model.probabilities 
    num_outcomes: dimension of the output hilbert space
    """
    vecout = np.zeros((num_outcomes))
    for key in outcomes.keys():
        vecout[int(key[0], 2)] = outcomes[key]
    return(vecout)

def matrix_from_jacob(jacob, num_outcomes):
    """
    returns a matrx from a pygsti probability jacobian
    
    --- Arguments ---
    jacob: jacobian returned by pygsti.model.dprobs 
    num_outcomes: dimension of the output hilbert space
    """
    matout = np.zeros((num_outcomes, len(jacob['0'*int(np.log2(num_outcomes))])))
    for key in jacob.keys():
        matout[int(key[0], 2), :] = np.array(jacob[key])
    return matout

def tensor_from_hessian(hessian, hilbert_dims):
    """
    returns a 3-tensor from a pygsti probability hessian
    
    --- Arguments ---
    hessian: hessian returned by pygsti.model.hprobs 
    num_outcomes: dimension of the output hilbert space
    """
    num_params = len(hessian['0'*int(np.log2(hilbert_dims))])
    tensor_out = np.zeros((hilbert_dims, num_params, num_params))
    for key in hessian.keys():
        tensor_out[int(key[0], 2), :, :] = hessian[key]
    return tensor_out

def categorical_covar(prob_vec):
    """
    Outputs the covariance of a categorical random variable
    drawn from the provided probability vector
    
    --- Arguments ---
    prob_vec: underlying probability vector 
    """
    return np.diag(prob_vec) - np.outer(prob_vec, prob_vec)

def dirichlet_covar(count_vec):
    total_counts = sum(count_vec)
    hdims = len(count_vec)
    mean_frequency = ( count_vec + np.ones(hdims) )/( total_counts + hdims )
    return 1/(total_counts + hdims + 1) * categorical_covar(mean_frequency)

class ExtendedKalmanFilter():
    """
    An extended Kalman filter for gate-set tomography
    
    --- Parameters ---
    model: an underlying pygsti model
    num_params: number of parameters in the pygsti model
    P: current covariance matrix
    """
    def __init__(self, model, P0):
        self.model = model.copy()
        self.P = P0
        
        self.param_history = [self.model.to_vector()]
        self.covar_history = [self.P]
        
    def update(self, circ, count_vec, clip_range=[-1,1], Q=None, R_additional=None, max_itr=1, itr_eps=1e-4):
        """
        Makes an exact update to the model
        where the jacobian is calculated as the current estimate
        
        --- Arguments ---
        circ: pygsti circuit used in the update
        count_vec: vector of observed counts
        clip_range: reasonable clipping range for the parameter update
        Q: state-space covariance 
        R_additional: additional measurement covariance
        max_itr: maximum number of iterations to the update
        itr_eps: epsilon for minimum difference to end iterated updates
        
        --- Returns --- 
        innovation: the prior innovation
        kgain: the Kalman gain
        """
        prior_covar = self.P
        prior_state = self.model.to_vector()
        hilbert_dims = 2**(circ.width)
        
        for itr in range(max_itr):
            # find the predicted frequency for the circuit outcome under the model
            probs = self.model.probabilities(circ)
            p_model = vector_from_outcomes(probs, hilbert_dims)
            
            # calculate the observed frequency
            total_counts = sum(count_vec)
            observation = count_vec/total_counts

            # calculate jacobian
            jacob = matrix_from_jacob(self.model.sim.dprobs(circ), 2**circ.width)

            # calculate the covaraiance of the observation
            mean_frequency = ( count_vec+np.ones(len(count_vec)) )/( sum(count_vec)+len(count_vec) )
            R = (1/(sum(count_vec)+len(count_vec)+1))*categorical_covar(mean_frequency)
            
            # add any additional noise
            if R_additional is not None:
                R += R_additional
            if Q is None: 
                Q = 0*np.eye(self.model.num_params)

            # Kalman gain
            P = prior_covar + Q
            kgain = P@jacob.T@np.linalg.pinv(jacob@P@jacob.T + R, 1e-15)
            
            # Kalman update
            innovation = observation - p_model
            post_state = prior_state + kgain@innovation
            post_state = np.clip(post_state, clip_range[0], clip_range[1])
            
            # check if iteration should end
            if np.linalg.norm(post_state - prior_state) < itr_eps:
                break
            else:
                prior_state = post_state
                self.model.from_vector(post_state)
        
        # update class parameters
        self.P = (np.eye(self.model.num_params) - kgain@jacob)@P
        self.model.from_vector(post_state)
        
        self.param_history.append(post_state)
        self.covar_history.append(self.P)
        
        return innovation, kgain 
            
    def update_approx(self, circ, count_vec, p0, jac0, hess0, clip_range=[-1, 1], max_itr=1, itr_eps=1e-4, Q=None, R_additional=None):
        """
        Makes an approximate update to the model
        where the jacobian is approximated
        
        --- Arguments ---
        circ: pygsti circuit used in the update
        count_vec: vector of observed counts
        p0: target model prediction
        jac0: target model jacobian
        hess0: target model hessian
        clip_range: reasonable clipping range for the parameter update
        Q: state-space covariance 
        R_additional: additional measurement covariance
        max_itr: maximum number of iterations to the update
        itr_eps: epsilon for minimum difference to end iterated updates
        
        --- Returns --- 
        innov: the innovation
        kgain: the Kalman gain
        """
        prior_covar = self.P
        prior_state = self.model.to_vector()
        hilbert_dims = 2**(circ.width)
        
        for itr in range(max_itr):
            # approximate predicted frequency for the circuit outcome under the model
            p_model = p0 + jac0@prior_state + prior_state@hess0@prior_state
            
            # approximate the jacobian at the current estimate
            jacob = jac0 + hess0@prior_state
            
            # calculate the observed frequency
            total_counts = sum(count_vec)
            observation = count_vec/total_counts

            # calculate the covaraiance of the observation
            mean_frequency = ( count_vec+np.ones(len(count_vec)) )/( sum(count_vec)+len(count_vec) )
            R = (1/(sum(count_vec)+len(count_vec)+1))*categorical_covar(mean_frequency)
            
            # add any additional noise
            if R_additional is not None:
                R += R_additional
            if Q is None: 
                Q = 0*np.eye(self.model.num_params)

            # Kalman gain
            P = prior_covar + Q
            kgain = P@jacob.T@np.linalg.pinv(jacob@P@jacob.T + R, 1e-9)
            
            # Kalman update
            innovation = observation - p_model
            post_state = prior_state + kgain@innovation
            post_state = np.clip(post_state, clip_range[0], clip_range[1])
            
            # check if iteration should end
            if np.linalg.norm(post_state - prior_state) < itr_eps:
                break
            else:
                prior_state = post_state
                self.model.from_vector(post_state)
        
        # update class parameters
        self.P = (np.eye(self.model.num_params) - kgain@jacob)@P
        self.model.from_vector(post_state)
        
        self.param_history.append(post_state)
        self.covar_history.append(self.P)
        return innovation, kgain
    
    def update_fast(self, circ, count_vec, jac0, hess0, clip_range=[-1, 1], max_itr=1, itr_eps=1e-4, Q=None, R_additional=None):
        """
        Makes an approximate update to the model
        where the jacobian is approximated
        
        --- Arguments ---
        circ: pygsti circuit used in the update
        count_vec: vector of observed counts
        p0: target model prediction
        jac0: target model jacobian
        hess0: target model hessian
        clip_range: reasonable clipping range for the parameter update
        Q: state-space covariance 
        R_additional: additional measurement covariance
        max_itr: maximum number of iterations to the update
        itr_eps: epsilon for minimum difference to end iterated updates
        
        --- Returns --- 
        innov: the innovation
        kgain: the Kalman gain
        """
        prior_covar = self.P
        prior_state = self.model.to_vector()
        hilbert_dims = 2**(circ.width)
        
        for itr in range(max_itr):
            # approximate predicted frequency for the circuit outcome under the model
            p_model = vector_from_outcomes(self.model.probabilities(circ), 2**circ.width)
            
            # approximate the jacobian at the current estimate
            jacob = jac0 + hess0@prior_state
            
            # calculate the observed frequency
            total_counts = sum(count_vec)
            observation = count_vec/total_counts

            # calculate the covaraiance of the observation
            mean_frequency = ( count_vec+np.ones(len(count_vec)) )/( sum(count_vec)+len(count_vec) )
            R = (1/(sum(count_vec)+len(count_vec)+1))*categorical_covar(mean_frequency)
            
            # add any additional noise
            if R_additional is not None:
                R += R_additional
            if Q is None: 
                Q = 0*np.eye(self.model.num_params)

            # Kalman gain
            P = prior_covar + Q
            kgain = P@jacob.T@np.linalg.pinv(jacob@P@jacob.T + R, 1e-9)
            
            # Kalman update
            innovation = observation - p_model
            post_state = prior_state + kgain@innovation
            post_state = np.clip(post_state, clip_range[0], clip_range[1])
            
            # check if iteration should end
            if np.linalg.norm(post_state - prior_state) < itr_eps:
                break
            else:
                prior_state = post_state
                self.model.from_vector(post_state)
        
        # update class parameters
        self.P = (np.eye(self.model.num_params) - kgain@jacob)@P
        self.model.from_vector(post_state)
        
        self.param_history.append(post_state)
        self.covar_history.append(self.P)
        return innovation, kgain
    
    def filter_dataset(self, circuit_list, dataset):
        """
        batch filter of the given circuits in the dataset using exact updates
        """
        for circ in tqdm(circuit_list):
            count_vec = vector_from_outcomes(dataset[circ].counts, 2**circ.width)
            self.update(circ, count_vec)
        
    
    def fast_filter_dataset(self, circuit_list, dataset, jacobians, hessians):
        """
        batch filter of the given circuits in the dataset using fast updates
        """
        for circ in tqdm(circuit_list):
            count_vec = vector_from_outcomes(dataset[circ].counts, 2**circ.width)
            self.update_fast(circ, count_vec, jacobians[circ], hessians[circ])
        
        
    
    
    
def pickle_dict(obj, filename):
    """
    dump object into a pickle file
    """
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_dict(filename):
    """
    load pickle object
    """
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)


def mserror(model1, model2):
    """
    return the mean-squared error between the two input models
    """
    evec = model1.to_vector() - model2.to_vector()
    return evec.T@evec


def make_error_plot(plt_title, model_vectors, true_params, filter_covars, y_range, mle_estimates=None, germ_length_ranges=None, plt_spacing=25):
    """
    Make a plot of the evolution of the logarithm of error parameters and their uncertainty 
    
    --- Arguments ---
    plt_title: title of the plot
    model_vectors: list of model parameter vectors after each update
    true_params: parameters used in the datagen model
    filter_covars: list of model covariance matrices after each update
    y_range: display range for the y-axis
    mle_estimates: list of mle models after each increase in germ length
    germ_length_ranges: list of ranges of different germ lengths for the gst circuits
    plt_space: how frequently points are plotted 
    """
    fig, axs = plt.subplots(1, 2)
    
    fig.suptitle(plt_title)

    axs[0].set_title('Estimate')
    axs[1].set_title('Uncertainty')

    for i in range(0, len(model_vectors), plt_spacing):
        params = model_vectors[i]
        error = np.log10((params-true_params)@(params-true_params))
        axs[0].scatter(i, error, c='black')
        axs[1].scatter(i, np.log10((np.trace(filter_covars[i]))), c='black')
        
    if mle_estimates is not None:
        if germ_length_ranges is None:
            raise ValueError("no germ_length_ranges. Please set circuit ranges for the mle estimates")
        for i, mdl in enumerate(mle_estimates):
            mle_error = np.log10((mdl.to_vector()-true_params)@(mdl.to_vector()-true_params))
            axs[0].plot(germ_length_ranges[i], (mle_error, mle_error), c='red')
        
    axs[0].set_ylim(y_range[0], y_range[1])
    axs[1].set_ylim(y_range[0], y_range[1])
    axs[0].set_xlabel('Circuit Index')
    axs[1].set_xlabel('Circuit Index')
    axs[0].set_ylabel('MSE [log]')
    
def setup_extended(target_model, covar, x0=None):
    """
    Setup and extended Kalman filter 
    with a covariance that is a multiple of the identity 
    
    --- Arguments --- 
    target_model: ideal model that the filter is based on 
    covar_strength: covar matrix
    x0: initial parameter estimate if given
    """
    filter_model = target_model.copy()
    if x0 is not None:
        filter_model.from_vector(x0)
    return ExtendedKalmanFilter(filter_model, covar)

def random_copy(max_error_rate, target_model):
    """
    makes a random copy of the target model
    """
    output_model = target_model.copy()
    er = max_error_rate * np.random.rand(target_model.num_params)
    output_model.from_vector(er)
    return output_model

def make_mle_estimates(dataset, model_pack, target_mdl, max_lengths):
    """
    find mle estimates on the dataset with increasing germ length
    """
    proto = pygsti.protocols.GST(target_mdl, gaugeopt_suite=None, verbosity=1)
    estimates = []
    edesigns = []
    for l in max_lengths:
        edesign = model_pack.create_gst_experiment_design(l)
        data = pygsti.protocols.ProtocolData(edesign, dataset)
        results_after = proto.run(data)
        fit = results_after.estimates['GateSetTomography'].models['final iteration estimate']
        estimates.append(fit)
        edesigns.append(edesign)
    return estimates, edesigns


from scipy.stats import multinomial

def experimental_loglikelihood(circuit_list, dataset, model):
    """
    multinomial likelihood of observations under the given model 
    
    assumes all the circuits have the same number of qubits
    """
    hdims = 2**circuit_list[0].width
    count_matrix = np.zeros([0, hdims])
    pmat = np.zeros([0, hdims])
    for idx, circ in enumerate(circuit_list):
        count_vec = vector_from_outcomes(dataset[circ].counts, hdims)
        count_matrix = np.vstack([count_matrix, count_vec])
        p_model = vector_from_outcomes(model.probabilities(circ), hdims)
        pmat = np.vstack([pmat, p_model])
    return sum(multinomial.logpmf(count_matrix, sum(count_matrix[0, :]), pmat))

def max_loglikelihood(circuit_list, dataset):
    """
    max loglikelihood 
    """
    hdims = 2**circuit_list[0].width
    count_matrix = np.zeros([0, hdims])
    pmat = np.zeros([0, hdims])
    for idx, circ in enumerate(circuit_list):
        count_vec = vector_from_outcomes(dataset[circ].counts, hdims)
        count_matrix = np.vstack([count_matrix, count_vec])
        frequency = count_vec/sum(count_vec)
        pmat = np.vstack([pmat, frequency])
    return sum(multinomial.logpmf(count_matrix, sum(count_matrix[0, :]), pmat))