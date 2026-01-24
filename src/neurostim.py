import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import sys
import os
import torch
import gpytorch
import argparse
import time
import scipy.io
import pickle
import math

from SALib.analyze import sobol as salib_sobol
from SALib.sample import sobol as sobol_sampler

from concurrent.futures import ThreadPoolExecutor
import warnings
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

### MODULES HANDLING ###
from utils.neurostim_datasets import *
from models.gaussians import NeuralSobolGP, NeuralAdditiveGP, NeuralExactGP, optimize, maximize_acq
from models.sobols import NeuralSobol

warnings.filterwarnings("ignore", category=FutureWarning, module="SALib.util")

### Method to calculate true Sobol Interactions

def build_surrogate(X, Y, type, 
                    D=2, pce_degree=3):
    if type == "rf":
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
        model.fit(X, Y)

        def predict(Xq):
            model.predict(Xq)

        return predict
    
    elif type == "gp":

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(D), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2)
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(X, Y)
        
        def predict(Xq):
            # GPR predict returns mean, std; use mean
            mu = gpr.predict(Xq, return_std=False)
            return mu
        
        return predict
    
    elif type == 'pce':
        degree = pce_degree
        polyf = PolynomialFeatures(degree=degree, include_bias=True)
        Xpoly = polyf.fit_transform(X)
        ridge = Ridge(alpha=1e-6, fit_intercept=False)
        ridge.fit(Xpoly, Y)
        def predict(Xq):
            return ridge.predict(polyf.transform(Xq))
        return predict

def sobol_interactions(dataset_type, surrogate='rf', N=4096):

    print(f'Sobol 2nd order interactions for {dataset_type}')
    options = set_experiment(dataset_type)

    S2 = []

    for m_i in range(options['n_subjects']):

        subject = load_data2(dataset_type, m_i)
        X = subject['ch2xy'].astype(float)

        s1 = []
        s2 = []
        n_reps = 30

        for e_i in range(len(subject['emgs'])):

            Y = subject['sorted_respMean'][:,e_i]
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_scaled = (X - X_min) / (X_max - X_min)

            D = X_scaled.shape[1]
            problem = {'num_vars': D, 'names': [f"x{i}" for i in range(D)], 'bounds': [[0,1]]*D}
            
            
            predictor = build_surrogate(X_scaled, Y, surrogate, D = D)
            params = sobol_sampler.sample(problem, N, calc_second_order=True)  
            Y_samp = predictor(params)

            Si = salib_sobol.analyze(problem, Y_samp, calc_second_order=True, print_to_console=False)  
            s1.append(Si['S1'])
            s2.append(Si['S2'])

            emg_avg = []
            for rep_i in range(n_reps):
                predictor = build_surrogate(X_scaled, Y, surrogate, D = D)
                params = sobol_sampler.sample(problem, N, calc_second_order=True)  
                Y_samp = predictor(params)
                Si = salib_sobol.analyze(problem, Y_samp, calc_second_order=True, print_to_console=False)  
                emg_avg.append(Si['S2'])

            for i in range(D):
                for j in range(i+1, D):

                    interactions = []
                    for rep_i in range(n_reps):
                        ref = emg_avg[rep_i]
                        interactions.append(ref[i,j])

                    interactions = np.asarray(interactions)
                    avg_interaction = np.mean(interactions, axis=0)
                    print(f"subject {m_i}, emg {e_i} | {problem['names'][i]} & {problem['names'][j]}: S2 = {avg_interaction:.4f}")
            
        s2 = np.array(s2)
        
        S2.append(np.mean(s2, axis=0))

    overall_avg = np.mean(np.array(S2), axis=0)
    print("\nAvg second-order interaction terms:")
    for i in range(D):
        for j in range(i+1, D):
            print(f"x{i} & x{j}: Avg S2 = {overall_avg[i,j]:.4f}")


### --- BO methods --- ###

def neurostim_bo(dataset, model_cls, kappas, device='cpu'):

    np.random.seed(0)

    # Experiment parameters initialization
    options = set_experiment(dataset)
    device = torch.device(device)
    nRep = options['n_reps']
    nrnd = options['n_rnd']
    nSubjects = options['n_subjects']
    nEmgs = options['n_emgs']
    MaxQueries = options['n_queries']
    ndims = options['n_dims']

    #Metrics initialization
    PP = torch.zeros((nSubjects,max(nEmgs),len(kappas),nRep, MaxQueries), device=device)
    PP_t = torch.zeros((nSubjects,max(nEmgs), len(kappas),nRep, MaxQueries), device=device)
    Q = torch.zeros((nSubjects,max(nEmgs),len(kappas),nRep, MaxQueries), device=device)
    Train_time = torch.zeros((nSubjects,max(nEmgs), len(kappas),nRep, MaxQueries), device=device)
    Cum_train =  torch.zeros((nSubjects,max(nEmgs), len(kappas),nRep, MaxQueries), device=device)
    RSQ = torch.zeros((nSubjects,max(nEmgs), len(kappas), nRep), device=device)
    REGRETS = torch.zeros((nSubjects,max(nEmgs), len(kappas), nRep, MaxQueries), device=device)
    SOBOLS = np.empty((nSubjects,max(nEmgs),len(kappas),nRep, MaxQueries), dtype=object)

    for s_idx in range(nSubjects):

        subject = load_data2(dataset, s_idx) 
        subject['ch2xy'] = torch.tensor(subject['ch2xy'], device=device)

        for k_idx, kappa in enumerate(kappas):

            for e_i in range(len(subject['emgs'])):

                # "Ground truth" map
                MPm= torch.tensor(subject['sorted_respMean'][:,e_i]).float()  
                # Best known channel
                mMPm= torch.max(MPm)

                # priors and kernel handling
                priorbox = gpytorch.priors.SmoothedBoxPrior(a=math.log(options['rho_low']),b= math.log(options['rho_high']), sigma=0.01)
                outputscale_priorbox= gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01) 

                prior_lik= gpytorch.priors.SmoothedBoxPrior(a=options['noise_min']**2,b= options['noise_max']**2, sigma=0.01) # gaussian noise variance
                likf= gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
                likf.initialize(noise=torch.tensor(1.0, device=device, dtype=torch.get_default_dtype()))

                if device =='cuda':
                    likf=likf.cuda()

                # Metrics initialization
                # Then run the sequential optimization
                DimSearchSpace = subject['DimSearchSpace'] 
                perf_explore= torch.zeros((nRep, MaxQueries), device=device)
                perf_exploit= torch.zeros((nRep, MaxQueries), device=device)
                perf_rsq= torch.zeros((nRep), device=device)
                P_test =  torch.zeros((nRep, MaxQueries, 2), device=device) #storing all queries
                train_time = torch.zeros((nRep, MaxQueries), device=device)
                cum_time = torch.zeros((nRep, MaxQueries), device=device)
                regret = np.empty((nRep, MaxQueries), dtype=np.float32)
                sobol_interactions = np.empty((nRep, MaxQueries), dtype=object)


                for rep_i in range(nRep):

                    print(f'subject {s_idx + 1} \ {nSubjects} | emg: {e_i} | kappa {kappa} | {rep_i+1} / {nRep}')

                    
                    # maximum response obtained in this round, used to normalize all responses between zero and one.
                    MaxSeenResp=0
                    q=0 # query number
                    timer = 0.0
                    order_this= torch.randperm(DimSearchSpace, device=device) # random permutation of each entry of the search space
                    P_max=[]

                    
                    executor = ThreadPoolExecutor(max_workers=4)
                    space_reconfiguration = None
                    interactions = None

                    while q < MaxQueries:
                        t0 = time.time()
                        # Query selection
                        if q>=nrnd:
                            # Max of acquisition map
                        
                            AcquisitionMap = ymu + kappa*torch.nan_to_num(torch.sqrt(ys2)) # UCB acquisition
                            Next_Elec= torch.where(AcquisitionMap.reshape(len(AcquisitionMap))==torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))
                            Next_Elec = Next_Elec[0][np.random.randint(len(Next_Elec))]  if len(Next_Elec[0]) > 1 else Next_Elec[0][0]
                            P_test[rep_i][q][0]= Next_Elec
                            #except:
                                #print(f'AcquisitionMap: {AcquisitionMap}')
                                #print(f'Next_Elect: {torch.where(AcquisitionMap.reshape(len(AcquisitionMap))==torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))}')
                                #print(f'ymu: {observed_pred.mean}')
                                #print(f'model and likelihood? model: {model} likelihood: {likf}')
                        else:
                            P_test[rep_i][q][0]= int(order_this[q])
                        query_elec = P_test[rep_i][q][0]

                        # Read response
                        sample_resp = torch.tensor(subject['sorted_resp'][int(query_elec)][e_i][subject['sorted_isvalid'][int(query_elec)][e_i]!=0])
                        if len(sample_resp) == 0:
                            sample_resp = torch.tensor(subject['sorted_resp'][int(query_elec)][e_i])
                            valid_responses = subject['sorted_isvalid'][int(query_elec)][e_i]
                            print(f'sample_response: {sample_resp} \n valid_responses: {valid_responses} \n and query_elec: {query_elec}\n')
                            test_respo = 1e-9
                        else:
                            test_respo = sample_resp[np.random.randint(len(sample_resp))]
                       
                        std = (0.02 * torch.mean(test_respo)).clamp(min=0.0)   
                        noise = torch.randn((), device=test_respo.device, dtype=test_respo.dtype) * std
                        test_respo = test_respo + noise
                        # done reading response
                        P_test[rep_i][q][1]= test_respo

                        if (test_respo > MaxSeenResp) or (MaxSeenResp==0): # updated maximum response obtained in this round
                            MaxSeenResp=test_respo

                        x= subject['ch2xy'][P_test[rep_i][:q+1,0].long(),:].float() # search space position
                        x = x.reshape((len(x), ndims))
                        y= P_test[rep_i][:q+1,1]/MaxSeenResp # test result


                        # Model initialization and model update
                        if q == 0:
                            model = model_cls(x, y, likf, priorbox, outputscale_priorbox)
                            if model.name == 'NeuralSobolGP':
                                sobol = NeuralSobol(dataset).to(device)
                                model.sobol = sobol  # Initialize sobol
                                interactions = np.zeros((ndims), dtype=float) #model.sobol.method(x, y, surrogate)
                        else:
                            model.set_train_data(x, y, strict=False)

                        # Model training
                        model.train()
                        likf.train()
                        model, likf, _ = optimize(model, x, y)


                        # Model evaluation
                        model.eval()
                        likf.eval()

                        with torch.no_grad():
                            X_test= subject['ch2xy'].float()
                            observed_pred = likf(model(X_test))

                            ymu= observed_pred.mean
                            ys2= observed_pred.variance

                        Tested= torch.unique(P_test[rep_i][:q+1,0]).long()
                        MapPredictionTested=ymu[Tested]
                        BestQuery = Tested if len(Tested) == 1 else Tested[(MapPredictionTested==torch.max(MapPredictionTested)).reshape(len(MapPredictionTested))]
                        if len(BestQuery) > 1:
                            BestQuery = np.array([BestQuery[np.random.randint(len(BestQuery))]])
                        
                        # Store metrics
                        P_max.append(BestQuery[0]) # Maximum response at time q
                        #msr[m_i,e_i,k_i,rep_i,q] = MaxSeenResp

                        # log regret calculations
                        chosen_pref = MPm[BestQuery].item()
                        instant_regret = mMPm.item() - chosen_pref
                        regret[rep_i, q] = instant_regret

                        # Pending Sobol job fetch
                        if space_reconfiguration is not None and space_reconfiguration.done():
                            interactions = space_reconfiguration.result()
                            new_partition = sobol.update_partition(interactions)
                            model.update_partition(new_partition)
                            space_reconfiguration = None
                           
                            
                        # Update partitions
                        if model.name == 'NeuralSobolGP':
                            sobol_interactions[rep_i, q] = interactions.copy()

                            if (space_reconfiguration is None or space_reconfiguration.done()) and (q > (MaxQueries // 4)):
                                surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=prior_lik)
                                surrogate = NeuralExactGP(x, y, surrogate_likelihood, priorbox, outputscale_priorbox)
                                space_reconfiguration = executor.submit(sobol.method, x, y, surrogate)
                            

                        # computation time calculations
                        t1 = time.time()
                        elapsed = t1 - t0
                        timer += elapsed
                        train_time[rep_i, q] = elapsed
                        cum_time[rep_i, q] = timer
                        q+=1
                    
                    executor.shutdown(wait=False)

                    # estimate current exploration performance: knowledge of best stimulation point
                    perf_explore[rep_i,:]=MPm[P_max].reshape((len(MPm[P_max])))/mMPm
                    # estimate current exploitation performance: knowledge of best stimulation point
                    perf_exploit[rep_i,:]= P_test[rep_i][:,0].long()
                    # R^2 correlation with ground truth
                    y_true = MPm
                    y_pred = ymu

                    ss_res = torch.sum((y_true - y_pred) ** 2)
                    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    perf_rsq[rep_i] = r_squared 
  

                # store all performance metrics
                PP[s_idx,e_i,k_idx]=perf_explore
                Q[s_idx,e_i,k_idx] = P_test[:,:,0]
                PP_t[s_idx,e_i,k_idx]= MPm[perf_exploit.long().cpu()]/mMPm
                Train_time[s_idx,e_i,k_idx] = train_time.mean(dim=0).detach().cpu()
                Cum_train[s_idx, e_i, k_idx] = cum_time.mean(dim=0).detach().cpu()
                RSQ[s_idx,e_i,k_idx]=perf_rsq
                REGRETS[s_idx,e_i,k_idx, :] = torch.log(torch.tensor(regret, dtype=torch.float32, device=device) + 1e-8) 
                SOBOLS[s_idx, e_i, k_idx] = sobol_interactions # mean_mats.mean(axis=0) #?# some mean operation

    # Saving variables
    output_dir = os.path.join('output', 'neurostim_experiments', dataset)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'{dataset}_{model.name}_budget{MaxQueries}_{nRep}reps.npz'
    results_path = os.path.join(output_dir,fname)
    np.savez_compressed(results_path,
            RSQ=RSQ.cpu(), PP=PP.cpu(), PP_t=PP_t.cpu(), 
            kappas=np.array(kappas),
            SOBOLS = SOBOLS,
            REGRETS = REGRETS.cpu(),
            Train_time = Train_time.cpu(),
            Cum_train = Cum_train.cpu()
            )
    print(f'saved results to {results_path}')

    
### --- Parser handling --- ###

def _parse_list_of_floats(text):
    if text is None:
        return None
    try:
        return [float(x) for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of numbers (e.g. 0.5,1,3)')

def _parse_list_of_ints(text):
    if text is None:
        return None
    try:
        return [int(x) for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of ints (e.g. 1,2,3)')

def _parse_list_of_strings(text):
    if text is None:
        return None
    try:
        return [x.strip() for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of strings')

def main(argv=None):
    parser = argparse.ArgumentParser(description='Neurostimulation Bayesian Optimization runner with CLI options')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='5d_rat', help='Neurostimulation dataset type')

    # Model selection
    parser.add_argument('--model_cls', type=str, default='NeuralExactGP', help='Model class name (NeuralExactGP, NeuralAdditiveGP, NeuralSobolGP)')

    # Method-specific params
    parser.add_argument('--kappas', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for experiments')

    # Device selection
    parser.add_argument('--device', type=str, default='cpu', help='Device for computation (e.g., cpu, cuda:0)')
    parser.add_argument('--devices', type=_parse_list_of_strings, default=None, help='Comma-separated devices for parallel execution (e.g., cuda:0,cuda:1,cpu)')

    # Misc
    parser.add_argument('--list_models', action='store_true')

    args = parser.parse_args(argv)

    # Allowed mappings (whitelist)
    model_map = {
        'NeuralExactGP': NeuralExactGP,
        'NeuralAdditiveGP': NeuralAdditiveGP,
        'NeuralSobolGP': NeuralSobolGP,
    }

    if args.list_models:
        print('Available model classes:')
        for k in model_map.keys():
            print(' -', k)
        return

    if args.model_cls not in model_map:
        raise ValueError(f"Unknown model_cls '{args.model_cls}'. Use --list_models to see options.")

    model_cls = model_map[args.model_cls]

    # Determine device
    device = args.devices[0] if args.devices else args.device

    # dispatch
    try:
        result = neurostim_bo(args.dataset, model_cls, kappas=args.kappas, device=device)
        print('Completed')

    except Exception as e:
        print('ERROR during execution:', e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':

    main()
    #neurostim_bo('nhp', NeuralSobolGP, kappas=[3.0], device='cpu')