import numpy as np
import pickle
import torch
import scipy

### --- Data processing methods --- ###

def sort_valid_5drats(resp, param, ch2xy):

    resp_mu = np.mean(resp,axis=2)
    resp_sigma = np.std(resp, axis=2)
    #std_map = np.std(resp, axis=(0, 2)) Current method considers mean std w/in repetitions. Change this for global
    val_pw = np.unique(param[:,0])
    val_freq = np.unique(param[:,1])
    val_duration = np.unique(param[:,2])
    mean_map = np.zeros((resp.shape[1],len(val_pw),len(val_freq),len(val_duration),8,4))
    std_map = np.zeros((resp.shape[1],len(val_pw),len(val_freq),len(val_duration),8,4))

    for e in range(resp.shape[1]):
        for i in range(len(param)):

            idx_pw = np.where(np.isclose(val_pw, param[i, 0]))[0][0]
            idx_freq = np.where(np.isclose(val_freq, param[i, 1]))[0][0]
            idx_duration = np.where(np.isclose(val_duration, param[i, 2]))[0][0]

            x_ch = int(ch2xy[i,3]) -1
            y_ch = int(ch2xy[i,4]) -1 

            mean_map[e, idx_pw, idx_freq, idx_duration, x_ch, y_ch] = resp_mu[i,e]
            std_map[e, idx_pw, idx_freq, idx_duration, x_ch, y_ch] = resp_sigma[i,e]

    n_cond, n_emgs, n_reps = resp.shape
    valid_resp = np.zeros_like(resp, dtype=np.int64)
    for i in range(n_cond):
        # find indices in the map corresponding to this condition
        idx_pw = np.where(np.isclose(val_pw, param[i, 0]))[0][0]
        idx_freq = np.where(np.isclose(val_freq, param[i, 1]))[0][0]
        idx_duration = np.where(np.isclose(val_duration, param[i, 2]))[0][0]
        x_ch = int(ch2xy[i, 3]) - 1
        y_ch = int(ch2xy[i, 4]) - 1

        for j in range(n_emgs):

            # extract mean/std vectors for this condition
            mean_vec = mean_map[j, idx_pw, idx_freq, idx_duration, x_ch, y_ch] #np.mean(resp_mu, axis=0)[j]   # shape (1,)
            std_vec = std_map[j, idx_pw, idx_freq, idx_duration, x_ch, y_ch] #std_map[j] 

            deviation = resp[i, j, :] - mean_vec
            valid_mask = np.abs(deviation) <= 3*std_vec

            #print(f'mean_vec: {mean_vec} and std_vec: {std_vec}')
            deviation = np.abs(resp[i, j, :] - mean_vec)
            #print(f'deviation: {deviation}')
            valid_mask = deviation <= 2*std_vec
            if np.all(~valid_mask):
                print(f'issue with stim {i} on emg {j}: {3*std_vec} is higher than |{deviation}| so gives invalid mask: {valid_mask}')
                
            valid_resp[i, j, :] = valid_mask.astype(np.int64)

    return valid_resp

def set_experiment(dataset_type):

    options = {}
    if dataset_type == 'nhp':
        options['noise_min']= 0.009 #Non-zero to avoid numerical instability
        options['kappa']=7
        options['rho_high']=3.0
        options['rho_low']=0.1
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.011
        options['n_subjects']=4
        options['n_queries']=96
        options['n_emgs'] = [6, 8, 4, 4]
        options['n_dims'] = 2
    elif dataset_type == 'rat':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=6
        options['n_queries']=32
        options['n_emgs'] = [6, 7, 8, 6, 5, 8]
        options['n_dims'] = 2
    elif dataset_type == '5d_rat':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']= 2 #TODO after analysis 3
        options['n_queries']= 100
        options['n_emgs'] = [4, 4] #TODO after analysis [4, 4, 5]
        options['n_dims'] = 5
    elif dataset_type == 'spinal':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=11
        options['n_queries']=64
        options['n_emgs'] = [8, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8]
        options['n_dims'] = 2
    
    options['n_reps'] = 20
    options['n_rnd'] = 1

    return options

def load_data(dataset_type, m_i):
    '''
    Input: 
        - dataset_type: str characterizing the modality of the experiment
        - m_i: int of subject

    Output:
        - dictionary of neurostimulation data
    
        Important: sorted response shape: (nChan, nEmgs, nReps)
    '''
    path_to_dataset = f'./datasets/{dataset_type}'

    if dataset_type == '5d_rat':
        
        match m_i:
            case 2:
                data = scipy.io.loadmat(f'{path_to_dataset}/BCI00_5D.mat')
                emgs = ['left extensor carpi radialis', 'biceps', 'triceps',
                                 'left flexor carpi ulnaris', 'unknown']
                dim_sizes = np.array([8, 4, 4, 4, 4])
            case 1:
                data = scipy.io.loadmat(f'{path_to_dataset}/rCer1.5_5D.mat')
                emgs = ['left extensor carpi radialis', 'left flexor carpi ulnaris', ' left triceps',
                                 'left biceps']
                dim_sizes = np.array([8, 4, 4, 4, 4])
            case 0:
                data = scipy.io.loadmat(f'{path_to_dataset}/rData03_5D.mat')
                emgs = ['left extensor carpi radialis', 'left flexor carpi ulnaris', ' left triceps',
                                    'left pectoralis']
                dim_sizes = np.array([8, 4, 3, 4, 4])

        resp = data['emg_response']
        param = data['stim_combinations']
        ch2xy = param[:, [0,1,2,5,6]]
        peak_resp = resp[:, :, :, 2].transpose((2, 1, 0)) # resp[:, :, :, 0] for unormalized response ; (nb_reps, nb_emgs, params) to (params, nb_emgs, nb_reps)
        resp_mean = np.mean(resp[:, :, :,2], axis=0).transpose((1, 0)) #(params, nb_emgs)
        sorted_isvalid = sort_valid_5drats(peak_resp, param, ch2xy)

        subject = {
            'emgs': emgs,
            'nChan': 32,
            'sorted_resp': peak_resp,
            'sorted_respMean': resp_mean,
            'sorted_isvalid': sorted_isvalid, 
            'ch2xy': ch2xy,
            'dim_sizes': dim_sizes,
            'DimSearchSpace' : np.prod(dim_sizes)
        }                
        return subject  
    elif dataset_type=='nhp':
        if m_i==0:
            data = scipy.io.loadmat(path_to_dataset+'/Cebus1_M1_190221.mat')['Cebus1_M1_190221'][0][0]
        elif m_i==1:
            data = scipy.io.loadmat(path_to_dataset+'/Cebus2_M1_200123.mat')['Cebus2_M1_200123'][0][0]
        elif m_i==2:    
            data = scipy.io.loadmat(path_to_dataset+'/Macaque1_M1_181212.mat')['Macaque1_M1_181212'][0][0]
        elif m_i==3:
            data  = scipy.io.loadmat(path_to_dataset+'/Macaque2_M1_190527.mat')['Macaque2_M1_190527'][0][0]

        if m_i >= 2:
            #macaques
            mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_evoked': 10, 'sampFreqEMG': 11, 'resp_region': 12, 'map': 13, 'ch2xy': 14, 
                'sorted_respMean': 15, 'sorted_respSD': 16
            }
        else:
            # cebus
            mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_respMean': 10, 'sorted_respSD': 11, 'sorted_evoked': 12, 'sampFreqEMG': 13, 
                'resp_region': 14, 'map': 15, 'ch2xy': 16
            }

        nChan = data[mapping['nChan']][0][0]

        rN = data[mapping['sorted_isvalid']]
        j1, j2, j3 = rN.shape[0], rN.shape[1], rN[0][0].shape[0]
        sorted_isvalid = np.stack([np.squeeze(rN[i, j]) for i in range(j1) for j in range(j2)], axis=0)
        sorted_isvalid = sorted_isvalid.reshape(j1, j2, j3)

        ch2xy = data[mapping['ch2xy']] - 1
        se = data[mapping['sorted_evoked']]
        i1, i2, i3, i4 = se.shape[0], se.shape[1], se[0][0].shape[0], se[0][0].shape[1]
        sorted_evoked = np.stack([np.squeeze(se[i, j]) for i in range(i1) for j in range(i2)], axis=0)
        sorted_evoked = sorted_evoked.reshape(i1, i2, i3, i4)
        sorted_filtered = sorted_evoked

        stim_channel = data[mapping['stim_channel']]
        if stim_channel.shape[0] == 1:
            stim_channel = stim_channel[0]

        fs = data[mapping['sampFreqEMG']][0][0]
        resp_region = data[mapping['resp_region']][0]

        stimProfile = data[mapping['stimProfile']][0]
        
        # compute baseline
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 30 * 10**(-3))
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, where_zero - window_size : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        sorted_filtered = sorted_filtered - baseline[..., np.newaxis]
        sorted_resp = np.max(sorted_filtered[:,:,:n_rep,resp_region[0]:resp_region[1]], axis=-1)

        # Create a masked array where invalid points are masked
        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)
        
        # Compute the mean over the last axis, ignoring masked (invalid) values
        sorted_respMean = masked_resp.mean(axis=-1)

        emgs = data[0][0]

        return {
        'emgs': emgs,
        'nChan': nChan, 
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'ch2xy': ch2xy,
        'DimSearchSpace': 96
        }
    elif dataset_type=='rat':  # rat dataset has 6 subjects
        if m_i==0:
            data = scipy.io.loadmat(path_to_dataset+'/rat1_M1_190716.mat')['rat1_M1_190716'][0][0]
        elif m_i==1:
            data = scipy.io.loadmat(path_to_dataset+'/rat2_M1_190617.mat')['rat2_M1_190617'][0][0]     
        elif m_i==2:
            data = scipy.io.loadmat(path_to_dataset+'/rat3_M1_190728.mat')['rat3_M1_190728'][0][0]                  
        elif m_i==3:
            data = scipy.io.loadmat(path_to_dataset+'/rat4_M1_191109.mat')['rat4_M1_191109'][0][0]                  
        elif m_i==4:
            data = scipy.io.loadmat(path_to_dataset+'/rat5_M1_191112.mat')['rat5_M1_191112'][0][0]                  
        elif m_i==5:
            data = scipy.io.loadmat(path_to_dataset+'/rat6_M1_200218.mat')['rat6_M1_200218'][0][0]   

        mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_evoked': 10, 'sampFreqEMG': 11, 'resp_region': 12, 'map': 13, 'ch2xy': 14, 
                'sorted_respMean': 15, 'sorted_respSD': 16
            }
        
        nChan = data[mapping['nChan']][0][0]

        rN = data[mapping['sorted_isvalid']]
        j1, j2, j3 = rN.shape[0], rN.shape[1], rN[0][0].shape[0]
        sorted_isvalid = np.stack([np.squeeze(rN[i, j]) for i in range(j1) for j in range(j2)], axis=0)
        sorted_isvalid = sorted_isvalid.reshape(j1, j2, j3)

        ch2xy = data[mapping['ch2xy']] - 1
        se = data[mapping['sorted_evoked']]
        i1, i2, i3, i4 = se.shape[0], se.shape[1], se[0][0].shape[0], se[0][0].shape[1]
        sorted_evoked = np.stack([np.squeeze(se[i, j]) for i in range(i1) for j in range(i2)], axis=0)
        sorted_evoked = sorted_evoked.reshape(i1, i2, i3, i4)
        sorted_filtered = sorted_evoked

        stim_channel = data[mapping['stim_channel']]
        if stim_channel.shape[0] == 1:
            stim_channel = stim_channel[0]

        fs = data[mapping['sampFreqEMG']][0][0]
        resp_region = data[mapping['resp_region']][0]

        stimProfile = data[mapping['stimProfile']][0]
        
        # compute baseline
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 30 * 10**(-3))
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, where_zero - window_size : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        sorted_filtered = sorted_filtered - baseline[..., np.newaxis]
        sorted_resp = np.max(sorted_filtered[:,:,:n_rep,resp_region[0]:resp_region[1]], axis=-1)
        # Create a masked array where invalid points are masked
        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)
        
        # Compute the mean over the last axis, ignoring masked (invalid) values
        sorted_respMean = masked_resp.mean(axis=-1)

        emgs = data[0][0]

        return {
        'emgs': emgs,
        'nChan': nChan, 
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'ch2xy': ch2xy,
        'DimSearchSpace': 32
        } 
    elif dataset_type =='spinal':

        subject_map = {
            0: 'rat0_C5_500uA.pkl', 1: 'rat1_C5_500uA.pkl', 2: 'rat1_C5_700uA.pkl', 3: 'rat1_midC4_500uA.pkl',
            4: 'rat2_C4_300uA.pkl', 5: 'rat2_C5_300uA.pkl', 6: 'rat2_C6_300uA.pkl', 7: 'rat3_C4_300uA.pkl',
            8: 'rat3_C5_200uA.pkl', 9: 'rat3_C5_350uA.pkl', 10: 'rat3_C6_300uA.pkl' 
        }
        
        #load data
        with open(f'{path_to_dataset}/{subject_map[m_i]}', "rb") as f:
            data = pickle.load(f)
        
        ch2xy, emgs = data['ch2xy'], data['emgs']
        evoked_emg, filtered_emg = data['evoked_emg'], data['filtered_emg']
        maps = data['map']
        parameters = data['parameters']
        resp_region = data['resp_region']
        fs = data['sampFreqEMG']
        sorted_evoked = data['sorted_evoked']
        sorted_filtered = data['sorted_filtered']
        sorted_resp = data['sorted_resp']
        sorted_isvalid = data['sorted_isvalid']
        sorted_respMean = data['sorted_respMean']
        sorted_respSD = data['sorted_respSD']
        stim_channel = data['stim_channel']
        stimProfile=data['stimProfile']
        n_muscles = emgs.shape[0]

        #Computing baseline for filtered signal
        nChan = parameters['nChan'][0]
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 35 * 10**(-3))
        baseline = []
        n_rep = 10000 # Globally define n_rep cutoff
        for iChan in range(nChan):
            reps= np.where(stim_channel == iChan + 1)[0]
            if len(reps) < n_rep:
                n_rep = len(reps)
        for iChan in range(nChan):
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)

        #remove baseline from filtered signal
        sorted_filtered[:, :, :n_rep, :] = sorted_filtered[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_filtered[:, :, :n_rep, int(resp_region[0]): int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:, :, :n_rep] == 0, sorted_resp)
        sorted_respMean = masked_resp.mean(axis=-1)

         # compute baseline for evoked signal
        baseline = []
        for iChan in range(nChan):
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_evoked[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        #remove baseline from evoked signal
        sorted_evoked[:, :, :n_rep, :] = sorted_evoked[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_evoked[:,:,:n_rep,int(resp_region[0]) :int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:,:,:n_rep] == 0, sorted_resp)

        #mask sorted_isvalid by n_rep
        sorted_isvalid = sorted_isvalid[:, :, :n_rep]

        subject = {
            'emgs': emgs,
            'nChan': 64,
            'DimSearchSpace': 64,
            'sorted_respMean': sorted_respMean,
            'ch2xy': ch2xy,
            'evoked_emg': evoked_emg, 'filtered_emg':filtered_emg, 'sorted_resp': sorted_resp,  
            'sorted_isvalid': sorted_isvalid, 'sorted_respSD': sorted_respSD,
            'sorted_filtered': sorted_filtered, 'stim_channel': stim_channel, 'fs': fs,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'resp_region': resp_region, 'stimProfile': stimProfile,  'baseline' : baseline    
        }
        
        return subject   
    else:
        raise ValueError('The dataset type should be 5d_rat, nhp, rat or spinal' )
        
