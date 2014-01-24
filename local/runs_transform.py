from __future__ import division
import numpy as np
import argparse
from template_speech_rec import configParserWrapper
from scipy.io import wavfile
import template_speech_rec.get_train_data as gtrd
import filterbank as fb    
from transforms import binary_phase_features, preemphasis, process_wav, spectrogram_magnitude_gradients, spectrogram
import matplotlib.pyplot as plt
from nitime.algorithms.spectral import dpss_windows
from scipy.ndimage.filters import correlate
from phoneclassification.stride_parts import extract_patches, extract_spec_patches, get_patch_matrix
from pyscat import runs

def main(args):
    config_d = configParserWrapper.load_settings(open(args.c,'r'))
    
    # setup the transform
    # should return the features and the underlying spectrogram
    if args.binary_features == 'edges':
        run_transform = lambda x: (lambda S: (S,
                                              gtrd.get_edge_features_use_config(S.T,config_d['EDGES'])))(gtrd.get_spectrogram_use_config(preemphasis(process_wav(x),preemph=config_d['SPECTROGRAM']['preemphasis']),config_d['SPECTROGRAM'],return_sample_mapping=True)[0])
        winsize = config_d['SPECTROGRAM']['num_window_samples']
        winstep = config_d['SPECTROGRAM']['num_window_step_samples']
        sample_rate= config_d['SPECTROGRAM']['sample_rate']
        oversampling=None
    elif args.binary_features == 'phase_peaks':
        htemp, dhtemp, ddhtemp, tttemp = fb.hermite_window(
        config_d['BPF']['winsize']-1,
        config_d['BPF']['order'],
        config_d['BPF']['half_time_support'])

        h = np.zeros((htemp.shape[0],
                  htemp.shape[1]+1))
        h[:,:-1] = htemp

        dh = np.zeros((dhtemp.shape[0],
                   dhtemp.shape[1]+1))
        dh[:,:-1] = dhtemp


        tt = (2*tttemp[-1] -tttemp[-2])*np.ones(tttemp.shape[0]+1)
        tt[:-1] = tttemp
    
        oversampling=config_d['BPF']['oversampling']
        T_s=config_d['OBJECT']['t_s']

        gsigma = config_d['BPF']['gsigma']
        gfilter= fb.get_gauss_filter(config_d['BPF']['gt'],
                                 config_d['BPF']['gf'],
                                 gsigma)

        run_transform = lambda x: binary_phase_features(
        preemphasis(process_wav(x),preemph=config_d['BPF']['preemph']),
        config_d['BPF']['sample_rate'],
        config_d['BPF']['freq_cutoff'],
        config_d['BPF']['winsize'],
        config_d['BPF']['nfft'],
        oversampling,
        h,
        dh,
        tt,
        gfilter,
        gsigma,
        config_d['BPF']['fthresh'],
        config_d['BPF']['othresh'],
        spread_length=config_d['BPF']['spread_length'],
            return_midpoints=True)
        winsize = config_d['BPF']['winsize']
        sample_rate= config_d['SPECTROGRAM']['sample_rate']
        oversampling = config_d['BPF']['oversampling']
        winstep = None
    elif args.binary_features == 'edges_transform_spec':
        if config_d['SPECTROGRAM']['wintype'] == 'hermite':
            # need the length of the windows to be odd but we can pad
            htemp, dhtemp, ddhtemp, tttemp = fb.hermite_window(
            config_d['SPECTROGRAM']['winsize']-1,
            config_d['SPECTROGRAM']['order'],
            config_d['SPECTROGRAM']['half_time_support'],
            pad_windows=1)
        elif config_d['SPECTROGRAM']['wintype'] == 'dpss':
            htemp = dpss_windows(config_d['SPECTROGRAM']['winsize'],
                     config_d['SPECTROGRAM']['half_time_support'],
                             config_d['SPECTROGRAM']['order']
                     )[0]

        oversampling=config_d['SPECTROGRAM']['oversampling']


        run_transform = lambda x: (lambda S: (np.log(S),
gtrd.get_edge_features_use_config(np.log(S.T),config_d['EDGES'])))(spectrogram(
        preemphasis(process_wav(x),preemph=config_d['SPECTROGRAM']['preemphasis']),
        config_d['SPECTROGRAM']['sample_rate'],
        config_d['SPECTROGRAM']['freq_cutoff'],
        config_d['SPECTROGRAM']['winsize'],
        config_d['SPECTROGRAM']['nfft'],
        oversampling,
        htemp))
        winsize = config_d['SPECTROGRAM']['winsize']
        sample_rate = config_d['SPECTROGRAM']['sample_rate']
        winstep = None
        oversampling = config_d['SPECTROGRAM']['oversampling']
        
        

    runs_edges = np.load(args.runs_edges)
    runs_edges_shape = np.array(args.runs_edges_shape)
    log_inv_prob_runs_edges = np.log(1-runs_edges)
    log_odds_runs_edges = np.log(runs_edges)-log_inv_prob_runs_edges
    const_run_edges = np.sum(log_inv_prob_runs_edges,1)
    uniform_probs = np.load(args.uniform_edge_probs)
    log_inv_unif_probs = np.log(1-uniform_probs)
    log_odds_unif_probs = np.log(uniform_probs) - np.log(1-uniform_probs)
    w_filter  = log_odds_runs_edges - (np.ones(runs_edges_shape) * log_odds_unif_probs).reshape(runs_edges.shape[1])
    const_unif_probs = log_inv_unif_probs.sum()* np.prod(runs_edges_shape[:-1])
    b_const = const_run_edges - const_unif_probs
    
    runs_specs = np.load(args.runs_specs)
    runs_specs = runs_specs.reshape(runs_specs.shape[0],
                                    runs_edges_shape[0],
                                    runs_edges_shape[1])
        
    for wfile_id, wfile in enumerate(np.loadtxt(args.wav_files,dtype=str)):
        spec,features = run_transform(wavfile.read(wfile)[1])



        feature_patches = extract_patches(features,patch_shape=tuple(runs_edges_shape) )
        cur_spec_patches = extract_spec_patches(spec,patch_shape=tuple(runs_specs.shape[1:]) )
        X = get_patch_matrix(feature_patches)
        Y = np.dot(X,log_odds_runs_edges.T) + const_run_edges
        Y = Y.reshape(feature_patches.shape[0],feature_patches.shape[1],log_odds_runs_edges.shape[0])
        Z = np.dot(X,w_filter.T) + b_const
        Z = Z.reshape(feature_patches.shape[0],feature_patches.shape[1],w_filter.shape[0])
        import pdb; pdb.set_trace()
        
        if wfile_id ==0:
            # get an average over the features
            avg = np.zeros(features.shape[-1])

        for run_id, run_filter in enumerate(eight_runs):
            if len(patches[run_id]) > 10000: continue
            F =correlate(features[:,:,run_filter['edge_id']],run_filter['block_filter'],mode='constant',cval=0)
            patch_times, patch_freqs = np.where(F>=args.bfeature_min_count)
            
            patch_freqs = patch_freqs - int(run_filter['block_filter'].shape[1]/2)
            patch_times = patch_times - int(run_filter['block_filter'].shape[1]/2)
            

            feature_patches = extract_patches(features,patch_shape=run_filter['block_filter'].shape + (features.shape[-1],) )
            cur_spec_patches = extract_spec_patches(spec,patch_shape=run_filter['block_filter'].shape )

            use_locs = (patch_times < feature_patches.shape[0]) * (patch_freqs < feature_patches.shape[1])
            patch_freqs = patch_freqs[use_locs]
            patch_times = patch_times[use_locs]

            patches[run_id].extend(feature_patches[patch_times,patch_freqs])
            try:
                spec_patches[run_id].extend(cur_spec_patches[patch_times,patch_freqs].astype(np.float32))
            except: import pdb; pdb.set_trace()
            print run_id, len(patches[run_id])
            
        # compute averages
        new_count = feature_patches.size + count
        avg += (feature_patches.mean(0).mean(0).mean(0).mean(0) - avg) *feature_patches.size/new_count

    np.save('%s_avg_uniform_edges.npy' % (args.save_path),avg)
    for run_id, run_filter in enumerate(eight_runs):
        np.save('%s_%dRun_edges.npy' % (args.save_path, run_id),patches[run_id])
        np.save('%s_%dRun_specs.npy' % (args.save_path, run_id),spec_patches[run_id])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Small script to create 
    plots of binary features overlayed onto spectrograms""")
    parser.add_argument('-c',type=str,default='conf/main.config',help='configuration file')
    parser.add_argument('--binary_features',type=str,default="edges",help='which binary feature transform to use')
    parser.add_argument('--wav_files',type=str,default="sa1.wav",help='path to a list of wav files')
    parser.add_argument('--save_path',type=str,default="plots/sa1_edges_overlayed",help='where to save the overlaying plot (should be without the .png extension since that is added, could be multiple plots output if there are multiple binary features')
    parser.add_argument('--aspect',type=float,default=1.5,help='plot aspect for the overlayed plot')
    parser.add_argument('--bfeature_paths',type=str,help='path to where the binary feature runs path is contained')
    parser.add_argument('--bfeature_min_count',type=int,help='minimum count for the size of the run')
    parser.add_argument('--runs_edges',type=str,help='path to where the runs are being stored')
    parser.add_argument('--uniform_edge_probs',type=str,help='path to where the uniform edge probabilities are being stored')
    parser.add_argument('--runs_edges_shape',type=int,nargs='+',help='path to where the runs are being stored')
    parser.add_argument('--runs_specs',type=str,help='path to where the spectrograms for the runs templates are beint stored')
    main(parser.parse_args())
