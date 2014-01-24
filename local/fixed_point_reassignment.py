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
from pyscat import binary_phase_meanshift
from visualize_binary_features import overlayed_plot

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
        
        
    for fl, fl_name in np.loadtxt(args.wav_files,dtype=str):
        print fl, fl_name
        spec,features = run_transform(wavfile.read(fl)[1])[:2]

        Y = binary_phase_meanshift.frequency_constrained_meanshift(features[:,:,0],3,1,10,niter=10)

        overlayed_plot(Y,np.log(spec),winsize,sample_rate,'%s_%s_%d' % (args.save_path,fl_name,0),winstep,oversampling,use_aspect=args.aspect)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Small script to create 
    plots of binary features overlayed onto spectrograms""")
    parser.add_argument('-c',type=str,default='conf/main.config',help='configuration file')
    parser.add_argument('--binary_features',type=str,default="edges",help='which binary feature transform to use')
    parser.add_argument('--wav_files',type=str,default="sa1.wav",help='path to the list of wav files')
    parser.add_argument('--save_path',type=str,default="plots/sa1_edges_overlayed",help='where to save the overlaying plot (should be without the .png extension since that is added, could be multiple plots output if there are multiple binary features')
    parser.add_argument('--aspect',type=float,default=1.5,help='plot aspect for the overlayed plot')
    main(parser.parse_args())
