from __future__ import division
import numpy as np
import argparse
from template_speech_rec import configParserWrapper
from scipy.io import wavfile
import template_speech_rec.get_train_data as gtrd
import filterbank as fb    
from transforms import binary_phase_features, preemphasis, process_wav, spectrogram_magnitude_gradients, spectrogram
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter, convolve
from nitime.algorithms.spectral import dpss_windows

def main(args):
    config_d = configParserWrapper.load_settings(open(args.c,'r'))


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


    if args.do_gradients:
        run_transform = lambda x: spectrogram_magnitude_gradients(
        preemphasis(process_wav(x),preemph=config_d['SPECTROGRAM']['preemphasis']),
        config_d['SPECTROGRAM']['sample_rate'],
        config_d['SPECTROGRAM']['freq_cutoff'],
        config_d['SPECTROGRAM']['winsize'],
        config_d['SPECTROGRAM']['nfft'],
        oversampling,
        htemp,dhtemp,tttemp)

        M,dMdt,dMdw = run_transform(wavfile.read(args.wav_file)[1])

        # clip extreme values
        v = np.sort(dMdt.ravel())
        low_tail = v[int(config_d['SPECGRADIENT']['derivative_clip_alpha']*len(v)+.5)]
        high_tail = v[int((1-config_d['SPECGRADIENT']['derivative_clip_alpha'])*len(v)+.5)]
        dMdt = np.clip(dMdt,low_tail,high_tail)
    
        v = np.sort(dMdw.ravel())
        low_tail = v[int(config_d['SPECGRADIENT']['derivative_clip_alpha']*len(v)+.5)]
        high_tail = v[int((1-config_d['SPECGRADIENT']['derivative_clip_alpha'])*len(v)+.5)]
        dMdw = np.clip(dMdw,low_tail,high_tail)


        plt.imshow(dMdt.T,origin='lower',cmap='bone',interpolation='nearest')
        plt.savefig('%s_dlogMdt.png' % args.save_path,bbox_inches='tight')

        plt.close('all')
        plt.imshow(dMdw.T,origin='lower',cmap='bone',interpolation='nearest')
        plt.savefig('%s_dlogMdw.png' % args.save_path,bbox_inches='tight')
    

    else:
        run_transform = lambda x: spectrogram(
        preemphasis(process_wav(x),preemph=config_d['SPECTROGRAM']['preemphasis']),
        config_d['SPECTROGRAM']['sample_rate'],
        config_d['SPECTROGRAM']['freq_cutoff'],
        config_d['SPECTROGRAM']['winsize'],
        config_d['SPECTROGRAM']['nfft'],
        oversampling,
        htemp)

        M = run_transform(wavfile.read(args.wav_file)[1])

    # compute edges using derivative gaussian filter
    plt.close('all')
    plt.imshow(np.log(M).T,origin='lower',cmap='hot',interpolation='nearest')
    plt.savefig('%s_M.png' % args.save_path,bbox_inches='tight')

    plt.close('all')




    X = gtrd.get_edge_features_use_config(np.log(M).T,config_d['EDGES'])

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Small script to create 
    plots of binary features overlayed onto spectrograms""")
    parser.add_argument('-c',type=str,default='conf/main.config',help='configuration file')
    parser.add_argument('--binary_features',type=str,default="edges",help='which binary feature transform to use')
    parser.add_argument('--wav_file',type=str,default="sa1.wav",help='path to the wav file')
    parser.add_argument('--do_gradients',action='store_true',help='whether to compute the gradients of the spectrogram--currently only available for hermite windows')
    parser.add_argument('--save_path',type=str,default="plots/sa1_edges_overlayed",help='where to save the overlaying plot (should be without the .png extension since that is added, could be multiple plots output if there are multiple binary features')
    parser.add_argument('--aspect',type=float,default=1.5,help='plot aspect for the overlayed plot')
    main(parser.parse_args())
