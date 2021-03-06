datadir=/home/mark/phoneclassification/data/local/data
local=`pwd`/local
mkdir -p $local
utils=`pwd`/utils
mkdir -p $utils
conf=`pwd`/conf
mkdir -p $conf

exp=`pwd`/exp/binary_phase_meanshift
scripts=`pwd`/scripts
mkdir -p $exp
mkdir -p $scripts
plots=$exp/plots
mkdir -p $plots

echo -e "
[SPECTROGRAM]
sample_rate=16000
num_window_samples=320
num_window_step_samples=80
fft_length=512
kernel_length=7
freq_cutoff=3000
preemphasis=.95
use_mel=False
do_mfccs=False
no_use_dpss=False
mel_nbands=40
num_ceps=13
liftering=.6
include_energy=False
include_deltas=False
include_double_deltas=False
delta_window=9
do_freq_smoothing=False
mel_smoothing_kernel=-1

[BPF]
sample_rate=16000
fft_length=256
winsize=256
order=5
half_time_support=4
oversampling=2
t_s=3200
gsigma=6
gt=8
gf=8
preemph=.95
nfft=256
fthresh=.12
othresh=.05
sample_rate=16000
freq_cutoff=5000
spread_length=3


[OBJECT]
T_s=3072


[SVM]
example_length=.2
kernel='linear'
penalty_list=['little_reg','0.1',
                                                 'reg_plus', '0.01',
                                                 'reg_plus_plus','0.001',
						 'reg_plus_plus_plus','0.0001']


[EDGES]
block_length=40
spread_length=1
threshold=.7
magnitude_block_length=0
abst_threshold=(0.025,  0.025,  0.015,  0.015,  0.02 ,  0.02 ,  0.02 ,  0.02 )
magnitude_threshold=.4
magnitude_spread=1
magnitude_and_edge_features=False
magnitude_features=False
mag_smooth_freq=0
mag_downsample_freq=0
auxiliary_data=False
auxiliary_threshold=.5
num_mag_channels=10
num_axiliary_data=3
save_parts_S=False

[EM]
n_components=6
n_init=6
n_iter=200
random_seed=0
[SPECTROGRAM]
sample_rate=16000
num_window_samples=320
num_window_step_samples=80
fft_length=512
kernel_length=7
freq_cutoff=3000
preemphasis=.95
use_mel=False
do_mfccs=False
no_use_dpss=False
mel_nbands=40
num_ceps=13
liftering=.6
include_energy=False
include_deltas=False
include_double_deltas=False
delta_window=9
do_freq_smoothing=False
mel_smoothing_kernel=-1

" > $conf/main.config

paste <( head -n 40 /home/mark/Template-Speech-Recognition/Development/051913/train.wav | sed s:^:/home/mark/Template-Speech-Recognition/Development/051913/:)  <( head -n 40 /home/mark/Template-Speech-Recognition/Development/051913/train.wav | sed s:.wav:: | sed s:/:_:g ) > $exp/display_wav_fls


# # process and input signal
# compute the binary phase features and see what comes out
python $local/mean_shift_binary_features.py -c $conf/main.config\
    --binary_features phase_peaks \
    --wav_files $exp/display_wav_fls \
    --save_path $plots/mean_shift_binary_phase_peaks \
    --aspect 3

#
# time to get setup for an experiment and see if we can
# great a more invariant speech recognition
# architecture

