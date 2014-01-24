datadir=/home/mark/phoneclassification/data/local/data
local=`pwd`/local
mkdir -p $local
utils=`pwd`/utils
mkdir -p $utils
conf=`pwd`/conf
mkdir -p $conf

exp=`pwd`/exp/learning_ridges
scripts=`pwd`/scripts
mkdir -p $exp
mkdir -p $scripts
plots=$exp/plots
mkdir -p $plots


#
# axoid paths
#
echo -e "
0 1 2 3 4 5 6
0 1 1 2 2 3 3
0 0 0 0 0 0 0
0 -1 -1 -2 -2 -3 -3
0 -1 -2 -3 -4 -5 -6
" > $exp/bfeature_paths

#
# extract the axoids
# include multiple files
winsize=256
wintype=dpss
halftimesupport=4
# setup the configuration file
echo -e "
[SPECTROGRAM]
sample_rate=16000
winsize=$winsize
wintype='$wintype'
order=5
half_time_support=$halftimesupport
oversampling=2
nfft=$winsize
freq_cutoff=5000
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

[SPECGRADIENT]
derivative_clip_alpha=.1
gsigma=6
gt=8
gf=8

[BPF]
sample_rate=16000
fft_length=256
winsize=256
order=5
half_time_support=6
oversampling=1.5
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
threshold=.9
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
random_seed=0" >  $conf/spectrogram_edges_${halftimesupport}HTS_${winsize}WS_${wintype}WT.config

head -n 40 /home/mark/Template-Speech-Recognition/Development/051913/train.wav | sed s:^:/home/mark/Template-Speech-Recognition/Development/051913/: > $exp/patch_files

python $local/binary_guided_axoid_extraction.py -c $conf/spectrogram_edges_${halftimesupport}HTS_${winsize}WS_${wintype}WT.config\
    --binary_features edges_transform_spec\
    --wav_files $exp/patch_files\
    --save_path $exp/run_patches8runs_spectrogram_edges_${halftimesupport}HTS_${winsize}WS_${wintype}WT\
    --bfeature_paths $exp/bfeature_paths\
    --bfeature_min_count 18

#
# now we train the part models and see whether we get 
# any prominent contours
mkdir -p $plots/eight_runs_parts
for nmix in 10 20 30 40 80 ; do
   for n in `seq 0 1 7` ; do
       python $local/train_bernoulli_model.py --patches $exp/run_patches8runs_spectrogram_edges_4HTS_256WS_dpssWT_${n}Run_edges.npy\
           --save_path $exp/run_patches8runs_saved_models_4HTS_256WS_dpssWT_${n}Run_${nmix}NMIX\
           --spec_patches $exp/run_patches8runs_spectrogram_edges_4HTS_256WS_dpssWT_${n}Run_specs.npy\
           --n_components $nmix\
           -v  --viz_spec_parts $plots/eight_runs_parts/viz_parts_${n}Run_${nmix}NMIX.png
   done
done

winsize=256
wintype=dpss
halftimesupport=4
       
nmix=10
n=2
python $local/runs_transform.py -c $conf/spectrogram_edges_${halftimesupport}HTS_${winsize}WS_${wintype}WT.config\
    --binary_features edges_transform_spec\
    --wav_files $exp/patch_files\
    --save_path $exp/run_patches8runs_spectrogram_edges_${halftimesupport}HTS_${winsize}WS_${wintype}WT\
    --bfeature_paths $exp/bfeature_paths\
    --bfeature_min_count 18\
    --runs_edges $exp/run_patches8runs_saved_models_4HTS_256WS_dpssWT_${n}Run_${nmix}NMIX_means.npy\
    --uniform_edge_probs $exp/run_patches8runs_spectrogram_edges_${halftimesupport}HTS_${winsize}WS_${wintype}WT_avg_uniform_edges.npy\
    --runs_edges_shape 9 9 8\
    --runs_specs $exp/run_patches8runs_saved_models_4HTS_256WS_dpssWT_${n}Run_${nmix}NMIX_S_clusters.npy


for nmix in 10 20 30 40 80 ; do
   for n in `seq 0 1 7` ; do

python $local/display_bernoulli_model.py --patches $exp/run_patches8runs_spectrogram_edges_4HTS_256WS_dpssWT_${n}Run_edges.npy\
           --save_path $exp/run_patches8runs_saved_models_4HTS_256WS_dpssWT_${n}Run_${nmix}NMIX\
           --spec_patches $exp/run_patches8runs_spectrogram_edges_4HTS_256WS_dpssWT_${n}Run_specs.npy\
           --n_components $nmix\
           -v  --display_bernoulli_parts $plots/eight_runs_parts/viz_bernoulli_parts_${n}Run_${nmix}NMIX.png

done
done
