datadir=/home/mark/phoneclassification/data/local/data
local=`pwd`/local
mkdir -p $local
utils=`pwd`/utils
mkdir -p $utils
conf=`pwd`/conf
mkdir -p $conf

exp=`pwd`/exp/formant_design
scripts=`pwd`/scripts
mkdir -p $exp
mkdir -p $scripts
plots=$exp/plots
mkdir -p $plots


# setup the configuration file
cat << "EOF" >> $conf/main.config
[SPECTROGRAM]
sample_rate=16000
num_window_samples=320
num_window_step_samples=80
fft_length=512
kernel_length=7
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
EOF

python $local/visualize_binary_features.py -c $conf/main.config\
    --binary_features edges\
    --wav_file /home/mark/Template-Speech-Recognition/Development/051913/train/dr1/fcjf0/sa1.wav \
    --save_path $plots/binary_features_on_spectrogram_fcjf0sa1_edges


python $local/draw_axoid_axis_wheel.py --axoid_coordinates 1.61803398875 0.6180339887498547 --nroots 8 --above_offset -.25 0.5 --below_offset .25 -0.5 > $plots/axoid_edges1

python $local/draw_axoid_axis_wheel.py --axoid_coordinates 1.61803398875 -0.6180339887498547 --nroots 8 --above_offset .25 0.5 --below_offset -.05 -0.5 > $plots/axoid_edges2

#
# now we test edges on synthetic data to confirm the edge
# directions that we presume work


python $local/generate_synthetic_formant.py --binary_sequence 1 1 1 1 \
    --nsequences 5 --radius 2 -c $conf/main.config --save_path $plots/seq1111edge_grid.png

python $local/generate_synthetic_formant.py --binary_sequence 1 0 1 0 --nsequences 5 --radius 2 -c $conf/main.config --save_path $plots/seq1010edge_grid.png

python $local/generate_synthetic_formant.py --binary_sequence 0 0 0 0 \
    --nsequences 2 --radius 2 -c $conf/main.config --save_path $plots/seq0000edge_grid.png

python $local/generate_synthetic_formant.py --binary_sequence -1 0 -1 0 --nsequences 5 --radius 2 -c $conf/main.config --save_path $plots/seq-10-10edge_grid.png

python $local/generate_synthetic_formant.py --binary_sequence -1 -1 -1 -1 \
    --nsequences 5 --radius 2 -c $conf/main.config --save_path $plots/seq-1-1-1-1edge_grid.png

python $local/axoid_five_family.py --freq_radius 4\
   --n_times 5\
   --obj_prob .8\
   --bgd_prob .1\
   --save_path $exp/axoid_4FR_5T_.8O_.1B.npy



python $local/visualize_axoid_transform.py -c $conf/main.config\
    --binary_features edges\
    --wav_file /home/mark/Template-Speech-Recognition/Development/051913/train/dr1/fcjf0/sa1.wav \
    --save_path $plots/axoid_features_on_spectrogram_fcjf0sa1_edges \
    --axoids $exp/axoid_4FR_5T_.8O_.1B.npy

# setup the configuration file
cat << "EOF" >> $conf/main_longer.config
[SPECTROGRAM]
sample_rate=16000
num_window_samples=320
num_window_step_samples=20
fft_length=512
kernel_length=7
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
threshold=.85
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
EOF

python $local/visualize_axoid_transform.py -c $conf/main_longer.config\
    --binary_features edges\
    --wav_file /home/mark/Template-Speech-Recognition/Development/051913/train/dr1/fcjf0/sa1.wav \
    --save_path $plots/axoid_features_on_spectrogram_fcjf0sa1_longer_axoids \
    --axoids $exp/axoid_4FR_5T_.8O_.1B.npy\
    --aspect 4

python $local/visualize_binary_features.py -c $conf/main_longer.config\
    --binary_features edges\
    --wav_file /home/mark/Template-Speech-Recognition/Development/051913/train/dr1/fcjf0/sa1.wav \
    --save_path $plots/binary_features_on_spectrogram_fcjf0sa1_longer_edges\
    --aspect 4
