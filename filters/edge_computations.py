from __future__ import division
import filterbank as fb
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import transforms
from scipy.special import gamma
from scipy.signal import hilbert
from scipy.ndimage.filters import median_filter, convolve, maximum_filter
from _reassignment import reassign

taper_length=255
order=6
half_time_support=6
h_temp,dh_temp,ddh_temp, tt_temp = fb.hermite_window(taper_length,
                   order,
                   half_time_support)
h = np.zeros((h_temp.shape[0],
              h_temp.shape[1]+1))
h[:,:-1] = h_temp

dh = np.zeros((dh_temp.shape[0],
              dh_temp.shape[1]+1))
dh[:,:-1] = dh_temp

ddh = np.zeros((ddh_temp.shape[0],
              ddh_temp.shape[1]+1))
ddh[:,:-1] = ddh_temp

tt = (2*tt_temp[-1] -tt_temp[-2])*np.ones(tt_temp.shape[0]+1)
tt[:-1] = tt_temp

sr,x = wavfile.read('/home/mark/Template-Speech-Recognition/Development/051913/train/dr1/fcjf0/sa1.wav')
phones = np.loadtxt('/home/mark/Template-Speech-Recognition/Development/051913/train/dr1/fcjf0/sa1.phn',dtype=str)
x = x.astype(float)/2**15
x[1:] = x[1:] - .95*x[:-1]
sample_rate=16000
freq_cutoff=5000

N = x.size
N1 = 256
winsize=taper_length+1

oversampling=1.5
nframes = int(.5 + N/N1*2**oversampling)

greater_than_winlength = winsize*np.ones((nframes,N1)) > np.arange(N1)

indices = (np.arange(N1,dtype=int)-int(N1/2))[np.newaxis, : ] + int(N1/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]

indices *= (2*(indices > 0)-1)
# symmetrize the tail
tail_indices = indices > N-1
indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)


avg = np.zeros(indices.shape,dtype=np.complex128)
avg_d = np.zeros(indices.shape,dtype=np.complex128)
abs_avg = np.zeros(indices.shape)

for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    avg += f
    abs_avg += np.abs(f_mv)

avg/=5
abs_avg/=5

midpoints = indices[:,int(indices.shape[1]/2)]

gsigma=.5
glength = 11
gt = np.arange(glength) - int(glength/2)
gt_mid = (gt[:-1] + gt[1:])/2
make_gfilter = lambda gsigma,gt: np.exp(-gt**2/(2*gsigma**2)) /(gsigma*np.sqrt(2*np.pi))
make_g1filter = lambda gsigma, gt: -gt/gsigma**2 * make_gfilter(gsigma,gt)
g1filter = make_g1filter(gsigma,gt)
g1filter_mid = make_g1filter(gsigma,gt_mid)
g1filter_hilbert = np.imag(hilbert(g1filter))
g1filter_hilbert_mid = np.imag(hilbert(g1filter_mid))

g2filter = (gt**2 - gsigma**2)/(gsigma**5 * np.sqrt(2*np.pi)) * np.exp(-gt**2/(2*gsigma**2))
g2filter_mid = (gt_mid**2 - gsigma**2)/(gsigma**5 * np.sqrt(2*np.pi)) * np.exp(-gt_mid**2/(2*gsigma**2))
g2filter_hilbert = np.imag(hilbert(g2filter))
g2filter_hilbert_mid = np.imag(hilbert(g2filter_mid))
ggamma = 1
orthog_gfilt = make_gfilter(gsigma*ggamma,gt)
orthog_gfilt_mid = make_gfilter(gsigma*ggamma,gt_mid)


verte_filter = np.outer(orthog_gfilt,g1filter)
verte_filter_hilbert = np.outer(orthog_gfilt,g1filter)
horize_filter = verte_filter.T
horize_filter_hilbert= verte_filter_hilbert.T


# get the diagonal


#
# plot the basic multitaper hermite spectrogram
#
#
plt.imshow(np.log(abs_avg).T[:100],origin='lower',interpolation='nearest'); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:100:16], (np.arange(abs_avg.shape[1])[:100:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Multitaper Hermite')
plt.savefig('multitaper_hermite_sa1.png',bbox_inches='tight')

plt.imshow(np.angle(avg).T[:100],origin='lower',interpolation='nearest',cmap='hsv'); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:100:16], (np.arange(abs_avg.shape[1])[:100:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Multitaper Hermite (Phase)')
plt.savefig('phase_multitaper_hermite_sa1.png',bbox_inches='tight')

#
# now compute the phase derivatives
#

avg_dphi_dt = np.zeros(indices.shape)
avg_dphi_dw = np.zeros(indices.shape)

for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f #* np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
    df_mv = df #* np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*h[i]))
    tf_mv = tf #* np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    abs_f_mv = np.abs(f_mv)**2
    dphi_dt = np.imag((df_mv * f_mv.conj())/abs_f_mv)
    dphi_dw = - np.real((tf_mv * f_mv.conj())/abs_f_mv)
    avg_dphi_dt += dphi_dt
    avg_dphi_dw += dphi_dw

const=5/256*2*np.pi
avg_dphi_dt/=const
avg_dphi_dw/=const


plt.imshow(avg_dphi_dt.T[:100],origin='lower',interpolation='nearest',vmin=-2,vmax=2,cmap='bone'); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:100:16], (np.arange(abs_avg.shape[1])[:100:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Multitaper Phase Time Partial Derivative')
plt.savefig('multitaper_hermite_dphi_dt_sa1.png',bbox_inches='tight')

gsigma=5
gfilter = np.exp(-((np.mgrid[:8,:8]-3.5)**2).sum(0)/(2*gsigma))/(gsigma*np.sqrt(2*np.pi))
gfilter /= gfilter.sum()
gdwfilter = -(np.mgrid[:8,:8]-3.5)[0]/gsigma * gfilter
gdtfilter = -(np.mgrid[:8,:8]-3.5)[1]/gsigma * gfilter
gdw2filter = ((np.mgrid[:8,:8]-3.5)[0]**2 - gsigma)/gsigma**2 * gfilter
gdt2filter = ((np.mgrid[:8,:8]-3.5)[1]**2 - gsigma)/gsigma**2 * gfilter

plt.close('all')
plt.imshow(convolve(avg_dphi_dt.T[:80],gdwfilter)>1,origin='lower',interpolation='nearest',cmap='bone',aspect=1.5); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:80:16], (np.arange(abs_avg.shape[1])[:80:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.savefig('multitaper_hermite_dphi_dt_thresh_diff_sa1.png',bbox_inches='tight')

plt.close('all')
plt.imshow(np.log(abs_avg).T[:80],origin='lower',interpolation='nearest',aspect=1.5); 
fcoords,tcoords = np.where(convolve(avg_dphi_dt.T[:80],gdwfilter)>1 )
plt.scatter(tcoords,fcoords,linewidth=.1,c='black',marker='+')
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:80:16], (np.arange(abs_avg.shape[1])[:80:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.savefig('multitaper_hermite_d2phi_dtdw_thresh_diff_overlayed_sa1.png',bbox_inches='tight')

plt.imshow(convolve(avg_dphi_dt.T[:80],gdtfilter)>1,origin='lower',interpolation='nearest',cmap='bone'); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:80:16], (np.arange(abs_avg.shape[1])[:80:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Thresholded Multitaper Phase Time Partial Derivative -- Gaussian difference filter')
plt.savefig('multitaper_hermite_dphi_dt_thresh_diff_sa1.png',bbox_inches='tight')


plt.close('all')
plt.imshow(convolve(np.log(abs_avg[:80]),gdw2filter)>1,origin='lower',interpolation='nearest',cmap='bone',aspect=1.5); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:80:16], (np.arange(abs_avg.shape[1])[:80:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.savefig('multitaper_hermite_d2logM_dw2_sm_thresh_sa1.png',bbox_inches='tight')

           


plt.imshow(np.log(abs_avg).T[:100],origin='lower',interpolation='nearest'); 
fcoords,tcoords = np.where(convolve(avg_dphi_dt.T[:100],gdwfilter)>1 )
plt.scatter(tcoords,fcoords,linewidth=.1,c='black',marker='+')
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:100:16], (np.arange(abs_avg.shape[1])[:100:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Thresholded Multitaper Phase Time Partial Derivative -- Gaussian difference filter')
plt.savefig('multitaper_hermite_dphi_dt_thresh_diff_sa1.png',bbox_inches='tight')

           

plt.imshow(np.log(abs_avg).T[:100],origin='lower',interpolation='nearest'); 
plt.scatter(tcoords,fcoords,linewidth=.1,c='black',marker='+')
plt.show()

# then we look at doing reassignment constrained by these masks
reassign_mask = ((convolve(np.log(abs_avg),-gdw2filter)>-.40)*(convolve(avg_dphi_dt,gdwfilter)>.75)*(np.log(abs_avg)>-8)).astype(int)
           
RS = np.zeros(abs_avg.shape)
RT = np.zeros(abs_avg.shape)
RF = np.zeros(abs_avg.shape)
reassign(RS,
         abs_avg,
         np.zeros(abs_avg.shape),
         avg_dphi_dt,
         RT,
         RF,
         2*reassign_mask)



# look at the second-order partial derivatives

avg_d2phi_dtdw = np.zeros(indices.shape)
avg_d2phi_dw2 = np.zeros(indices.shape)
avg_d2phi_dt2 = np.zeros(indices.shape)

for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f * np.exp(-2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
    df_mv = df * np.exp(-2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*h[i]))
    tf_mv = tf * np.exp(-2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tdf = np.fft.fft((x[indices]*greater_than_winlength) * np.outer(indices[:,0],np.arange(indices.shape[1]))(tt*dh[i]))
    tdf_mv = tdf * np.exp(-2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    ddf = np.fft.fft((x[indices]*greater_than_winlength) * ddh[i])
    ddf_mv = ddf * np.exp(-2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    ttf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*tt*h[i]))
    ttf_mv = ttf * np.exp(-2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    abs_f_mv = np.abs(f_mv)**2
    d2phi_dtdw = np.real(tdf_mv * f_mv.conj()/ abs_f_mv) - np.real(tf_mv * df_mv/ f_mv**2)
    d2phi_dt2 = np.imag(ddf_mv * f_mv.conj()/ abs_f_mv) - np.imag((df_mv * f_mv.conj()/abs_f_mv)**2) 
    d2phi_dw2 = np.imag((tf_mv * f_mv.conj()/abs_f_mv)**2) - np.imag(ttf_mv * f_mv.conj()/ abs_f_mv)
    avg_d2phi_dtdw += d2phi_dtdw
    avg_d2phi_dw2 += d2phi_dw2
    avg_d2phi_dt2 += d2phi_dt2



avg_d2phi_dtdw/=5
avg_d2phi_dw2 /= 5
avg_d2phi_dt2 /= 5

tlocs, flocs = np.mgrid[:indices.shape[0],:indices.shape[1]]

plt.imshow(avg_d2phi_dt2.T[:100],origin='lower',interpolation='nearest',vmin=-.05,vmax=.05,cmap='bone'); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:100:16], (np.arange(abs_avg.shape[1])[:100:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Multitaper Phase Time Partial Derivative')
plt.show()
plt.savefig('multitaper_hermite_d2phi_dtdw_sa1.png',bbox_inches='tight')

avg_dphi_dt = np.zeros(indices.shape)
avg_dphi_dw = np.zeros(indices.shape)
avg_d2phi_dtdw = np.zeros(indices.shape)
avg_d2phi_dw2 = np.zeros(indices.shape)
avg_d2phi_dt2 = np.zeros(indices.shape)
avg_dlogM_dw = np.zeros(indices.shape)
avg_dlogM_dt = np.zeros(indices.shape)
avg_d2logM_dw2 = np.zeros(indices.shape)
avg_d2logM_dt2 = np.zeros(indices.shape)
avg_d2logM_dwdt = np.zeros(indices.shape)
avg_RS = np.zeros(indices.shape)


gsigma=1
gfilter = np.exp(-((np.mgrid[:8,:8]-3.5)**2).sum(0)/(2*gsigma))
gdwfilter = -(np.mgrid[:8,:8]-3.5)[0]/gsigma * gfilter
gdtfilter = -(np.mgrid[:8,:8]-3.5)[1]/gsigma * gfilter
gdw2filter = ((np.mgrid[:8,:8]-3.5)[0]**2 - gsigma)/gsigma**2 * gfilter

for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f 
    df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
    df_mv = df 
    tf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*h[i]))
    tf_mv = tf 
    tdf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*dh[i]))
    tdf_mv = tdf 
    ddf = np.fft.fft((x[indices]*greater_than_winlength) * ddh[i])
    ddf_mv = ddf 
    ttf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*tt*h[i]))
    ttf_mv = ttf 
    abs_f_mv = np.abs(f_mv)**2
    d2phi_dtdw = np.real(tdf_mv * f_mv.conj()/ abs_f_mv) - np.real(tf_mv * df_mv/ f_mv**2)
    d2phi_dt2 = np.imag(ddf_mv * f_mv.conj()/ abs_f_mv) - np.imag((df_mv * f_mv.conj()/abs_f_mv)**2) 
    d2phi_dw2 = np.imag((tf_mv * f_mv.conj()/abs_f_mv)**2) - np.imag(ttf_mv * f_mv.conj()/ abs_f_mv)
    avg_d2phi_dtdw += d2phi_dtdw
    avg_d2phi_dw2 += d2phi_dw2
    avg_d2phi_dt2 += d2phi_dt2
    dphi_dt = -np.imag((df_mv * f_mv.conj())/abs_f_mv)
    dphi_dw =  np.real((tf_mv * f_mv.conj())/abs_f_mv)
    avg_dphi_dt += dphi_dt
    avg_dphi_dw += dphi_dw
    dlogM_dt = convolve(np.real(df_mv * f_mv.conj()/abs_f_mv),gfilter)
    dlogM_dw = convolve(np.imag(tf_mv * f_mv.conj()/abs_f_mv),gfilter)
    d2logM_dt2 = convolve(np.real(df_mv * f_mv.conj()/abs_f_mv),gdtfilter)
    d2logM_dw2 = convolve(np.imag(tf_mv * f_mv.conj()/abs_f_mv),gdwfilter)
    d2logM_dwdt = convolve(np.imag(tf_mv * f_mv.conj()/abs_f_mv),gdtfilter)
    avg_dlogM_dw += dlogM_dw
    avg_dlogM_dt = dlogM_dt
    avg_d2logM_dw2 += d2logM_dw2
    avg_d2logM_dt2 = d2logM_dt2
    avg_d2logM_dwdt += d2logM_dwdt
    RS = np.zeros(abs_avg.shape)
    RT = np.zeros(abs_avg.shape)
    RF = np.zeros(abs_avg.shape)
    reassign(RS,
             abs_f_mv,
             np.zeros(abs_f_mv.shape),
             dphi_dt,
             RT,
             RF,
             2*(abs_f_mv > 1e-3).astype(int))
    avg_RS += RS


avg_RS /=5
const = 5/256*2*np.pi
avg_dphi_dt/=const
avg_dphi_dw/=const
avg_d2phi_dtdw/=const
avg_d2phi_dw2 /= const
avg_d2phi_dt2 /= const
avg_dlogM_dt/=const
avg_dlogM_dw/=const
avg_d2logM_dwdt/=const
avg_d2logM_dw2 /= const
avg_d2logM_dt2 /= const


gsigma=1
gfilter = np.exp(-((np.mgrid[:8,:8]-3.5)**2).sum(0)/(2*gsigma))
gdwfilter = -(np.mgrid[:8,:8]-3.5)[1]/gsigma * gfilter
gdtfilter = -(np.mgrid[:8,:8]-3.5)[0]/gsigma * gfilter




RS = np.zeros(abs_avg.shape)
RT = np.zeros(abs_avg.shape)
RF = np.zeros(abs_avg.shape)
reassign(RS,
abs_avg,
np.zeros(abs_avg.shape),
avg_dphi_dt,
             RT,
             RF,
             2*((abs_avg > 1e-3) * (np.abs(convolve(avg_d2phi_dt2,gfilter)) < .3)).astype(int))


gsigma=2; gw=20
gfilter = np.exp(-((np.mgrid[:gw,:gw]-gw/2)**2).sum(0)/(2*gsigma))
gdwfilter = -(np.mgrid[:gw,:gw]-gw/2)[1]/gsigma * gfilter
gdtfilter = -(np.mgrid[:gw,:gw]-gw/2)[0]/gsigma * gfilter

sm6_avg_d2phi_dtdw = convolve(avg_d2phi_dtdw,gfilter)

plt.imshow(sm6_avg_d2phi_dtdw.T[:100],origin='lower',interpolation='nearest',vmin=-.05,vmax=.05,cmap='bone'); 
plt.xticks( np.arange(abs_avg.shape[0])[::80], (midpoints/16)[::80].astype(int))
plt.xlabel('ms')
plt.yticks( np.arange(abs_avg.shape[1])[:100:16], (np.arange(abs_avg.shape[1])[:100:16]/winsize * sample_rate).astype(int))
plt.ylabel('Hz')
plt.title('Multitaper Phase Time Partial Derivative')
plt.show()


avg_sm_d2phi_dtdw = np.zeros(frames.shape)
avg_sm_d2phi_dw2 = np.zeros(frames.shape)
avg_sm_d2phi_dt2 = np.zeros(frames.shape)
avg_dlogM_dw = np.zeros(frames.shape)
avg_dlogM_dt = np.zeros(frames.shape)
avg_d2logM_dw2 = np.zeros(frames.shape)
avg_d2logM_dt2 = np.zeros(frames.shape)
avg_d2logM_dwdt = np.zeros(frames.shape)

gsigma=6
gfilter = np.exp(-((np.mgrid[:8,:8]-3.5)**2).sum(0)/(2*gsigma))
gdwfilter = -(np.mgrid[:8,:8]-3.5)[1]/gsigma * gfilter
gdtfilter = -(np.mgrid[:8,:8]-3.5)[0]/gsigma * gfilter




N1 = 256
oversampling=2
nframes = int(.5 + N/N1*2**oversampling)

greater_than_winlength = 256*np.ones((nframes,N1)) > np.arange(N1)

indices = (np.arange(N1,dtype=int)-int(N1/2))[np.newaxis, : ] + int(N1/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]

indices *= (2*(indices > 0)-1)
# symmetrize the tail
tail_indices = indices > N-1
indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)




taper_length=511
order=6
half_time_support=6
h_temp,dh_temp,ddh_temp, tt = fb.hermite_window(taper_length,
                   order,
                   half_time_support)
h = np.zeros((h_temp.shape[0],
              h_temp.shape[1]+1))
h[:,:-1] = h_temp

dh = np.zeros((dh_temp.shape[0],
              dh_temp.shape[1]+1))
dh[:,:-1] = dh_temp

ddh = np.zeros((ddh_temp.shape[0],
              ddh_temp.shape[1]+1))
ddh[:,:-1] = ddh_temp


sr,x = wavfile.read('/home/mark/Research/phoneclassification/sa1.wav')
x = x.astype(float)/2**15
x[1:] = x[1:] - .95*x[:-1]

oversampling=3
N = x.size
N1 = 512
nframes = int(.5 + N/N1*2**oversampling)

greater_than_winlength = 386*np.ones((nframes,N1)) > np.arange(N1)

indices = (np.arange(N1,dtype=int)-int(N1/2))[np.newaxis, : ] + int(N1/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]

indices *= (2*(indices > 0)-1)
# symmetrize the tail
tail_indices = indices > N-1
indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)

frames = np.fft.fft((x[indices]*greater_than_winlength) * h[0])
frames_mv = frames * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)

dframes = np.fft.fft((x[indices]*greater_than_winlength) * dh[0])
dframes_mv = dframes * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)

w_hat_delta = np.imag((dframes_mv * frames_mv.conj())/np.abs(frames)**2)
w_hat_delta_unnorm = np.imag((dframes_mv * frames_mv.conj()))
w_hat_delta_compressed = np.log(w_hat_delta * np.sign(w_hat_delta) +1e-8) * np.sign(w_hat_delta)


taper_length=255
order=6
half_time_support=6
h_temp,dh_temp, tt = hermite_window(taper_length,
                   order,
                   half_time_support)
h = np.zeros((h_temp.shape[0],
              h_temp.shape[1]+1))
h[:,:-1] = h_temp

dh = np.zeros((dh_temp.shape[0],
              dh_temp.shape[1]+1))
dh[:,:-1] = dh_temp



N1 = 256
oversampling=2
nframes = int(.5 + N/N1*2**oversampling)

greater_than_winlength = 256*np.ones((nframes,N1)) > np.arange(N1)

indices = (np.arange(N1,dtype=int)-int(N1/2))[np.newaxis, : ] + int(N1/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]

indices *= (2*(indices > 0)-1)
# symmetrize the tail
tail_indices = indices > N-1
indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)

frames = np.fft.fft((x[indices]*greater_than_winlength) * h[0])
frames_mv = frames * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)

dframes = np.fft.fft((x[indices]*greater_than_winlength) * dh[0])
dframes_mv = dframes * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)

w_hat_delta = np.imag((dframes_mv * frames_mv.conj())/np.abs(frames)**2)
w_hat_delta_unnorm = np.imag((dframes_mv * frames_mv.conj()))
w_hat_delta_compressed = np.log(w_hat_delta * np.sign(w_hat_delta) +1e-8) * np.sign(w_hat_delta)

plt.subplot(2,1,1);plt.imshow(w_hat_delta.T[:100],interpolation='nearest',origin='lower',aspect=2,vmin=-.03,vmax=.03,cmap='bone'); plt.subplot(2,1,2); plt.imshow(np.log(np.abs(frames)).T[:100],origin='lower',interpolation='nearest'); plt.show() 

avg = np.zeros(frames.shape,dtype=np.complex128)
avg_d = np.zeros(dframes.shape,dtype=np.complex128)
abs_avg = np.zeros(frames.shape)
w_hat_delta_avg = np.zeros(frames.shape)

for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    avg += f
    abs_avg += np.abs(f_mv)
    df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
    df_mv = df * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    avg_d += np.abs(df_mv)
    w_hat_delta_avg += np.imag((df_mv * f_mv.conj())/np.abs(f_mv)**2)

avg/=5
avg_d/=5
abs_avg/=5
w_hat_delta_avg/=5

w_hat_delta = np.imag((avg_d * avg.conj())/np.abs(avg)**2)
w_hat_delta_unnorm = np.imag((avg_d * avg.conj()))
w_hat_delta_compressed = np.log(w_hat_delta * np.sign(w_hat_delta) +1e-8) * np.sign(w_hat_delta)

plt.subplot(2,1,1);plt.imshow(w_hat_delta.T[:100],interpolation='nearest',origin='lower',aspect=2,vmin=-.03,vmax=.03,cmap='bone'); plt.subplot(2,1,2); plt.imshow(np.log(np.abs(avg)).T[:100],origin='lower',interpolation='nearest'); plt.show() 

plt.subplot(2,1,1);plt.imshow(w_hat_delta_avg.T[:100],interpolation='nearest',origin='lower',aspect=2,vmin=-.03,vmax=.03,cmap='bone'); plt.subplot(2,1,2); plt.imshow(np.log(np.abs(avg)).T[:100],origin='lower',interpolation='nearest'); plt.show() 

# now getting the hessian
taper_length=255
order=6
half_time_support=6
h_temp,dh_temp,ddh_temp, tt_temp = hermite_window(taper_length,
                   order,
                   half_time_support)
h = np.zeros((h_temp.shape[0],
              h_temp.shape[1]+1))
h[:,:-1] = h_temp

dh = np.zeros((dh_temp.shape[0],
              dh_temp.shape[1]+1))
dh[:,:-1] = dh_temp

ddh = np.zeros((ddh_temp.shape[0],
              ddh_temp.shape[1]+1))
ddh[:,:-1] = ddh_temp

tt = (2*tt_temp[-1] -tt_temp[-2])*np.ones(tt_temp.shape[0]+1)
tt[:-1] = tt_temp

avg = np.zeros(frames.shape,dtype=np.complex128)
avg_d = np.zeros(dframes.shape,dtype=np.complex128)
abs_avg = np.zeros(frames.shape)
avg_dphi_dt = np.zeros(frames.shape)
avg_dphi_dw = np.zeros(frames.shape)
avg_d2phi_dtdw = np.zeros(frames.shape)
avg_d2phi_dw2 = np.zeros(frames.shape)
avg_d2phi_dt2 = np.zeros(frames.shape)
avg_sm_d2phi_dtdw = np.zeros(frames.shape)
avg_sm_d2phi_dw2 = np.zeros(frames.shape)
avg_sm_d2phi_dt2 = np.zeros(frames.shape)
avg_dlogM_dw = np.zeros(frames.shape)
avg_dlogM_dt = np.zeros(frames.shape)
avg_d2logM_dw2 = np.zeros(frames.shape)
avg_d2logM_dt2 = np.zeros(frames.shape)
avg_d2logM_dwdt = np.zeros(frames.shape)

gfilter = np.exp(-((np.mgrid[:8,:8]-3.5)**2).sum(0)/12)
gdwfilter = -(np.mgrid[:8,:8]-3.5)[1]/6 * gfilter
gdtfilter = -(np.mgrid[:8,:8]-3.5)[0]/6 * gfilter


for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    avg += f
    abs_avg += np.abs(f_mv)
    df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
    df_mv = df * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*h[i]))
    tf_mv = tf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tdf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*dh[i]))
    tdf_mv = tdf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    ddf = np.fft.fft((x[indices]*greater_than_winlength) * ddh[i])
    ddf_mv = ddf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    ttf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*tt*h[i]))
    ttf_mv = ttf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    abs_f_mv = np.abs(f_mv)**2
    dphi_dt = np.imag((df_mv * f_mv.conj())/abs_f_mv)
    dphi_dw = - np.real((tf_mv * f_mv.conj())/abs_f_mv)
    dlogM_dt = convolve(np.real(df_mv * f_mv.conj()/abs_f_mv),gfilter)
    dlogM_dw = convolve(np.imag(tf_mv * f_mv.conj()/abs_f_mv),gfilter)
    d2logM_dt2 = convolve(np.real(df_mv * f_mv.conj()/abs_f_mv),gdtfilter)
    d2logM_dw2 = convolve(np.imag(tf_mv * f_mv.conj()/abs_f_mv),gdwfilter)
    d2logM_dwdt = convolve(np.imag(tf_mv * f_mv.conj()/abs_f_mv),gdtfilter)
    d2phi_dtdw = np.real(tdf_mv * f_mv.conj()/ abs_f_mv) - np.real(tf_mv * df_mv/ f_mv**2)
    d2phi_dw2 = np.imag((tf_mv * f_mv.conj()/abs_f_mv)**2) - np.imag(ttf_mv * f_mv.conj()/ abs_f_mv)
    d2phi_dt2 = np.imag(ddf_mv * f_mv.conj()/ abs_f_mv) - np.imag((df_mv * f_mv.conj()/abs_f_mv)**2) 
    # sm_d2phi_dtdw = convolve(np.real(tdf_mv * f_mv.conj()/ abs_f_mv) - np.real(tf_mv * df_mv/ f_mv**2),gfilter)
    # sm_d2phi_dw2 = convolve(np.imag((tf_mv * f_mv.conj()/abs_f_mv)**2) - np.imag(ttf_mv * f_mv.conj()/ abs_f_mv),gfilter)
    # sm_d2phi_dt2 = convolve(np.imag(ddf_mv * f_mv.conj()/ abs_f_mv) - np.imag((df_mv * f_mv.conj()/abs_f_mv)**2),gfilter)
    sm_d2phi_dtdw = median_filter(np.real(tdf_mv * f_mv.conj()/ abs_f_mv) - np.real(tf_mv * df_mv/ f_mv**2),size=(5,5))
    sm_d2phi_dw2 =median_filter(np.imag((tf_mv * f_mv.conj()/abs_f_mv)**2) - np.imag(ttf_mv * f_mv.conj()/ abs_f_mv),size=(5,5))
    sm_d2phi_dt2 = median_filter(np.imag(ddf_mv * f_mv.conj()/ abs_f_mv) - np.imag((df_mv * f_mv.conj()/abs_f_mv)**2),size=(5,5))
    avg_dphi_dt += dphi_dt
    avg_dphi_dw += dphi_dw
    avg_d2phi_dtdw += d2phi_dtdw
    avg_d2phi_dw2 += d2phi_dw2
    avg_d2phi_dt2 += d2phi_dt2
    avg_sm_d2phi_dtdw += sm_d2phi_dtdw
    avg_sm_d2phi_dw2 += sm_d2phi_dw2
    avg_sm_d2phi_dt2 += sm_d2phi_dt2
    avg_dlogM_dw += dlogM_dw
    avg_dlogM_dt = dlogM_dt
    avg_d2logM_dw2 += d2logM_dw2
    avg_d2logM_dt2 = d2logM_dt2
    avg_d2logM_dwdt += d2logM_dwdt


avg/=5
abs_avg/=5
avg_dphi_dt /= 5
avg_dphi_dw /= 5
avg_dlogM_dt /= 5
avg_dlogM_dw /= 5
avg_d2logM_dt2 /= 5
avg_d2logM_dw2 /= 5
avg_d2logM_dwdt /= 5
avg_d2phi_dtdw /=5 
avg_d2phi_dw2 /= 5
avg_d2phi_dt2 /= 5
avg_sm_d2phi_dtdw /=5 
avg_sm_d2phi_dw2 /= 5
avg_sm_d2phi_dt2 /= 5

    
    

# get the eigenvectors
tau = (avg_d2phi_dw2 - avg_d2phi_dt2)/(2*avg_d2phi_dtdw)
t = np.sign(tau)/(np.abs(tau) + np.sqrt(1+tau**2))
c = 1/np.sqrt(1+t**2)
s = c*t

# compute the hessian eigenvectors
l1 = avg_d2phi_dt2 - t* avg_d2phi_dtdw
l2 = avg_d2phi_dw2 + t* avg_d2phi_dtdw


order = (l2 > l1).astype(int)

ltrue1 = l2*(1-order) + l1*order
etrue1_t = s*(1-order) + c*order
etrue1_w = c*(1-order) - s*order
inner_prod = etrue1_t * dphi_dt + etrue1_w * dphi_dw

#across frequency
lfreq = ltrue1 * (ltrue1 < 0) * order

E = np.zeros(avg_dphi_dt.T.shape)
E[:-1] = avg_dphi_dt.T[1:] - avg_dphi_dt.T[:-1]

plt.subplot(2,1,1); plt.imshow(avg_dphi_dt.T[:100],cmap='binary',origin='lower',vmin=-.05,vmax=.05); plt.subplot(2,1,2); plt.imshow(np.log(abs_avg).T[:100],origin='lower'); plt.show()

trinary_dlogM_dw = lambda t: np.sign(avg_dlogM_dw)*( np.abs(avg_dlogM_dw) > t)
plt.imshow(trinary_dlogM_dw(.5).T[:100],origin='lower',cmap='bone',vmin=-1,vmax=1); plt.show()
plt.imshow(avg_dlogM_dw.T[:100],origin='lower',cmap='bone',vmin=-1,vmax=1); plt.show()


from scipy.ndimage.filters import median_filter, convolve
E = median_filter(avg_dphi_dt,size=(3,3))
E_filtered = np.sign(E)*(np.abs(E) > .02)

filter_d2logM_dw2 = convolve(avg_dlogM_dw,gdfilter)

filter_d2phi_dwdt = convolve(avg_dphi_dt,gdwfilter)
filter_d2phi_dt2 = convolve(avg_dphi_dt,gdtfilter)
filter_d2phi_dw2 = convolve(avg_dphi_dw,gdwfilter)

# get the eigenvectors
tau = (filter_d2phi_dw2 - filter_d2phi_dt2)/(2*filter_d2phi_dwdt)
t = np.sign(tau)/(np.abs(tau) + np.sqrt(1+tau**2))
c = np.nan_to_num(1/np.sqrt(1+t**2))
s = c*t

# eigenvalues
l1 = filter_d2phi_dt2 - t* filter_d2phi_dwdt
l2 = filter_d2phi_dw2 + t* filter_d2phi_dwdt


np.abs(s) > np.abs(c)


plt.subplot(2,1,1); plt.imshow(filter_d2phi_dwdt.T[:100]>.1,origin='lower',cmap='bone'); plt.subplot(2,1,2); plt.imshow(np.log(abs_avg).T[:100],origin='lower'); plt.show() 

trinary_d2phi_dt2 = lambda t: np.sign(filter_d2phi_dt2)*( np.abs(filter_d2phi_dt2) > t)

plt.subplot(2,1,1); plt.imshow(((filter_d2phi_dwdt>.12)*trinary_d2phi_dt2(.01)).T[:100],origin='lower',cmap='bone'); plt.subplot(2,1,2); plt.imshow(np.log(abs_avg).T[:100],origin='lower'); plt.show() 


hs = []
for i in xrange(4,10):
    taper_length=255
    order=6
    half_time_support=i
    h_temp1,dh_temp1,ddh_temp1, tt_temp1 = hermite_window(taper_length,
                                                          order,
                                                          half_time_support)
    hs.append(h_temp1[0])



# now getting the hessian also get a wider filter
# this doesn
taper_length=255
order=6
half_time_support=12
h_temp,dh_temp,ddh_temp, tt_temp = hermite_window(taper_length,
                   order,
                   half_time_support)
h = np.zeros((h_temp.shape[0],
              h_temp.shape[1]+1))
h[:,:-1] = h_temp

dh = np.zeros((dh_temp.shape[0],
              dh_temp.shape[1]+1))
dh[:,:-1] = dh_temp

ddh = np.zeros((ddh_temp.shape[0],
              ddh_temp.shape[1]+1))
ddh[:,:-1] = ddh_temp

tt = (2*tt_temp[-1] -tt_temp[-2])*np.ones(tt_temp.shape[0]+1)
tt[:-1] = tt_temp

avg = np.zeros(frames.shape,dtype=np.complex128)
avg_d = np.zeros(dframes.shape,dtype=np.complex128)
abs_avg = np.zeros(frames.shape)
avg_dphi_dt = np.zeros(frames.shape)
avg_dphi_dw = np.zeros(frames.shape)
avg_d2phi_dtdw = np.zeros(frames.shape)
avg_d2phi_dw2 = np.zeros(frames.shape)
avg_d2phi_dt2 = np.zeros(frames.shape)

for i in xrange(5):
    f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
    f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    avg += f
    abs_avg += np.abs(f_mv)
    df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
    df_mv = df * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*h[i]))
    tf_mv = tf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    tdf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*dh[i]))
    tdf_mv = tdf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    ddf = np.fft.fft((x[indices]*greater_than_winlength) * ddh[i])
    ddf_mv = ddf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    ttf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*tt*h[i]))
    ttf_mv = ttf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/N1)
    abs_f_mv = np.abs(f_mv)**2
    dphi_dt = np.imag((df_mv * f_mv.conj())/abs_f_mv)
    dphi_dw = - np.real((tf_mv * f_mv.conj())/abs_f_mv)
    d2phi_dtdw = np.real(tdf_mv * f_mv.conj()/ abs_f_mv) - np.real(tf_mv * df_mv/ f_mv**2)
    d2phi_dw2 = np.imag((tf_mv * f_mv.conj()/abs_f_mv)**2) - np.imag(ttf_mv * f_mv.conj()/ abs_f_mv)
    d2phi_dt2 = np.imag(ddf_mv * f_mv.conj()/ abs_f_mv) - np.imag((df_mv * f_mv.conj()/abs_f_mv)**2) 
    avg_dphi_dt += dphi_dt
    avg_dphi_dw += dphi_dw
    avg_d2phi_dtdw += d2phi_dtdw
    avg_d2phi_dw2 += d2phi_dw2
    avg_d2phi_dt2 += d2phi_dt2

avg/=5
abs_avg/=5
avg_dphi_dt /= 5
avg_dphi_dw /= 5
avg_d2phi_dtdw /=5 
avg_d2phi_dw2 /= 5
avg_d2phi_dt2 /= 5

    
    

# get the eigenvectors
tau = (avg_d2phi_dw2 - avg_d2phi_dt2)/(2*avg_d2phi_dtdw)
t = np.sign(tau)/(np.abs(tau) + np.sqrt(1+tau**2))
c = 1/np.sqrt(1+t**2)
s = c*t

# compute the hessian eigenvectors
l1 = avg_d2phi_dt2 - t* avg_d2phi_dtdw
l2 = avg_d2phi_dw2 + t* avg_d2phi_dtdw


order = (l2 > l1).astype(int)

ltrue1 = l2*(1-order) + l1*order
etrue1_t = s*(1-order) + c*order
etrue1_w = c*(1-order) - s*order
inner_prod = etrue1_t * dphi_dt + etrue1_w * dphi_dw

#across frequency
lfreq = ltrue1 * (ltrue1 < 0) * order

E = np.zeros(avg_dphi_dt.T.shape)
E[:-1] = avg_dphi_dt.T[1:] - avg_dphi_dt.T[:-1]

plt.subplot(2,1,1); plt.imshow(avg_dphi_dt.T[:100],cmap='binary',origin='lower',vmin=-.05,vmax=.05); plt.subplot(2,1,2); plt.imshow(np.log(abs_avg).T[:100],origin='lower'); plt.show()
