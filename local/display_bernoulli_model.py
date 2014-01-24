from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
from amitgroup.stats import bernoullimm


def main(args):
    means = np.load('%s_means.npy' % args.save_path)
    S_clusters = np.load('%s_S_clusters.npy' % args.save_path)
    
    means = means.reshape( *( S_clusters.shape + ( int(means.size/S_clusters.size),)))


    if True:
        plt.close('all')
        fig = plt.figure(1, (6, 6))
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                             nrows_ncols = (means.shape[0],means.shape[-1]+1 ), # creates 2x2 grid of axes
                             axes_pad=0.001, # pad between axes in inch.
                     )

        ncols = means.shape[-1] + 1
        for i in xrange(S_clusters.shape[0]):
            try:
                grid[i*ncols].imshow(S_clusters[i],cmap=cm.binary,interpolation='nearest')
                grid[i*ncols].spines['bottom'].set_color('red')
                grid[i*ncols].spines['top'].set_color('red')
                grid[i*ncols].spines['left'].set_color('red')
                grid[i*ncols].spines['right'].set_color('red')
                for a in grid[i*ncols].axis.values():
                    a.toggle(all=False)
            except:
                import pdb; pdb.set_trace()
            for j in xrange(ncols-1):
                try:
                    grid[i*ncols+j+1].imshow(means[i,:,:,j],cmap=cm.binary,interpolation='nearest')
                    grid[i*ncols+j+1].spines['bottom'].set_color('red')
                    grid[i*ncols+j+1].spines['top'].set_color('red')
                    grid[i*ncols+j+1].spines['left'].set_color('red')
                    grid[i*ncols+j+1].spines['right'].set_color('red')
                    for a in grid[i*ncols+j+1].axis.values():
                        a.toggle(all=False)
                except:
                    import pdb; pdb.set_trace()

            

                
        plt.savefig('%s' % args.display_bernoulli_parts
                                           ,bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Clustering and output for bernoulli data with paired spectrum data""")
    parser.add_argument('--save_path',type=str,default='',help='path to save the trained models to')
    parser.add_argument('--patches',type=str,default='',help='path to the patches')
    parser.add_argument('--spec_patches',type=str,default='',help='path to the spectrogram patches')
    parser.add_argument('--n_components',type=int,default=20,help='number of parts')
    parser.add_argument('-v',action='store_true',help='whether it is verbose')
    parser.add_argument('--display_bernoulli_parts',type=str,default=None,help='path to save the learned parts to')
    main(parser.parse_args())

    
