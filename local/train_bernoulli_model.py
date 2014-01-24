from __future__ import division
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
from amitgroup.stats import bernoullimm


def main(args):
    X = np.load(args.patches)
    S = np.load(args.spec_patches)
    bmm = bernoullimm.BernoulliMM(n_components=args.n_components,
                                  n_init= 10,
                                  n_iter= 500,
                                  random_state=0,
                                  verbose=args.v,
                                  tol=1e-6)
    bmm.fit(X)
    use_means = bmm.predict_proba(X).sum(0) > 30
    spec_shape = S.shape[1:]
    S_clusters = bmm.cluster_underlying_data(S.reshape(len(S),np.prod(spec_shape)),X).reshape(
            *( (bmm.n_components,) + spec_shape))[use_means]
    ncols = int(np.sqrt(args.n_components))
    nrows = int(np.ceil(args.n_components/ncols))

    np.save('%s_means.npy' % args.save_path, bmm.means_)
    np.save('%s_S_clusters.npy' % args.save_path, S_clusters)
    np.save('%s_weights.npy' % args.save_path, bmm.weights_)

    if args.viz_spec_parts is not None:
        plt.close('all')
        fig = plt.figure(1, (6, 6))
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                             nrows_ncols = (nrows,ncols ), # creates 2x2 grid of axes
                             axes_pad=0.001, # pad between axes in inch.
                     )

        for i in xrange(S_clusters.shape[0]):

            try:
                grid[i].imshow(S_clusters[i],cmap=cm.binary,interpolation='nearest')
                grid[i].spines['bottom'].set_color('red')
                grid[i].spines['top'].set_color('red')
                grid[i].spines['left'].set_color('red')
                grid[i].spines['right'].set_color('red')
                for a in grid[i].axis.values():
                    a.toggle(all=False)
            except:
                import pdb; pdb.set_trace()

        for i in xrange(S_clusters.shape[0],nrows*ncols):
            try:
                grid[i].spines['bottom'].set_color('red')
            except: import pdb; pdb.set_trace()
            grid[i].spines['top'].set_color('red')
            grid[i].spines['left'].set_color('red')
            grid[i].spines['right'].set_color('red')
        
            for a in grid[i].axis.values():
                a.toggle(all=False)
                
        plt.savefig('%s' % args.viz_spec_parts
                                           ,bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Clustering and output for bernoulli data with paired spectrum data""")
    parser.add_argument('--save_path',type=str,default='',help='path to save the trained models to')
    parser.add_argument('--patches',type=str,default='',help='path to the patches')
    parser.add_argument('--spec_patches',type=str,default='',help='path to the spectrogram patches')
    parser.add_argument('--n_components',type=int,default=20,help='number of parts')
    parser.add_argument('-v',action='store_true',help='whether it is verbose')
    parser.add_argument('--viz_spec_parts',type=str,default=None,help='path to save the learned parts to')
    main(parser.parse_args())

    
