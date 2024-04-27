import numpy as np
import pandas as pd
import torch
# from torch_geometric.data import Data
# import  torch_geometric
# import simtk
# import networkx as nx
# import mdtraj as md
# from openmm.app.pdbfile import PDBFile
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
def get_mol_graph_per_traj(save_forlder,
                           index_trj,
                           visulaize=False):

    current_pdb_obj=PDBFile(save_forlder+'output-'+str(index_trj)+'.pdb')
    current_state_df=pd.read_csv(save_forlder+'scalars-'+str(index_trj)+'.csv')
    mol_topology=current_pdb_obj.getTopology()
    ''' get bonds in the chane'''
    bonds_index=[]
    for bond_inf in mol_topology.bonds():
        bonds_index.append([bond_inf[0].index, bond_inf[1].index])

    ''' get the atoms names'''
    atom_names=[]
    for atim_indx in mol_topology.atoms():
        atom_names.append(atim_indx.name)
    ''' crete the edges information for the graph'''
    bonds_index=np.array(bonds_index)
    bonds_index=np.concatenate([bonds_index, bonds_index[:, [1, 0]]], axis=0)
    edge_index = torch.tensor(bonds_index, dtype=torch.long)
    ''' generate the graph for each frame (time-index) of the data'''
    total_single_traj_data=[]
    total_positions=[]
    total_phi=[]
    total_psi=[]
    for frm_indx in range(current_pdb_obj.getNumFrames()):
        positions=current_pdb_obj.getPositions(asNumpy=True, frame=frm_indx)
        total_positions.append(np.array(positions))

        phi, psi = compute_ala2_phipsi(positions, mol_topology)
        total_psi.append(psi.squeeze())
        total_phi.append(phi.squeeze())
        compute_ala2_phipsi(positions,mol_topology)
        x = torch.tensor(np.array(positions), dtype=torch.float)
        total_single_traj_data.append(Data(x=x, edge_index=edge_index.t().contiguous()))
    ''' visualize the graph'''
    if visulaize:
        g = torch_geometric.utils.to_networkx(total_single_traj_data[0], to_undirected=True)
        nx.draw(g,labels=dict(zip(np.arange(len(atom_names)), atom_names)))
    else:
        pass
    ''' generate the outputs'''
    # psi = np.array(total_psi)
    psi=np.array(total_psi) - np.array(total_psi).mean()
    psi[np.where(psi > 1. * np.pi)[0]] = psi[np.where(psi > 1. * np.pi)[0]] - 2 * np.pi
    psi[np.where(psi < -1. * np.pi)[0]] = psi[np.where(psi < -1. * np.pi)[0]] + 2 * np.pi
    # phi = np.array(total_phi)
    phi = np.array(total_phi) - np.array(total_phi).mean()
    phi[np.where(phi > 1. * np.pi)[0]] = phi[np.where(phi > 1. * np.pi)[0]] - 2 * np.pi
    phi[np.where(phi < -1. * np.pi)[0]] = phi[np.where(phi < -1. * np.pi)[0]] + 2 * np.pi
    traj_dict = {
        'positions':np.array(total_positions),
        'graph_data':total_single_traj_data,
        'atom_names': dict(zip(np.arange(len(atom_names)), atom_names)),
        'Temperature (K)': current_state_df['Temperature (K)'].to_numpy(),
        'Potential Energy (kJ/mole)': current_state_df['Potential Energy (kJ/mole)'].to_numpy(),
        'Total Energy (kJ/mole)': current_state_df['Total Energy (kJ/mole)'].to_numpy(),
        'Time (ps)': current_state_df['#"Time (ps)"'].to_numpy(),
        'phi':phi,
        'phi_mean': np.array(total_phi).mean(),
        'psi':psi ,
        'psi_mean': np.array(total_psi).mean(),
    }
    return traj_dict


def compute_ala2_phipsi(samples, topology):
    """
    Compute Ala2 Ramachandran angles
    Parameters
    ----------
    traj : mdtraj.Trajectory
    """

    traj = md.Trajectory(xyz=samples, topology=topology)

    phi_atoms_idx = [4, 6, 8, 14]
    phi = md.compute_dihedrals(traj, indices=[phi_atoms_idx], periodic=True)[:, 0]
    # phi[np.where(phi >=1.5*np.pi)[0]]=phi[np.where(phi >=1.5*np.pi)[0]]-2*np.pi
    # phi[np.where(phi <= -2 * np.pi)[0]] = phi[np.where(phi <= -2 * np.pi)[0]] +2 * np.pi
    psi_atoms_idx = [6, 8, 14, 16]
    psi = md.compute_dihedrals(traj, indices=[psi_atoms_idx], periodic=True)[:, 0]
    # psi[np.where(psi >= 1.5 * np.pi)[0]] = psi[np.where(psi >= 1.5 * np.pi)[0]] - 2 * np.pi
    # psi[np.where(psi <= -1.5 * np.pi)[0]] = psi[np.where(psi <= -1.5 * np.pi)[0]] + 2 * np.pi

    return phi, psi



import numpy as np
from scipy.ndimage import filters
from scipy.signal.windows import gaussian
import torch
from scipy import stats
from sklearn import metrics
def gaussian_kernel_smoother(y, sigma, window):
    b = gaussian(window, sigma)
    y_smooth = np.zeros(y.shape)
    neurons = y.shape[1]
    for neuron in range(neurons):
        y_smooth[:, neuron] = np.convolve(y[:, neuron], b/b.sum(),'same')
    return y_smooth



def get_diagonal(matrix):
    output=torch.zeros((matrix.shape[0],matrix.shape[1]))
    for ii in range( matrix.shape[0]):
        output[ii,:] = torch.diagonal(torch.squeeze(matrix[ii]))
    return output

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth
def calInformetiveChan(data,minNumSpiks):
    return np.where(np.sum(data,axis=0)>minNumSpiks)

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h* X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i,  :]= (PadX[i:h+i,:]).reshape([-1,])
    return XDsgn

def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth
def calInformetiveChan(data,minNumSpiks):
    return np.where(np.sum(data,axis=0)>minNumSpiks)

def get_normalized(x, config):
    if config['supervised']:
        return (x-x.mean())/x.std()
    else:
        return  (x-x.mean())/x.std()

def get_cdf(data):
    count, bins_count = np.histogram(data, bins=10)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return cdf

def get_metrics(z, z_hat):
    cc=[]
    mse=[]
    mae=[]
    for ii in range(z.shape[1]):
        cc.append(stats.pearsonr(z[:,ii],z_hat[:,ii])[0])
        mse.append(metrics.mean_squared_error(z[:, ii], z_hat[:, ii]))
        mae.append(metrics.mean_absolute_error(z[:, ii], z_hat[:, ii]))

    return np.mean(cc), np.mean(mse),np.mean(mae)

def get_mask_imputation(length,missing_rate):
    mask=torch.zeros(length, dtype=torch.bool)

    # np.random.choice(range(length), 10, replace=np.floor((missing_rate/100)*length).astype('int'))
    true_indx=np.random.choice(range(length), np.floor((missing_rate/100)*length).astype('int'), replace=False)
    mask[true_indx]=1
    return mask

def get_mask_forcasting(length,forcasting_rate):
    mask=torch.zeros(length, dtype=torch.bool)
    true_indx=torch.arange(np.floor((100-forcasting_rate)/100*length).astype('int'), length)
    mask[true_indx]=1
    return mask
def get_1d_pdf(data, n_bins):
    # Create a kernel density estimator
    kde = gaussian_kde(data)
    # Evaluate the KDE on a grid of points to get the PDF values
    x_grid = np.linspace(min(data), max(data), n_bins)
    pdf_values = kde(x_grid)
    return x_grid, pdf_values

def get_eigens(X,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    # Get the eigenvalues from the explained_variance_ attribute
    eigenvalues = pca.explained_variance_
    return eigenvalues, pca.components_

def moving_average(data, window_size, axis=0):
    """Apply a moving average filter along a specified axis."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_data(x_in,z_in,downsample_rate, smooth_window_size):
    z = z_in
    for ii in range(z.shape[-1]):
        for jj in range(z.shape[-2]):
            z[:,jj,ii]=moving_average(z_in[:,jj,ii], smooth_window_size, axis=0)
    x=x_in
    for ii in range(x.shape[-1]):
        for jj in range(x.shape[-2]):
            x[:,jj,ii]=moving_average(x_in[:,jj,ii], smooth_window_size, axis=0)
    data_len = x_in.shape[0]

    x = 2 * (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) - 1
    z = 2 * (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0)) - 1

    ''' normalization'''


    ''' down sampling rate'''

    x = x[np.arange(0, data_len, downsample_rate), :,:]
    # x = calDesignMatrix_V2(x, 3).squeeze()
    z = z[np.arange(0, data_len, downsample_rate), :,:]
    return x,z


def plot_hsit(axs, data,bin_edges, label,prediction=False ):
    hist_,_=np.histogram(data.reshape([-1,]), density=True, bins=bin_edges)
    hist_/=hist_.sum()
    kde = stats.gaussian_kde(data.reshape([-1, ]))
    kde_vals = kde.evaluate(bin_edges)
    kde_vals/=kde_vals.sum()
    if prediction:
        # axs.hist(data.reshape([-1, ]), density=True,stacked=True, bins=bin_edges, color='goldenrod', label='Histogram_pred_', alpha=.6)
        axs.plot(bin_edges, kde_vals, 'goldenrod', label='KDE Interpolation')
    else:
        # axs.hist(data.reshape([-1,]),density=True,stacked=True, bins=bin_edges,color='k', label='Histogram_tru_', alpha=.6)
        axs.plot(bin_edges, kde_vals, 'k', label='KDE Interpolation')
    axs.set_ylabel(label)
    return hist_

import numpy as np

def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)
# Generate sample data (replace with your own data)
# Calculate MMD using the RBF kernel

