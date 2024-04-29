import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from sklearn.metrics.pairwise import cosine_similarity
########## R-squared (R2) ##########

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """
    y_test = np.nan_to_num(y_test)
    y_test_pred = np.nan_to_num(y_test_pred)
    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s
def get_MAE(y_test,y_test_pred):
    y_test = np.nan_to_num(y_test)
    y_test_pred = np.nan_to_num(y_test_pred)
    MAE_list=[] #Initialize a list that
    for i in range(y_test.shape[1]): #Loop through outputs

        MAE=np.mean((y_test_pred[:,i]-y_test[:,i]))
        MAE_list.append(MAE) #
    MAE_array=np.array(MAE_list)
    return MAE_array #

def get_RMSE(y_test,y_test_pred):
    y_test = np.nan_to_num(y_test)
    y_test_pred = np.nan_to_num(y_test_pred)
    RMSE_list=[] #Initialize a list that
    for i in range(y_test.shape[1]): #Loop through outputs

        RMSE=np.sqrt(np.mean((y_test_pred[:,i]-y_test[:,i])**2))
        RMSE_list.append(RMSE) #
    RMSE_array=np.array(RMSE_list)
    return RMSE_array #

def get_pearsonr(y_test,y_test_pred):
    y_test=np.nan_to_num(y_test)
    y_test_pred=np.nan_to_num(y_test_pred)
    CC_list=[] #Initialize a list that
    for i in range(y_test.shape[1]): #Loop through outputs

        CC=pearsonr(y_test_pred[:,i],y_test[:,i])[0]
        if np.isnan(CC):
            CC_list.append(0)
        else:
            CC_list.append(CC) #
    CC_array=np.array(CC_list)
    return CC_array #


def get_vendi(y_tru,y_test_pred):
    # V_list=[] #Initialize a list that
    # for i in range(y_test_pred.shape[0]): #Loop through outputs
    #     k = lambda a, b: np.exp(-np.abs(a - b))
    #     VV=vendi.score(y_test_pred[i,:].reshape([-1,]),k)
    #     V_list.append(VV) #
    # V_array=np.array(V_list)
    return score_K((cosine_similarity(y_test_pred, y_tru))) #

########## Pearson's correlation (rho) ##########

def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """

    rho_list=[] #Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho) #Append rho of this output to the list
    rho_array=np.array(rho_list)
    return rho_array #Return the array of rhos
