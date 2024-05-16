import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage
from torch.nn import Parameter
import scipy.sparse.linalg
from kernels import kernel_factory
from scipy.stats import mode
import sklearn.preprocessing


### ------------ training of the multi-view clustering ------------ ###
def mvclustering(args, feats, dims, scales, init_image_size, kernel, kernel_param, k_clusters, assign_func, eta, rho, resize_ratio, im_name='', feats_test=None):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      kernel: which kernel to construct the multi-view clustering
      k_clusters: how many clusters to conduct
      assign_func: use uncoupled-view ensemble or mean-view ensemble of the final clusters prediction
      eta: regularization coefficient on the weights of the multi-view clustering method
      rho: to balance the RKM loss (rho) and Tensor-based multi-view loss (1-rho)
      feats_test: whether to extend results to out-of-sample cases
    """
    # we consider single image here
    # nuber of views is 1,3,6 or 18
    num_views = feats.size(0)

    # normalize each view
    if args.normalize == "l2":
        feats = F.normalize(feats, p=2, dim=-1)
    elif args.normalize == "l1":
        feats = F.normalize(feats, p=1, dim=-1)

    # prepare kernel function for each view
    kernels = [kernel_factory(kernel, kernel_param) for view in range(num_views)]
    
    # Build Omega for each view
    Omegas = omega_mv(feats, feats, kernels)

    # apply tau
    if args.tau_kernel is not None:
        for view in range(num_views):
            #print(f"Omegas[view].max(): {Omegas[view].max()}")
            #print(f"Omegas[view].min(): {Omegas[view].min()}")
            if not args.binary_graph:
                # Apply a lower-bound threshold
                Omegas[view][Omegas[view] < (Omegas[view].max() * args.tau_kernel[view])] = 1e-5
            else:
                # Create a binary graph
                Omegas[view] = (Omegas[view] > (Omegas[view].max() * args.tau_kernel[view]))
                Omegas[view] = np.where(Omegas[view].astype(float) == 0, 1e-5, Omegas[view])


    # Compute Dinv, dinv for centering
    Dadd, Dinv, dinv = degree_matrix(Omegas)

    # Compute the centered kernel matrices
    OmegasCentered = centered_omegas(Omegas, Dinv, dinv)

    # Compute the degree matrix D ON THE CENTERED KERNEL MATRIX
    #Dadd, _, _ = degree_matrix(OmegasCentered)

    OmegasCentered_weighted = [args.kappa[view] * OmegasCentered[view] for view in range(num_views)]
    OmegasCentered_add = rho * np.add.reduce(OmegasCentered_weighted) + (1 - rho) * np.multiply.reduce(OmegasCentered)
    

    #print(f"add:{np.add.reduce(OmegasCentered_weighted)}")
    #print(f"multiply:{np.multiply.reduce(OmegasCentered)}")


    # Build matrices
    R = Dadd
    L = 1 / eta * OmegasCentered_add
    eigenValues, H = eigh(L, R, subset_by_index=[len(L)-2,len(L)-1])
    H = np.expand_dims(H[:,-1], axis=-1)
    eigenvectors = H

    # Compute score variables on the training data
    etrain = []
    for v in range(num_views):
        etrain.append(np.matmul(OmegasCentered[v], H))
    
    # if feats_test is None, then we do not consider the out-of-sample extension
    if feats_test == None:
        feats_test = feats

    # return the clusters prediction on the feats_test
    results = multiview_pred(args, feats_test, etrain, H, Dinv, dinv, feats, num_views, kernels, assign_func)

    # mode takes the most frequent cluster assignment of each embedding among the views
    # partition is a (#tokens -class token by 1 binary array)
    partition = mode(results['y_pred'], axis=1)[0]


    #This is a place holder to avoid an error in further processing
    if all(i[0] == 0 for i in partition):
        partition[0,0] = 1
        print(f"All tokens were assigned to the same class; no bipartition found.")
    if all(i[0] == 1 for i in partition):
        partition[0,0] = 0
        print(f"All tokens were assigned to the same class; no bipartition found.")
    #assert not (all(i[0] == 0 for i in partition) or all(i[0] == 1 for i in partition)), "All tokens were assigned to the same class; no bipartition found."

    # use the cluster similarities to determine which one is the background (is more connected to the entire graph according to Tokencut)
    group_A_idx = partition == 1
    group_B_idx = partition != 1
    feats_A = feats[0,group_A_idx[:,0]]
    feats_B = feats[0,group_B_idx[:,0]]
    feats_A = F.normalize(feats_A, p=2)
    feats_B = F.normalize(feats_B, p=2)
    similarity_A = (torch.mm(feats_A, feats_A.t())).sum()/feats_A.size(0)
    similarity_B = (torch.mm(feats_B, feats_B.t())).sum()/feats_B.size(0)
    if similarity_A > similarity_B:
        partition = np.logical_not(partition)
        eigenvectors = eigenvectors * -1


    # I DONT THINK WE NEED ALL THIS:
    # etest = results['etest']
    # if etest[0][partition==1][0] <= 0:
    #     etest[0] = etest[0] * -1
    
    if eigenvectors[partition==1][0] <= 0:
        # object should have the eigenvector larger than 0
        eigenvectors = eigenvectors * -1

    # if args.tau_box is not None:
    #     shrink_idx = np.logical_and(etest[0] > 0, etest[0] < (args.tau_box * max(etest[0])).item())
    #     partition[shrink_idx] = 0


    # flipping the test score and the eigenvector if necessary
    # etest = results['etest']
    # if etest[0][partition==1][0] <= 0:
        # object should have the score larger than 0
    #    etest[0] = etest[0] * -1
    #   eigenvectors = eigenvectors * -1
   # if eigenvectors[partition==1][0] <= 0:
        # object should have the eigenvector larger than 0
    #    eigenvectors = eigenvectors * -1
    # for tighter bounding box
   # if args.tau_box is not None:
    #    shrink_idx = np.logical_and(etest[0] > 0, etest[0] < (args.tau_box * max(etest[0])).item())
     #   partition[shrink_idx] = 0

    # reshape the array to [w_featmap, h_featmap]
    partition = partition.reshape(dims).astype(float)

    # find the largest connected foreground component in the bipartition of the graph
    pred, _, objects,cc = detect_box(partition, dims, resize_ratio, initial_im_size=init_image_size[1:], scales=scales) 
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    # remove paddings
    eigenvectors = eigenvectors.reshape(dims)

    return np.asarray(pred), objects, mask, eigenvectors



def detect_box(partition, dims, resize_ratio, initial_im_size=None, scales=None, principle_object=True):
    """
    only when we conduct bipartition, a.k.a k=2, should we use bounding box.
    returns the bounding box coordinates in image space and feature space, 
    as well as the mask in feature space, for the largest connected component in the graph.
    also returns the objects mask
    """
    objects, num_objects = ndimage.label(partition) 

    num_components = []
    for i in range(1, num_objects+1):
        num_components.append(np.sum(objects == i))
    largest_idx = np.argmax(num_components)
    cc = largest_idx + 1

    mask = np.where(objects == cc)
    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    pred_feats = [ymin, xmin, ymax, xmax]

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
    # Check not out of image size (used when padding)
    if resize_ratio is not None:
        tmp= [int(round(item * resize_ratio)) for item in pred]
        pred = tmp
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    return pred, pred_feats, objects, mask


### -------------------- codebook related functions --------------- ###
def codebook(e):
    """
    Finds the codebook for encoding matrix e
    :param e: N x (k-1) matrix of -1, 1 entries
    :return: list of the k most frequent encodings
    """
    k = e.shape[1] + 1
    # k = e.shape[1]
    c, counts = np.unique(e, axis=0, return_counts=True)
    return [t[0] for t in sorted(zip(c, counts), key=lambda x: -x[1])[:k]]

def closest_code(e, codebook, alphat=None, alphaCenters=None):
    """
    Finds closest encoding vector in codebook
    :param e: N x (k-1) matrix of -1, 1 entries
    :param codebook: list of the k codes of length (k-1)
    :return: array of length N, closest element in codebook to e
    """
    from sklearn.metrics import DistanceMetric
    dist = DistanceMetric.get_metric('hamming')
    dist2 = DistanceMetric.get_metric('euclidean')
    d = dist.pairwise(e, np.array(codebook))
    qtrain = np.argmin(d, axis=1)
    if alphat is not None and alphaCenters is not None and d.shape[1] > 1 and qtrain.shape[0] <= alphat.shape[0]:
        #Break ties
        sorted_d = np.sort(d, axis=1)
        nidx = sorted_d[:,0] == sorted_d[:,1]
        if np.sum(nidx) > 0:
            nidx_test = nidx
            if nidx.shape[0] < alphat.shape[0]:
                nidx_test = np.concatenate([nidx, np.zeros((alphat.shape[0] - nidx.shape[0]), dtype=bool)])
            d2 = dist2.pairwise(alphat[nidx_test], alphaCenters)
            qtrain[nidx] = np.argmin(d2, axis=1)
    return qtrain

def assign_mean(etrain, beta=None):
    """
    Mean decision rule.
    :param etrain: list of V N x (k-1) matrices of score variables for view v
    :param beta: weights for each view in the decision rule. If None, beta[i] = 1/V
    :return: new score variable as a list of V N x (k-1) matrices
    """
    from math import isclose
    N, V = etrain[0].shape[0], len(etrain)
    if beta is None:
        beta = np.array([1. / V for v in range(V)])
    else:
        if type(beta) == omegaconf.listconfig.ListConfig or type(beta) == list:
            beta = np.array(list(beta))
        else:
            if type(beta["value"]) == str and (beta["value"] == "None" or beta["value"] == "null"):
                beta["value"] = None
            if beta["value"] is not None:
                beta = np.array(list(beta["value"]))
            elif beta["value"] is None:
                beta = np.array([beta[f"beta{v + 1}"] for v in range(V) if beta[f"beta{v + 1}"] is not None])
                if len(beta) == 0:
                    beta = np.array([1. / V for v in range(V)])
                else:
                    beta = beta / sum(beta)
    if not isclose(sum(beta), 1.0):
        beta = beta / sum(beta)
    assert len(beta) == V
    assert isclose(sum(beta), 1.0)

    encoding_total = np.array(etrain)
    dim_array = np.ones((1, encoding_total.ndim), int).ravel()
    dim_array[0] = -1
    beta_reshaped = beta.reshape(dim_array)
    encoding_total = encoding_total * beta_reshaped
    encoding_total = np.sum(encoding_total, axis=0)
    return [encoding_total for v in range(V)]

def assign_uncoupled(etrain, beta=None):
    """
    Uncoupled decision rule.
    :param etrain: list of V N x (k-1) matrices of score variables for view v
    :return: new score variable as a list of V N x (k-1) matrices
    """
    N, V = etrain[0].shape[0], len(etrain)
    return [etrain[v] for v in range(V)]

def compute_alphaCenters(alphat, etrain):
    assert alphat.shape[1] == etrain.shape[1]
    assert alphat.shape[0] == etrain.shape[0]

    N, d = alphat.shape
    k = d + 1
    # k = d
    alphaCenters = np.zeros((k, d))
    c, m, uniquecw = np.unique(my_sign(etrain), return_index=True, return_inverse=True, axis=0)
    cwsizes = np.zeros((len(m)))
    for i in range(len(m)):
        cwsizes[i] = np.sum(uniquecw == i)
    j= np.argsort(-cwsizes, kind='mergesort')
    if len(m) < k:
        k = len(m)
    qtrain = np.zeros((alphat.shape[0],))
    for i in range(k):
        qtrain[uniquecw == j[i]] = i + 1
    for i in range(k):
        alphaCenters[i] = np.median(alphat[qtrain == (i+1)], axis=0)
    return alphaCenters

def my_sign(x):
    return np.sign(x) + (x == 0)

### -------------------------------------------------------------------


### ------- Training: eigen decomposition related functions ------- ###
def omega_mv(X1, X2, kernels):
    V = X1.size(0)
    # Build Omega for each view
    Omegas = []
    for v in range(V):
        Omegas_tmp = kernels[v](X1[v].t(), X2[v].t()).numpy()
        Omegas.append((0.5 * (Omegas_tmp + Omegas_tmp.transpose())) if Omegas_tmp.shape[0] == Omegas_tmp.shape[1] else Omegas_tmp.T)
    return Omegas

def degree_matrix(Omegas):
    V = len(Omegas)
    Ntest, N = Omegas[0].shape
    # Compute the kernel matrix Omega, and the degree matrix D
    Dinv = []
    dinv = []
    Dadd = np.zeros((Ntest, Ntest))
    for v in range(V):
        d = np.sum(Omegas[v], axis=1)
        dinv.append(np.nan_to_num((1. / d)).reshape(Ntest, 1))
        Dinv.append(np.nan_to_num(np.diag(1. / d)))
        Dadd += np.diag(d)
    return Dadd, Dinv, dinv

def centered_omegas(Omegas, Dinv, dinv, Dinvtrain=None, dinvtrain=None):
    Dinvtrain = Dinvtrain if Dinvtrain is not None else Dinv
    dinvtrain = dinvtrain if dinvtrain is not None else dinv
    V = len(Omegas)
    Ntest, N = Omegas[0].shape
    # Compute the centered kernel matrices
    OmegasCentered = []
    for v in range(V):
        md = np.eye(Ntest) - np.matmul(np.ones((Ntest, 1)), dinv[v].transpose()) / np.sum(dinv[v])
        kd = np.eye(N) - np.matmul(np.matmul(Dinvtrain[v], np.ones((N, 1))), np.ones((1, N))) / np.sum(dinvtrain[v])
        OmegasCentered.append(np.matmul(np.matmul(md, Omegas[v]), kd))
    return OmegasCentered

### -------------------------------------------------------------------


### ----------- Testing: sample prediction related functions ------ ###
def compute_etest(args, X, x_test, H, Dinv, dinv, kernels):
    Omegas_test = omega_mv(X, x_test, kernels)
    num_views = x_test.size(0)

    # for sparse Kernel matrix:
    if args.tau_kernel is not None:
        if not args.binary_graph : 
            for view in range(num_views):
                Omegas_test[view][Omegas_test[view] < (Omegas_test[view].max()) * args.tau_kernel[view]] = 1e-5
        else: 
            for view in range(num_views):
                Omegas_test[view] = (Omegas_test[view] > (Omegas_test[view].max()) * args.tau_kernel[view])
                Omegas_test[view] = np.where(Omegas_test[view].astype(float) == 0, 1e-5, Omegas_test[view])

    Dadd_test, Dinv_test, dinv_test = degree_matrix(Omegas_test)
    OmegasCentered_test = centered_omegas(Omegas_test, Dinv_test, dinv_test, Dinv, dinv)
    etest = [np.matmul(OmegasCentered_test[v], H) for v in range(len(kernels))]
    return etest, OmegasCentered_test

def f_generic(args, x_test, Ktest_mem, etrain, H, Dinv, dinv, X, num_views, kernels, assign_f, beta=None):
    hash_X = sum([hash(str(x_test[v])) for v in range(num_views)])
    if hash_X not in Ktest_mem:
        Ktest_mem[hash_X] = omega_mv(x_test, x_test, kernels)
    Ktest = Ktest_mem[hash_X]

    etest, OmegasCentered_test = compute_etest(args, X, x_test, H, Dinv, dinv, kernels)
    etrainused = assign_f(etrain, beta)
    etestused = assign_f(etest, beta)
    codebooks = [np.array(codebook(my_sign(etrainused[v]))) for v in range(num_views)]
    alphaCenters = [compute_alphaCenters(H, etrainused[v]) for v in range(num_views)]
    q = np.array([closest_code(my_sign(etestused[v]), codebooks[v], H, alphaCenters[v]) for v in range(num_views)]).transpose()
    # q = np.array([closest_code(my_sign(etrainused[v]), codebooks[v], H, alphaCenters[v]) for v in range(num_views)]).transpose()
    return {"y_pred": q, "K": OmegasCentered_test, "etest": etestused, "etrain": etrainused, "Ktest": Ktest, "xtest": x_test, "kernels": kernels}

def multiview_pred(args, x_test, etrain, H, Dinv, dinv, X, num_views, kernels, assign_func, beta=None):
    Ktest_mem = {}
    if assign_func == "uncoupled":
        results = f_generic(args, x_test, Ktest_mem, etrain, H, Dinv, dinv, X, num_views, kernels, assign_uncoupled, beta=None)
    elif assign_func == "mean":
        results = f_generic(args, x_test, Ktest_mem, etrain, H, Dinv, dinv, X, num_views, kernels, assign_mean, beta=None)

    return results

### -------------------------------------------------------------------


if __name__ == '__main__':
    # six heads for the transformer features
    feats = torch.rand(1, 6, 673, 64)

    k_clusters = 3
    assign_func = 'mean'
    eta = 1.0
    rho = 0.0
    kernel = "rbf"
    kernel_param = {'sigma2':50.0}
    dims = [32, 21]
    init_image_size = torch.Size([3, 500, 332])
    scales = [16, 16]

    pred, objects, foreground, seed, eigenvector = mvclustering(feats, dims, scales, init_image_size, kernel, kernel_param, k_clusters, assign_func, eta, rho)
