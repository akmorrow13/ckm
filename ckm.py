from sklearn.kernel_approximation import RBFSampler
from skimage.measure import block_reduce

import random

import scipy
import numpy as np
from numpy import ndarray

import datetime
import time
from mnist import MNIST


st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
RANDOM_STATE= int(random.random()*100)
RANDOM_STATE= 0
flatten = ndarray.flatten


def apply_patch_rbf(X,imsize, patches, rbf_weights, rbf_offset):

    new_shape = patches.shape[:-3] + (patches.shape[-3]*patches.shape[-1]*patches.shape[-2],)
    patches = patches.reshape(new_shape)
    X_lift = np.zeros((X.shape[0], patches.shape[1], len(rbf_offset)))
    k = 0
    for n in range(X.shape[0]):
        image_patches = patches[n, :]
        flat_patch_norm = np.maximum(np.linalg.norm(image_patches, axis=1), 1e-4)[:,np.newaxis]
        flat_patch_normalized = image_patches/flat_patch_norm
        projection = flat_patch_normalized.dot(rbf_weights)
        projection += rbf_offset
        np.cos(projection, projection)
        X_lift[n] = flat_patch_norm*projection

    X_lift = X_lift.reshape(X_lift.shape[0], X_lift.shape[1]*X_lift.shape[2])
    # Contrast normalization
    return X_lift

def pool(x, pool_size, imsize, func=np.average):
    x = x.reshape(imsize)
    spatial_indices = np.indices(x.shape[:2])
    x_pool = block_reduce(x, block_size = pool_size + (1,), func=func)
    return flatten(x_pool)

def gaussian_pool(x, pool_size, imsize):
    x = x.reshape(imsize)
    x_out =  x.copy()
    for i in range(x_out.shape[2]):
        x_out[:,:,i] = scipy.ndimage.filters.gaussian_filter(x[:,:,i], pool_size/np.sqrt(2))
    return flatten(x_out[::pool_size, ::pool_size,:])

def patchify_all_imgs(X, patch_shape, pad=True, pad_mode='constant', cval=0):
    out = []
    for x in X:
        dim = x.shape[0]
        x = x.reshape(int(np.sqrt(dim)), int(np.sqrt(dim)), x.shape[1])
        patches = patchify(x, patch_shape, pad, pad_mode, cval)
        out_shape = patches.shape
        out.append(patches.reshape(out_shape[0]*out_shape[1], patch_shape[0], patch_shape[1], -1))
    return np.array(out)

def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (patch_shape[0]/2, patch_shape[0]/2)
        img = np.pad(img, (pad_size, pad_size, (0,0)),  mode=pad_mode, constant_values=cval)

    img = np.ascontiguousarray(img)  # won't make a copy if not needed

    X, Y, Z = img.shape
    x, y= patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y*Z, Z, Y*Z, Z, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches

def learn_gamma(patches, sample_size=3000, percentile=10):
    patches = patches.reshape(-1,patches.shape[2]*patches.shape[3]*patches.shape[-1])
    x_indices = np.random.choice(patches.shape[0], sample_size)
    y_indices = np.random.choice(patches.shape[0], sample_size)
    x = patches[x_indices]
    y = patches[y_indices]
    x_norm = np.maximum(np.linalg.norm(x, axis=1), 1e-16)[:,np.newaxis]
    y_norm = np.maximum(np.linalg.norm(y, axis=1), 1e-16)[:,np.newaxis]
    x = x/x_norm
    y = y/y_norm
    diff = x - y
    norms = np.linalg.norm(diff, axis=1)
    return 1.0/((1.0/np.sqrt(2) * np.median(norms))**2)



def ckm_apply(X_train, X_test, patch_shape, gamma, n_components, pool=True, random_state=RANDOM_STATE):
    patches_train = patchify_all_imgs(X_train, patch_shape, pad=False)
    if (gamma == None):
        #print "USING LEARNED GAMMA ", learn_gamma(patches_train)
        gamma = learn_gamma(patches_train)

    patch_rbf = RBFSampler(gamma=gamma, random_state=random_state, n_components=n_components).fit(np.zeros((1,patch_shape[0]*patch_shape[1]*X_train.shape[-1])))
    #print "Generated train patches"
    #print "Patches_train size", patches_train.shape
    #print "RBF map shape", patch_rbf.random_weights_.shape
    X_patch_lift_train = apply_patch_rbf(X_train, X_train.shape[1], patches_train, patch_rbf.random_weights_, patch_rbf.random_offset_)
    #print "Lifted train"
    patches_test = patchify_all_imgs(X_test, patch_shape, pad=False)
    #print "Generated test patches"
    X_patch_lift_test = apply_patch_rbf(X_test, X_test.shape[1], patches_test, patch_rbf.random_weights_, patch_rbf.random_offset_)
    #print "Lifted test"
    #print "Pre pool shape", X_patch_lift_train.shape
    #print patches_train.shape
    imsize = (np.sqrt(patches_train.shape[1]), np.sqrt(patches_train.shape[1]), patch_rbf.n_components)
    if (pool):
        X_out_train = np.array([ gaussian_pool(x,2,imsize)  for x in X_patch_lift_train])
        X_out_test = np.array([ gaussian_pool(x,2,imsize)  for x in X_patch_lift_test])
    else:
        X_out_train = X_patch_lift_train
        X_out_test = X_patch_lift_test

    X_out_train = X_out_train.reshape(X_train.shape[0],-1, 50)
    X_out_test = X_out_test.reshape(X_test.shape[0],-1, 50)
    return X_out_train, X_out_test
