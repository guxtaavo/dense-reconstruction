import numpy as np
from scipy import ndimage


def plane_sweep_ncc(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation. """
    m,n = im_l.shape

    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))
    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))

    # compute mean of patch
    ndimage.uniform_filter(im_l,wid,mean_l)
    ndimage.uniform_filter(im_r,wid,mean_r)

    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r

    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums

      ndimage.uniform_filter(norm_l*np.roll(norm_r,displ+start),wid,s) # sum nominator
      ndimage.uniform_filter(norm_l*norm_l,wid,s_l)
      ndimage.uniform_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,s_r) # sum denominator
      # store ncc scores
      dmaps[:,:,displ] = s/np.sqrt(np.absolute(s_l*s_r))


    # pick best depth for each pixel
    best_map = np.argmax(dmaps,axis=2) + start

    return best_map


def plane_sweep_gauss(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation
    with Gaussian weighted neigborhoods. """
    m,n = im_l.shape

    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))

    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))

    # compute mean
    ndimage.gaussian_filter(im_l,wid,0,mean_l)
    ndimage.gaussian_filter(im_r,wid,0,mean_r)

    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r

    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums
      ndimage.gaussian_filter(norm_l*np.roll(norm_r,displ+start),wid,0,s)  # sum nominator
      ndimage.gaussian_filter(norm_l*norm_l,wid,0,s_l)
      ndimage.gaussian_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,0,s_r) # sum denominator

      # store ncc scores
      dmaps[:,:,displ] = s/np.sqrt(s_l*s_r)

      # pick best depth for each pixel
      best_map = np.argmax(dmaps,axis=2)+ start


    return best_map
