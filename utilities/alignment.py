### adapted from ImageAnalysis3 - for alignment of images based on fiducial beads
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.spatial.distance import pdist, squareform, euclidean

def correct_image3D_by_microscope_param(image3D:np.ndarray, microscope_params:dict):
    """Correct 3D image with microscopy parameter"""
    _image = image3D.copy()
    if not isinstance(microscope_params, dict):
        raise TypeError(f"Wrong inputt ype for microscope_params, should be a dict")
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _image = _image.transpose((0,2,1))
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _image = np.flip(_image, 2)
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _image = np.flip(_image, 1)
    return _image


def _find_boundary(_ct, _radius, _im_size):
    _bds = []
    for _c, _sz in zip(_ct, _im_size):
        _bds.append([max(_c-_radius, 0), min(_c+_radius, _sz)])
    
    return np.array(_bds, dtype=int)


def generate_drift_crops(single_im_size, coord_sel=None, drift_size=None):
    """Function to generate drift crop from a selected center and given drift size
    keywards:
        single_im_size: single image size to generate crops, np.ndarray like;
        coord_sel: selected center coordinate to split image into 4 rectangles, np.ndarray like;
        drift_size: size of drift crop, int or np.int;
    returns:
        crops: 4x3x2 np.ndarray. 
    """
    # check inputs
    _single_im_size = np.array(single_im_size)
    if coord_sel is None:
        coord_sel = np.array(_single_im_size/2, dtype=int)
    if coord_sel[-2] >= _single_im_size[-2] or coord_sel[-1] >= _single_im_size[-1]:
        raise ValueError(f"wrong input coord_sel:{coord_sel}, should be smaller than single_im_size:{single_im_size}")
    if drift_size is None:
        drift_size = int(np.max(_single_im_size)/4)
        
    # generate crop centers
    crop_cts = [
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1],]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1],]),                               
    ]
    # generate boundaries
    crops = [_find_boundary(_ct, _radius=drift_size/2, _im_size=single_im_size) for _ct in crop_cts]
        
    return np.array(crops)

def shift_spots(spots, drift):
    """shift the spots based on drift"""
    if np.shape(spots)[1] == 11: # this means 3d fitting result
        coords = np.array(spots).copy()[:,1:4]
    else:
        raise ValueError(f"Wrong input coords")
    # apply drift
    corr_coords = coords + drift
    # update spots
    output_coords = np.array(spots).copy()
    output_coords[:, 1:4] = corr_coords
    return output_coords

def align_image(
    src_im:np.ndarray, 
    ref_im:np.ndarray, 
    crop_list=None,
    precision_fold=100, 
    min_good_drifts=3, drift_diff_th=1., drift_pixel_threshold = 150):
    """Function to align one image by either FFT or spot_finding
        both source and reference images should be corrected
    """

    single_im_size = src_im.shape
    # check crop_list:
    if crop_list is None:
        crop_list = generate_drift_crops(single_im_size)
    for _crop in crop_list:
        if np.shape(np.array(_crop)) != (3,2):
            raise IndexError(f"crop should be 3x2 np.ndarray.")
    
    # define result flag
    _result_flag = 0
    
    if np.shape(src_im) != np.shape(ref_im):
        raise IndexError(f"shape of target image:{np.shape(src_im)} and reference image:{np.shape(ref_im)} doesnt match!")

    ## crop images
    _crop_src_ims, _crop_ref_ims = [], []
    for _crop in crop_list:
        _s = tuple([slice(*np.array(_c,dtype=int)) for _c in _crop])
        _crop_src_ims.append(src_im[_s])
        _crop_ref_ims.append(ref_im[_s])

    ## align two images
    _drifts = []
    for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
        # calculate drift with autocorr
        _dft, _error, _phasediff = phase_cross_correlation(_rim, _sim, 
                                                               upsample_factor=precision_fold)
        # append 
        if (len(np.where(np.abs(_dft)>drift_pixel_threshold)[0])==0): # drift is usually not more than 150 pixels
            _drifts.append(_dft)
        if (len(np.where(_dft==0)[0])<=1): # drift is does not usually have two dim equal to zero
            _drifts.append(_dft) 

        # detect variance within existing drifts
        _mean_dft = np.nanmean(_drifts, axis=0)
        if len(_drifts) >= min_good_drifts:
            _dists = np.linalg.norm(_drifts-_mean_dft, axis=1)
            _kept_drift_inds = np.where(_dists <= drift_diff_th)[0]
            if len(_kept_drift_inds) >= min_good_drifts:
                _updated_mean_dft = np.nanmean(np.array(_drifts)[_kept_drift_inds], axis=0)
                _result_flag = 'Optimal alignment'
                break
    
    if len(_drifts)==0:
        return [0,0,0], 'Failed alignment'
    elif len(_drifts)==1:
        return _drifts[0], 'Poor alignment'

    if '_updated_mean_dft' not in locals():
        _drifts = np.array(_drifts)
        # select top 3 drifts
        _dist_mat = squareform(pdist(_drifts))
        np.fill_diagonal(_dist_mat, np.inf)
        # select closest pair
        _sel_inds = np.array(np.unravel_index(np.argmin(_dist_mat), np.shape(_dist_mat)))
        _sel_drifts = list(_drifts[_sel_inds])
        # select closest 3rd drift
        third_drift = _drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))]
        diff_first_third = euclidean(third_drift, _sel_drifts[0])
        diff_second_third = euclidean(third_drift, _sel_drifts[1])
        diff_first_second = euclidean(_sel_drifts[0], _sel_drifts[1])
        _cv = np.std([diff_first_second, diff_first_third, diff_second_third])/np.mean([diff_first_second, diff_first_third, diff_second_third])
        if _cv<=0.5:
            _sel_drifts.append(_drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))])
        # return mean
        _updated_mean_dft = np.nanmean(_sel_drifts, axis=0)
        _result_flag = 'Suboptimal alignment'

    return  _updated_mean_dft, _result_flag