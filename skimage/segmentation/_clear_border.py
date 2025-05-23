import numpy as np

from scipy.ndimage import generate_binary_structure

from ._clear_border_cy import _clear_border_flat
from ..measure import label
from ..util import crop
from ..morphology._util import _offsets_to_raveled_neighbors


def clear_border(labels, buffer_size=0, bgval=0, mask=None, *, out=None):
    """Clear objects connected to the label image border.

    Parameters
    ----------
    labels : (M[, N[, ..., P]]) array of int or bool
        Imaging data labels.
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.
    mask : ndarray of bool, same shape as `image`, optional.
        Image data mask. Objects in labels image overlapping with
        False pixels of mask will be removed. If defined, the
        argument buffer_size will be ignored.
    out : ndarray
        Array of the same shape as `labels`, into which the
        output is placed. By default, a new array is created.

    Returns
    -------
    out : (M[, N[, ..., P]]) array
        Imaging data labels with cleared borders

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.segmentation import clear_border
    >>> labels = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
    ...                    [1, 1, 0, 0, 1, 0, 0, 1, 0],
    ...                    [1, 1, 0, 1, 0, 1, 0, 0, 0],
    ...                    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    ...                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> clear_border(labels)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> mask = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1],
    ...                  [0, 0, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
    >>> clear_border(labels, mask=mask)
    array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    """
    if any(buffer_size >= s for s in labels.shape) and mask is None:
        # ignore buffer_size if mask
        raise ValueError("buffer size may not be greater than labels size")

    if out is None:
        out = labels.copy()

    if mask is not None:
        err_msg = (
            f'labels and mask should have the same shape but '
            f'are {out.shape} and {mask.shape}'
        )
        if out.shape != mask.shape:
            raise (ValueError, err_msg)
        if mask.dtype != bool:
            raise TypeError("mask should be of type bool.")
        borders = ~mask
    else:
        # create borders with buffer_size
        borders = np.zeros_like(out, dtype=bool)
        ext = buffer_size + 1
        slstart = slice(ext)
        slend = slice(-ext, None)
        slices = [slice(None) for _ in out.shape]
        for d in range(out.ndim):
            slices[d] = slstart
            borders[tuple(slices)] = True
            slices[d] = slend
            borders[tuple(slices)] = True
            slices[d] = slice(None)
            
    return _clear_border_fast(out, borders, bgval)

def _clear_border_fast(out, initial_indices, bgval):
    #################
    # Most preparation logic is the similar as of morphology.flood_fill
    ################
    
    # ensure C contiguity, as required by nonzero checking    
    if out.flags.c_contiguous is True:
        out_view = out.ravel(order='C')
        order = 'C'
    elif initial_indices.flags.f_contiguous is True:
        out_view = out.ravel(order='F')
        order = 'F'
    else:
        # worst case scenario, out_view will be a copy
        out_view = out.ravel(order='C')
        order = 'C'
        
    # Shortcut for rank zero
    if 0 in out.shape:
        out[...] = bgval
        return out
    
    footprint = generate_binary_structure(out.ndim, out.ndim)
    
    center = tuple(s // 2 for s in footprint.shape)
    
    # if the initial indices are already background, skip them
    initial_indices[out == bgval] = False
    
    # initial indices from the mask
    initial_indices = np.nonzero(initial_indices.ravel(order=order))[0]
    
    visited = np.zeros((len(out_view)), dtype=np.uint8, order=order)
    visited[initial_indices] = True

    n_indices = len(initial_indices)
    
    # used within the algorithm to store the indices left to explore
    indices_container = np.empty(out_view.shape, dtype=np.uintp)
    indices_container[:n_indices] = initial_indices

    # Stride-aware neighbors 
    neighbor_offsets = _offsets_to_raveled_neighbors(
        out.shape, footprint, center=center, order=order
    ).astype(np.intp)
    
    neighbor_offsets = np.sort(neighbor_offsets, )
    print(neighbor_offsets)
    
    import time
    
    s = time.time()
    _clear_border_flat(
        out_view,
        indices_container,
        neighbor_offsets,
        visited,
        n_indices,
        bgval
    )
    print("Core took", time.time() - s)
    
    visited = visited.reshape(out.shape, order=order)
    
    out[visited == 1] = bgval
    
    return out
    
    