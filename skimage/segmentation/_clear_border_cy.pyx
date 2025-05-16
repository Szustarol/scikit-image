#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""Cython code used in _clear_border.py."""

cimport numpy as cnp
cnp.import_array()

ctypedef fused dtype_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t

cpdef inline void _clear_border_flat(dtype_t[::1] mask,
                                    Py_ssize_t[::1] indices,
                                    Py_ssize_t[::1] neighbor_offsets,
                                    Py_ssize_t n_indices,
                                    ):
   
    cdef Py_ssize_t i, index, current_index, neighbor_index, offset
   
    with nogil:
        for i in range(n_indices):
            # set all starting indices to zero to avoid adding them more than once
            mask[indices[i]] = 0

        while index < n_indices:
            current_index = indices[index]
            # at this point mask[current_index] should already be 0
            for i in range(neighbor_offsets.shape[0]):
                offset = neighbor_offsets[i]
                neighbor_index = current_index + offset
                
                # if neighbour is "on", 
                if mask[neighbor_index]:
                    # add neighbour to queue
                    indices[n_indices] = neighbor_index
                    n_indices += 1
                    # turn the neighbour off, to avoid adding it again
                    mask[neighbor_index] = 0
            index += 1
