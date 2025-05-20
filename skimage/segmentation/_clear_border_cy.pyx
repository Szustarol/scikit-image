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

cpdef inline void _clear_border_flat(dtype_t[::1] out,
                                    Py_ssize_t[::1] indices,
                                    Py_ssize_t[::1] neighbor_offsets,
                                    Py_ssize_t n_indices,
                                    dtype_t bgval
                                    ):
    # n_indices indicates the total amount of indices to process
    #   and is altered during the processing
    cdef Py_ssize_t i, current_index, neighbor_index
    cdef Py_ssize_t n_offsets = neighbor_offsets.shape[0]
   
    with nogil:
        # normally, the initial indices should be zeroed manually
        # but this is done before the function call to utilize numpy 
        # for i in range(n_indices):
        #     # set all starting indices to bgval to avoid adding them more than once
        #     out[indices[i]] = bgval

        # to preserve spatial locality, the indices are processed as a stack, not a FIFO queue
        # this helps to avoid cache misses
        while n_indices > 0:
            # get the current element and pop the stack
            current_index = indices[n_indices-1]
            n_indices -= 1
            # at this point out[current_index] should already be 0
            for i in range(n_offsets):
                neighbor_index = current_index + neighbor_offsets[i]
                
                # if neighbour is "on", 
                if out[neighbor_index] != bgval:
                    # add neighbour to queue
                    indices[n_indices] = neighbor_index
                    n_indices += 1
                    # remove the neighbour 
                    out[neighbor_index] = bgval
