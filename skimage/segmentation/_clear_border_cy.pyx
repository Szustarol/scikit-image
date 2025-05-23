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

cpdef inline void _clear_border_flat(dtype_t[::1] out_view,
                                    Py_ssize_t[::1] indices,
                                    Py_ssize_t[::1] neighbor_offsets,
                                    cnp.uint8_t[::1] visited,
                                    Py_ssize_t n_indices,
                                    dtype_t bgval
                                    ):
    # n_indices indicates the total amount of indices to process
    #   and is altered during the processing
    cdef size_t n_total = out_view.shape[0]
    cdef size_t i, current_index, neighbor_index
    cdef size_t n_offsets = neighbor_offsets.shape[0]
    cdef dtype_t current_value
   
    with nogil:
        # to preserve spatial locality, the indices are processed as a stack, not a FIFO queue
        # this helps to avoid cache misses
        while n_indices > 0:
            # get the current element and pop the stack
            n_indices -= 1
            current_index = indices[n_indices]
            current_value = out_view[current_index]
            for i in range(n_offsets):
                neighbor_index = current_index + neighbor_offsets[i]
                # some unsigned magic - since the value is unsigned,
                # it will wrap around if it is negative, so only
                # upper bound check is needed
                if neighbor_index > n_total:
                    # out of bounds
                    continue
                # if neighbour is "on", 
                if not visited[neighbor_index] and out_view[neighbor_index] == current_value:
                    # add neighbour to queue
                    indices[n_indices] = neighbor_index
                    n_indices += 1
                    visited[neighbor_index] = 1
