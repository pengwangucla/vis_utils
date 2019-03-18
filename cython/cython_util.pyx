from __future__ import division
import numpy as np
cimport numpy as np 

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "src/manager.hh":
    void depth2normals(DTYPE_t* normal,
                       DTYPE_t* depth,
                       DTYPE_t* intrinsic,
                       int width,
                       int height,
                       int batch)


def depth2normals_np(np.ndarray[DTYPE_t, ndim=2] depth,
                     np.ndarray[DTYPE_t, ndim=1] intrinsic):
    cdef int height = depth.shape[0];
    cdef int width = depth.shape[1];
    cdef int batch = 1;

    cdef np.ndarray[DTYPE_t, ndim=3] normal = np.zeros((3, height, width), dtype=DTYPE)
    depth2normals(&normal[0, 0, 0],
                  &depth[0, 0],
                  &intrinsic[0],
                  width, height, batch)

    return normal


def gen_depth_map(np.ndarray[DTYPE_t, ndim=2] proj, int height, int width, int get_id):

    cdef np.ndarray[DTYPE_t, ndim=2] depth = np.zeros((height, width), dtype=DTYPE)
    cdef int pix_num = proj.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] index = np.zeros((height, width),
      dtype=np.int32)

    cdef int x
    cdef int y

    for i in range(pix_num):
        x = int(proj[0, i])
        y = int(proj[1, i])
        if 0 < x and x <= width and 0 < y and y <= height:
            cur_depth = depth[y-1][x-1]
            if cur_depth == 0 or cur_depth > proj[2, i]:
                depth[y-1][x-1] = np.float32(proj[2, i])
                index[y-1][x-1] = np.int32(i)

    if get_id == 1:
      return depth, index
    else:
      return depth


def gen_depth_map_with_ref_depth(np.ndarray[DTYPE_t, ndim=2] proj, 
        int height, int width, np.ndarray[DTYPE_t, ndim=2] ref_depth):

    cdef np.ndarray[DTYPE_t, ndim=2] depth = np.zeros((height, width), dtype=DTYPE)
    cdef int pix_num = proj.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1] index = np.zeros(pix_num, dtype=np.int32)

    cdef int x
    cdef int y

    for i in range(pix_num):
        x = int(proj[0, i])
        y = int(proj[1, i])
        if 0 < x and x <= width and 0 < y and y <= height:
            cur_depth = depth[y-1][x-1]
            if cur_depth == 0 or cur_depth > proj[2, i]:
                depth[y-1][x-1] = np.float32(proj[2, i])
                index[i] = 1
            if np.abs(proj[2, i] - ref_depth[y-1][x-1]) < 0.1:
                index[i] = 1

    return depth, index

