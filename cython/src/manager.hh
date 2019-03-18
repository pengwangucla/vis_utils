#ifndef MANAGER_HH_
#define MANAGER_HH_

#include <iostream>
#include <assert.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#endif

extern void depth2normals(float* out, const float* depth,
  const float* intrinsics, int x_size, int y_size, int z_size);

#endif
