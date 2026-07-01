#ifndef MACROS_H
#define MACROS_H

#ifdef HAS_CUDA
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#endif // MACROS_H
