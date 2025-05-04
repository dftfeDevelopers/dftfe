#include "linearSolverCGDeviceKernels.h"

namespace dftfe
{
  template <typename Type, dftfe::Int blockSize>
  __global__ void
  applyPreconditionAndComputeDotProductKernel(Type            *d_dvec,
                                              Type            *d_devSum,
                                              const Type      *d_rvec,
                                              const Type      *d_jacobi,
                                              const dftfe::Int N)
  {
    __shared__ Type smem[blockSize];

    dftfe::Int tid = threadIdx.x;
    dftfe::Int idx = threadIdx.x + blockIdx.x * (blockSize * 2);

    Type localSum;

    if (idx < N)
      {
        Type jacobi = d_jacobi[idx];
        Type r      = d_rvec[idx];

        localSum    = jacobi * r * r;
        d_dvec[idx] = jacobi * r;
      }
    else
      localSum = 0;

    if (idx + blockSize < N)
      {
        Type jacobi = d_jacobi[idx + blockSize];
        Type r      = d_rvec[idx + blockSize];
        localSum += jacobi * r * r;
        d_dvec[idx + blockSize] = jacobi * r;
      }

    smem[tid] = localSum;
    __syncthreads();

#pragma unroll
    for (dftfe::Int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        __syncthreads();
      }

    if (tid < dftfe::utils::DEVICE_WARP_SIZE)
      {
        if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
          localSum += smem[tid + dftfe::utils::DEVICE_WARP_SIZE];

#pragma unroll
        for (dftfe::Int offset = dftfe::utils::DEVICE_WARP_SIZE / 2; offset > 0;
             offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#endif
          }
      }

    if (tid == 0)
      atomicAdd(&d_devSum[0], localSum);
  }


  template <typename Type, dftfe::Int blockSize>
  __global__ void
  applyPreconditionComputeDotProductAndSaddKernel(Type            *d_qvec,
                                                  Type            *d_devSum,
                                                  const Type      *d_rvec,
                                                  const Type      *d_jacobi,
                                                  const dftfe::Int N)
  {
    __shared__ Type smem[blockSize];

    dftfe::Int tid = threadIdx.x;
    dftfe::Int idx = threadIdx.x + blockIdx.x * (blockSize * 2);

    Type localSum;

    if (idx < N)
      {
        Type jacobi = d_jacobi[idx];
        Type r      = d_rvec[idx];

        localSum    = jacobi * r * r;
        d_qvec[idx] = -1 * jacobi * r;
      }
    else
      localSum = 0;

    if (idx + blockSize < N)
      {
        Type jacobi = d_jacobi[idx + blockSize];
        Type r      = d_rvec[idx + blockSize];
        localSum += jacobi * r * r;
        d_qvec[idx + blockSize] = -1 * jacobi * r;
      }

    smem[tid] = localSum;
    __syncthreads();

#pragma unroll
    for (dftfe::Int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];
        __syncthreads();
      }

    if (tid < dftfe::utils::DEVICE_WARP_SIZE)
      {
        if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
          localSum += smem[tid + dftfe::utils::DEVICE_WARP_SIZE];

#pragma unroll
        for (dftfe::Int offset = dftfe::utils::DEVICE_WARP_SIZE / 2; offset > 0;
             offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#endif
          }
      }

    if (tid == 0)
      atomicAdd(&d_devSum[0], localSum);
  }


  template <typename Type, dftfe::Int blockSize>
  __global__ void
  scaleXRandComputeNormKernel(Type            *x,
                              Type            *d_rvec,
                              Type            *d_devSum,
                              const Type      *d_qvec,
                              const Type      *d_dvec,
                              const Type       alpha,
                              const dftfe::Int N)
  {
    __shared__ Type smem[blockSize];

    dftfe::Int tid = threadIdx.x;
    dftfe::Int idx = threadIdx.x + blockIdx.x * (blockSize * 2);

    Type localSum;

    if (idx < N)
      {
        Type rNew;
        Type rOld = d_rvec[idx];
        x[idx] += alpha * d_qvec[idx];
        rNew        = rOld + alpha * d_dvec[idx];
        localSum    = rNew * rNew;
        d_rvec[idx] = rNew;
      }
    else
      localSum = 0;

    if (idx + blockSize < N)
      {
        Type rNew;
        Type rOld = d_rvec[idx + blockSize];
        x[idx + blockSize] += alpha * d_qvec[idx + blockSize];
        rNew = rOld + alpha * d_dvec[idx + blockSize];
        localSum += rNew * rNew;
        d_rvec[idx + blockSize] = rNew;
      }

    smem[tid] = localSum;
    __syncthreads();

#pragma unroll
    for (dftfe::Int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

        __syncthreads();
      }

    if (tid < dftfe::utils::DEVICE_WARP_SIZE)
      {
        if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
          localSum += smem[tid + dftfe::utils::DEVICE_WARP_SIZE];

#pragma unroll
        for (dftfe::Int offset = dftfe::utils::DEVICE_WARP_SIZE / 2; offset > 0;
             offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#endif
          }
      }

    if (tid == 0)
      atomicAdd(&d_devSum[0], localSum);
  }
  void
  applyPreconditionAndComputeDotProductDevice(double          *d_dvec,
                                              double          *d_devSum,
                                              const double    *d_rvec,
                                              const double    *d_jacobi,
                                              const dftfe::Int N)
  {
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);
    DFTFE_LAUNCH_KERNEL(DFTFE_KERNEL_NAME(
                          applyPreconditionAndComputeDotProductKernel<
                            double,
                            dftfe::utils::DEVICE_BLOCK_SIZE>),
                        blocks,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        0,
                        d_dvec,
                        d_devSum,
                        d_rvec,
                        d_jacobi,
                        N);
  }


  void
  applyPreconditionComputeDotProductAndSaddDevice(double          *d_qvec,
                                                  double          *d_devSum,
                                                  const double    *d_rvec,
                                                  const double    *d_jacobi,
                                                  const dftfe::Int N)
  {
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);


    DFTFE_LAUNCH_KERNEL(DFTFE_KERNEL_NAME(
                          applyPreconditionComputeDotProductAndSaddKernel<
                            double,
                            dftfe::utils::DEVICE_BLOCK_SIZE>),
                        blocks,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        0,
                        d_qvec,
                        d_devSum,
                        d_rvec,
                        d_jacobi,
                        N);
  }


  void
  scaleXRandComputeNormDevice(double          *x,
                              double          *d_rvec,
                              double          *d_devSum,
                              const double    *d_qvec,
                              const double    *d_dvec,
                              const double     alpha,
                              const dftfe::Int N)
  {
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);


    DFTFE_LAUNCH_KERNEL(
      DFTFE_KERNEL_NAME(
        scaleXRandComputeNormKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      x,
      d_rvec,
      d_devSum,
      d_qvec,
      d_dvec,
      alpha,
      N);
  }

} // namespace dftfe
