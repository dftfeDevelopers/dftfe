#include "linearSolverCGDeviceKernels.h"

namespace dftfe
{
  template <typename Type, dftfe::Int blockSize>
  DFTFE_CREATE_KERNEL_SMEM_S(
    Type,
    blockSize,
    void,
    applyPreconditionAndComputeDotProductKernel,
    DFTFE_KERNEL_ARGUMENT({
      Type       localSum;
      dftfe::Int idx = threadId + blockId * (blockSize * 2);

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

      smem[threadId] = localSum;
      SYNCTHREADS;

      _Pragma("unroll") for (dftfe::Int size =
                               dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                             size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
                             size /= 2)
      {
        if ((blockSize >= size) && (threadId < size / 2))
          smem[threadId] = localSum = localSum + smem[threadId + size / 2];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(ind.get_group());
#endif
      }

      if (threadId < dftfe::utils::DEVICE_WARP_SIZE)
        {
          if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
            localSum += smem[threadId + dftfe::utils::DEVICE_WARP_SIZE];

          _Pragma("unroll") for (dftfe::Int offset =
                                   dftfe::utils::DEVICE_WARP_SIZE / 2;
                                 offset > 0;
                                 offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#elif DFTFE_WITH_DEVICE_LANG_SYCL
            localSum +=
              sycl::shift_group_left(ind.get_sub_group(), localSum, offset);
#endif
          }
        }

      if (threadId == 0)
        dftfe::utils::atomicAddWrapper(&d_devSum[0], localSum);
    }),
    Type            *d_dvec,
    Type            *d_devSum,
    const Type      *d_rvec,
    const Type      *d_jacobi,
    const dftfe::Int N);


  template <typename Type, dftfe::Int blockSize>
  DFTFE_CREATE_KERNEL_SMEM_S(
    Type,
    blockSize,
    void,
    applyPreconditionComputeDotProductAndSaddKernel,
    DFTFE_KERNEL_ARGUMENT({
      dftfe::Int idx = threadId + blockId * (blockSize * 2);

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

      smem[threadId] = localSum;
      SYNCTHREADS;

      _Pragma("unroll") for (dftfe::Int size =
                               dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                             size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
                             size /= 2)
      {
        if ((blockSize >= size) && (threadId < size / 2))
          smem[threadId] = localSum = localSum + smem[threadId + size / 2];
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(ind.get_group());
#endif
      }

      if (threadId < dftfe::utils::DEVICE_WARP_SIZE)
        {
          if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
            localSum += smem[threadId + dftfe::utils::DEVICE_WARP_SIZE];

          _Pragma("unroll") for (dftfe::Int offset =
                                   dftfe::utils::DEVICE_WARP_SIZE / 2;
                                 offset > 0;
                                 offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#elif DFTFE_WITH_DEVICE_LANG_SYCL
            localSum +=
              sycl::shift_group_left(ind.get_sub_group(), localSum, offset);
#endif
          }
        }

      if (threadId == 0)
        dftfe::utils::atomicAddWrapper(&d_devSum[0], localSum);
    }),
    Type            *d_qvec,
    Type            *d_devSum,
    const Type      *d_rvec,
    const Type      *d_jacobi,
    const dftfe::Int N);


  template <typename Type, dftfe::Int blockSize>
  DFTFE_CREATE_KERNEL_SMEM_S(
    Type,
    blockSize,
    void,
    scaleXRandComputeNormKernel,
    DFTFE_KERNEL_ARGUMENT({
      dftfe::Int idx = threadId + blockId * (blockSize * 2);

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

      smem[threadId] = localSum;
      SYNCTHREADS;

      _Pragma("unroll") for (dftfe::Int size =
                               dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                             size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
                             size /= 2)
      {
        if ((blockSize >= size) && (threadId < size / 2))
          smem[threadId] = localSum = localSum + smem[threadId + size / 2];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(ind.get_group());
#endif
      }

      if (threadId < dftfe::utils::DEVICE_WARP_SIZE)
        {
          if (blockSize >= 2 * dftfe::utils::DEVICE_WARP_SIZE)
            localSum += smem[threadId + dftfe::utils::DEVICE_WARP_SIZE];

          _Pragma("unroll") for (dftfe::Int offset =
                                   dftfe::utils::DEVICE_WARP_SIZE / 2;
                                 offset > 0;
                                 offset /= 2)
          {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            unsigned mask = 0xffffffff;
            localSum += __shfl_down_sync(mask, localSum, offset);
#elif DFTFE_WITH_DEVICE_LANG_HIP
            localSum +=
              __shfl_down(localSum, offset, dftfe::utils::DEVICE_WARP_SIZE);
#elif DFTFE_WITH_DEVICE_LANG_SYCL
            localSum +=
              sycl::shift_group_left(ind.get_sub_group(), localSum, offset);
#endif
          }
        }

      if (threadId == 0)
        dftfe::utils::atomicAddWrapper(&d_devSum[0], localSum);
    }),
    Type            *x,
    Type            *d_rvec,
    Type            *d_devSum,
    const Type      *d_qvec,
    const Type      *d_dvec,
    const Type       alpha,
    const dftfe::Int N);

  template <typename Type>
  DFTFE_CREATE_KERNEL(
    void,
    saddKernel,
    {
      for (dftfe::uInt idx = globalThreadId; idx < size;
           idx += nThreadsPerBlock * nThreadBlock)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    },
    Type             *y,
    Type             *x,
    const Type        beta,
    const dftfe::uInt size);


  void
  applyPreconditionAndComputeDotProductDevice(double          *d_dvec,
                                              double          *d_devSum,
                                              const double    *d_rvec,
                                              const double    *d_jacobi,
                                              const dftfe::Int N)
  {
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);
    DFTFE_LAUNCH_KERNEL_SMEM_S(DFTFE_KERNEL_ARGUMENT(
                                 applyPreconditionAndComputeDotProductKernel<
                                   double,
                                   dftfe::utils::DEVICE_BLOCK_SIZE>),
                               blocks,
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                               double,
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                               dftfe::utils::defaultStream,
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
    DFTFE_LAUNCH_KERNEL_SMEM_S(
      DFTFE_KERNEL_ARGUMENT(applyPreconditionComputeDotProductAndSaddKernel<
                            double,
                            dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      double,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
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
    DFTFE_LAUNCH_KERNEL_SMEM_S(
      DFTFE_KERNEL_ARGUMENT(
        scaleXRandComputeNormKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      double,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      x,
      d_rvec,
      d_devSum,
      d_qvec,
      d_dvec,
      alpha,
      N);
  }

  void
  sadd(double *y, double *x, const double beta, const dftfe::uInt size)
  {
    const dftfe::uInt gridSize =
      (size / dftfe::utils::DEVICE_BLOCK_SIZE) +
      (size % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
    DFTFE_LAUNCH_KERNEL(saddKernel,
                        gridSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::defaultStream,
                        y,
                        x,
                        beta,
                        size);
  }


} // namespace dftfe
