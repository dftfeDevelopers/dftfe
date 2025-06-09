#include "linearSolverCGDeviceKernels.h"

namespace dftfe
{
  template <typename Type, dftfe::Int blockSize>
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
  __global__ void
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
  void
#endif
  applyPreconditionAndComputeDotProductKernel(
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::nd_item<1> item,
#endif
    Type            *d_dvec,
    Type            *d_devSum,
    const Type      *d_rvec,
    const Type      *d_jacobi,
    const dftfe::Int N
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    ,
    Type *smem
#endif
  )
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __shared__ Type smem[blockSize];
    dftfe::Int      tid = threadIdx.x;
    dftfe::Int      idx = threadIdx.x + blockIdx.x * (blockSize * 2);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    dftfe::Int tid = item.get_local_id(0);
    dftfe::Int idx = item.get_local_id(0) + item.get_group(0) * (blockSize * 2);
#endif

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
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif


#pragma unroll
    for (dftfe::Int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(item.get_group());
#endif
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
#elif DFTFE_WITH_DEVICE_LANG_SYCL
            localSum +=
              sycl::shift_group_left(item.get_sub_group(), localSum, offset);
#endif
          }
      }

    if (tid == 0)
      dftfe::utils::atomicAddWrapper(&d_devSum[0], localSum);
  }


  template <typename Type, dftfe::Int blockSize>
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
  __global__ void
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
  void
#endif
  applyPreconditionComputeDotProductAndSaddKernel(
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::nd_item<1> item,
#endif
    Type            *d_qvec,
    Type            *d_devSum,
    const Type      *d_rvec,
    const Type      *d_jacobi,
    const dftfe::Int N
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    ,
    Type *smem
#endif
  )
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __shared__ Type smem[blockSize];
    dftfe::Int      tid = threadIdx.x;
    dftfe::Int      idx = threadIdx.x + blockIdx.x * (blockSize * 2);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    dftfe::Int tid = item.get_local_id(0);
    dftfe::Int idx = item.get_local_id(0) + item.get_group(0) * (blockSize * 2);
#endif

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
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif


#pragma unroll
    for (dftfe::Int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(item.get_group());
#endif
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
#elif DFTFE_WITH_DEVICE_LANG_SYCL
            localSum +=
              sycl::shift_group_left(item.get_sub_group(), localSum, offset);
#endif
          }
      }

    if (tid == 0)
      dftfe::utils::atomicAddWrapper(&d_devSum[0], localSum);
  }


  template <typename Type, dftfe::Int blockSize>
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
  __global__ void
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
  void
#endif
  scaleXRandComputeNormKernel(
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::nd_item<1> item,
#endif
    Type            *x,
    Type            *d_rvec,
    Type            *d_devSum,
    const Type      *d_qvec,
    const Type      *d_dvec,
    const Type       alpha,
    const dftfe::Int N
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    ,
    Type *smem
#endif
  )
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __shared__ Type smem[blockSize];
    dftfe::Int      tid = threadIdx.x;
    dftfe::Int      idx = threadIdx.x + blockIdx.x * (blockSize * 2);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    dftfe::Int tid = item.get_local_id(0);
    dftfe::Int idx = item.get_local_id(0) + item.get_group(0) * (blockSize * 2);
#endif

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
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif


#pragma unroll
    for (dftfe::Int size = dftfe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
         size >= 4 * dftfe::utils::DEVICE_WARP_SIZE;
         size /= 2)
      {
        if ((blockSize >= size) && (tid < size / 2))
          smem[tid] = localSum = localSum + smem[tid + size / 2];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
        __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
        sycl::group_barrier(item.get_group());
#endif
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
#elif DFTFE_WITH_DEVICE_LANG_SYCL
            localSum +=
              sycl::shift_group_left(item.get_sub_group(), localSum, offset);
#endif
          }
      }

    if (tid == 0)
      dftfe::utils::atomicAddWrapper(&d_devSum[0], localSum);
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
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    DFTFE_LAUNCH_KERNEL(DFTFE_KERNEL_NAME(
                          applyPreconditionAndComputeDotProductKernel<
                            double,
                            dftfe::utils::DEVICE_BLOCK_SIZE>),
                        blocks,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::defaultStream,
                        d_dvec,
                        d_devSum,
                        d_rvec,
                        d_jacobi,
                        N);
#elif DFTFE_WITH_DEVICE_LANG_SYCL
    dftfe::utils::defaultStream.submit([=](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> SMem_acc(dftfe::utils::DEVICE_BLOCK_SIZE,
                                               cgh);
      cgh.parallel_for(
        sycl::nd_range<1>(blocks * dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          applyPreconditionAndComputeDotProductKernel<
            double,
            dftfe::utils::DEVICE_BLOCK_SIZE>(item,
                                             d_dvec,
                                             d_devSum,
                                             d_rvec,
                                             d_jacobi,
                                             N,
                                             SMem_acc.get_pointer());
        });
    });
#endif
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
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    DFTFE_LAUNCH_KERNEL(DFTFE_KERNEL_NAME(
                          applyPreconditionComputeDotProductAndSaddKernel<
                            double,
                            dftfe::utils::DEVICE_BLOCK_SIZE>),
                        blocks,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::defaultStream,
                        d_qvec,
                        d_devSum,
                        d_rvec,
                        d_jacobi,
                        N);
#elif DFTFE_WITH_DEVICE_LANG_SYCL
    dftfe::utils::defaultStream.submit([=](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> SMem_acc(dftfe::utils::DEVICE_BLOCK_SIZE,
                                               cgh);
      cgh.parallel_for(
        sycl::nd_range<1>(blocks * dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          applyPreconditionComputeDotProductAndSaddKernel<
            double,
            dftfe::utils::DEVICE_BLOCK_SIZE>(item,
                                             d_qvec,
                                             d_devSum,
                                             d_rvec,
                                             d_jacobi,
                                             N,
                                             SMem_acc.get_pointer());
        });
    });
#endif
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
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    const dftfe::Int blocks = (N + (dftfe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
                              (dftfe::utils::DEVICE_BLOCK_SIZE * 2);


    DFTFE_LAUNCH_KERNEL(
      DFTFE_KERNEL_NAME(
        scaleXRandComputeNormKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>),
      blocks,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      x,
      d_rvec,
      d_devSum,
      d_qvec,
      d_dvec,
      alpha,
      N);
#elif DFTFE_WITH_DEVICE_LANG_SYCL
    dftfe::utils::defaultStream.submit([=](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> SMem_acc(dftfe::utils::DEVICE_BLOCK_SIZE,
                                               cgh);
      cgh.parallel_for(
        sycl::nd_range<1>(blocks * dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          scaleXRandComputeNormKernel<double, dftfe::utils::DEVICE_BLOCK_SIZE>(
            item,
            x,
            d_rvec,
            d_devSum,
            d_qvec,
            d_dvec,
            alpha,
            N,
            SMem_acc.get_pointer());
        });
    });
#endif
  }

} // namespace dftfe
