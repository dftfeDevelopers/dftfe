#include "matrixFreeDeviceKernels.h"

namespace dftfe
{
  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
  __global__ void
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
  void
#endif
  computeAXKernelPoisson(
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::nd_item<1> item,
#endif
    Type             *V,
    const Type       *U,
    const Type       *P,
    const Type       *J,
    const dftfe::Int *map
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    ,
    Type *SMem
#endif
  )
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    extern __shared__ Type SMem[];
    const dftfe::Int       blockDimX  = blockDim.x;
    const dftfe::Int       blockIdxX  = blockIdx.x;
    const dftfe::Int       threadIdxX = threadIdx.x;
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    const dftfe::Int blockDimX  = item.get_local_range(0);
    const dftfe::Int blockIdxX  = item.get_group(0);
    const dftfe::Int threadIdxX = item.get_local_id(0);
#endif
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // P(q*p), D(q*q), PT(p*q), DT(q*q)
    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[N * N * N];
    Type *sharedZ  = &sharedY[N * N * N];
    Type *sharedT  = &sharedZ[N * N * N];
    Type *sharedP  = &sharedT[N * N * N];
    Type *sharedD  = &sharedP[N * K];
    Type *sharedPT = &sharedD[N * N];
    Type *sharedDT = &sharedPT[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const dftfe::Int mapShift = blockIdxX * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (dftfe::Int i = threadIdxX; i < 2 * N * (K + N); i += blockDimX)
      sharedP[i] = P[i];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif
    //////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // V -> UPPP
    // Z -> VDz
    // Y -> VDy
    // X -> VDx

    // 1st GEMM of P
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < M; i += blockDimX)
      {
        Type x[N], u[K];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < K; k++)
          {
            u[k] = U[map[i + k * M + mapShift]];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * u[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[i + j * M] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 2nd GEMM of P
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < K * N; i += blockDimX)
      {
        Type y[N], x[K];

        dftfe::Int a = i % K;
        dftfe::Int b = i / K;

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < K; k++)
          {
            x[k] = sharedX[a + k * K + b * M];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              y[j] += sharedP[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedY[a + (j + b * N) * K] = y[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 3rd GEMM of P
    // X Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type x[N], y[K];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < K; k++)
          {
            y[k] = sharedY[k + i * K];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * y[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[j + i * N] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 1st GEMM of D
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type y[N], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              y[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedY[i + j * N * N] = y[j];
      }

    // 2nd GEMM of D
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type z[N], x[N];

        dftfe::Int a = i % N;
        dftfe::Int b = i / N;

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          z[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[a + (k + b * N) * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedZ[a + (j + b * N) * N] = z[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type t[N], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          t[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              t[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedT[j + i * N] = t[j];
      }

    //////////////////////////////////////////////////////////////////
    // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
    // directions

    const dftfe::Int JShift = blockIdxX * dim * dim;

    // Copy Jacobian Factor to shared memory
#pragma unroll
    for (dftfe::Int i = threadIdxX; i < dim * dim; i += blockDimX)
      sharedJ[i] = J[i + JShift];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // Gemm with Jacobian Factor
#pragma unroll
    for (dftfe::Int i = threadIdxX; i < N * N * N; i += blockDimX)
      {
        Type v[3];

        v[2] = sharedY[i];
        v[1] = sharedZ[i];
        v[0] = sharedT[i];

        sharedY[i] = sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
        sharedZ[i] = sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
        sharedT[i] = sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // Integration
    // Z -> Z(DT)z
    // Y -> Y(DT)y
    // X -> X(DT)x
    // V -> (Z + Y + X)(PT)(PT)(PT)

    // 1st GEMM of DT
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type x[N], y[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            y[k] = sharedY[i + k * N * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              x[j] += sharedDT[j + k * N] * y[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[i + j * N * N] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 2nd GEMM of DT
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type y[N], z[N];

        dftfe::Int a = i % N;
        dftfe::Int b = i / N;

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            z[k] = sharedZ[a + (k + b * N) * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[a + (j + b * N) * N] += y[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 3rd GEMM of DT
    // X Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type z[N], t[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          z[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            t[k] = sharedT[k + i * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              z[j] += sharedDT[j + k * N] * t[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[j + i * N] += z[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 1st GEMM of PT
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type y[K], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (dftfe::Int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          sharedY[i + j * N * N] = y[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 2nd GEMM of PT
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < N * K; i += blockDimX)
      {
        Type x[K], y[N];

        dftfe::Int a = i % N;
        dftfe::Int b = i / N;

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            y[k] = sharedY[a + (k + b * N) * N];

#pragma unroll
            for (dftfe::Int j = 0; j < K; j++)
              x[j] += sharedPT[j + k * K] * y[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          sharedX[a + (j + b * K) * N] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 3rd GEMM of PT
    // X Direction
    for (dftfe::Int i = threadIdxX; i < M; i += blockDimX)
      {
        Type y[K], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (dftfe::Int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          dftfe::utils::atomicAddWrapper(&V[map[j + i * K + mapShift]], y[j]);
      }
  }

  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
  __global__ void
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
  void
#endif
  computeAXKernelHelmholtz(
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::nd_item<1> item,
#endif
    Type             *V,
    const Type       *U,
    const Type       *P,
    const Type       *J,
    const dftfe::Int *map,
    const Type        coeffHelmholtz
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    ,
    Type *SMem
#endif
  )
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    extern __shared__ Type SMem[];
    const dftfe::Int       blockDimX  = blockDim.x;
    const dftfe::Int       blockIdxX  = blockIdx.x;
    const dftfe::Int       threadIdxX = threadIdx.x;
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    const dftfe::Int blockDimX  = item.get_local_range(0);
    const dftfe::Int blockIdxX  = item.get_group(0);
    const dftfe::Int threadIdxX = item.get_local_id(0);
#endif
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // P(q*p), D(q*q), PT(p*q), DT(q*q)

    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[N * N * N];
    Type *sharedZ  = &sharedY[N * N * N];
    Type *sharedT  = &sharedZ[N * N * N];
    Type *sharedP  = &sharedT[N * N * N];
    Type *sharedD  = &sharedP[N * K];
    Type *sharedPT = &sharedD[N * N];
    Type *sharedDT = &sharedPT[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const dftfe::Int mapShift = blockIdxX * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (dftfe::Int i = threadIdxX; i < 2 * N * (K + N); i += blockDimX)
      sharedP[i] = P[i];

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    //////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // V -> UPPP
    // Z -> VDz
    // Y -> VDy
    // X -> VDx

    // 1st GEMM of P
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < M; i += blockDimX)
      {
        Type x[N], u[K];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < K; k++)
          {
            u[k] = U[map[i + k * M + mapShift]];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * u[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[i + j * M] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 2nd GEMM of P
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < K * N; i += blockDimX)
      {
        Type y[N], x[K];

        dftfe::Int a = i % K;
        dftfe::Int b = i / K;

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < K; k++)
          {
            x[k] = sharedX[a + k * K + b * M];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              y[j] += sharedP[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedY[a + (j + b * N) * K] = y[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 3rd GEMM of P
    // X Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type x[N], y[K];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < K; k++)
          {
            y[k] = sharedY[k + i * K];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * y[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[j + i * N] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 1st GEMM of D
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type y[N], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              y[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedY[i + j * N * N] = y[j];
      }

    // 2nd GEMM of D
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type z[N], x[N];

        dftfe::Int a = i % N;
        dftfe::Int b = i / N;

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          z[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[a + (k + b * N) * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedZ[a + (j + b * N) * N] = z[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type t[N], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          t[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              t[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedT[j + i * N] = t[j];
      }

    //////////////////////////////////////////////////////////////////
    // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
    // directions

    const dftfe::Int JShift = blockIdxX * dim * dim;

    // Copy Jacobian Factor to shared memory
#pragma unroll
    for (dftfe::Int i = threadIdxX; i < dim * dim; i += blockDimX)
      sharedJ[i] = J[i + JShift];

    Type detJ;

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // Gemm with Jacobian Factor
#pragma unroll
    for (dftfe::Int i = threadIdxX; i < N * N * N; i += blockDimX)
      {
        Type v[3];

        v[2] = sharedY[i];
        v[1] = sharedZ[i];
        v[0] = sharedT[i];

        sharedY[i] = sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
        sharedZ[i] = sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
        sharedT[i] = sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];

        detJ =
          sharedJ[0] * (sharedJ[4] * sharedJ[8] - sharedJ[5] * sharedJ[7]) -
          sharedJ[1] * (sharedJ[3] * sharedJ[8] - sharedJ[5] * sharedJ[6]) +
          sharedJ[2] * (sharedJ[3] * sharedJ[7] - sharedJ[4] * sharedJ[6]);
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // Integration
    // Z -> Z(DT)z
    // Y -> Y(DT)y
    // X -> X(DT)x
    // V -> (Z + Y + X)(PT)(PT)(PT)

    // 1st GEMM of DT
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type x[N], y[N], h[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            y[k] = sharedY[i + k * N * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              x[j] += sharedDT[j + k * N] * y[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          {
            h[j]                   = sharedX[i + j * N * N];
            sharedX[i + j * N * N] = coeffHelmholtz * detJ * h[j] + x[j];
          }
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 2nd GEMM of DT
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type y[N], z[N];

        dftfe::Int a = i % N;
        dftfe::Int b = i / N;

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            z[k] = sharedZ[a + (k + b * N) * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[a + (j + b * N) * N] += y[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 3rd GEMM of DT
    // X Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type z[N], t[N];

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          z[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            t[k] = sharedT[k + i * N];

#pragma unroll
            for (dftfe::Int j = 0; j < N; j++)
              z[j] += sharedDT[j + k * N] * t[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < N; j++)
          sharedX[j + i * N] += z[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 1st GEMM of PT
    // Z Direction
    for (dftfe::Int i = threadIdxX; i < N * N; i += blockDimX)
      {
        Type y[K], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (dftfe::Int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          sharedY[i + j * N * N] = y[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 2nd GEMM of PT
    // Y Direction
    for (dftfe::Int i = threadIdxX; i < N * K; i += blockDimX)
      {
        Type x[K], y[N];

        dftfe::Int a = i % N;
        dftfe::Int b = i / N;

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          x[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            y[k] = sharedY[a + (k + b * N) * N];

#pragma unroll
            for (dftfe::Int j = 0; j < K; j++)
              x[j] += sharedPT[j + k * K] * y[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          sharedX[a + (j + b * K) * N] = x[j];
      }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    __syncthreads();
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::group_barrier(item.get_group());
#endif

    // 3rd GEMM of PT
    // X Direction
    for (dftfe::Int i = threadIdxX; i < M; i += blockDimX)
      {
        Type y[K], x[N];

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          y[j] = 0.0;

        for (dftfe::Int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (dftfe::Int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (dftfe::Int j = 0; j < K; j++)
          dftfe::utils::atomicAddWrapper(&V[map[j + i * K + mapShift]], y[j]);
      }
  }


  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  void
  matrixFreeDeviceKernels<Type, M, N, K, dim>::computeAXDevicePoisson(
    const dftfe::Int  blocks,
    const dftfe::Int  threads,
    const dftfe::Int  smem,
    Type             *V,
    const Type       *U,
    const Type       *P,
    const Type       *J,
    const dftfe::Int *map)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    DFTFE_LAUNCH_KERNEL(DFTFE_KERNEL_NAME(
                          computeAXKernelPoisson<Type, M, N, K, dim>),
                        blocks,
                        threads,
                        smem,
                        dftfe::utils::defaultStream,
                        V,
                        U,
                        P,
                        J,
                        map);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    dftfe::utils::defaultStream.submit([=](sycl::handler &cgh) {
      sycl::local_accessor<Type, 1> SMem_acc(smem / sizeof(Type), cgh);
      cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                       [=](sycl::nd_item<1> item) {
                         computeAXKernelPoisson<Type, M, N, K, dim>(
                           item, V, U, P, J, map, SMem_acc.get_pointer());
                       });
    });
#endif
  }

  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  void
  matrixFreeDeviceKernels<Type, M, N, K, dim>::computeAXDeviceHelmholtz(
    const dftfe::Int  blocks,
    const dftfe::Int  threads,
    const dftfe::Int  smem,
    Type             *V,
    const Type       *U,
    const Type       *P,
    const Type       *J,
    const dftfe::Int *map,
    const Type        coeffHelmholtz)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
    DFTFE_LAUNCH_KERNEL(DFTFE_KERNEL_NAME(
                          computeAXKernelHelmholtz<Type, M, N, K, dim>),
                        blocks,
                        threads,
                        smem,
                        dftfe::utils::defaultStream,
                        V,
                        U,
                        P,
                        J,
                        map,
                        coeffHelmholtz);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    dftfe::utils::defaultStream.submit([=](sycl::handler &cgh) {
      sycl::local_accessor<Type, 1> SMem_acc(smem / sizeof(Type), cgh);
      cgh.parallel_for(
        sycl::nd_range<1>(blocks * threads, threads),
        [=](sycl::nd_item<1> item) {
          computeAXKernelHelmholtz<Type, M, N, K, dim>(
            item, V, U, P, J, map, coeffHelmholtz, SMem_acc.get_pointer());
        });
    });
#endif
  }
  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  void
  matrixFreeDeviceKernels<Type, M, N, K, dim>::
    computeAXDevicePoissonSetAttributes(const dftfe::Int smem)
  {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    cudaFuncSetAttribute(computeAXKernelPoisson<double, M, N, K, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
#endif
  }
  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  void
  matrixFreeDeviceKernels<Type, M, N, K, dim>::
    computeAXDeviceHelmholtzSetAttributes(const dftfe::Int smem)
  {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    cudaFuncSetAttribute(computeAXKernelHelmholtz<double, M, N, K, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
#endif
  }
  template class matrixFreeDeviceKernels<double, 4, 2, 2, 3>;
  template class matrixFreeDeviceKernels<double, 9, 3, 3, 3>;
  template class matrixFreeDeviceKernels<double, 16, 4, 4, 3>;
  template class matrixFreeDeviceKernels<double, 25, 5, 5, 3>;
  template class matrixFreeDeviceKernels<double, 36, 6, 6, 3>;
  template class matrixFreeDeviceKernels<double, 49, 7, 7, 3>;
  template class matrixFreeDeviceKernels<double, 64, 8, 8, 3>;
  template class matrixFreeDeviceKernels<double, 81, 9, 9, 3>;
  template class matrixFreeDeviceKernels<double, 100, 10, 10, 3>;
  template class matrixFreeDeviceKernels<double, 121, 11, 11, 3>;
  template class matrixFreeDeviceKernels<double, 144, 12, 12, 3>;
  template class matrixFreeDeviceKernels<double, 169, 13, 13, 3>;
  template class matrixFreeDeviceKernels<double, 196, 14, 14, 3>;
  template class matrixFreeDeviceKernels<double, 225, 15, 15, 3>;
  template class matrixFreeDeviceKernels<double, 256, 16, 16, 3>;
  template class matrixFreeDeviceKernels<double, 289, 17, 17, 3>;

} // namespace dftfe
