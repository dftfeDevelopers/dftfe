#include "matrixFreeDeviceKernels.h"

namespace dftfe
{
  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  DFTFE_CREATE_KERNEL_SMEM_D(
    Type,
    void,
    computeAXKernelPoisson,
    DFTFE_KERNEL_ARGUMENT({
      // V = AU
      // gridDim.x = cells;
      // First index is fastest convention used
      // sharedT is used to temporarily store UP^T/UP
      // P(q*p), D(q*q), PT(p*q), DT(q*q)
      Type *sharedX  = smem;
      Type *sharedY  = &sharedX[N * N * N];
      Type *sharedZ  = &sharedY[N * N * N];
      Type *sharedT  = &sharedZ[N * N * N];
      Type *sharedP  = &sharedT[N * N * N];
      Type *sharedD  = &sharedP[N * K];
      Type *sharedPT = &sharedD[N * N];
      Type *sharedDT = &sharedPT[K * N];
      Type *sharedJ  = &sharedDT[N * N];

      const dftfe::Int mapShift = blockId * M * K;

      // Copy Shape Function Values and Gradients to shared memory
      _Pragma("unroll") for (dftfe::Int i = threadId; i < 2 * N * (K + N);
                             i += nThreadsPerBlock) sharedP[i] = P[i];

      SYNCTHREADS;

      //////////////////////////////////////////////////////////////
      // Interpolation combined with Extraction
      // V -> UPPP
      // Z -> VDz
      // Y -> VDy
      // X -> VDx

      // 1st GEMM of P
      // Z Direction
      for (dftfe::Int i = threadId; i < M; i += nThreadsPerBlock)
        {
          Type x[N], u[K];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < K; k++)
            {
              u[k] = U[map[i + k * M + mapShift]];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] +=
                sharedP[j + k * N] * u[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[i + j * M] = x[j];
        }

      SYNCTHREADS;

      // 2nd GEMM of P
      // Y Direction
      for (dftfe::Int i = threadId; i < K * N; i += nThreadsPerBlock)
        {
          Type y[N], x[K];

          dftfe::Int a = i % K;
          dftfe::Int b = i / K;

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < K; k++)
            {
              x[k] = sharedX[a + k * K + b * M];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] +=
                sharedP[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedY[a + (j + b * N) * K] = y[j];
        }

      SYNCTHREADS;

      // 3rd GEMM of P
      // X Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type x[N], y[K];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < K; k++)
            {
              y[k] = sharedY[k + i * K];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] +=
                sharedP[j + k * N] * y[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[j + i * N] = x[j];
        }

      SYNCTHREADS;

      // 1st GEMM of D
      // Z Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type y[N], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[i + k * N * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] +=
                sharedD[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedY[i + j * N * N] = y[j];
        }

      SYNCTHREADS;

      for (dftfe::Int i = threadId; i < M; i += nThreadsPerBlock)
        {
          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            dftfe::utils::atomicAddWrapper(&V[map[i + j * M + mapShift]],
                                           sharedY[i + j * M]);
        }

      // 2nd GEMM of D
      // Y Direction
      /*for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type z[N], x[N];

          dftfe::Int a = i % N;
          dftfe::Int b = i / N;

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[a + (k + b * N) * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] +=
                sharedD[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedZ[a + (j + b * N) * N] = z[j];
        }

      // 3rd GEMM of D
      // X Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type t[N], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) t[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[k + i * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) t[j] +=
                sharedD[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedT[j + i * N] = t[j];
        }

      //////////////////////////////////////////////////////////////////
      // sharedT, sharedZ, sharedY have the respective
      gemms of X, Y,
        Z
        // directions

        const dftfe::Int JShift = blockId * dim * dim;

      // Copy Jacobian Factor to shared memory
      _Pragma("unroll") for (dftfe::Int i = threadId; i < dim * dim;
                             i += nThreadsPerBlock) sharedJ[i] = J[i + JShift];

      SYNCTHREADS;

      // Gemm with Jacobian Factor
      _Pragma("unroll") for (dftfe::Int i = threadId; i < N * N * N;
                             i += nThreadsPerBlock)
      {
        Type v[3];

        v[2] = sharedY[i];
        v[1] = sharedZ[i];
        v[0] = sharedT[i];

        sharedY[i] = sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
        sharedZ[i] = sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
        sharedT[i] = sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];
      }

      SYNCTHREADS;

      // Integration
      // Z -> Z(DT)z
      // Y -> Y(DT)y
      // X -> X(DT)x
      // V -> (Z + Y + X)(PT)(PT)(PT)

      // 1st GEMM of DT
      // Z Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type x[N], y[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              y[k] = sharedY[i + k * N * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] +=
                sharedDT[j + k * N] * y[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[i + j * N * N] = x[j];
        }

      SYNCTHREADS;

      // 2nd GEMM of DT
      // Y Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type y[N], z[N];

          dftfe::Int a = i % N;
          dftfe::Int b = i / N;

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              z[k] = sharedZ[a + (k + b * N) * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] +=
                sharedDT[j + k * N] * z[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[a + (j + b * N) * N] += y[j];
        }

      SYNCTHREADS;

      // 3rd GEMM of DT
      // X Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type z[N], t[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              t[k] = sharedT[k + i * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] +=
                sharedDT[j + k * N] * t[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[j + i * N] += z[j];
        }

      SYNCTHREADS;

      // 1st GEMM of PT
      // Z Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type y[K], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[i + k * N * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] +=
                sharedPT[j + k * K] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            sharedY[i + j * N * N] = y[j];
        }

      SYNCTHREADS;

      // 2nd GEMM of PT
      // Y Direction
      for (dftfe::Int i = threadId; i < N * K; i += nThreadsPerBlock)
        {
          Type x[K], y[N];

          dftfe::Int a = i % N;
          dftfe::Int b = i / N;

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              y[k] = sharedY[a + (k + b * N) * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) x[j] +=
                sharedPT[j + k * K] * y[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            sharedX[a + (j + b * K) * N] = x[j];
        }

      SYNCTHREADS;

      // 3rd GEMM of PT
      // X Direction
      for (dftfe::Int i = threadId; i < M; i += nThreadsPerBlock)
        {
          Type y[K], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[k + i * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] +=
                sharedPT[j + k * K] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            dftfe::utils::atomicAddWrapper(&V[map[j + i * K + mapShift]], y[j]);
        } //*/
    }),
    Type             *V,
    const Type       *U,
    const Type       *P,
    const Type       *J,
    const dftfe::Int *map);


  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  DFTFE_CREATE_KERNEL_SMEM_D(
    Type,
    void,
    computeAXKernelHelmholtz,
    DFTFE_KERNEL_ARGUMENT({
      // V = AU
      // gridDim.x = cells;
      // First index is fastest convention used
      // sharedT is used to temporarily store UP^T/UP
      // P(q*p), D(q*q), PT(p*q), DT(q*q)

      Type *sharedX  = smem;
      Type *sharedY  = &sharedX[N * N * N];
      Type *sharedZ  = &sharedY[N * N * N];
      Type *sharedT  = &sharedZ[N * N * N];
      Type *sharedP  = &sharedT[N * N * N];
      Type *sharedD  = &sharedP[N * K];
      Type *sharedPT = &sharedD[N * N];
      Type *sharedDT = &sharedPT[K * N];
      Type *sharedJ  = &sharedDT[N * N];

      const dftfe::Int mapShift = blockId * M * K;

      // Copy Shape Function Values and Gradients to shared memory
      _Pragma("unroll") for (dftfe::Int i = threadId; i < 2 * N * (K + N);
                             i += nThreadsPerBlock) sharedP[i] = P[i];

      SYNCTHREADS;

      //////////////////////////////////////////////////////////////
      // Interpolation combined with Extraction
      // V -> UPPP
      // Z -> VDz
      // Y -> VDy
      // X -> VDx

      // 1st GEMM of P
      // Z Direction
      for (dftfe::Int i = threadId; i < M; i += nThreadsPerBlock)
        {
          Type x[N], u[K];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < K; k++)
            {
              u[k] = U[map[i + k * M + mapShift]];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] +=
                sharedP[j + k * N] * u[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[i + j * M] = x[j];
        }

      SYNCTHREADS;

      // 2nd GEMM of P
      // Y Direction
      for (dftfe::Int i = threadId; i < K * N; i += nThreadsPerBlock)
        {
          Type y[N], x[K];

          dftfe::Int a = i % K;
          dftfe::Int b = i / K;

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < K; k++)
            {
              x[k] = sharedX[a + k * K + b * M];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] +=
                sharedP[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedY[a + (j + b * N) * K] = y[j];
        }

      SYNCTHREADS;

      // 3rd GEMM of P
      // X Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type x[N], y[K];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < K; k++)
            {
              y[k] = sharedY[k + i * K];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] +=
                sharedP[j + k * N] * y[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[j + i * N] = x[j];
        }

      SYNCTHREADS;

      // 1st GEMM of D
      // Z Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type y[N], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[i + k * N * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] +=
                sharedD[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedY[i + j * N * N] = y[j];
        }

      // 2nd GEMM of D
      // Y Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type z[N], x[N];

          dftfe::Int a = i % N;
          dftfe::Int b = i / N;

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[a + (k + b * N) * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] +=
                sharedD[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedZ[a + (j + b * N) * N] = z[j];
        }

      // 3rd GEMM of D
      // X Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type t[N], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) t[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[k + i * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) t[j] +=
                sharedD[j + k * N] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedT[j + i * N] = t[j];
        }

      //////////////////////////////////////////////////////////////////
      // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
      // directions

      const dftfe::Int JShift = blockId * dim * dim;

      // Copy Jacobian Factor to shared memory
      _Pragma("unroll") for (dftfe::Int i = threadId; i < dim * dim;
                             i += nThreadsPerBlock) sharedJ[i] = J[i + JShift];

      Type detJ;

      SYNCTHREADS;
      // Gemm with Jacobian Factor
      _Pragma("unroll") for (dftfe::Int i = threadId; i < N * N * N;
                             i += nThreadsPerBlock)
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

      SYNCTHREADS;

      // Integration
      // Z -> Z(DT)z
      // Y -> Y(DT)y
      // X -> X(DT)x
      // V -> (Z + Y + X)(PT)(PT)(PT)

      // 1st GEMM of DT
      // Z Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type x[N], y[N], h[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              y[k] = sharedY[i + k * N * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) x[j] +=
                sharedDT[j + k * N] * y[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
          {
            h[j]                   = sharedX[i + j * N * N];
            sharedX[i + j * N * N] = coeffHelmholtz * detJ * h[j] + x[j];
          }
        }

      SYNCTHREADS;

      // 2nd GEMM of DT
      // Y Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type y[N], z[N];

          dftfe::Int a = i % N;
          dftfe::Int b = i / N;

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              z[k] = sharedZ[a + (k + b * N) * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) y[j] +=
                sharedDT[j + k * N] * z[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[a + (j + b * N) * N] += y[j];
        }

      SYNCTHREADS;

      // 3rd GEMM of DT
      // X Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type z[N], t[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              t[k] = sharedT[k + i * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++) z[j] +=
                sharedDT[j + k * N] * t[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < N; j++)
            sharedX[j + i * N] += z[j];
        }

      SYNCTHREADS;

      // 1st GEMM of PT
      // Z Direction
      for (dftfe::Int i = threadId; i < N * N; i += nThreadsPerBlock)
        {
          Type y[K], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[i + k * N * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] +=
                sharedPT[j + k * K] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            sharedY[i + j * N * N] = y[j];
        }

      SYNCTHREADS;

      // 2nd GEMM of PT
      // Y Direction
      for (dftfe::Int i = threadId; i < N * K; i += nThreadsPerBlock)
        {
          Type x[K], y[N];

          dftfe::Int a = i % N;
          dftfe::Int b = i / N;

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) x[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              y[k] = sharedY[a + (k + b * N) * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) x[j] +=
                sharedPT[j + k * K] * y[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            sharedX[a + (j + b * K) * N] = x[j];
        }

      SYNCTHREADS;

      // 3rd GEMM of PT
      // X Direction
      for (dftfe::Int i = threadId; i < M; i += nThreadsPerBlock)
        {
          Type y[K], x[N];

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] = 0.0;

          for (dftfe::Int k = 0; k < N; k++)
            {
              x[k] = sharedX[k + i * N];

              _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++) y[j] +=
                sharedPT[j + k * K] * x[k];
            }

          _Pragma("unroll") for (dftfe::Int j = 0; j < K; j++)
            dftfe::utils::atomicAddWrapper(&V[map[j + i * K + mapShift]], y[j]);
        }
    }),
    Type             *V,
    const Type       *U,
    const Type       *P,
    const Type       *J,
    const dftfe::Int *map,
    const Type        coeffHelmholtz);

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
    DFTFE_LAUNCH_KERNEL_SMEM_D(DFTFE_KERNEL_ARGUMENT(
                                 computeAXKernelPoisson<Type, M, N, K, dim>),
                               blocks,
                               threads,
                               Type,
                               smem / sizeof(Type),
                               dftfe::utils::defaultStream,
                               V,
                               U,
                               P,
                               J,
                               map);
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
    DFTFE_LAUNCH_KERNEL_SMEM_D(DFTFE_KERNEL_ARGUMENT(
                                 computeAXKernelHelmholtz<Type, M, N, K, dim>),
                               blocks,
                               threads,
                               Type,
                               smem / sizeof(Type),
                               dftfe::utils::defaultStream,
                               V,
                               U,
                               P,
                               J,
                               map,
                               coeffHelmholtz);
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
