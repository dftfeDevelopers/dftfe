// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Gourab Panigrahi
//

#include <MatrixFree.h>

namespace dftfe
{
  template <int nDofsPerDim, int nQuadPointsPerDim>
  MatrixFree<nDofsPerDim, nQuadPointsPerDim>::MatrixFree(
    const MPI_Comm &mpi_comm,
    const int       nCells)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
    , d_nCells(nCells)
    , d_nDofsPerCell(nDofsPerDim * nDofsPerDim * nDofsPerDim)
    , d_nQuadsPerCell(nQuadPointsPerDim * nQuadPointsPerDim * nQuadPointsPerDim)
  {}


  template <typename Type, int M, int N, int K, int dim>
  __global__ void
  computeAXKernel(Type *      V,
                  const Type *U,
                  const Type *P,
                  const Type *J,
                  const int * map)
  {
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // P(q*p), D(q*q), PT(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[N * N * N];
    Type *sharedZ  = &sharedY[N * N * N];
    Type *sharedT  = &sharedZ[N * N * N];
    Type *sharedP  = &sharedT[N * N * N];
    Type *sharedD  = &sharedP[N * K];
    Type *sharedPT = &sharedD[N * N];
    Type *sharedDT = &sharedPT[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const int mapShift = blockIdx.x * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < 2 * N * (K + N); i += blockDim.x)
      sharedP[i] = P[i];

    __syncthreads();

    //////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // V -> UPPP
    // Z -> VDz
    // Y -> VDy
    // X -> VDx

    // 1st GEMM of P
    // Z Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type x[N], u[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            u[k] = U[map[i + k * M + mapShift]];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * u[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[i + j * M] = x[j];
      }

    __syncthreads();

    // 2nd GEMM of P
    // Y Direction
    for (int i = threadIdx.x; i < K * N; i += blockDim.x)
      {
        Type y[N], x[K];

        int a = i % K;
        int b = i / K;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            x[k] = sharedX[a + k * K + b * M];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedP[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[a + (j + b * N) * K] = y[j];
      }

    __syncthreads();

    // 3rd GEMM of P
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            y[k] = sharedY[k + i * K];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] = x[j];
      }

    __syncthreads();

    // 1st GEMM of D
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[i + j * N * N] = y[j];
      }

    // 2nd GEMM of D
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], x[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedZ[a + (j + b * N) * N] = z[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type t[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          t[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              t[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedT[j + i * N] = t[j];
      }

    //////////////////////////////////////////////////////////////////
    // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
    // directions

    const int JShift = blockIdx.x * dim * dim;

    // Copy Jacobian Factor to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
      sharedJ[i] = J[i + JShift];

    __syncthreads();

    // Gemm with Jacobian Factor
#pragma unroll
    for (int i = threadIdx.x; i < N * N * N; i += blockDim.x)
      {
        Type v[3];

        v[2] = sharedY[i];
        v[1] = sharedZ[i];
        v[0] = sharedT[i];

        sharedY[i] = sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
        sharedZ[i] = sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
        sharedT[i] = sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];
      }

    __syncthreads();

    // Integration
    // Z -> Z(DT)z
    // Y -> Y(DT)y
    // X -> X(DT)x
    // V -> (Z + Y + X)(PT)(PT)(PT)

    // 1st GEMM of DT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedDT[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[i + j * N * N] = x[j];
      }

    __syncthreads();

    // 2nd GEMM of DT
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], z[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            z[k] = sharedZ[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[a + (j + b * N) * N] += y[j];
      }

    __syncthreads();

    // 3rd GEMM of DT
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], t[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            t[k] = sharedT[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedDT[j + k * N] * t[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] += z[j];
      }

    __syncthreads();

    // 1st GEMM of PT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedY[i + j * N * N] = y[j];
      }

    __syncthreads();

    // 2nd GEMM of PT
    // Y Direction
    for (int i = threadIdx.x; i < N * K; i += blockDim.x)
      {
        Type x[K], y[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < K; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              x[j] += sharedPT[j + k * K] * y[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedX[a + (j + b * K) * N] = x[j];
      }

    __syncthreads();

    // 3rd GEMM of PT
    // X Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          atomicAdd(&V[map[j + i * K + mapShift]], y[j]);
      }
  }


  template <typename Type, int M, int N, int K, int dim>
  __global__ void
  computeAXKernelcoeff(Type *      V,
                       const Type *U,
                       const Type *P,
                       const Type *J,
                       const int * map,
                       const Type  coeffHelmholtz)
  {
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // P(q*p), D(q*q), PT(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[N * N * N];
    Type *sharedZ  = &sharedY[N * N * N];
    Type *sharedT  = &sharedZ[N * N * N];
    Type *sharedP  = &sharedT[N * N * N];
    Type *sharedD  = &sharedP[N * K];
    Type *sharedPT = &sharedD[N * N];
    Type *sharedDT = &sharedPT[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const int mapShift = blockIdx.x * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < 2 * N * (K + N); i += blockDim.x)
      sharedP[i] = P[i];

    __syncthreads();

    //////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // V -> UPPP
    // Z -> VDz
    // Y -> VDy
    // X -> VDx

    // 1st GEMM of P
    // Z Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type x[N], u[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            u[k] = U[map[i + k * M + mapShift]];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * u[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[i + j * M] = x[j];
      }

    __syncthreads();

    // 2nd GEMM of P
    // Y Direction
    for (int i = threadIdx.x; i < K * N; i += blockDim.x)
      {
        Type y[N], x[K];

        int a = i % K;
        int b = i / K;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            x[k] = sharedX[a + k * K + b * M];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedP[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[a + (j + b * N) * K] = y[j];
      }

    __syncthreads();

    // 3rd GEMM of P
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            y[k] = sharedY[k + i * K];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] = x[j];
      }

    __syncthreads();

    // 1st GEMM of D
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[i + j * N * N] = y[j];
      }

    // 2nd GEMM of D
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], x[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedZ[a + (j + b * N) * N] = z[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type t[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          t[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              t[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedT[j + i * N] = t[j];
      }

    //////////////////////////////////////////////////////////////////
    // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
    // directions

    const int JShift = blockIdx.x * dim * dim;

    // Copy Jacobian Factor to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
      sharedJ[i] = J[i + JShift];

    Type detJ;

    __syncthreads();

    // Gemm with Jacobian Factor
#pragma unroll
    for (int i = threadIdx.x; i < N * N * N; i += blockDim.x)
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

    __syncthreads();

    // Integration
    // Z -> Z(DT)z
    // Y -> Y(DT)y
    // X -> X(DT)x
    // V -> (Z + Y + X)(PT)(PT)(PT)

    // 1st GEMM of DT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[N], h[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedDT[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          {
            h[j]                   = sharedX[i + j * N * N];
            sharedX[i + j * N * N] = coeffHelmholtz * detJ * h[j] + x[j];
          }
      }

    __syncthreads();

    // 2nd GEMM of DT
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], z[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            z[k] = sharedZ[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[a + (j + b * N) * N] += y[j];
      }

    __syncthreads();

    // 3rd GEMM of DT
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], t[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            t[k] = sharedT[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedDT[j + k * N] * t[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] += z[j];
      }

    __syncthreads();

    // 1st GEMM of PT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedY[i + j * N * N] = y[j];
      }

    __syncthreads();

    // 2nd GEMM of PT
    // Y Direction
    for (int i = threadIdx.x; i < N * K; i += blockDim.x)
      {
        Type x[K], y[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < K; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              x[j] += sharedPT[j + k * K] * y[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedX[a + (j + b * K) * N] = x[j];
      }

    __syncthreads();

    // 3rd GEMM of PT
    // X Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          atomicAdd(&V[map[j + i * K + mapShift]], y[j]);
      }
  }


  template <int nDofsPerDim, int nQuadPointsPerDim>
  void
  MatrixFree<nDofsPerDim, nQuadPointsPerDim>::reinit(
    const dealii::MatrixFree<3, double> *matrixFreeDataPtr,
    const unsigned int &                 dofHandlerID,
    const int                            matrixFreeQuadratureID)
  {
    constexpr int p = nDofsPerDim;
    constexpr int q = nQuadPointsPerDim;

    auto dofInfo = matrixFreeDataPtr->get_dof_info(dofHandlerID);
    auto shapeData =
      matrixFreeDataPtr->get_shape_info(dofHandlerID, matrixFreeQuadratureID)
        .get_shape_data();
    auto mappingData =
      matrixFreeDataPtr->get_mapping_info().cell_data[matrixFreeQuadratureID];

    // Shape Function Values, Gradients and their Transposes
    // P(q*p), D(q*q), PT(p*q), DT(q*q)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      shapeFunction(2 * q * (p + q));

    for (int i = 0; i < p; i++)
      for (int j = 0; j < q; j++)
        {
#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          double value = shapeData.shape_values[j + i * q] *
                         std::sqrt(shapeData.quadrature.weight(j));
#else
          double value = shapeData.shape_values[j + i * q][0] *
                         std::sqrt(shapeData.quadrature.weight(j));
#endif
          shapeFunction[j + i * q]               = value;
          shapeFunction[i + j * p + q * (p + q)] = value;
        }

    for (int i = 0; i < q; i++)
      for (int j = 0; j < q; j++)
        {
#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          double grad = shapeData.shape_gradients_collocation[j + i * q] *
                        std::sqrt(shapeData.quadrature.weight(j)) /
                        std::sqrt(shapeData.quadrature.weight(i));
#else
          double grad = shapeData.shape_gradients_collocation[j + i * q][0] *
                        std::sqrt(shapeData.quadrature.weight(j)) /
                        std::sqrt(shapeData.quadrature.weight(i));
#endif
          shapeFunction[j + i * q + q * p]           = grad;
          shapeFunction[i + j * q + (2 * p + q) * q] = grad;
        }

    constexpr int dim = 3;

    // Jacobian
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      jacobianFactor(dim * dim * d_nCells);

    auto             cellOffsets    = mappingData.data_index_offsets;
    constexpr double coeffLaplacian = 1.0 / (4.0 * M_PI);

    for (int cellIdx = 0; cellIdx < d_nCells; cellIdx++)
      for (int k = 0; k < dim; k++)
        for (int i = 0; i < dim; i++)
          for (int j = 0; j < dim; j++)
            jacobianFactor[j + i * dim + cellIdx * dim * dim] +=
              coeffLaplacian *
              mappingData
                .JxW_values[cellOffsets[cellIdx / dofInfo.vectorization_length]]
                           [0] *
              mappingData
                .jacobians[0]
                          [cellOffsets[cellIdx / dofInfo.vectorization_length]]
                          [k][j][0] *
              mappingData
                .jacobians[0]
                          [cellOffsets[cellIdx / dofInfo.vectorization_length]]
                          [k][i][0];

    // Map making
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::HOST> map(
      d_nDofsPerCell * d_nCells);

    for (auto cellIdx = 0; cellIdx < d_nCells; ++cellIdx)
      std::memcpy(map.data() + cellIdx * d_nDofsPerCell,
                  ((dofInfo.row_starts[cellIdx].second ==
                    dofInfo.row_starts[cellIdx + 1].second) &&
                   (dofInfo.row_starts_plain_indices[cellIdx] ==
                    dealii::numbers::invalid_unsigned_int)) ?
                    dofInfo.dof_indices.data() +
                      dofInfo.row_starts[cellIdx].first :
                    dofInfo.plain_dof_indices.data() +
                      dofInfo.row_starts_plain_indices[cellIdx],
                  d_nDofsPerCell * sizeof(unsigned int));

    // Construct the device vectors
    d_shapeFunction.resize(shapeFunction.size());
    d_shapeFunction.copyFrom(shapeFunction);

    d_jacobianFactor.resize(jacobianFactor.size());
    d_jacobianFactor.copyFrom(jacobianFactor);

    d_map.resize(map.size());
    d_map.copyFrom(map);

    d_shapeFunctionPtr  = d_shapeFunction.data();
    d_jacobianFactorPtr = d_jacobianFactor.data();
    d_mapPtr            = d_map.data();

    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    cudaFuncSetAttribute(computeAXKernelcoeff<double, p * p, q, p, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
#endif
  }


  template <int nDofsPerDim, int nQuadPointsPerDim>
  void
  MatrixFree<nDofsPerDim, nQuadPointsPerDim>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x)
  {
    constexpr int         dim     = 3;
    constexpr int         p       = nDofsPerDim;
    constexpr int         q       = nQuadPointsPerDim;
    constexpr int         threads = 64;
    const int             blocks  = d_nCells;
    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeAXKernel<double, p * p, q, p, dim><<<blocks, threads, smem>>>(
      Ax.begin(), x.begin(), d_shapeFunctionPtr, d_jacobianFactorPtr, d_mapPtr);

#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                         computeAXKernel<double, p * p, q, p, dim>),
                       blocks,
                       threads,
                       smem,
                       0,
                       Ax.begin(),
                       x.begin(),
                       d_shapeFunctionPtr,
                       d_jacobianFactorPtr,
                       d_mapPtr);
#endif
  }


  template <int nDofsPerDim, int nQuadPointsPerDim>
  void
  MatrixFree<nDofsPerDim, nQuadPointsPerDim>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x,
    double                        coeffHelmholtz)
  {
    constexpr int         dim     = 3;
    constexpr int         p       = nDofsPerDim;
    constexpr int         q       = nQuadPointsPerDim;
    constexpr int         threads = 64;
    const int             blocks  = d_nCells;
    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeAXKernelcoeff<double, p * p, q, p, dim>
      <<<blocks, threads, smem>>>(Ax.begin(),
                                  x.begin(),
                                  d_shapeFunctionPtr,
                                  d_jacobianFactorPtr,
                                  d_mapPtr,
                                  coeffHelmholtz);

#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                         computeAXKernelcoeff<double, p * p, q, p, dim>),
                       blocks,
                       threads,
                       smem,
                       0,
                       Ax.begin(),
                       x.begin(),
                       d_shapeFunctionPtr,
                       d_jacobianFactorPtr,
                       d_mapPtr,
                       coeffHelmholtz);
#endif
  }

#include "MatrixFree.inst.cc"
} // namespace dftfe
