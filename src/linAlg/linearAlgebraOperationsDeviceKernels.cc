
#include "linearAlgebraOperationsDeviceKernels.h"

namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
    namespace
    {
      __global__ void
      addSubspaceRotatedBlockToXKernel(const unsigned int BDof,
                                       const unsigned int BVec,
                                       const float       *rotatedXBlockSP,
                                       double            *X,
                                       const unsigned int startingDofId,
                                       const unsigned int startingVecId,
                                       const unsigned int N)
      {
        const unsigned int numEntries = BVec * BDof;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int ibdof = i / BVec;
            const unsigned int ivec  = i % BVec;

            *(X + N * (startingDofId + ibdof) + startingVecId + ivec) +=
              rotatedXBlockSP[ibdof * BVec + ivec];
          }
      }

      __global__ void
      addSubspaceRotatedBlockToXKernel(
        const unsigned int                      BDof,
        const unsigned int                      BVec,
        const dftfe::utils::deviceFloatComplex *rotatedXBlockSP,
        dftfe::utils::deviceDoubleComplex      *X,
        const unsigned int                      startingDofId,
        const unsigned int                      startingVecId,
        const unsigned int                      N)
      {
        const unsigned int numEntries = BVec * BDof;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int ibdof = i / BVec;
            const unsigned int ivec  = i % BVec;

            *(X + N * (startingDofId + ibdof) + startingVecId + ivec) =
              dftfe::utils::add(*(X + N * (startingDofId + ibdof) +
                                  startingVecId + ivec),
                                rotatedXBlockSP[ibdof * BVec + ivec]);
          }
      }


      __global__ void
      copyFromOverlapMatBlockToDPSPBlocksKernel(
        const unsigned int B,
        const unsigned int D,
        const double      *overlapMatrixBlock,
        double            *overlapMatrixBlockDP,
        float             *overlapMatrixBlockSP)
      {
        const unsigned int numEntries = B * D;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int ibdof = i / D;
            const unsigned int ivec  = i % D;

            if (ivec < B)
              overlapMatrixBlockDP[ibdof * B + ivec] = overlapMatrixBlock[i];
            else
              overlapMatrixBlockSP[ibdof * (D - B) + (ivec - B)] =
                overlapMatrixBlock[i];
          }
      }


      __global__ void
      copyFromOverlapMatBlockToDPSPBlocksKernel(
        const unsigned int                       B,
        const unsigned int                       D,
        const dftfe::utils::deviceDoubleComplex *overlapMatrixBlock,
        dftfe::utils::deviceDoubleComplex       *overlapMatrixBlockDP,
        dftfe::utils::deviceFloatComplex        *overlapMatrixBlockSP)
      {
        const unsigned int numEntries = B * D;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int ibdof = i / D;
            const unsigned int ivec  = i % D;

            if (ivec < B)
              dftfe::utils::copyValue(overlapMatrixBlockDP + ibdof * B + ivec,
                                      overlapMatrixBlock[i]);
            else
              dftfe::utils::copyValue(overlapMatrixBlockSP + ibdof * (D - B) +
                                        (ivec - B),
                                      overlapMatrixBlock[i]);
          }
      }

      __global__ void
      computeDiagQTimesXKernel(const double      *diagValues,
                               double            *X,
                               const unsigned int N,
                               const unsigned int M)
      {
        const unsigned int numEntries = N * M;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int idof = i / N;
            const unsigned int ivec = i % N;

            *(X + N * idof + ivec) = *(X + N * idof + ivec) * diagValues[ivec];
          }
      }


      __global__ void
      computeDiagQTimesXKernel(
        const dftfe::utils::deviceDoubleComplex *diagValues,
        dftfe::utils::deviceDoubleComplex       *X,
        const unsigned int                       N,
        const unsigned int                       M)
      {
        const unsigned int numEntries = N * M;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int idof = i / N;
            const unsigned int ivec = i % N;

            *(X + N * idof + ivec) =
              dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
          }
      }


      __global__ void
      computeDiagQTimesXKernel(const double                      *diagValues,
                               dftfe::utils::deviceDoubleComplex *X,
                               const unsigned int                 N,
                               const unsigned int                 M)
      {
        const unsigned int numEntries = N * M;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int idof = i / N;
            const unsigned int ivec = i % N;

            *(X + N * idof + ivec) =
              dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernel(const unsigned int numVectors,
                                  const unsigned int numDofs,
                                  const unsigned int N,
                                  const unsigned int startingVecId,
                                  const double      *eigenValues,
                                  const double      *x,
                                  const double      *y,
                                  double            *r)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int dofIndex  = i / numVectors;
            const unsigned int waveIndex = i % numVectors;
            r[i] = y[i] - x[dofIndex * N + startingVecId + waveIndex] *
                            eigenValues[startingVecId + waveIndex];
            r[i] = r[i] * r[i];
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernel(const unsigned int numVectors,
                                  const unsigned int numDofs,
                                  const unsigned int N,
                                  const unsigned int startingVecId,
                                  const double      *eigenValues,
                                  const dftfe::utils::deviceDoubleComplex *X,
                                  const dftfe::utils::deviceDoubleComplex *Y,
                                  double                                  *r)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int                      dofIndex  = i / numVectors;
            const unsigned int                      waveIndex = i % numVectors;
            const dftfe::utils::deviceDoubleComplex diff =
              dftfe::utils::makeComplex(
                Y[i].x - X[dofIndex * N + startingVecId + waveIndex].x *
                           eigenValues[startingVecId + waveIndex],
                Y[i].y - X[dofIndex * N + startingVecId + waveIndex].y *
                           eigenValues[startingVecId + waveIndex]);
            r[i] = diff.x * diff.x + diff.y * diff.y;
          }
      }

      __global__ void
      setZeroKernel(const unsigned int BVec,
                    const unsigned int M,
                    const unsigned int N,
                    double            *yVec,
                    const unsigned int startingXVecId)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const unsigned int gangBlockId = blockIdx.x / numGangsPerBVec;
        const unsigned int localThreadId =
          globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

        if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
            localThreadId < BVec)
          {
            *(yVec + gangBlockId * N + startingXVecId + localThreadId) = 0.0;
          }
      }


      __global__ void
      setZeroKernel(const unsigned int                 BVec,
                    const unsigned int                 M,
                    const unsigned int                 N,
                    dftfe::utils::deviceDoubleComplex *yVec,
                    const unsigned int                 startingXVecId)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const unsigned int gangBlockId = blockIdx.x / numGangsPerBVec;
        const unsigned int localThreadId =
          globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

        if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
            localThreadId < BVec)
          {
            *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
              dftfe::utils::makeComplex(0.0, 0.0);
          }
      }



      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernelGeneralised(const unsigned int numVectors,
                                             const unsigned int numDofs,
                                             const unsigned int N,
                                             const unsigned int startingVecId,
                                             const double      *y,
                                             double            *r)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int dofIndex  = i / numVectors;
            const unsigned int waveIndex = i % numVectors;
            r[i]                         = y[i] * y[i];
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernelGeneralised(
        const unsigned int                       numVectors,
        const unsigned int                       numDofs,
        const unsigned int                       N,
        const unsigned int                       startingVecId,
        const dftfe::utils::deviceDoubleComplex *Y,
        double                                  *r)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int dofIndex  = i / numVectors;
            const unsigned int waveIndex = i % numVectors;
            r[i]                         = Y[i].x * Y[i].x + Y[i].y * Y[i].y;
          }
      }



    } // namespace

    template <typename ValueType1, typename ValueType2>
    void
    addSubspaceRotatedBlockToX(const unsigned int            BDof,
                               const unsigned int            BVec,
                               const ValueType1             *rotatedXBlockSP,
                               ValueType2                   *X,
                               const unsigned int            startingDofId,
                               const unsigned int            startingVecId,
                               const unsigned int            N,
                               dftfe::utils::deviceStream_t &streamCompute)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      addSubspaceRotatedBlockToXKernel<<<
        (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        streamCompute>>>(BDof,
                         BVec,
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           rotatedXBlockSP),
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         startingDofId,
                         startingVecId,
                         N);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(addSubspaceRotatedBlockToXKernel,
                         (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         streamCompute,
                         BDof,
                         BVec,
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           rotatedXBlockSP),
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         startingDofId,
                         startingVecId,
                         N);
#endif
    }
    template <typename ValueType1, typename ValueType2>
    void
    copyFromOverlapMatBlockToDPSPBlocks(
      const unsigned int            B,
      const unsigned int            D,
      const ValueType1             *overlapMatrixBlock,
      ValueType1                   *overlapMatrixBlockDP,
      ValueType2                   *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      copyFromOverlapMatBlockToDPSPBlocksKernel<<<
        (D * B + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        streamDataMove>>>(
        B,
        D,
        dftfe::utils::makeDataTypeDeviceCompatible(overlapMatrixBlock),
        dftfe::utils::makeDataTypeDeviceCompatible(overlapMatrixBlockDP),
        dftfe::utils::makeDataTypeDeviceCompatible(overlapMatrixBlockSP));
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        copyFromOverlapMatBlockToDPSPBlocksKernel,
        (D * B + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        streamDataMove,
        B,
        D,
        dftfe::utils::makeDataTypeDeviceCompatible(overlapMatrixBlock),
        dftfe::utils::makeDataTypeDeviceCompatible(overlapMatrixBlockDP),
        dftfe::utils::makeDataTypeDeviceCompatible(overlapMatrixBlockSP));
#endif
    }
    template <typename ValueType1, typename ValueType2>
    void
    computeDiagQTimesX(const ValueType1  *diagValues,
                       ValueType2        *X,
                       const unsigned int N,
                       const unsigned int M)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeDiagQTimesXKernel<<<(M * N +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE,
                                 dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(diagValues),
        dftfe::utils::makeDataTypeDeviceCompatible(X),
        N,
        M);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(computeDiagQTimesXKernel,
                         (M * N + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         dftfe::utils::makeDataTypeDeviceCompatible(diagValues),
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         N,
                         M);
#endif
    }

    template <typename ValueType>
    void
    computeResidualDevice(const unsigned int numVectors,
                          const unsigned int numDofs,
                          const unsigned int N,
                          const unsigned int startingVecId,
                          const double      *eigenValues,
                          const ValueType   *X,
                          const ValueType   *Y,
                          double            *r)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeResidualDeviceKernel<<<(numVectors +
                                     (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                      dftfe::utils::DEVICE_BLOCK_SIZE * numDofs,
                                    dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numVectors,
        numDofs,
        N,
        startingVecId,
        eigenValues,
        dftfe::utils::makeDataTypeDeviceCompatible(X),
        dftfe::utils::makeDataTypeDeviceCompatible(Y),
        r);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(computeResidualDeviceKernel,
                         (numVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE * numDofs,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         numVectors,
                         numDofs,
                         N,
                         startingVecId,
                         eigenValues,
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         dftfe::utils::makeDataTypeDeviceCompatible(Y),
                         r);
#endif
    }

    template <typename ValueType>
    void
    computeGeneralisedResidualDevice(const unsigned int numVectors,
                                     const unsigned int numDofs,
                                     const unsigned int N,
                                     const unsigned int startingVecId,
                                     const ValueType   *X,
                                     double            *residualSqDevice)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeResidualDeviceKernelGeneralised<<<
        (numVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * numDofs,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numVectors,
        numDofs,
        N,
        startingVecId,
        dftfe::utils::makeDataTypeDeviceCompatible(X),
        residualSqDevice);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(computeResidualDeviceKernelGeneralised,
                         (numVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE * numDofs,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         numVectors,
                         numDofs,
                         N,
                         startingVecId,
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         residualSqDevice);
#endif
    }



    template <typename ValueType>
    void
    setZero(const unsigned int BVec,
            const unsigned int M,
            const unsigned int N,
            ValueType         *yVec,
            const unsigned int startingXVecId)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      setZeroKernel<<<(BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                        dftfe::utils::DEVICE_BLOCK_SIZE * M,
                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        BVec,
        M,
        N,
        dftfe::utils::makeDataTypeDeviceCompatible(yVec),
        startingXVecId);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(setZeroKernel,
                         (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE * M,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         BVec,
                         M,
                         N,
                         dftfe::utils::makeDataTypeDeviceCompatible(yVec),
                         startingXVecId);
#endif
    }


    template void
    addSubspaceRotatedBlockToX(const unsigned int            BDof,
                               const unsigned int            BVec,
                               const float                  *rotatedXBlockSP,
                               double                       *X,
                               const unsigned int            startingDofId,
                               const unsigned int            startingVecId,
                               const unsigned int            N,
                               dftfe::utils::deviceStream_t &streamCompute);
    template void
    addSubspaceRotatedBlockToX(const unsigned int            BDof,
                               const unsigned int            BVec,
                               const std::complex<float>    *rotatedXBlockSP,
                               std::complex<double>         *X,
                               const unsigned int            startingDofId,
                               const unsigned int            startingVecId,
                               const unsigned int            N,
                               dftfe::utils::deviceStream_t &streamCompute);
    template void
    copyFromOverlapMatBlockToDPSPBlocks(
      const unsigned int            B,
      const unsigned int            D,
      const double                 *overlapMatrixBlock,
      double                       *overlapMatrixBlockDP,
      float                        *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove);
    template void
    copyFromOverlapMatBlockToDPSPBlocks(
      const unsigned int            B,
      const unsigned int            D,
      const std::complex<double>   *overlapMatrixBlock,
      std::complex<double>         *overlapMatrixBlockDP,
      std::complex<float>          *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove);

    template void
    computeDiagQTimesX(const double      *diagValues,
                       double            *X,
                       const unsigned int N,
                       const unsigned int M);
    template void
    computeDiagQTimesX(const std::complex<double> *diagValues,
                       std::complex<double>       *X,
                       const unsigned int          N,
                       const unsigned int          M);

    template void
    computeDiagQTimesX(const double         *diagValues,
                       std::complex<double> *X,
                       const unsigned int    N,
                       const unsigned int    M);

    template void
    computeResidualDevice(const unsigned int numVectors,
                          const unsigned int numDofs,
                          const unsigned int N,
                          const unsigned int startingVecId,
                          const double      *eigenValues,
                          const double      *X,
                          const double      *Y,
                          double            *r);
    template void
    computeResidualDevice(const unsigned int          numVectors,
                          const unsigned int          numDofs,
                          const unsigned int          N,
                          const unsigned int          startingVecId,
                          const double               *eigenValues,
                          const std::complex<double> *X,
                          const std::complex<double> *Y,
                          double                     *r);

    template void
    computeGeneralisedResidualDevice(const unsigned int numVectors,
                                     const unsigned int numDofs,
                                     const unsigned int N,
                                     const unsigned int startingVecId,
                                     const double      *X,
                                     double            *residualSqDevice);

    template void
    computeGeneralisedResidualDevice(const unsigned int          numVectors,
                                     const unsigned int          numDofs,
                                     const unsigned int          N,
                                     const unsigned int          startingVecId,
                                     const std::complex<double> *X,
                                     double *residualSqDevice);

    template void
    setZero(const unsigned int BVec,
            const unsigned int M,
            const unsigned int N,
            double            *yVec,
            const unsigned int startingXVecId);
    template void
    setZero(const unsigned int    BVec,
            const unsigned int    M,
            const unsigned int    N,
            std::complex<double> *yVec,
            const unsigned int    startingXVecId);

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
