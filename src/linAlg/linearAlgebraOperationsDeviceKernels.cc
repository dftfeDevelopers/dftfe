
#include "linearAlgebraOperationsDeviceKernels.h"

namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
    namespace
    {
      __global__ void
      addSubspaceRotatedBlockToXKernel(const dftfe::uInt BDof,
                                       const dftfe::uInt BVec,
                                       const float      *rotatedXBlockSP,
                                       double           *X,
                                       const dftfe::uInt startingDofId,
                                       const dftfe::uInt startingVecId,
                                       const dftfe::uInt N)
      {
        const dftfe::uInt numEntries = BVec * BDof;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt ibdof = i / BVec;
            const dftfe::uInt ivec  = i % BVec;

            *(X + N * (startingDofId + ibdof) + startingVecId + ivec) +=
              rotatedXBlockSP[ibdof * BVec + ivec];
          }
      }

      __global__ void
      addSubspaceRotatedBlockToXKernel(
        const dftfe::uInt                       BDof,
        const dftfe::uInt                       BVec,
        const dftfe::utils::deviceFloatComplex *rotatedXBlockSP,
        dftfe::utils::deviceDoubleComplex      *X,
        const dftfe::uInt                       startingDofId,
        const dftfe::uInt                       startingVecId,
        const dftfe::uInt                       N)
      {
        const dftfe::uInt numEntries = BVec * BDof;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt ibdof = i / BVec;
            const dftfe::uInt ivec  = i % BVec;

            *(X + N * (startingDofId + ibdof) + startingVecId + ivec) =
              dftfe::utils::add(*(X + N * (startingDofId + ibdof) +
                                  startingVecId + ivec),
                                rotatedXBlockSP[ibdof * BVec + ivec]);
          }
      }


      __global__ void
      copyFromOverlapMatBlockToDPSPBlocksKernel(
        const dftfe::uInt B,
        const dftfe::uInt D,
        const double     *overlapMatrixBlock,
        double           *overlapMatrixBlockDP,
        float            *overlapMatrixBlockSP)
      {
        const dftfe::uInt numEntries = B * D;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt ibdof = i / D;
            const dftfe::uInt ivec  = i % D;

            if (ivec < B)
              overlapMatrixBlockDP[ibdof * B + ivec] = overlapMatrixBlock[i];
            else
              overlapMatrixBlockSP[ibdof * (D - B) + (ivec - B)] =
                overlapMatrixBlock[i];
          }
      }


      __global__ void
      copyFromOverlapMatBlockToDPSPBlocksKernel(
        const dftfe::uInt                        B,
        const dftfe::uInt                        D,
        const dftfe::utils::deviceDoubleComplex *overlapMatrixBlock,
        dftfe::utils::deviceDoubleComplex       *overlapMatrixBlockDP,
        dftfe::utils::deviceFloatComplex        *overlapMatrixBlockSP)
      {
        const dftfe::uInt numEntries = B * D;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt ibdof = i / D;
            const dftfe::uInt ivec  = i % D;

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
      computeDiagQTimesXKernel(const double     *diagValues,
                               double           *X,
                               const dftfe::uInt N,
                               const dftfe::uInt M)
      {
        const dftfe::uInt numEntries = N * M;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt idof = i / N;
            const dftfe::uInt ivec = i % N;

            *(X + N * idof + ivec) = *(X + N * idof + ivec) * diagValues[ivec];
          }
      }


      __global__ void
      computeDiagQTimesXKernel(
        const dftfe::utils::deviceDoubleComplex *diagValues,
        dftfe::utils::deviceDoubleComplex       *X,
        const dftfe::uInt                        N,
        const dftfe::uInt                        M)
      {
        const dftfe::uInt numEntries = N * M;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt idof = i / N;
            const dftfe::uInt ivec = i % N;

            *(X + N * idof + ivec) =
              dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
          }
      }


      __global__ void
      computeDiagQTimesXKernel(const double                      *diagValues,
                               dftfe::utils::deviceDoubleComplex *X,
                               const dftfe::uInt                  N,
                               const dftfe::uInt                  M)
      {
        const dftfe::uInt numEntries = N * M;
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt idof = i / N;
            const dftfe::uInt ivec = i % N;

            *(X + N * idof + ivec) =
              dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernel(const dftfe::uInt numVectors,
                                  const dftfe::uInt numDofs,
                                  const dftfe::uInt N,
                                  const dftfe::uInt startingVecId,
                                  const double     *eigenValues,
                                  const double     *x,
                                  const double     *y,
                                  double           *r)
      {
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt dofIndex  = i / numVectors;
            const dftfe::uInt waveIndex = i % numVectors;
            r[i] = y[i] - x[dofIndex * N + startingVecId + waveIndex] *
                            eigenValues[startingVecId + waveIndex];
            r[i] = r[i] * r[i];
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernel(const dftfe::uInt numVectors,
                                  const dftfe::uInt numDofs,
                                  const dftfe::uInt N,
                                  const dftfe::uInt startingVecId,
                                  const double     *eigenValues,
                                  const dftfe::utils::deviceDoubleComplex *X,
                                  const dftfe::utils::deviceDoubleComplex *Y,
                                  double                                  *r)
      {
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt                       dofIndex  = i / numVectors;
            const dftfe::uInt                       waveIndex = i % numVectors;
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
      setZeroKernel(const dftfe::uInt BVec,
                    const dftfe::uInt M,
                    const dftfe::uInt N,
                    double           *yVec,
                    const dftfe::uInt startingXVecId)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const dftfe::uInt gangBlockId = blockIdx.x / numGangsPerBVec;
        const dftfe::uInt localThreadId =
          globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

        if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
            localThreadId < BVec)
          {
            *(yVec + gangBlockId * N + startingXVecId + localThreadId) = 0.0;
          }
      }


      __global__ void
      setZeroKernel(const dftfe::uInt                  BVec,
                    const dftfe::uInt                  M,
                    const dftfe::uInt                  N,
                    dftfe::utils::deviceDoubleComplex *yVec,
                    const dftfe::uInt                  startingXVecId)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const dftfe::uInt gangBlockId = blockIdx.x / numGangsPerBVec;
        const dftfe::uInt localThreadId =
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
      computeResidualDeviceKernelGeneralised(const dftfe::uInt numVectors,
                                             const dftfe::uInt numDofs,
                                             const dftfe::uInt N,
                                             const dftfe::uInt startingVecId,
                                             const double     *y,
                                             double           *r)
      {
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt dofIndex  = i / numVectors;
            const dftfe::uInt waveIndex = i % numVectors;
            r[i]                        = y[i] * y[i];
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernelGeneralised(
        const dftfe::uInt                        numVectors,
        const dftfe::uInt                        numDofs,
        const dftfe::uInt                        N,
        const dftfe::uInt                        startingVecId,
        const dftfe::utils::deviceDoubleComplex *Y,
        double                                  *r)
      {
        for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt dofIndex  = i / numVectors;
            const dftfe::uInt waveIndex = i % numVectors;
            r[i]                        = Y[i].x * Y[i].x + Y[i].y * Y[i].y;
          }
      }



    } // namespace

    template <typename ValueType1, typename ValueType2>
    void
    addSubspaceRotatedBlockToX(const dftfe::uInt             BDof,
                               const dftfe::uInt             BVec,
                               const ValueType1             *rotatedXBlockSP,
                               ValueType2                   *X,
                               const dftfe::uInt             startingDofId,
                               const dftfe::uInt             startingVecId,
                               const dftfe::uInt             N,
                               dftfe::utils::deviceStream_t &streamCompute)
    {
      DFTFE_LAUNCH_KERNEL(
        addSubspaceRotatedBlockToXKernel,
        (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        streamCompute,
        BDof,
        BVec,
        dftfe::utils::makeDataTypeDeviceCompatible(rotatedXBlockSP),
        dftfe::utils::makeDataTypeDeviceCompatible(X),
        startingDofId,
        startingVecId,
        N);
    }
    template <typename ValueType1, typename ValueType2>
    void
    copyFromOverlapMatBlockToDPSPBlocks(
      const dftfe::uInt             B,
      const dftfe::uInt             D,
      const ValueType1             *overlapMatrixBlock,
      ValueType1                   *overlapMatrixBlockDP,
      ValueType2                   *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove)
    {
      DFTFE_LAUNCH_KERNEL(
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
    }
    template <typename ValueType1, typename ValueType2>
    void
    computeDiagQTimesX(const ValueType1 *diagValues,
                       ValueType2       *X,
                       const dftfe::uInt N,
                       const dftfe::uInt M)
    {
      DFTFE_LAUNCH_KERNEL(computeDiagQTimesXKernel,
                          (M * N + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          0,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            diagValues),
                          dftfe::utils::makeDataTypeDeviceCompatible(X),
                          N,
                          M);
    }

    template <typename ValueType>
    void
    computeResidualDevice(const dftfe::uInt numVectors,
                          const dftfe::uInt numDofs,
                          const dftfe::uInt N,
                          const dftfe::uInt startingVecId,
                          const double     *eigenValues,
                          const ValueType  *X,
                          const ValueType  *Y,
                          double           *r)
    {
      DFTFE_LAUNCH_KERNEL(computeResidualDeviceKernel,
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
    }

    template <typename ValueType>
    void
    computeGeneralisedResidualDevice(const dftfe::uInt numVectors,
                                     const dftfe::uInt numDofs,
                                     const dftfe::uInt N,
                                     const dftfe::uInt startingVecId,
                                     const ValueType  *X,
                                     double           *residualSqDevice)
    {
      DFTFE_LAUNCH_KERNEL(computeResidualDeviceKernelGeneralised,
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
    }



    template <typename ValueType>
    void
    setZero(const dftfe::uInt BVec,
            const dftfe::uInt M,
            const dftfe::uInt N,
            ValueType        *yVec,
            const dftfe::uInt startingXVecId)
    {
      DFTFE_LAUNCH_KERNEL(setZeroKernel,
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
    }


    template void
    addSubspaceRotatedBlockToX(const dftfe::uInt             BDof,
                               const dftfe::uInt             BVec,
                               const float                  *rotatedXBlockSP,
                               double                       *X,
                               const dftfe::uInt             startingDofId,
                               const dftfe::uInt             startingVecId,
                               const dftfe::uInt             N,
                               dftfe::utils::deviceStream_t &streamCompute);
    template void
    addSubspaceRotatedBlockToX(const dftfe::uInt             BDof,
                               const dftfe::uInt             BVec,
                               const std::complex<float>    *rotatedXBlockSP,
                               std::complex<double>         *X,
                               const dftfe::uInt             startingDofId,
                               const dftfe::uInt             startingVecId,
                               const dftfe::uInt             N,
                               dftfe::utils::deviceStream_t &streamCompute);
    template void
    copyFromOverlapMatBlockToDPSPBlocks(
      const dftfe::uInt             B,
      const dftfe::uInt             D,
      const double                 *overlapMatrixBlock,
      double                       *overlapMatrixBlockDP,
      float                        *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove);
    template void
    copyFromOverlapMatBlockToDPSPBlocks(
      const dftfe::uInt             B,
      const dftfe::uInt             D,
      const std::complex<double>   *overlapMatrixBlock,
      std::complex<double>         *overlapMatrixBlockDP,
      std::complex<float>          *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove);

    template void
    computeDiagQTimesX(const double     *diagValues,
                       double           *X,
                       const dftfe::uInt N,
                       const dftfe::uInt M);
    template void
    computeDiagQTimesX(const std::complex<double> *diagValues,
                       std::complex<double>       *X,
                       const dftfe::uInt           N,
                       const dftfe::uInt           M);

    template void
    computeDiagQTimesX(const double         *diagValues,
                       std::complex<double> *X,
                       const dftfe::uInt     N,
                       const dftfe::uInt     M);

    template void
    computeResidualDevice(const dftfe::uInt numVectors,
                          const dftfe::uInt numDofs,
                          const dftfe::uInt N,
                          const dftfe::uInt startingVecId,
                          const double     *eigenValues,
                          const double     *X,
                          const double     *Y,
                          double           *r);
    template void
    computeResidualDevice(const dftfe::uInt           numVectors,
                          const dftfe::uInt           numDofs,
                          const dftfe::uInt           N,
                          const dftfe::uInt           startingVecId,
                          const double               *eigenValues,
                          const std::complex<double> *X,
                          const std::complex<double> *Y,
                          double                     *r);

    template void
    computeGeneralisedResidualDevice(const dftfe::uInt numVectors,
                                     const dftfe::uInt numDofs,
                                     const dftfe::uInt N,
                                     const dftfe::uInt startingVecId,
                                     const double     *X,
                                     double           *residualSqDevice);

    template void
    computeGeneralisedResidualDevice(const dftfe::uInt           numVectors,
                                     const dftfe::uInt           numDofs,
                                     const dftfe::uInt           N,
                                     const dftfe::uInt           startingVecId,
                                     const std::complex<double> *X,
                                     double *residualSqDevice);

    template void
    setZero(const dftfe::uInt BVec,
            const dftfe::uInt M,
            const dftfe::uInt N,
            double           *yVec,
            const dftfe::uInt startingXVecId);
    template void
    setZero(const dftfe::uInt     BVec,
            const dftfe::uInt     M,
            const dftfe::uInt     N,
            std::complex<double> *yVec,
            const dftfe::uInt     startingXVecId);

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
