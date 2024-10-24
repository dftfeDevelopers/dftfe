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
// @author Sambit Das, Phani Motamarri



#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryStorage.h>
#include <dftUtils.h>
#include <linearAlgebraOperationsDevice.h>
#include <linearAlgebraOperationsInternal.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>


namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
    namespace
    {
      __global__ void
      combinedDeviceKernel(const unsigned int contiguousBlockSize,
                           const unsigned int numContiguousBlocks,
                           double *           x,
                           double *           y,
                           const double       a,
                           const double       b,
                           const double       scalar,
                           const double       scalarOld,
                           const double *     invSqrtMassVec,
                           const double *     sqrtMassVec)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            *(y + index) *= (*(sqrtMassVec + blockIndex) * 1.0 / scalarOld);
            *(x + index) *= (*(invSqrtMassVec + blockIndex));
            y[index] = a * x[index] + b * y[index];
            *(x + index) *= (*(invSqrtMassVec + blockIndex) * scalar);
            *(y + index) *= (*(sqrtMassVec + blockIndex));
          }
      }


      __global__ void
      combinedDeviceKernel(const unsigned int contiguousBlockSize,
                           const unsigned int numContiguousBlocks,
                           dftfe::utils::deviceDoubleComplex *X,
                           dftfe::utils::deviceDoubleComplex *Y,
                           const double                       a,
                           const double                       b,
                           const double                       scalar,
                           const double                       scalarOld,
                           const double *                     invSqrtMassVec,
                           const double *                     sqrtMassVec)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            *(Y + index)            = dftfe::utils::makeComplex(
              (Y + index)->x * (*(sqrtMassVec + blockIndex) * 1.0 / scalarOld),
              (Y + index)->y * (*(sqrtMassVec + blockIndex) * 1.0 / scalarOld));
            *(X + index) = dftfe::utils::makeComplex(
              (X + index)->x * (*(invSqrtMassVec + blockIndex)),
              (X + index)->y * (*(invSqrtMassVec + blockIndex)));
            Y[index] =
              dftfe::utils::makeComplex(a * X[index].x + b * Y[index].x,
                                        a * X[index].y + b * Y[index].y);
            *(X + index) = dftfe::utils::makeComplex(
              (X + index)->x * (*(invSqrtMassVec + blockIndex) * scalar),
              (X + index)->y * (*(invSqrtMassVec + blockIndex) * scalar));
            *(Y + index) = dftfe::utils::makeComplex(
              (Y + index)->x * (*(sqrtMassVec + blockIndex)),
              (Y + index)->y * (*(sqrtMassVec + blockIndex)));
          }
      }



      __global__ void
      addSubspaceRotatedBlockToXKernel(const unsigned int BDof,
                                       const unsigned int BVec,
                                       const float *      rotatedXBlockSP,
                                       double *           X,
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
        dftfe::utils::deviceDoubleComplex *     X,
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
      copyFromOverlapMatBlockToDPSPBlocks(const unsigned int B,
                                          const unsigned int D,
                                          const double *     overlapMatrixBlock,
                                          double *overlapMatrixBlockDP,
                                          float * overlapMatrixBlockSP)
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
      copyFromOverlapMatBlockToDPSPBlocks(
        const unsigned int                       B,
        const unsigned int                       D,
        const dftfe::utils::deviceDoubleComplex *overlapMatrixBlock,
        dftfe::utils::deviceDoubleComplex *      overlapMatrixBlockDP,
        dftfe::utils::deviceFloatComplex *       overlapMatrixBlockSP)
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
      computeDiagQTimesXKernel(const double *     diagValues,
                               double *           X,
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
        dftfe::utils::deviceDoubleComplex *      X,
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


      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualDeviceKernel(const unsigned int numVectors,
                                  const unsigned int numDofs,
                                  const unsigned int N,
                                  const unsigned int startingVecId,
                                  const double *     eigenValues,
                                  const double *     x,
                                  const double *     y,
                                  double *           r)
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
                                  const double *     eigenValues,
                                  const dftfe::utils::deviceDoubleComplex *X,
                                  const dftfe::utils::deviceDoubleComplex *Y,
                                  double *                                 r)
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
      copyFloatArrToDoubleArrLocallyOwned(
        const unsigned int  contiguousBlockSize,
        const unsigned int  numContiguousBlocks,
        const float *       floatArr,
        const unsigned int *locallyOwnedFlagArr,
        double *            doubleArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            if (locallyOwnedFlagArr[blockIndex] == 1)
              doubleArr[index] = floatArr[index];
          }
      }


      __global__ void
      copyFloatArrToDoubleArrLocallyOwned(
        const unsigned int                      contiguousBlockSize,
        const unsigned int                      numContiguousBlocks,
        const dftfe::utils::deviceFloatComplex *floatArr,
        const unsigned int *                    locallyOwnedFlagArr,
        dftfe::utils::deviceDoubleComplex *     doubleArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            if (locallyOwnedFlagArr[blockIndex] == 1)
              dftfe::utils::copyValue(doubleArr + index, floatArr[index]);
          }
      }
    } // namespace


    void
    chebyshevFilterOverlapComputeCommunication(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X2,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y2,
      const unsigned int                                                    m,
      const double                                                          a,
      const double                                                          b,
      const double                                                          a0)
    {
      double e, c, sigma, sigma1, sigma2, gamma, alpha1Old, alpha2Old;
      e      = (b - a) / 2.0;
      c      = (b + a) / 2.0;
      sigma  = e / (a0 - c);
      sigma1 = sigma;
      gamma  = 2.0 / sigma1;


      //
      // create YArray
      // initialize to zeros.
      // x
      Y1.setValue(dataTypes::number(0.0));
      Y2.setValue(dataTypes::number(0.0));


      //
      // call HX
      //


      double alpha1 = sigma1 / e, alpha2 = -c;
      operatorMatrix.HXCheby(X1, alpha1, 0.0, alpha1 * alpha2, Y1);
      X2.updateGhostValues();
      operatorMatrix.HXCheby(
        X2, alpha1, 0.0, alpha1 * alpha2, Y2, false, false, true, true);
      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2    = 1.0 / (gamma - sigma);
          alpha1Old = alpha1, alpha2Old = alpha2;
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          if (degree == 2)
            {
              operatorMatrix.HXCheby(X2,
                                     alpha1Old,
                                     0.0,
                                     alpha1Old * alpha2Old,
                                     Y2,
                                     false,
                                     true,
                                     false,
                                     true);
              Y1.updateGhostValuesBegin();
              operatorMatrix.HXCheby(X2,
                                     alpha1Old,
                                     0.0,
                                     alpha1Old * alpha2Old,
                                     Y2,
                                     false,
                                     true,
                                     true,
                                     false);
              Y1.updateGhostValuesEnd();
              Y2.accumulateAddLocallyOwnedBegin();
            }
          else
            {
              operatorMatrix.HXCheby(Y2,
                                     alpha1Old,
                                     alpha2Old,
                                     -c * alpha1Old,
                                     X2,
                                     false,
                                     true,
                                     false,
                                     true);
              Y1.updateGhostValuesBegin();
              operatorMatrix.HXCheby(Y2,
                                     alpha1Old,
                                     alpha2Old,
                                     -c * alpha1Old,
                                     X2,
                                     false,
                                     true,
                                     true,
                                     false);
              Y1.updateGhostValuesEnd();
              X2.accumulateAddLocallyOwnedBegin();
            }


          //
          // call HX
          //
          operatorMatrix.HXCheby(
            Y1, alpha1, alpha2, -c * alpha1, X1, false, false, true, true);
          if (degree == 2)
            {
              Y2.accumulateAddLocallyOwnedEnd();
              Y2.zeroOutGhosts();
            }
          else
            {
              X2.accumulateAddLocallyOwnedEnd();
              X2.zeroOutGhosts();
              X2.swap(Y2);
            }

          operatorMatrix.HXCheby(
            Y1, alpha1, alpha2, -c * alpha1, X1, false, true, false, true);
          Y2.updateGhostValuesBegin();
          operatorMatrix.HXCheby(
            Y1, alpha1, alpha2, -c * alpha1, X1, false, true, true, false);
          Y2.updateGhostValuesEnd();
          X1.accumulateAddLocallyOwnedBegin();
          operatorMatrix.HXCheby(
            Y2, alpha1, alpha2, -c * alpha1, X2, false, false, true, true);
          X1.accumulateAddLocallyOwnedEnd();
          X1.zeroOutGhosts();

          //
          // XArray = YArray
          //
          X1.swap(Y1);

          if (degree == m)
            {
              operatorMatrix.HXCheby(
                Y2, alpha1, alpha2, -c * alpha1, X2, false, true, false, false);
              X2.accumulateAddLocallyOwned();
              X2.zeroOutGhosts();
              X2.swap(Y2);
            }

          //
          // YArray = YNewArray
          //
          sigma = sigma2;
        }

      // copy back YArray to XArray
      X1 = Y1;
      X2 = Y2;
    }

    void
    chebyshevFilterOverlapComputeCommunicationSinglePrec(
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                                  BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X2,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y2,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &X1_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &Y1_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &X2_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &                 Y2_SP,
      std::vector<double> eigenvalues,
      const unsigned int  m,
      const double        a,
      const double        b,
      const double        a0)
    {
      double e, c, sigma, sigma1, sigma2, gamma, alpha1Old, alpha2Old;
      e      = (b - a) / 2.0;
      c      = (b + a) / 2.0;
      sigma  = e / (a0 - c);
      sigma1 = sigma;
      gamma  = 2.0 / sigma1;

      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        eigenValuesFiltered, eigenValuesFiltered1, eigenValuesFiltered2;
      eigenValuesFiltered.resize(eigenvalues.size());
      eigenValuesFiltered.copyFrom(eigenvalues);
      eigenValuesFiltered1 = eigenValuesFiltered;
      eigenValuesFiltered2 = eigenValuesFiltered;
      eigenValuesFiltered1.setValue(1.0);

      //
      // create YArray
      // initialize to zeros.
      // x
      operatorMatrix.HXCheby(X1, 1.0, 0.0, 0.0, Y1);


      //
      // call HX
      //


      double alpha1 = sigma1 / e, alpha2 = -c;
      eigenValuesFiltered2.setValue(alpha1 * alpha2);
      BLASWrapperPtr->ApaBD(1,
                            eigenValuesFiltered2.size(),
                            alpha1,
                            eigenValuesFiltered2.data(),
                            eigenValuesFiltered1.data(),
                            eigenValuesFiltered.data(),
                            eigenValuesFiltered2.data());
      BLASWrapperPtr->ApaBD(X1.locallyOwnedSize(),
                            X1.numVectors(),
                            -1.0,
                            Y1.data(),
                            X1.data(),
                            eigenValuesFiltered.data(),
                            Y1.data());
      X1_SP.setValue(0.0);
      X2_SP.setValue(0.0);
      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
        X1.locallyOwnedSize() * X1.numVectors(), Y1.data(), Y1_SP.data());
      BLASWrapperPtr->xscal(Y1_SP.data(),
                            dataTypes::numberFP32(alpha1),
                            X2.locallyOwnedSize() * X2.numVectors());
      X2.updateGhostValues();
      operatorMatrix.HXCheby(X2, 1.0, 0.0, 0.0, Y2, false, false, true, true);
      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2    = 1.0 / (gamma - sigma);
          alpha1Old = alpha1, alpha2Old = alpha2;
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          if (degree == 2)
            {
              operatorMatrix.HXCheby(
                X2, 1.0, 0.0, 0.0, Y2, false, true, false, true);
              Y1_SP.updateGhostValuesBegin();
              operatorMatrix.HXCheby(
                X2, 1.0, 0.0, 0.0, Y2, false, true, true, false);
              Y1_SP.updateGhostValuesEnd();
              Y2.accumulateAddLocallyOwnedBegin();
            }
          else
            {
              operatorMatrix.HXCheby(Y2_SP,
                                     alpha1Old,
                                     alpha2Old,
                                     -c * alpha1Old,
                                     X2_SP,
                                     false,
                                     true,
                                     false,
                                     true);
              Y1_SP.updateGhostValuesBegin();
              operatorMatrix.HXCheby(Y2_SP,
                                     alpha1Old,
                                     alpha2Old,
                                     -c * alpha1Old,
                                     X2_SP,
                                     false,
                                     true,
                                     true,
                                     false);
              Y1_SP.updateGhostValuesEnd();
              X2_SP.accumulateAddLocallyOwnedBegin();
            }


          //
          // call HX
          //
          operatorMatrix.HXCheby(Y1_SP,
                                 alpha1,
                                 alpha2,
                                 -c * alpha1,
                                 X1_SP,
                                 false,
                                 false,
                                 true,
                                 true);
          if (degree == 2)
            {
              Y2.accumulateAddLocallyOwnedEnd();
              Y2.zeroOutGhosts();
              BLASWrapperPtr->ApaBD(X2.locallyOwnedSize(),
                                    X2.numVectors(),
                                    -1.0,
                                    Y2.data(),
                                    X2.data(),
                                    eigenValuesFiltered.data() +
                                      X1.numVectors(),
                                    Y2.data());
              BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                X2.locallyOwnedSize() * X2.numVectors(),
                Y2.data(),
                Y2_SP.data());
              BLASWrapperPtr->xscal(Y2_SP.data(),
                                    dataTypes::numberFP32(alpha1Old),
                                    X2.locallyOwnedSize() * X2.numVectors());
            }
          else
            {
              X2_SP.accumulateAddLocallyOwnedEnd();
              X2_SP.zeroOutGhosts();
              BLASWrapperPtr->ApaBD(X2_SP.locallyOwnedSize(),
                                    X2_SP.numVectors(),
                                    alpha1Old,
                                    X2_SP.data(),
                                    Y2.data(),
                                    eigenValuesFiltered2.data() +
                                      X1_SP.numVectors(),
                                    X2_SP.data());
              BLASWrapperPtr->axpby(eigenValuesFiltered2.size(),
                                    -c * alpha1Old,
                                    eigenValuesFiltered2.data(),
                                    alpha2Old,
                                    eigenValuesFiltered1.data());
              BLASWrapperPtr->ApaBD(1,
                                    eigenValuesFiltered1.size(),
                                    alpha1Old,
                                    eigenValuesFiltered1.data(),
                                    eigenValuesFiltered2.data(),
                                    eigenValuesFiltered.data(),
                                    eigenValuesFiltered1.data());
              X2_SP.swap(Y2_SP);
              eigenValuesFiltered1.swap(eigenValuesFiltered2);
            }

          operatorMatrix.HXCheby(Y1_SP,
                                 alpha1,
                                 alpha2,
                                 -c * alpha1,
                                 X1_SP,
                                 false,
                                 true,
                                 false,
                                 true);
          Y2_SP.updateGhostValuesBegin();
          operatorMatrix.HXCheby(Y1_SP,
                                 alpha1,
                                 alpha2,
                                 -c * alpha1,
                                 X1_SP,
                                 false,
                                 true,
                                 true,
                                 false);
          Y2_SP.updateGhostValuesEnd();
          X1_SP.accumulateAddLocallyOwnedBegin();
          operatorMatrix.HXCheby(Y2_SP,
                                 alpha1,
                                 alpha2,
                                 -c * alpha1,
                                 X2_SP,
                                 false,
                                 false,
                                 true,
                                 true);
          X1_SP.accumulateAddLocallyOwnedEnd();
          X1_SP.zeroOutGhosts();
          BLASWrapperPtr->ApaBD(X1_SP.locallyOwnedSize(),
                                X1_SP.numVectors(),
                                alpha1,
                                X1_SP.data(),
                                Y1.data(),
                                eigenValuesFiltered2.data(),
                                X1_SP.data());

          //
          // XArray = YArray
          //
          X1_SP.swap(Y1_SP);

          if (degree == m)
            {
              operatorMatrix.HXCheby(Y2_SP,
                                     alpha1,
                                     alpha2,
                                     -c * alpha1,
                                     X2_SP,
                                     false,
                                     true,
                                     false,
                                     false);
              X2_SP.accumulateAddLocallyOwned();
              X2_SP.zeroOutGhosts();
              BLASWrapperPtr->ApaBD(X2_SP.locallyOwnedSize(),
                                    X2_SP.numVectors(),
                                    alpha1,
                                    X2_SP.data(),
                                    Y2.data(),
                                    eigenValuesFiltered2.data() +
                                      X1_SP.numVectors(),
                                    X2_SP.data());
              BLASWrapperPtr->axpby(eigenValuesFiltered2.size(),
                                    -c * alpha1,
                                    eigenValuesFiltered2.data(),
                                    alpha2,
                                    eigenValuesFiltered1.data());
              BLASWrapperPtr->ApaBD(1,
                                    eigenValuesFiltered1.size(),
                                    alpha1,
                                    eigenValuesFiltered1.data(),
                                    eigenValuesFiltered2.data(),
                                    eigenValuesFiltered.data(),
                                    eigenValuesFiltered1.data());
              X2_SP.swap(Y2_SP);
              eigenValuesFiltered1.swap(eigenValuesFiltered2);
            }

          //
          // YArray = YNewArray
          //
          sigma = sigma2;
        }

      // copy back YArray to XArray
      BLASWrapperPtr->ApaBD(X1.locallyOwnedSize(),
                            X1.numVectors(),
                            1.0,
                            Y1_SP.data(),
                            X1.data(),
                            eigenValuesFiltered2.data(),
                            X1.data());

      BLASWrapperPtr->ApaBD(X2.locallyOwnedSize(),
                            X2.numVectors(),
                            1.0,
                            Y2_SP.data(),
                            X2.data(),
                            eigenValuesFiltered2.data() + X1.numVectors(),
                            X2.data());
    }


    void
    subspaceRotationSpectrumSplitScalapack(
      const dataTypes::number *X,
      dataTypes::number *      XFrac,
      const unsigned int       M,
      const unsigned int       N,
      const unsigned int       Nfr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool                                       rotationMatTranspose)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      const unsigned int vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, Nfr);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        rotationMatBlockHost;

      if (dftParams.allowFullCPUMemSubspaceRot)
        {
          rotationMatBlockHost.resize(N * Nfr, dataTypes::number(0));
          rotationMatBlockHost.setValue(0);
        }
      else
        {
          rotationMatBlockHost.resize(vectorsBlockSize * N,
                                      dataTypes::number(0));
          rotationMatBlockHost.setValue(0);
        }

      dftfe::utils::deviceStream_t streamCompute, streamDeviceCCL;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDeviceCCL);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&communEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlock(vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockNext(vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlock(Nfr * dofsBlockSize, dataTypes::number(0));

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          unsigned int BDof = 0;
          if (M >= idof)
            BDof = std::min(dofsBlockSize, M - idof);

          for (unsigned int jvec = 0; jvec < Nfr; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const unsigned int BVec = std::min(vectorsBlockSize, Nfr - jvec);

              const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
              const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

              if (dftParams.allowFullCPUMemSubspaceRot)
                {
                  if (idof == 0)
                    {
                      // Extract QBVec from parallel ScaLAPACK matrix Q
                      if (rotationMatTranspose)
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < N; ++i)
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  const unsigned int localRowId =
                                    globalToLocalRowIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::unordered_map<unsigned int,
                                                         unsigned int>::iterator
                                        it = globalToLocalColumnIdMap.find(
                                          j + jvec);
                                      if (it != globalToLocalColumnIdMap.end())
                                        *(rotationMatBlockHost.begin() +
                                          jvec * N + i * BVec + j) =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                    }
                                }
                        }
                      else
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < N; ++i)
                              if (globalToLocalColumnIdMap.find(i) !=
                                  globalToLocalColumnIdMap.end())
                                {
                                  const unsigned int localColumnId =
                                    globalToLocalColumnIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::unordered_map<unsigned int,
                                                         unsigned int>::iterator
                                        it =
                                          globalToLocalRowIdMap.find(j + jvec);
                                      if (it != globalToLocalRowIdMap.end())
                                        *(rotationMatBlockHost.begin() +
                                          jvec * N + i * BVec + j) =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                    }
                                }
                        }
                    }
                }
              else
                {
                  std::memset(rotationMatBlockHost.begin(),
                              0,
                              BVec * N * sizeof(dataTypes::number));

                  // Extract QBVec from parallel ScaLAPACK matrix Q
                  if (rotationMatTranspose)
                    {
                      if (processGrid->is_process_active())
                        for (unsigned int i = 0; i < N; ++i)
                          if (globalToLocalRowIdMap.find(i) !=
                              globalToLocalRowIdMap.end())
                            {
                              const unsigned int localRowId =
                                globalToLocalRowIdMap[i];
                              for (unsigned int j = 0; j < BVec; ++j)
                                {
                                  std::unordered_map<unsigned int,
                                                     unsigned int>::iterator
                                    it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                  if (it != globalToLocalColumnIdMap.end())
                                    *(rotationMatBlockHost.begin() + i * BVec +
                                      j) = rotationMatPar.local_el(localRowId,
                                                                   it->second);
                                }
                            }
                    }
                  else
                    {
                      if (processGrid->is_process_active())
                        for (unsigned int i = 0; i < N; ++i)
                          if (globalToLocalColumnIdMap.find(i) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const unsigned int localColumnId =
                                globalToLocalColumnIdMap[i];
                              for (unsigned int j = 0; j < BVec; ++j)
                                {
                                  std::unordered_map<unsigned int,
                                                     unsigned int>::iterator
                                    it = globalToLocalRowIdMap.find(j + jvec);
                                  if (it != globalToLocalRowIdMap.end())
                                    *(rotationMatBlockHost.begin() + i * BVec +
                                      j) =
                                      rotationMatPar.local_el(it->second,
                                                              localColumnId);
                                }
                            }
                    }
                }


              if (dftParams.allowFullCPUMemSubspaceRot)
                {
                  if (dftParams.useDeviceDirectAllReduce)
                    {
                      dftfe::utils::deviceMemcpyAsyncH2D(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockNext.begin()),
                        rotationMatBlockHost.begin() + jvec * N,
                        BVec * N * sizeof(dataTypes::number),
                        streamDeviceCCL);

                      if (idof == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              rotationMatBlockNext.begin(),
                              rotationMatBlockNext.begin(),
                              BVec * N,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDeviceCCL);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              rotationMatBlockNext.begin(),
                              rotationMatBlockNext.begin(),
                              BVec * N,
                              streamDeviceCCL);

                          dftfe::utils::deviceMemcpyAsyncD2H(
                            rotationMatBlockHost.begin() + jvec * N,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockNext.begin()),
                            BVec * N * sizeof(dataTypes::number),
                            streamDeviceCCL);
                        }
                    }
                  else
                    {
                      if (idof == 0)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      rotationMatBlockHost.begin() + jvec * N,
                                      BVec * N,
                                      dataTypes::mpi_type_id(
                                        rotationMatBlockHost.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      dftfe::utils::deviceMemcpyH2D(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlock.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockHost.begin() + jvec * N),
                        BVec * N * sizeof(dataTypes::number));
                    }
                }
              else
                {
                  if (dftParams.useDeviceDirectAllReduce)
                    {
                      dftfe::utils::deviceMemcpyAsyncH2D(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockNext.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockHost.begin()),
                        BVec * N * sizeof(dataTypes::number),
                        streamDeviceCCL);

                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          rotationMatBlockNext.begin(),
                          rotationMatBlockNext.begin(),
                          BVec * N,
                          tempReal.begin(),
                          tempImag.begin(),
                          streamDeviceCCL);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          rotationMatBlockNext.begin(),
                          rotationMatBlockNext.begin(),
                          BVec * N,
                          streamDeviceCCL);
                    }
                  else
                    {
                      MPI_Allreduce(MPI_IN_PLACE,
                                    rotationMatBlockHost.begin(),
                                    BVec * N,
                                    dataTypes::mpi_type_id(
                                      rotationMatBlockHost.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);

                      dftfe::utils::deviceMemcpyH2D(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlock.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockHost.begin()),
                        BVec * N * sizeof(dataTypes::number));
                    }
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                  dftfe::utils::deviceStreamWaitEvent(streamDeviceCCL,
                                                      computeEvents[blockCount],
                                                      0);

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  dftfe::utils::deviceEventRecord(communEvents[blockCount],
                                                  streamDeviceCCL);
                  if (dftfe::utils::deviceEventSynchronize(
                        communEvents[blockCount]) ==
                      dftfe::utils::deviceSuccess)
                    rotationMatBlock.swap(rotationMatBlockNext);
                }

              if (BDof != 0)
                {
                  BLASWrapperPtr->xgemm('N',
                                        'N',
                                        BVec,
                                        BDof,
                                        N,
                                        &scalarCoeffAlpha,
                                        rotationMatBlock.begin(),
                                        BVec,
                                        X + idof * N,
                                        N,
                                        &scalarCoeffBeta,
                                        rotatedVectorsMatBlock.begin() + jvec,
                                        Nfr);
                }

              blockCount++;
            } // block loop over vectors


          if (BDof != 0)
            {
              dftfe::utils::deviceMemcpyAsyncD2D(
                XFrac + idof * Nfr,
                dftfe::utils::makeDataTypeDeviceCompatible(
                  rotatedVectorsMatBlock.begin()),
                Nfr * BDof * sizeof(dataTypes::number),
                streamCompute);
            }

        } // block loop over dofs

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(communEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);
    }



    void
    subspaceRotationScalapack(
      dataTypes::number *X,
      const unsigned int M,
      const unsigned int N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool                                       rotationMatTranspose,
      const bool                                       isRotationMatLowerTria)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        rotationMatBlockHost;

      if (dftParams.allowFullCPUMemSubspaceRot)
        {
          rotationMatBlockHost.resize(N * N, dataTypes::number(0));
          rotationMatBlockHost.setValue(0);
        }
      else
        {
          rotationMatBlockHost.resize(vectorsBlockSize * N,
                                      dataTypes::number(0));
          rotationMatBlockHost.setValue(0);
        }


      dftfe::utils::deviceStream_t streamCompute, streamDeviceCCL;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDeviceCCL);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&communEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlock(vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockTemp(vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlock(N * dofsBlockSize, dataTypes::number(0));

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          unsigned int BDof = 0;
          if (M >= idof)
            BDof = std::min(dofsBlockSize, M - idof);

          for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

              const unsigned int D = isRotationMatLowerTria ? (jvec + BVec) : N;

              if ((jvec + BVec) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + BVec) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  const dataTypes::number scalarCoeffAlpha =
                    dataTypes::number(1.0);
                  const dataTypes::number scalarCoeffBeta =
                    dataTypes::number(0);

                  if (dftParams.allowFullCPUMemSubspaceRot)
                    {
                      if (idof == 0)
                        {
                          // Extract QBVec from parallel ScaLAPACK matrix Q
                          if (rotationMatTranspose)
                            {
                              if (processGrid->is_process_active())
                                for (unsigned int i = 0; i < D; ++i)
                                  if (globalToLocalRowIdMap.find(i) !=
                                      globalToLocalRowIdMap.end())
                                    {
                                      const unsigned int localRowId =
                                        globalToLocalRowIdMap[i];
                                      for (unsigned int j = 0; j < BVec; ++j)
                                        {
                                          std::unordered_map<
                                            unsigned int,
                                            unsigned int>::iterator it =
                                            globalToLocalColumnIdMap.find(j +
                                                                          jvec);
                                          if (it !=
                                              globalToLocalColumnIdMap.end())
                                            *(rotationMatBlockHost.begin() +
                                              jvec * N + i * BVec + j) =
                                              rotationMatPar.local_el(
                                                localRowId, it->second);
                                        }
                                    }
                            }
                          else
                            {
                              if (processGrid->is_process_active())
                                for (unsigned int i = 0; i < D; ++i)
                                  if (globalToLocalColumnIdMap.find(i) !=
                                      globalToLocalColumnIdMap.end())
                                    {
                                      const unsigned int localColumnId =
                                        globalToLocalColumnIdMap[i];
                                      for (unsigned int j = 0; j < BVec; ++j)
                                        {
                                          std::unordered_map<
                                            unsigned int,
                                            unsigned int>::iterator it =
                                            globalToLocalRowIdMap.find(j +
                                                                       jvec);
                                          if (it != globalToLocalRowIdMap.end())
                                            *(rotationMatBlockHost.begin() +
                                              jvec * N + i * BVec + j) =
                                              rotationMatPar.local_el(
                                                it->second, localColumnId);
                                        }
                                    }
                            }
                        }
                    }
                  else
                    {
                      std::memset(rotationMatBlockHost.begin(),
                                  0,
                                  BVec * N * sizeof(dataTypes::number));

                      // Extract QBVec from parallel ScaLAPACK matrix Q
                      if (rotationMatTranspose)
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < D; ++i)
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  const unsigned int localRowId =
                                    globalToLocalRowIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::unordered_map<unsigned int,
                                                         unsigned int>::iterator
                                        it = globalToLocalColumnIdMap.find(
                                          j + jvec);
                                      if (it != globalToLocalColumnIdMap.end())
                                        *(rotationMatBlockHost.begin() +
                                          i * BVec + j) =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                    }
                                }
                        }
                      else
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < D; ++i)
                              if (globalToLocalColumnIdMap.find(i) !=
                                  globalToLocalColumnIdMap.end())
                                {
                                  const unsigned int localColumnId =
                                    globalToLocalColumnIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::unordered_map<unsigned int,
                                                         unsigned int>::iterator
                                        it =
                                          globalToLocalRowIdMap.find(j + jvec);
                                      if (it != globalToLocalRowIdMap.end())
                                        *(rotationMatBlockHost.begin() +
                                          i * BVec + j) =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                    }
                                }
                        }
                    }

                  if (dftParams.allowFullCPUMemSubspaceRot)
                    {
                      if (dftParams.useDeviceDirectAllReduce)
                        {
                          dftfe::utils::deviceMemcpyAsyncH2D(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockTemp.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin() + jvec * N),
                            BVec * D * sizeof(dataTypes::number),
                            streamDeviceCCL);

                          if (idof == 0)
                            {
                              if (std::is_same<dataTypes::number,
                                               std::complex<double>>::value)
                                devicecclMpiCommDomain
                                  .deviceDirectAllReduceWrapper(
                                    rotationMatBlockTemp.begin(),
                                    rotationMatBlockTemp.begin(),
                                    BVec * D,
                                    tempReal.begin(),
                                    tempImag.begin(),
                                    streamDeviceCCL);
                              else
                                devicecclMpiCommDomain
                                  .deviceDirectAllReduceWrapper(
                                    rotationMatBlockTemp.begin(),
                                    rotationMatBlockTemp.begin(),
                                    BVec * D,
                                    streamDeviceCCL);

                              dftfe::utils::deviceMemcpyAsyncD2H(
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockHost.begin() + jvec * N),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockTemp.begin()),
                                BVec * D * sizeof(dataTypes::number),
                                streamDeviceCCL);
                            }
                        }
                      else
                        {
                          if (idof == 0)
                            MPI_Allreduce(MPI_IN_PLACE,
                                          rotationMatBlockHost.begin() +
                                            jvec * N,
                                          BVec * D,
                                          dataTypes::mpi_type_id(
                                            rotationMatBlockHost.begin()),
                                          MPI_SUM,
                                          mpiCommDomain);

                          dftfe::utils::deviceMemcpyH2D(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlock.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin() + jvec * N),
                            BVec * D * sizeof(dataTypes::number));
                        }
                    }
                  else
                    {
                      if (dftParams.useDeviceDirectAllReduce)
                        {
                          dftfe::utils::deviceMemcpyAsyncH2D(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockTemp.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin()),
                            BVec * D * sizeof(dataTypes::number),
                            streamDeviceCCL);

                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              rotationMatBlockTemp.begin(),
                              rotationMatBlockTemp.begin(),
                              BVec * D,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDeviceCCL);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              rotationMatBlockTemp.begin(),
                              rotationMatBlockTemp.begin(),
                              BVec * D,
                              streamDeviceCCL);
                        }
                      else
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        rotationMatBlockHost.begin(),
                                        BVec * D,
                                        dataTypes::mpi_type_id(
                                          rotationMatBlockHost.begin()),
                                        MPI_SUM,
                                        mpiCommDomain);

                          dftfe::utils::deviceMemcpyH2D(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlock.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin()),
                            BVec * D * sizeof(dataTypes::number));
                        }
                    }

                  if (dftParams.useDeviceDirectAllReduce)
                    {
                      // check for completion of compute of previous block in
                      // compute stream before proceeding to rewriting
                      // rotationMatBlock in communication stream
                      dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                      streamCompute);
                      dftfe::utils::deviceStreamWaitEvent(
                        streamDeviceCCL, computeEvents[blockCount], 0);

                      // synchronize host to communication stream before doing
                      // swap this automatically also makes sure the compute
                      // stream has the correct rotationMatBlock for dgemm
                      dftfe::utils::deviceEventRecord(communEvents[blockCount],
                                                      streamDeviceCCL);
                      if (dftfe::utils::deviceEventSynchronize(
                            communEvents[blockCount]) ==
                          dftfe::utils::deviceSuccess)
                        rotationMatBlock.swap(rotationMatBlockTemp);
                    }

                  if (BDof != 0)
                    {
                      BLASWrapperPtr->xgemm('N',
                                            'N',
                                            BVec,
                                            BDof,
                                            D,
                                            &scalarCoeffAlpha,
                                            rotationMatBlock.begin(),
                                            BVec,
                                            X + idof * N,
                                            N,
                                            &scalarCoeffBeta,
                                            rotatedVectorsMatBlock.begin() +
                                              jvec,
                                            N);
                    }
                } // band parallelization
              blockCount++;
            } // block loop over vectors


          if (BDof != 0)
            {
              dftfe::utils::deviceMemcpyAsyncD2D(
                dftfe::utils::makeDataTypeDeviceCompatible(X) + idof * N,
                dftfe::utils::makeDataTypeDeviceCompatible(
                  rotatedVectorsMatBlock.begin()),
                N * BDof * sizeof(dataTypes::number),
                streamCompute);
            }

        } // block loop over dofs


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(communEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);
    }

    void
    subspaceRotationCGSMixedPrecScalapack(
      dataTypes::number *X,
      const unsigned int M,
      const unsigned int N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool                                       rotationMatTranspose)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));


      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        rotationMatBlockHostSP(vectorsBlockSize * N);

      std::memset(rotationMatBlockHostSP.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        diagValuesHost;
      diagValuesHost.resize(N, 0);
      std::memset(diagValuesHost.begin(), 0, N * sizeof(dataTypes::number));

      dftfe::utils::deviceStream_t streamCompute, streamDeviceCCL;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDeviceCCL);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int          numberBlocks = (N / vectorsBlockSize);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&communEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockSP(vectorsBlockSize * N, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockSPTemp(vectorsBlockSize * N, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        diagValues(N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlockSP(vectorsBlockSize * dofsBlockSize,
                                 dataTypes::numberFP32(0));

      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP = dataTypes::numberFP32(0);


      // Extract DiagQ from parallel ScaLAPACK matrix Q
      if (rotationMatTranspose)
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalRowIdMap.find(i) != globalToLocalRowIdMap.end())
                {
                  const unsigned int localRowId = globalToLocalRowIdMap[i];
                  std::unordered_map<unsigned int, unsigned int>::iterator it =
                    globalToLocalColumnIdMap.find(i);
                  if (it != globalToLocalColumnIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(localRowId, it->second);
                    }
                }
        }
      else
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalColumnIdMap.find(i) !=
                  globalToLocalColumnIdMap.end())
                {
                  const unsigned int localColumnId =
                    globalToLocalColumnIdMap[i];
                  std::unordered_map<unsigned int, unsigned int>::iterator it =
                    globalToLocalRowIdMap.find(i);
                  if (globalToLocalRowIdMap.find(i) !=
                      globalToLocalRowIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(it->second, localColumnId);
                    }
                }
        }

      MPI_Allreduce(MPI_IN_PLACE,
                    diagValuesHost.begin(),
                    N,
                    dataTypes::mpi_type_id(diagValuesHost.begin()),
                    MPI_SUM,
                    mpiCommDomain);

      dftfe::utils::deviceMemcpyH2D(
        dftfe::utils::makeDataTypeDeviceCompatible(diagValues.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(diagValuesHost.begin()),
        N * sizeof(dataTypes::number));

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeDiagQTimesXKernel<<<(M * N +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE,
                                 dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(diagValues.begin()),
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
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           diagValues.begin()),
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         N,
                         M);
#endif

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

          const unsigned int D = jvec + BVec;

          std::memset(rotationMatBlockHostSP.begin(),
                      0,
                      BVec * N * sizeof(dataTypes::numberFP32));

          if ((jvec + BVec) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + BVec) >
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Extract QBVec from parallel ScaLAPACK matrix Q
              if (rotationMatTranspose)
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalRowIdMap.find(i) !=
                          globalToLocalRowIdMap.end())
                        {
                          const unsigned int localRowId =
                            globalToLocalRowIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalColumnIdMap.find(j + jvec);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    j) = rotationMatPar.local_el(localRowId,
                                                                 it->second);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalColumnIdMap.find(i);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    i - jvec) = dataTypes::numberFP32(0);
                                }
                            }
                        }
                }
              else
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalColumnIdMap.find(i) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j + jvec);
                              if (it != globalToLocalRowIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    j) = rotationMatPar.local_el(it->second,
                                                                 localColumnId);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    i - jvec) = dataTypes::numberFP32(0);
                                }
                            }
                        }
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  dftfe::utils::deviceMemcpyAsyncH2D(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSPTemp.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32),
                    streamDeviceCCL);

                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      rotationMatBlockSPTemp.begin(),
                      rotationMatBlockSPTemp.begin(),
                      BVec * D,
                      tempRealFP32.begin(),
                      tempImagFP32.begin(),
                      streamDeviceCCL);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      rotationMatBlockSPTemp.begin(),
                      rotationMatBlockSPTemp.begin(),
                      BVec * D,
                      streamDeviceCCL);
                }
              else
                {
                  MPI_Allreduce(MPI_IN_PLACE,
                                rotationMatBlockHostSP.begin(),
                                BVec * D,
                                dataTypes::mpi_type_id(
                                  rotationMatBlockHostSP.begin()),
                                MPI_SUM,
                                mpiCommDomain);

                  dftfe::utils::deviceMemcpyH2D(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSP.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32));
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                  dftfe::utils::deviceStreamWaitEvent(streamDeviceCCL,
                                                      computeEvents[blockCount],
                                                      0);

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  dftfe::utils::deviceEventRecord(communEvents[blockCount],
                                                  streamDeviceCCL);
                  if (dftfe::utils::deviceEventSynchronize(
                        communEvents[blockCount]) ==
                      dftfe::utils::deviceSuccess)
                    rotationMatBlockSP.swap(rotationMatBlockSPTemp);
                }

              for (unsigned int idof = 0; idof < maxNumLocalDofs;
                   idof += dofsBlockSize)
                {
                  // Correct block dimensions if block "goes off edge of" the
                  // matrix
                  unsigned int BDof = 0;
                  if (M >= idof)
                    BDof = std::min(dofsBlockSize, M - idof);

                  if (BDof != 0)
                    {
                      BLASWrapperPtr->xgemm('N',
                                            'N',
                                            BVec,
                                            BDof,
                                            D,
                                            &scalarCoeffAlphaSP,
                                            rotationMatBlockSP.begin(),
                                            BVec,
                                            XSP.begin() + idof * N,
                                            N,
                                            &scalarCoeffBetaSP,
                                            rotatedVectorsMatBlockSP.begin(),
                                            BVec);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                      addSubspaceRotatedBlockToXKernel<<<
                        (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        streamCompute>>>(
                        BDof,
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(X),
                        idof,
                        jvec,
                        N);
#elif DFTFE_WITH_DEVICE_LANG_HIP
                      hipLaunchKernelGGL(
                        addSubspaceRotatedBlockToXKernel,
                        (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        streamCompute,
                        BDof,
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(X),
                        idof,
                        jvec,
                        N);
#endif
                    }
                } // block loop over dofs
            }     // band parallalelization loop
          blockCount++;
        } // block loop over vectors

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(communEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);
    }

    void
    subspaceRotationRRMixedPrecScalapack(
      dataTypes::number *X,
      const unsigned int M,
      const unsigned int N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool                                       rotationMatTranspose)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));


      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        rotationMatBlockHostSP(vectorsBlockSize * N);

      std::memset(rotationMatBlockHostSP.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        diagValuesHost;
      diagValuesHost.resize(N, 0);
      std::memset(diagValuesHost.begin(), 0, N * sizeof(dataTypes::number));

      dftfe::utils::deviceStream_t streamCompute, streamDeviceCCL;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDeviceCCL);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int          numberBlocks = (N / vectorsBlockSize);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&communEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockSP(vectorsBlockSize * N, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockSPTemp(vectorsBlockSize * N, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        diagValues(N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlockSP(vectorsBlockSize * dofsBlockSize,
                                 dataTypes::numberFP32(0));

      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP = dataTypes::numberFP32(0);


      // Extract DiagQ from parallel ScaLAPACK matrix Q
      if (rotationMatTranspose)
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalRowIdMap.find(i) != globalToLocalRowIdMap.end())
                {
                  const unsigned int localRowId = globalToLocalRowIdMap[i];
                  std::unordered_map<unsigned int, unsigned int>::iterator it =
                    globalToLocalColumnIdMap.find(i);
                  if (it != globalToLocalColumnIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(localRowId, it->second);
                    }
                }
        }
      else
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalColumnIdMap.find(i) !=
                  globalToLocalColumnIdMap.end())
                {
                  const unsigned int localColumnId =
                    globalToLocalColumnIdMap[i];
                  std::unordered_map<unsigned int, unsigned int>::iterator it =
                    globalToLocalRowIdMap.find(i);
                  if (globalToLocalRowIdMap.find(i) !=
                      globalToLocalRowIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(it->second, localColumnId);
                    }
                }
        }

      MPI_Allreduce(MPI_IN_PLACE,
                    diagValuesHost.begin(),
                    N,
                    dataTypes::mpi_type_id(diagValuesHost.begin()),
                    MPI_SUM,
                    mpiCommDomain);

      dftfe::utils::deviceMemcpyH2D(
        dftfe::utils::makeDataTypeDeviceCompatible(diagValues.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(diagValuesHost.begin()),
        N * sizeof(dataTypes::number));

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      computeDiagQTimesXKernel<<<(M * N +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE,
                                 dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(diagValues.begin()),
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
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           diagValues.begin()),
                         dftfe::utils::makeDataTypeDeviceCompatible(X),
                         N,
                         M);
#endif

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

          const unsigned int D = N;

          if ((jvec + BVec) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + BVec) >
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              std::memset(rotationMatBlockHostSP.begin(),
                          0,
                          BVec * N * sizeof(dataTypes::numberFP32));

              // Extract QBVec from parallel ScaLAPACK matrix Q
              if (rotationMatTranspose)
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalRowIdMap.find(i) !=
                          globalToLocalRowIdMap.end())
                        {
                          const unsigned int localRowId =
                            globalToLocalRowIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalColumnIdMap.find(j + jvec);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    j) = rotationMatPar.local_el(localRowId,
                                                                 it->second);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalColumnIdMap.find(i);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    i - jvec) = dataTypes::numberFP32(0);
                                }
                            }
                        }
                }
              else
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalColumnIdMap.find(i) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j + jvec);
                              if (it != globalToLocalRowIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    j) = rotationMatPar.local_el(it->second,
                                                                 localColumnId);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    i - jvec) = dataTypes::numberFP32(0);
                                }
                            }
                        }
                }


              if (dftParams.useDeviceDirectAllReduce)
                {
                  dftfe::utils::deviceMemcpyAsyncH2D(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSPTemp.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32),
                    streamDeviceCCL);

                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      rotationMatBlockSPTemp.begin(),
                      rotationMatBlockSPTemp.begin(),
                      BVec * D,
                      tempRealFP32.begin(),
                      tempImagFP32.begin(),
                      streamDeviceCCL);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      rotationMatBlockSPTemp.begin(),
                      rotationMatBlockSPTemp.begin(),
                      BVec * D,
                      streamDeviceCCL);
                }
              else
                {
                  MPI_Allreduce(MPI_IN_PLACE,
                                rotationMatBlockHostSP.begin(),
                                BVec * D,
                                dataTypes::mpi_type_id(
                                  rotationMatBlockHostSP.begin()),
                                MPI_SUM,
                                mpiCommDomain);

                  dftfe::utils::deviceMemcpyH2D(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSP.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32));
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                  dftfe::utils::deviceStreamWaitEvent(streamDeviceCCL,
                                                      computeEvents[blockCount],
                                                      0);

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  dftfe::utils::deviceEventRecord(communEvents[blockCount],
                                                  streamDeviceCCL);
                  if (dftfe::utils::deviceEventSynchronize(
                        communEvents[blockCount]) ==
                      dftfe::utils::deviceSuccess)
                    rotationMatBlockSP.swap(rotationMatBlockSPTemp);
                }

              for (unsigned int idof = 0; idof < maxNumLocalDofs;
                   idof += dofsBlockSize)
                {
                  // Correct block dimensions if block "goes off edge of" the
                  // matrix
                  unsigned int BDof = 0;
                  if (M >= idof)
                    BDof = std::min(dofsBlockSize, M - idof);

                  if (BDof != 0)
                    {
                      BLASWrapperPtr->xgemm('N',
                                            'N',
                                            BVec,
                                            BDof,
                                            D,
                                            &scalarCoeffAlphaSP,
                                            rotationMatBlockSP.begin(),
                                            BVec,
                                            XSP.begin() + idof * N,
                                            N,
                                            &scalarCoeffBetaSP,
                                            rotatedVectorsMatBlockSP.begin(),
                                            BVec);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                      addSubspaceRotatedBlockToXKernel<<<
                        (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        streamCompute>>>(
                        BDof,
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(X),
                        idof,
                        jvec,
                        N);
#elif DFTFE_WITH_DEVICE_LANG_HIP
                      hipLaunchKernelGGL(
                        addSubspaceRotatedBlockToXKernel,
                        (BVec * BDof + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        streamCompute,
                        BDof,
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(X),
                        idof,
                        jvec,
                        N);
#endif
                    }
                } // block loop over dofs
            }     // band parallelization
          blockCount++;
        } // block loop over vectors


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(communEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);
    }

    void
    fillParallelOverlapMatScalapack(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlock(N * vectorsBlockSize, dataTypes::number(0));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHost;
      overlapMatrixBlockHost.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      dftfe::utils::deviceStream_t streamDeviceCCL = 0;

      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
        }

      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);


          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Comptute local XTrunc^{T}*XcBlock.
              BLASWrapperPtr->xgemm(
                'N',
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T',
                D,
                B,
                M,
                &scalarCoeffAlpha,
                X + ivec,
                N,
                X + ivec,
                N,
                &scalarCoeffBeta,
                overlapMatrixBlock.begin(),
                D);


              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      overlapMatrixBlock.begin(),
                      overlapMatrixBlock.begin(),
                      D * B,
                      tempReal.begin(),
                      tempImag.begin(),
                      streamDeviceCCL);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      overlapMatrixBlock.begin(),
                      overlapMatrixBlock.begin(),
                      D * B,
                      streamDeviceCCL);
                }

              dftfe::utils::deviceMemcpyD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHost.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlock.begin()),
                D * B * sizeof(dataTypes::number));

              // Sum local XTrunc^{T}*XcBlock across domain decomposition
              // processors
              if (!dftParams.useDeviceDirectAllReduce)
                MPI_Allreduce(MPI_IN_PLACE,
                              overlapMatrixBlockHost.begin(),
                              D * B,
                              dataTypes::mpi_type_id(
                                overlapMatrixBlockHost.begin()),
                              MPI_SUM,
                              mpiCommDomain);


              // Copying only the lower triangular part to the ScaLAPACK overlap
              // matrix
              if (processGrid->is_process_active())
                for (unsigned int i = 0; i < B; ++i)
                  if (globalToLocalColumnIdMap.find(i + ivec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const unsigned int localColumnId =
                        globalToLocalColumnIdMap[i + ivec];
                      for (unsigned int j = ivec + i; j < N; ++j)
                        {
                          std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(j);
                          if (it != globalToLocalRowIdMap.end())
                            overlapMatPar.local_el(it->second, localColumnId) =
                              overlapMatrixBlockHost[i * D + j - ivec];
                        }
                    }

            } // band parallelization
        }     // end block loop

      if (numberBandGroups > 1)
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
    }

    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication in the computation of overlap
    /// matrix/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times XBlock
    // COP denotes Device->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed.
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
    // scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void
    fillParallelOverlapMatScalapackAsyncComputeCommun(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for data movement and computation
      dftfe::utils::deviceStream_t streamCompute, streamDataMove;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDataMove);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      // create pinned memory used later to copy from Device->CPU
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHost;
      overlapMatrixBlockHost.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      // allocate device vectors to be used later
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlock(N * vectorsBlockSize, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockNext(N * vectorsBlockSize, dataTypes::number(0));

      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);
          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock.
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    D,
                    B,
                    M,
                    &scalarCoeffAlpha,
                    X + ivec,
                    N,
                    X + ivec,
                    N,
                    &scalarCoeffBeta,
                    overlapMatrixBlock.begin(),
                    D);

                  // record completion of compute for first block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                }

              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future operations in the streamDataMove will only occur after
              // both the compute on currentblock and swap is over. Note that at
              // this point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((dftfe::utils::deviceEventSynchronize(
                     computeEvents[blockCount]) ==
                   dftfe::utils::deviceSuccess) &&
                  (ivec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                overlapMatrixBlock.swap(overlapMatrixBlockNext);

              const unsigned int ivecNew = ivec + vectorsBlockSize;
              const unsigned int DNew    = N - ivecNew;
              const unsigned int BNew    = min(vectorsBlockSize, N - ivecNew);


              // start computations on the next block
              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // evaluate X^{T} times XBlock
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    DNew,
                    BNew,
                    M,
                    &scalarCoeffAlpha,
                    X + ivecNew,
                    N,
                    X + ivecNew,
                    N,
                    &scalarCoeffBeta,
                    overlapMatrixBlockNext.begin(),
                    DNew);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      overlapMatrixBlock.begin(),
                      overlapMatrixBlock.begin(),
                      D * B,
                      tempReal.begin(),
                      tempImag.begin(),
                      streamDataMove);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      overlapMatrixBlock.begin(),
                      overlapMatrixBlock.begin(),
                      D * B,
                      streamDataMove);
                }

              dftfe::utils::deviceMemcpyAsyncD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHost.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlock.begin()),
                D * B * sizeof(dataTypes::number),
                streamDataMove);

              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (!dftParams.useDeviceDirectAllReduce)
                    MPI_Allreduce(MPI_IN_PLACE,
                                  overlapMatrixBlockHost.begin(),
                                  D * B,
                                  dataTypes::mpi_type_id(
                                    overlapMatrixBlockHost.begin()),
                                  MPI_SUM,
                                  mpiCommDomain);


                  // Copying only the lower triangular part to the ScaLAPACK
                  // overlap matrix
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < B; ++i)
                      if (globalToLocalColumnIdMap.find(i + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i + ivec];
                          for (unsigned int j = ivec + i; j < N; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHost[i * D + j - ivec];
                            }
                        }
                }
            } // band parallelization

          blockCount += 1;
        } // end block loop


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(copyEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDataMove);

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }


    void
    fillParallelOverlapMatMixedPrecScalapack(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSP(N * vectorsBlockSize, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDP(vectorsBlockSize * vectorsBlockSize,
                             dataTypes::number(0));

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));

      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostDP;
      overlapMatrixBlockHostDP.resize(vectorsBlockSize * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostDP.begin(),
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostSP;
      overlapMatrixBlockHostSP.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostSP.begin(),
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      dftfe::utils::deviceStream_t streamDeviceCCL = 0;

      const dataTypes::number     scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number     scalarCoeffBeta  = dataTypes::number(0);
      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP = dataTypes::numberFP32(0);

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);


          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              BLASWrapperPtr->xgemm(
                'N',
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T',
                B,
                B,
                M,
                &scalarCoeffAlpha,
                X + ivec,
                N,
                X + ivec,
                N,
                &scalarCoeffBeta,
                overlapMatrixBlockDP.begin(),
                B);

              const unsigned int DRem = D - B;

              if (DRem != 0)
                {
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    DRem,
                    B,
                    M,
                    &scalarCoeffAlphaSP,
                    XSP.begin() + ivec + B,
                    N,
                    XSP.begin() + ivec,
                    N,
                    &scalarCoeffBetaSP,
                    overlapMatrixBlockSP.begin(),
                    DRem);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        B * B,
                        DRem * B,
                        tempReal.begin(),
                        tempRealFP32.begin(),
                        tempImag.begin(),
                        tempImagFP32.begin(),
                        streamDeviceCCL);
                  else
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        B * B,
                        DRem * B,
                        streamDeviceCCL);
                }

              dftfe::utils::deviceMemcpyD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostDP.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockDP.begin()),
                B * B * sizeof(dataTypes::number));

              dftfe::utils::deviceMemcpyD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostSP.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockSP.begin()),
                DRem * B * sizeof(dataTypes::numberFP32));

              if (!dftParams.useDeviceDirectAllReduce)
                {
                  // Sum local XTrunc^{T}*XcBlock for double precision across
                  // domain decomposition processors
                  MPI_Allreduce(MPI_IN_PLACE,
                                overlapMatrixBlockHostDP.begin(),
                                B * B,
                                dataTypes::mpi_type_id(
                                  overlapMatrixBlockHostDP.begin()),
                                MPI_SUM,
                                mpiCommDomain);

                  // Sum local XTrunc^{T}*XcBlock for single precision across
                  // domain decomposition processors
                  MPI_Allreduce(MPI_IN_PLACE,
                                overlapMatrixBlockHostSP.begin(),
                                DRem * B,
                                dataTypes::mpi_type_id(
                                  overlapMatrixBlockHostSP.begin()),
                                MPI_SUM,
                                mpiCommDomain);
                }

              // Copying only the lower triangular part to the ScaLAPACK overlap
              // matrix
              if (processGrid->is_process_active())
                for (unsigned int i = 0; i < B; ++i)
                  if (globalToLocalColumnIdMap.find(i + ivec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const unsigned int localColumnId =
                        globalToLocalColumnIdMap[i + ivec];
                      for (unsigned int j = ivec + i; j < ivec + B; ++j)
                        {
                          std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(j);
                          if (it != globalToLocalRowIdMap.end())
                            overlapMatPar.local_el(it->second, localColumnId) =
                              overlapMatrixBlockHostDP[i * B + j - ivec];
                        }

                      for (unsigned int j = ivec + B; j < N; ++j)
                        {
                          std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(j);
                          if (it != globalToLocalRowIdMap.end())
                            overlapMatPar.local_el(it->second, localColumnId) =
                              overlapMatrixBlockHostSP[i * DRem + j - ivec - B];
                        }
                    }
            } // band parallelization
        }     // end block loop

      if (numberBandGroups > 1)
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
    }


    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication in the computation of overlap matrix using
    /// mixed precision arithmetic/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times XBlock
    // COP denotes Device->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed.
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
    // scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void
    fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for Device->CPU copy and computation
      dftfe::utils::deviceStream_t streamCompute, streamDataMove;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDataMove);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSP(N * vectorsBlockSize, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDP(vectorsBlockSize * vectorsBlockSize,
                             dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSPNext(N * vectorsBlockSize,
                                 dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDPNext(vectorsBlockSize * vectorsBlockSize,
                                 dataTypes::number(0));

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));


      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostDP;
      overlapMatrixBlockHostDP.resize(vectorsBlockSize * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostDP.begin(),
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostSP;
      overlapMatrixBlockHostSP.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostSP.begin(),
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      const dataTypes::number     scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number     scalarCoeffBeta  = dataTypes::number(0);
      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP = dataTypes::numberFP32(0);

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);
          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    B,
                    B,
                    M,
                    &scalarCoeffAlpha,
                    X + ivec,
                    N,
                    X + ivec,
                    N,
                    &scalarCoeffBeta,
                    overlapMatrixBlockDP.begin(),
                    B);

                  const unsigned int DRem = D - B;

                  if (DRem != 0)
                    {
                      BLASWrapperPtr->xgemm(
                        'N',
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          'C' :
                          'T',
                        DRem,
                        B,
                        M,
                        &scalarCoeffAlphaSP,
                        XSP.begin() + ivec + B,
                        N,
                        XSP.begin() + ivec,
                        N,
                        &scalarCoeffBetaSP,
                        overlapMatrixBlockSP.begin(),
                        DRem);
                    }

                  // record completion of compute for first block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                }

              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future operations in the streamDataMove will only occur after
              // both the compute on currentblock and swap is over. Note that at
              // this point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((dftfe::utils::deviceEventSynchronize(
                     computeEvents[blockCount]) ==
                   dftfe::utils::deviceSuccess) &&
                  (ivec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                {
                  overlapMatrixBlockDP.swap(overlapMatrixBlockDPNext);
                  overlapMatrixBlockSP.swap(overlapMatrixBlockSPNext);
                }

              const unsigned int DRem = D - B;

              const unsigned int ivecNew = ivec + vectorsBlockSize;
              const unsigned int DNew    = N - ivecNew;
              const unsigned int BNew    = min(vectorsBlockSize, N - ivecNew);

              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // evaluate X^{T} times XBlock
                  BLASWrapperPtr->xgemm(
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    BNew,
                    BNew,
                    M,
                    &scalarCoeffAlpha,
                    X + ivecNew,
                    N,
                    X + ivecNew,
                    N,
                    &scalarCoeffBeta,
                    overlapMatrixBlockDPNext.begin(),
                    BNew);

                  const unsigned int DRemNew = DNew - BNew;

                  if (DRemNew != 0)
                    {
                      BLASWrapperPtr->xgemm(
                        'N',
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          'C' :
                          'T',
                        DRemNew,
                        BNew,
                        M,
                        &scalarCoeffAlphaSP,
                        XSP.begin() + ivecNew + BNew,
                        N,
                        XSP.begin() + ivecNew,
                        N,
                        &scalarCoeffBetaSP,
                        overlapMatrixBlockSPNext.begin(),
                        DRemNew);
                    }

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        B * B,
                        DRem * B,
                        tempReal.begin(),
                        tempRealFP32.begin(),
                        tempImag.begin(),
                        tempImagFP32.begin(),
                        streamDataMove);
                  else
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        B * B,
                        DRem * B,
                        streamDataMove);
                }

              dftfe::utils::deviceMemcpyAsyncD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostDP.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockDP.begin()),
                B * B * sizeof(dataTypes::number),
                streamDataMove);

              dftfe::utils::deviceMemcpyAsyncD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostSP.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockSP.begin()),
                DRem * B * sizeof(dataTypes::numberFP32),
                streamDataMove);

              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  const unsigned int DRem = D - B;

                  if (!dftParams.useDeviceDirectAllReduce)
                    {
                      // Sum local XTrunc^{T}*XcBlock for double precision
                      // across domain decomposition processors
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostDP.begin(),
                                    B * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostDP.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);

                      // Sum local XTrunc^{T}*XcBlock for single precision
                      // across domain decomposition processors
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostSP.begin(),
                                    DRem * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostSP.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);
                    }

                  // Copying only the lower triangular part to the ScaLAPACK
                  // overlap matrix
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < B; ++i)
                      if (globalToLocalColumnIdMap.find(i + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i + ivec];
                          for (unsigned int j = ivec + i; j < ivec + B; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostDP[i * B + j - ivec];
                            }

                          for (unsigned int j = ivec + B; j < N; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostSP[i * DRem + j - ivec -
                                                           B];
                            }
                        }
                }
            } // band parallelization

          blockCount += 1;

        } // end block loop


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(copyEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDataMove);

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }


    void
    fillParallelOverlapMatMixedPrecCommunScalapackAsyncComputeCommun(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for data movement and computation
      dftfe::utils::deviceStream_t streamCompute, streamDataMove;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDataMove);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      // create pinned memory used later to copy from Device->CPU
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostDP;
      overlapMatrixBlockHostDP.resize(vectorsBlockSize * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostDP.begin(),
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostSP;
      overlapMatrixBlockHostSP.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostSP.begin(),
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      // allocate device vectors to be used later
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlock(N * vectorsBlockSize, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockNext(N * vectorsBlockSize, dataTypes::number(0));


      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDP(vectorsBlockSize * vectorsBlockSize,
                             dataTypes::number(0));


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSP(N * vectorsBlockSize, dataTypes::numberFP32(0));


      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);
          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock.
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  BLASWrapperPtr->xgemm(
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    D,
                    B,
                    M,
                    &scalarCoeffAlpha,
                    X + ivec,
                    N,
                    X + ivec,
                    N,
                    &scalarCoeffBeta,
                    overlapMatrixBlock.begin(),
                    D);

                  // record completion of compute for first block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                }

              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future operations in the streamDataMove will only occur after
              // both the compute on currentblock and swap is over. Note that at
              // this point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((dftfe::utils::deviceEventSynchronize(
                     computeEvents[blockCount]) ==
                   dftfe::utils::deviceSuccess) &&
                  (ivec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                overlapMatrixBlock.swap(overlapMatrixBlockNext);

              const unsigned int ivecNew = ivec + vectorsBlockSize;
              const unsigned int DNew    = N - ivecNew;
              const unsigned int BNew    = min(vectorsBlockSize, N - ivecNew);


              // start computations on the next block
              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // evaluate X^{T} times XBlock
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    DNew,
                    BNew,
                    M,
                    &scalarCoeffAlpha,
                    X + ivecNew,
                    N,
                    X + ivecNew,
                    N,
                    &scalarCoeffBeta,
                    overlapMatrixBlockNext.begin(),
                    DNew);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }


              const unsigned int DRem = D - B;

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
              copyFromOverlapMatBlockToDPSPBlocks<<<
                (D * B + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                  dftfe::utils::DEVICE_BLOCK_SIZE,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                0,
                streamDataMove>>>(B,
                                  D,
                                  dftfe::utils::makeDataTypeDeviceCompatible(
                                    overlapMatrixBlock.begin()),
                                  dftfe::utils::makeDataTypeDeviceCompatible(
                                    overlapMatrixBlockDP.begin()),
                                  dftfe::utils::makeDataTypeDeviceCompatible(
                                    overlapMatrixBlockSP.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
              hipLaunchKernelGGL(copyFromOverlapMatBlockToDPSPBlocks,
                                 (D * B +
                                  (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE,
                                 dftfe::utils::DEVICE_BLOCK_SIZE,
                                 0,
                                 streamDataMove,
                                 B,
                                 D,
                                 dftfe::utils::makeDataTypeDeviceCompatible(
                                   overlapMatrixBlock.begin()),
                                 dftfe::utils::makeDataTypeDeviceCompatible(
                                   overlapMatrixBlockDP.begin()),
                                 dftfe::utils::makeDataTypeDeviceCompatible(
                                   overlapMatrixBlockSP.begin()));
#endif

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        B * B,
                        DRem * B,
                        tempReal.begin(),
                        tempRealFP32.begin(),
                        tempImag.begin(),
                        tempImagFP32.begin(),
                        streamDataMove);
                  else
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        overlapMatrixBlockDP.begin(),
                        overlapMatrixBlockSP.begin(),
                        B * B,
                        DRem * B,
                        streamDataMove);
                }

              dftfe::utils::deviceMemcpyAsyncD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostDP.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockDP.begin()),
                B * B * sizeof(dataTypes::number),
                streamDataMove);

              dftfe::utils::deviceMemcpyAsyncD2H(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostSP.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockSP.begin()),
                DRem * B * sizeof(dataTypes::numberFP32),
                streamDataMove);


              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  if (!dftParams.useDeviceDirectAllReduce)
                    {
                      // Sum local XTrunc^{T}*XcBlock for double precision
                      // across domain decomposition processors
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostDP.begin(),
                                    B * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostDP.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);

                      // Sum local XTrunc^{T}*XcBlock for single precision
                      // across domain decomposition processors
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostSP.begin(),
                                    DRem * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostSP.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);
                    }

                  // Copying only the lower triangular part to the ScaLAPACK
                  // overlap matrix
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < B; ++i)
                      if (globalToLocalColumnIdMap.find(i + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i + ivec];
                          for (unsigned int j = ivec + i; j < ivec + B; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostDP[i * B + j - ivec];
                            }

                          for (unsigned int j = ivec + B; j < N; ++j)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostSP[i * DRem + j - ivec -
                                                           B];
                            }
                        }
                }
            } // band parallelization

          blockCount += 1;
        } // end block loop


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(copyEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDataMove);

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }


    void
    computeEigenResidualNorm(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const std::vector<double> &                          eigenValues,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      const MPI_Comm &                                     interBandGroupComm,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      std::vector<double> &residualNorm,
      const dftParameters &dftParams,
      const bool           useBandParal)
    {
      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);


      const unsigned int vectorsBlockSize = dftParams.wfcBlockSize;
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        residualNormSquareDevice(N, 0);
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        HXBlockFull(vectorsBlockSize * M, dataTypes::number(0));
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        residualSqDevice(vectorsBlockSize * M, 0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        onesVecDevice(M, 1.0);


      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        eigenValuesDevice(N, 0);
      dftfe::utils::deviceMemcpyH2D(eigenValuesDevice.begin(),
                                    &eigenValues[0],
                                    N * sizeof(double));

      const bool   scaleFlag = false;
      const double scalar    = 1.0;
      const double alpha = 1.0, beta = 0;

      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - jvec);


          if (((jvec + B) <=
                 bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
               (jvec + B) >
                 bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]) ||
              !useBandParal)
            {
              const unsigned int chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                {
                  BLASWrapperPtr->stridedCopyToBlockConstantStride(
                    chebyBlockSize, N, M, k, X, XBlock.begin());

                  // evaluate H times XBlock^{T} and store in HXBlock^{T}
                  operatorMatrix.HX(XBlock, 1.0, 0.0, 0.0, HXBlock);
                  BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                    B,
                    chebyBlockSize,
                    M,
                    k - jvec,
                    HXBlock.begin(),
                    HXBlockFull.begin());
                }

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
              computeResidualDeviceKernel<<<
                (B + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                  dftfe::utils::DEVICE_BLOCK_SIZE * M,
                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                B,
                M,
                N,
                jvec,
                eigenValuesDevice.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(X),
                dftfe::utils::makeDataTypeDeviceCompatible(HXBlockFull.begin()),
                residualSqDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
              hipLaunchKernelGGL(computeResidualDeviceKernel,
                                 (B + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                   dftfe::utils::DEVICE_BLOCK_SIZE * M,
                                 dftfe::utils::DEVICE_BLOCK_SIZE,
                                 0,
                                 0,
                                 B,
                                 M,
                                 N,
                                 jvec,
                                 eigenValuesDevice.begin(),
                                 dftfe::utils::makeDataTypeDeviceCompatible(X),
                                 dftfe::utils::makeDataTypeDeviceCompatible(
                                   HXBlockFull.begin()),
                                 residualSqDevice.begin());
#endif

              BLASWrapperPtr->xgemm('N',
                                    'T',
                                    1,
                                    B,
                                    M,
                                    &alpha,
                                    onesVecDevice.begin(),
                                    1,
                                    residualSqDevice.begin(),
                                    B,
                                    &beta,
                                    residualNormSquareDevice.begin() + jvec,
                                    1);
            }
        }


      dftfe::utils::deviceMemcpyD2H(&residualNorm[0],
                                    residualNormSquareDevice.begin(),
                                    N * sizeof(double));

      MPI_Allreduce(
        MPI_IN_PLACE, &residualNorm[0], N, MPI_DOUBLE, MPI_SUM, mpiCommDomain);

      if (numberBandGroups > 1 || !useBandParal)
        MPI_Allreduce(MPI_IN_PLACE,
                      &residualNorm[0],
                      N,
                      MPI_DOUBLE,
                      MPI_SUM,
                      interBandGroupComm);


      if (dftParams.verbosity >= 4)
        {
          if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
            std::cout << "L-2 Norm of residue   :" << std::endl;
        }
      for (unsigned int iWave = 0; iWave < N; ++iWave)
        residualNorm[iWave] = std::sqrt(residualNorm[iWave]);

      if (dftParams.verbosity >= 4 &&
          dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
        for (unsigned int iWave = 0; iWave < N; ++iWave)
          std::cout << "eigen vector " << iWave << ": " << residualNorm[iWave]
                    << std::endl;

      if (dftParams.verbosity >= 4)
        if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
          std::cout << std::endl;
    }

    // X^{T}*HConj*XConj
    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
         const dataTypes::number *                            X,
         distributedDeviceVec<dataTypes::number> &            XBlock,
         distributedDeviceVec<dataTypes::number> &            HXBlock,
         const unsigned int                                   M,
         const unsigned int                                   N,
         std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
           &                                              BLASWrapperPtr,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         utils::DeviceCCLWrapper &devicecclMpiCommDomain,
         const MPI_Comm &         mpiCommDomain,
         const MPI_Comm &         interBandGroupComm,
         const dftParameters &    dftParams,
         const bool               onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);



      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projHamBlockHost;
      projHamBlockHost.resize(vectorsBlockSize * N, 0);
      std::memset(projHamBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));

      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const unsigned int chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                {
                  BLASWrapperPtr->stridedCopyToBlockConstantStride(
                    chebyBlockSize, N, M, k, X, XBlock.begin());

                  // evaluate XBlock^{T} times H^{T} and store in HXBlock
                  operatorMatrix.HX(
                    XBlock,
                    1.0,
                    0.0,
                    0.0,
                    HXBlock,
                    onlyHPrimePartForFirstOrderDensityMatResponse);

                  BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                    B,
                    chebyBlockSize,
                    M,
                    k - jvec,
                    HXBlock.begin(),
                    HXBlockFull.begin());
                }

              // Comptute local XTrunc^{T}*HConj*XConj.
              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const unsigned int D          = N - jvec;
              BLASWrapperPtr->xgemm(
                'N',
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T',
                D,
                B,
                M,
                &alpha,
                X + jvec,
                N,
                HXBlockFull.begin(),
                B,
                &beta,
                projHamBlock.begin(),
                D);

              dftfe::utils::deviceMemcpyD2H(
                projHamBlockHost.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  projHamBlock.begin()),
                D * B * sizeof(dataTypes::number));


              // Sum local projHamBlock across domain decomposition processors
              MPI_Allreduce(MPI_IN_PLACE,
                            projHamBlockHost.begin(),
                            D * B,
                            dataTypes::mpi_type_id(projHamBlockHost.begin()),
                            MPI_SUM,
                            mpiCommDomain);

              // Copying only the lower triangular part to the ScaLAPACK
              // projected Hamiltonian matrix
              if (processGrid->is_process_active())
                for (unsigned int j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const unsigned int localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (unsigned int i = j + jvec; i < N; ++i)
                        {
                          std::unordered_map<unsigned int,
                                             unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projHamPar.local_el(it->second, localColumnId) =
                              projHamBlockHost[j * D + i - jvec];
                        }
                    }

            } // band parallelization
        }


      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }

    // X^{T}*HConj*XConj  with overlap of computation and
    // communication
    void
    XtHXOverlapComputeCommun(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number *                            X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 mpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftParameters &                            dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      /////////////PSEUDO CODE for the implementation below for Overlapping
      /// compute and communication/////////////////
      //
      // In the algorithm below the communication and computation of two
      // consecutive blocks of wavefunctions: block i and block i+1 are
      // overlapped.
      // ----------------------------------------------------------
      // CMP denotes computuation of X^{T} times HXBlock
      // COP denotes Device->CPU copy of X^{T} times HXBlock
      // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to
      // scalapack matrix
      // ----------------------------------------------------------
      // Two Device streams are created: compute and copy
      // CMP is performed in compute Device stream and COP is performed in copy
      // Device stream. COP for a block can only start after the CMP for that
      // block in the compute stream is completed. COM is performed for a block
      // only after COP even for that block is completed.
      //
      // In a blocked loop do:
      // 1) [CMP] Call compute on first block (edge case only for first
      // iteration) 2) Wait for CMP event for current block to be completed. 3)
      // Swap current and next block memory (all iterations except edge case) 4)
      // [COP] Call copy on current block 5) [CMP] Call compute on next block 6)
      // Wait for COP event for current block to be completed 7) [COM] Perform
      // blocking MPI_Allreduce on curent block and copy to scalapack matrix
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);



      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for Device->CPU copy and computation
      dftfe::utils::deviceStream_t streamCompute, streamDataMove;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDataMove);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projHamBlockHost;
      projHamBlockHost.resize(vectorsBlockSize * N, 0);
      std::memset(projHamBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockNext(vectorsBlockSize * N, dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const unsigned int chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const unsigned int D          = N - jvec;

              // handle edge case for the first block or the first block in the
              // band group in case of band parallelization
              if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // compute HXBlockFull in an inner loop over blocks of B
                  // wavefunction vectors
                  for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate H times XBlock^{T} and store in HXBlock^{T}
                      operatorMatrix.HX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        HXBlock,
                        onlyHPrimePartForFirstOrderDensityMatResponse);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - jvec,
                        HXBlock.begin(),
                        HXBlockFull.begin());
                    }

                  // evalute X^{T} times HXBlock
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    D,
                    B,
                    M,
                    &alpha,
                    X + jvec,
                    N,
                    HXBlockFull.begin(),
                    B,
                    &beta,
                    projHamBlock.begin(),
                    D);

                  // record completion of compute for first block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                }


              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future calls in the streamDataMove will only occur after both
              // the compute on currentblock and swap is over. Note that at this
              // point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((dftfe::utils::deviceEventSynchronize(
                     computeEvents[blockCount]) ==
                   dftfe::utils::deviceSuccess) &&
                  (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                projHamBlock.swap(projHamBlockNext);

              const unsigned int jvecNew = jvec + vectorsBlockSize;
              const unsigned int DNew    = N - jvecNew;

              // start computations on the next block
              if (jvecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  for (unsigned int k = jvecNew; k < jvecNew + B;
                       k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate H times XBlock^{T} and store in HXBlock^{T}
                      operatorMatrix.HX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        HXBlock,
                        onlyHPrimePartForFirstOrderDensityMatResponse);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - jvecNew,
                        HXBlock.begin(),
                        HXBlockFull.begin());
                    }

                  // evalute X^{T} times HXBlock
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    DNew,
                    B,
                    M,
                    &alpha,
                    X + jvecNew,
                    N,
                    HXBlockFull.begin(),
                    B,
                    &beta,
                    projHamBlockNext.begin(),
                    DNew);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // Sum local projHamBlock across domain decomposition
                  // processors
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    {
                      devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                        projHamBlock.begin(),
                        projHamBlock.begin(),
                        D * B,
                        tempReal.begin(),
                        tempImag.begin(),
                        streamDataMove);
                    }
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      projHamBlock.begin(),
                      projHamBlock.begin(),
                      D * B,
                      streamDataMove);
                }

              dftfe::utils::deviceMemcpyAsyncD2H(
                projHamBlockHost.begin(),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  projHamBlock.begin()),
                D * B * sizeof(dataTypes::number),
                streamDataMove);

              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matrix
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  // Sum local projHamBlock across domain decomposition
                  // processors
                  if (!dftParams.useDeviceDirectAllReduce)
                    MPI_Allreduce(MPI_IN_PLACE,
                                  projHamBlockHost.begin(),
                                  D * B,
                                  dataTypes::mpi_type_id(
                                    projHamBlockHost.begin()),
                                  MPI_SUM,
                                  mpiCommDomain);

                  // Copying only the lower triangular part to the ScaLAPACK
                  // projected Hamiltonian matrix
                  if (processGrid->is_process_active())
                    for (unsigned int j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + jvec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[j + jvec];
                          for (unsigned int i = j + jvec; i < N; ++i)
                            {
                              std::unordered_map<unsigned int,
                                                 unsigned int>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                projHamPar.local_el(it->second, localColumnId) =
                                  projHamBlockHost[j * D + i - jvec];
                            }
                        }
                }

            } // band parallelization
          blockCount += 1;
        }

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(copyEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDataMove);

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }

    // X^{T}*HConj*XConj  (Xc denotes complex conjugate)
    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute
    /// and communication/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times HXBlock
    // COP denotes Device->CPU copy of X^{T} times HXBlock
    // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed.
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
    // scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    XtHXMixedPrecOverlapComputeCommun(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number *                            X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const unsigned int                                   Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 mpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftParameters &                            dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);


      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

      const unsigned int numberBlocks = N / vectorsBlockSize;

      // create device compute and copy streams
      dftfe::utils::deviceStream_t streamCompute, streamDataMove;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDataMove);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XFP32(M * N, dataTypes::numberFP32(0.0));

      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XFP32.begin());

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projHamBlockHost;
      projHamBlockHost.resize(vectorsBlockSize * N, 0);
      std::memset(projHamBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projHamBlockHostFP32;
      projHamBlockHostFP32.resize(vectorsBlockSize * N, 0);
      std::memset(projHamBlockHostFP32.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        HXBlockFullFP32(vectorsBlockSize * M, dataTypes::numberFP32(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockFP32(vectorsBlockSize * N, dataTypes::numberFP32(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockNext(vectorsBlockSize * N, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockFP32Next(vectorsBlockSize * N, dataTypes::numberFP32(0.0));


      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const unsigned int chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const dataTypes::numberFP32 alphaFP32 =
                                            dataTypes::numberFP32(1.0),
                                          betaFP32 = dataTypes::numberFP32(0.0);
              const unsigned int D                 = N - jvec;

              // handle edge case for the first block or the first block in the
              // band group in case of band parallelization
              if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // compute HXBlockFull or HXBlockFullFP32 in an inner loop
                  // over blocks of B wavefunction vectors
                  for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate H times XBlock^{T} and store in HXBlock^{T}
                      HXBlock.setValue(0);
                      const bool   scaleFlag = false;
                      const double scalar    = 1.0;
                      if (!(jvec + B > Noc))
                        {
                          XBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                          HXBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        }
                      operatorMatrix.HX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        HXBlock,
                        onlyHPrimePartForFirstOrderDensityMatResponse);
                      if (!(jvec + B > Noc))
                        {
                          XBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                          HXBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        }

                      if (jvec + B > Noc)
                        BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                          B,
                          chebyBlockSize,
                          M,
                          k - jvec,
                          HXBlock.begin(),
                          HXBlockFull.begin());
                      else
                        BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                          B,
                          chebyBlockSize,
                          M,
                          k - jvec,
                          HXBlock.begin(),
                          HXBlockFullFP32.begin());
                    }

                  // evaluate X^{T} times HXBlockFullConj or XFP32^{T} times
                  // HXBlockFullFP32Conj
                  if (jvec + B > Noc)
                    BLASWrapperPtr->xgemm(
                      'N',
                      std::is_same<dataTypes::number,
                                   std::complex<double>>::value ?
                        'C' :
                        'T',
                      D,
                      B,
                      M,
                      &alpha,
                      X + jvec,
                      N,
                      HXBlockFull.begin(),
                      B,
                      &beta,
                      projHamBlock.begin(),
                      D);
                  else
                    BLASWrapperPtr->xgemm(
                      'N',
                      std::is_same<dataTypes::numberFP32,
                                   std::complex<float>>::value ?
                        'C' :
                        'T',
                      D,
                      B,
                      M,
                      &alphaFP32,
                      XFP32.begin() + jvec,
                      N,
                      HXBlockFullFP32.begin(),
                      B,
                      &betaFP32,
                      projHamBlockFP32.begin(),
                      D);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                }

              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future calls in the streamDataMove will only occur after both
              // the compute on currentblock and swap is over. Note that at this
              // point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((dftfe::utils::deviceEventSynchronize(
                     computeEvents[blockCount]) ==
                   dftfe::utils::deviceSuccess) &&
                  (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                {
                  if (jvec + B > Noc)
                    projHamBlock.swap(projHamBlockNext);
                  else
                    projHamBlockFP32.swap(projHamBlockFP32Next);
                }

              const unsigned int jvecNew = jvec + vectorsBlockSize;
              const unsigned int DNew    = N - jvecNew;

              if (jvecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // compute HXBlockFull or HXBlockFullFP32 in an inner loop
                  // over blocks of B wavefunction vectors
                  for (unsigned int k = jvecNew; k < jvecNew + B;
                       k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate H times XBlock^{T} and store in HXBlock^{T}
                      HXBlock.setValue(0);
                      const bool   scaleFlag = false;
                      const double scalar    = 1.0;
                      if (!(jvecNew + B > Noc))
                        {
                          XBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                          HXBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        }
                      operatorMatrix.HX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        HXBlock,
                        onlyHPrimePartForFirstOrderDensityMatResponse);
                      if (!(jvecNew + B > Noc))
                        {
                          XBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                          HXBlock.setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        }

                      if (jvecNew + B > Noc)
                        BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                          B,
                          chebyBlockSize,
                          M,
                          k - jvecNew,
                          HXBlock.begin(),
                          HXBlockFull.begin());
                      else
                        BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                          B,
                          chebyBlockSize,
                          M,
                          k - jvecNew,
                          HXBlock.begin(),
                          HXBlockFullFP32.begin());
                    }

                  // evaluate X^{T} times HXBlockFullConj or XFP32^{T} times
                  // HXBlockFullFP32Conj
                  if (jvecNew + B > Noc)
                    BLASWrapperPtr->xgemm(
                      'N',
                      std::is_same<dataTypes::number,
                                   std::complex<double>>::value ?
                        'C' :
                        'T',
                      DNew,
                      B,
                      M,
                      &alpha,
                      X + jvecNew,
                      N,
                      HXBlockFull.begin(),
                      B,
                      &beta,
                      projHamBlockNext.begin(),
                      DNew);
                  else
                    BLASWrapperPtr->xgemm(
                      'N',
                      std::is_same<dataTypes::numberFP32,
                                   std::complex<float>>::value ?
                        'C' :
                        'T',
                      DNew,
                      B,
                      M,
                      &alphaFP32,
                      XFP32.begin() + jvecNew,
                      N,
                      HXBlockFullFP32.begin(),
                      B,
                      &betaFP32,
                      projHamBlockFP32Next.begin(),
                      DNew);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (jvec + B > Noc)
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlock.begin(),
                          projHamBlock.begin(),
                          D * B,
                          tempReal.begin(),
                          tempImag.begin(),
                          streamDataMove);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlock.begin(),
                          projHamBlock.begin(),
                          D * B,
                          streamDataMove);
                    }
                  else
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlockFP32.begin(),
                          projHamBlockFP32.begin(),
                          D * B,
                          tempRealFP32.begin(),
                          tempImagFP32.begin(),
                          streamDataMove);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlockFP32.begin(),
                          projHamBlockFP32.begin(),
                          D * B,
                          streamDataMove);
                    }
                }

              if (jvec + B > Noc)
                dftfe::utils::deviceMemcpyAsyncD2H(
                  projHamBlockHost.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projHamBlock.begin()),
                  D * B * sizeof(dataTypes::number),
                  streamDataMove);
              else
                dftfe::utils::deviceMemcpyAsyncD2H(
                  projHamBlockHostFP32.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projHamBlockFP32.begin()),
                  D * B * sizeof(dataTypes::numberFP32),
                  streamDataMove);

              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matrix
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  if (jvec + B > Noc)
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      projHamBlockHost.begin(),
                                      D * B,
                                      dataTypes::mpi_type_id(
                                        projHamBlockHost.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (unsigned int j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const unsigned int localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (unsigned int i = j + jvec; i < N; ++i)
                                {
                                  std::unordered_map<unsigned int,
                                                     unsigned int>::iterator
                                    it = globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHost[j * D + i - jvec];
                                }
                            }
                    }
                  else
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      projHamBlockHostFP32.begin(),
                                      D * B,
                                      dataTypes::mpi_type_id(
                                        projHamBlockHostFP32.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (unsigned int j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const unsigned int localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (unsigned int i = j + jvec; i < N; ++i)
                                {
                                  std::unordered_map<unsigned int,
                                                     unsigned int>::iterator
                                    it = globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHostFP32[j * D + i - jvec];
                                }
                            }
                    }
                }
            } // band parallelization
          blockCount += 1;
        }

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(copyEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDataMove);

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }

    // X^{T}*HConj*XConj  with overlap of computation and
    // communication
    void
    XtHXMixedPrecCommunOverlapComputeCommun(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number *                            X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const unsigned int                                   Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 mpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftParameters &                            dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      /////////////PSEUDO CODE for the implementation below for Overlapping
      /// compute and communication/////////////////
      //
      // In the algorithm below the communication and computation of two
      // consecutive blocks of wavefunctions: block i and block i+1 are
      // overlapped.
      // ----------------------------------------------------------
      // CMP denotes computuation of X^{T} times HXBlock
      // COP denotes Device->CPU copy of X^{T} times HXBlock
      // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to
      // scalapack matrix
      // ----------------------------------------------------------
      // Two Device streams are created: compute and copy
      // CMP is performed in compute Device stream and COP is performed in copy
      // Device stream. COP for a block can only start after the CMP for that
      // block in the compute stream is completed. COM is performed for a block
      // only after COP even for that block is completed.
      //
      // In a blocked loop do:
      // 1) [CMP] Call compute on first block (edge case only for first
      // iteration) 2) Wait for CMP event for current block to be completed. 3)
      // Swap current and next block memory (all iterations except edge case) 4)
      // [COP] Call copy on current block 5) [CMP] Call compute on next block 6)
      // Wait for COP event for current block to be completed 7) [COM] Perform
      // blocking MPI_Allreduce on curent block and copy to scalapack matrix
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


      std::unordered_map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::unordered_map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);



      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for Device->CPU copy and computation
      dftfe::utils::deviceStream_t streamCompute, streamDataMove;
      dftfe::utils::deviceStreamCreate(&streamCompute);
      dftfe::utils::deviceStreamCreate(&streamDataMove);

      // attach deviceblas handle to compute stream
      BLASWrapperPtr->setStream(streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projHamBlockHost;
      projHamBlockHost.resize(vectorsBlockSize * N, 0);
      std::memset(projHamBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projHamBlockHostFP32;
      projHamBlockHostFP32.resize(vectorsBlockSize * N, 0);
      std::memset(projHamBlockHostFP32.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        HXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlock(vectorsBlockSize * N, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockNext(vectorsBlockSize * N, dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockFP32(vectorsBlockSize * N, dataTypes::numberFP32(0.0));

      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempReal;
      dftfe::utils::MemoryStorage<dataTypes::numberValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImag;

      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempRealFP32;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32ValueType,
                                  dftfe::utils::MemorySpace::DEVICE>
        tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          tempReal.resize(vectorsBlockSize * N, 0);
          tempImag.resize(vectorsBlockSize * N, 0);
          tempRealFP32.resize(vectorsBlockSize * N, 0);
          tempImagFP32.resize(vectorsBlockSize * N, 0);
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const unsigned int chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const unsigned int D          = N - jvec;

              // handle edge case for the first block or the first block in the
              // band group in case of band parallelization
              if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // compute HXBlockFull in an inner loop over blocks of B
                  // wavefunction vectors
                  for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate H times XBlock^{T} and store in HXBlock^{T}
                      operatorMatrix.HX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        HXBlock,
                        onlyHPrimePartForFirstOrderDensityMatResponse);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - jvec,
                        HXBlock.begin(),
                        HXBlockFull.begin());
                    }

                  // evalute X^{T} times HXBlock
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    D,
                    B,
                    M,
                    &alpha,
                    X + jvec,
                    N,
                    HXBlockFull.begin(),
                    B,
                    &beta,
                    projHamBlock.begin(),
                    D);

                  // record completion of compute for first block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount],
                                                  streamCompute);
                }


              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future calls in the streamDataMove will only occur after both
              // the compute on currentblock and swap is over. Note that at this
              // point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((dftfe::utils::deviceEventSynchronize(
                     computeEvents[blockCount]) ==
                   dftfe::utils::deviceSuccess) &&
                  (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                projHamBlock.swap(projHamBlockNext);

              const unsigned int jvecNew = jvec + vectorsBlockSize;
              const unsigned int DNew    = N - jvecNew;

              // start computations on the next block
              if (jvecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  for (unsigned int k = jvecNew; k < jvecNew + B;
                       k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate H times XBlock^{T} and store in HXBlock^{T}
                      operatorMatrix.HX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        HXBlock,
                        onlyHPrimePartForFirstOrderDensityMatResponse);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - jvecNew,
                        HXBlock.begin(),
                        HXBlockFull.begin());
                    }

                  // evalute X^{T} times HXBlock
                  BLASWrapperPtr->xgemm(
                    'N',
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      'C' :
                      'T',
                    DNew,
                    B,
                    M,
                    &alpha,
                    X + jvecNew,
                    N,
                    HXBlockFull.begin(),
                    B,
                    &beta,
                    projHamBlockNext.begin(),
                    DNew);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }

              if (!(jvec + B > Noc))
                {
                  BLASWrapperPtr->setStream(streamDataMove);
                  BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                    D * B, projHamBlock.begin(), projHamBlockFP32.begin());
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (jvec + B > Noc)
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlock.begin(),
                          projHamBlock.begin(),
                          D * B,
                          tempReal.begin(),
                          tempImag.begin(),
                          streamDataMove);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlock.begin(),
                          projHamBlock.begin(),
                          D * B,
                          streamDataMove);
                    }
                  else
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlockFP32.begin(),
                          projHamBlockFP32.begin(),
                          D * B,
                          tempRealFP32.begin(),
                          tempImagFP32.begin(),
                          streamDataMove);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projHamBlockFP32.begin(),
                          projHamBlockFP32.begin(),
                          D * B,
                          streamDataMove);
                    }
                }

              if (jvec + B > Noc)
                dftfe::utils::deviceMemcpyAsyncD2H(
                  projHamBlockHost.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projHamBlock.begin()),
                  D * B * sizeof(dataTypes::number),
                  streamDataMove);
              else
                dftfe::utils::deviceMemcpyAsyncD2H(
                  projHamBlockHostFP32.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projHamBlockFP32.begin()),
                  D * B * sizeof(dataTypes::numberFP32),
                  streamDataMove);

              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matrix
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  if (jvec + B > Noc)
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      projHamBlockHost.begin(),
                                      D * B,
                                      dataTypes::mpi_type_id(
                                        projHamBlockHost.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (unsigned int j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const unsigned int localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (unsigned int i = j + jvec; i < N; ++i)
                                {
                                  std::unordered_map<unsigned int,
                                                     unsigned int>::iterator
                                    it = globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHost[j * D + i - jvec];
                                }
                            }
                    }
                  else
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      projHamBlockHostFP32.begin(),
                                      D * B,
                                      dataTypes::mpi_type_id(
                                        projHamBlockHostFP32.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (unsigned int j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const unsigned int localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (unsigned int i = j + jvec; i < N; ++i)
                                {
                                  std::unordered_map<unsigned int,
                                                     unsigned int>::iterator
                                    it = globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHostFP32[j * D + i - jvec];
                                }
                            }
                    }
                }

            } // band parallelization
          blockCount += 1;
        }

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(copyEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDataMove);

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }


  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
