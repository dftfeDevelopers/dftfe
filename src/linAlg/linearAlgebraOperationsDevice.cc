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


#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceBlasWrapper.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryStorage.h>
#include <dftUtils.h>
#include <linearAlgebraOperationsDevice.h>
#include <linearAlgebraOperationsInternal.h>
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


    //
    // evaluate upper bound of the spectrum using k-step Lanczos iteration
    //
    std::pair<double, double>
    lanczosLowerUpperBoundEigenSpectrum(
      operatorDFTDeviceClass &                 operatorMatrix,
      distributedDeviceVec<dataTypes::number> &Xb,
      distributedDeviceVec<dataTypes::number> &Yb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                       blockSize,
      const dftParameters &                    dftParams)
    {
      const unsigned int this_mpi_process =
        dealii::Utilities::MPI::this_mpi_process(
          operatorMatrix.getMPICommunicator());

      const unsigned int lanczosIterations =
        dftParams.reproducible_output ? 40 : 20;
      double beta;


      dataTypes::number alpha, alphaNeg;

      //
      // generate random vector v
      //
      distributedCPUVec<dataTypes::number> vVector, fVector, v0Vector;
      vVector.reinit(operatorMatrix.getParallelVecSingleComponent());
      fVector.reinit(operatorMatrix.getParallelVecSingleComponent());

      vVector = dataTypes::number(0), fVector = dataTypes::number(0);
      std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();

      for (unsigned int i = 0; i < local_size; i++)
        vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

      operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(vVector, 1);
      vVector.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVector /= vVector.l2_norm();
      // vVector.update_ghost_values();

      //
      // call matrix times X
      //
      std::vector<distributedCPUVec<dataTypes::number>> v(1), f(1);
      v[0] = vVector;
      f[0] = fVector;

      distributedCPUVec<dataTypes::number> &vvec = v[0];

      dftfe::utils::deviceMemcpyH2D_2D(
        dftfe::utils::makeDataTypeDeviceCompatible(Xb.begin()),
        blockSize * sizeof(dataTypes::number),
        vvec.begin(),
        1 * sizeof(dataTypes::number),
        1 * sizeof(dataTypes::number),
        local_size);

      Yb.setValue(0);
      operatorMatrix.HX(
        Xb, projectorKetTimesVector, local_size, blockSize, false, 1.0, Yb);

      distributedCPUVec<dataTypes::number> &fvec = f[0];
      dftfe::utils::deviceMemcpyD2H_2D(
        fvec.begin(),
        1 * sizeof(dataTypes::number),
        dftfe::utils::makeDataTypeDeviceCompatible(Yb.begin()),
        blockSize * sizeof(dataTypes::number),
        1 * sizeof(dataTypes::number),
        local_size);

      operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(v[0], 1);
      fVector = f[0];

      alpha = fVector * vVector;
      fVector.add(-1.0 * alpha, vVector);
      std::vector<dataTypes::number> Tlanczos(lanczosIterations *
                                                lanczosIterations,
                                              0);

      Tlanczos[0]    = alpha;
      unsigned index = 0;

      // filling only lower triangular part
      for (unsigned int j = 1; j < lanczosIterations; j++)
        {
          beta     = fVector.l2_norm();
          v0Vector = vVector;
          vVector.equ(1.0 / beta, fVector);
          v[0] = vVector, f[0] = fVector;
          // operatorMatrix.HX(v,f);

          distributedCPUVec<dataTypes::number> &vvec = v[0];
          dftfe::utils::deviceMemcpyH2D_2D(
            dftfe::utils::makeDataTypeDeviceCompatible(Xb.begin()),
            blockSize * sizeof(dataTypes::number),
            vvec.begin(),
            1 * sizeof(dataTypes::number),
            1 * sizeof(dataTypes::number),
            local_size);

          Yb.setValue(0);
          operatorMatrix.HX(
            Xb, projectorKetTimesVector, local_size, blockSize, false, 1.0, Yb);

          distributedCPUVec<dataTypes::number> &fvec = f[0];
          dftfe::utils::deviceMemcpyD2H_2D(
            fvec.begin(),
            1 * sizeof(dataTypes::number),
            dftfe::utils::makeDataTypeDeviceCompatible(Yb.begin()),
            blockSize * sizeof(dataTypes::number),
            1 * sizeof(dataTypes::number),
            local_size);

          operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(v[0], 1);
          fVector = f[0];
          fVector.add(-1.0 * beta, v0Vector); // beta is real
          alpha = fVector * vVector;
          fVector.add(-1.0 * alpha, vVector);
          index += 1;
          Tlanczos[index] = beta;
          index += lanczosIterations;
          Tlanczos[index] = alpha;
        }

      // eigen decomposition to find max eigen value of T matrix
      std::vector<double> eigenValuesT(lanczosIterations);
      char                jobz = 'N', uplo = 'L';
      const unsigned int  n = lanczosIterations, lda = lanczosIterations;
      int                 info;
      const unsigned int  lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
      std::vector<int>    iwork(liwork, 0);

#ifdef USE_COMPLEX
      const unsigned int                lrwork = 1 + 5 * n + 2 * n * n;
      std::vector<double>               rwork(lrwork, 0);
      std::vector<std::complex<double>> work(lwork);
      zheevd_(&jobz,
              &uplo,
              &n,
              &Tlanczos[0],
              &lda,
              &eigenValuesT[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);
#else
      std::vector<double> work(lwork, 0);
      dsyevd_(&jobz,
              &uplo,
              &n,
              &Tlanczos[0],
              &lda,
              &eigenValuesT[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);
#endif


      for (unsigned int i = 0; i < eigenValuesT.size(); i++)
        {
          eigenValuesT[i] = eigenValuesT[i];
        }
      std::sort(eigenValuesT.begin(), eigenValuesT.end());
      //
      const double fvectorNorm = fVector.l2_norm();
      if (dftParams.verbosity >= 5 && this_mpi_process == 0)
        {
          std::cout << "bUp1: " << eigenValuesT[lanczosIterations - 1]
                    << ", fvector norm: " << fvectorNorm << std::endl;
          std::cout << "aLow: " << eigenValuesT[0] << std::endl;
        }

      double lowerBound = std::floor(eigenValuesT[0]);
      double upperBound = std::ceil(
        eigenValuesT[lanczosIterations - 1] +
        (dftParams.reproducible_output ? fvectorNorm : fvectorNorm / 10));
      return (std::make_pair(lowerBound, upperBound));
    }



    void
    chebyshevFilter(
      operatorDFTDeviceClass &                     operatorMatrix,
      distributedDeviceVec<dataTypes::number> &    XArray,
      distributedDeviceVec<dataTypes::number> &    YArray,
      distributedDeviceVec<dataTypes::numberFP32> &tempFloatArray,
      distributedDeviceVec<dataTypes::number> &    projectorKetTimesVector,
      const unsigned int                           localVectorSize,
      const unsigned int                           numberVectors,
      const unsigned int                           m,
      const double                                 a,
      const double                                 b,
      const double                                 a0,
      const bool                                   mixedPrecOverall,
      const dftParameters &                        dftParams)
    {
      double e, c, sigma, sigma1, sigma2, gamma, device_time;
      e                                  = (b - a) / 2.0;
      c                                  = (b + a) / 2.0;
      sigma                              = e / (a0 - c);
      sigma1                             = sigma;
      gamma                              = 2.0 / sigma1;
      const unsigned int totalVectorSize = localVectorSize * numberVectors;
      int                inc             = 1;

      YArray.setValue(0);
      //
      // call HX
      //
      bool   scaleFlag = false;
      double scalar    = 1.0;

      operatorMatrix.HX(XArray,
                        projectorKetTimesVector,
                        localVectorSize,
                        numberVectors,
                        scaleFlag,
                        scalar,
                        YArray);

      double alpha1 = sigma1 / e, alpha2 = -c;
      double alpha1Old = alpha1;

      //
      // YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      dftfe::utils::deviceKernelsGeneric::axpby(
        totalVectorSize, XArray.begin(), YArray.begin(), alpha2, (double)1);

      dftfe::utils::deviceKernelsGeneric::ascal(totalVectorSize,
                                                YArray.begin(),
                                                alpha1);

      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          double coeff = -c * alpha1;

          if (degree == 2)
            {
              dftfe::utils::deviceKernelsGeneric::axpby(
                totalVectorSize, YArray.begin(), XArray.begin(), coeff, alpha2);

              // scale src vector with M^{-1/2}
              //
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                alpha1,
                operatorMatrix.getInvSqrtMassVec(),
                YArray.begin());


              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0,
                operatorMatrix.getSqrtMassVec(),
                XArray.begin());


              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray,
                                     tempFloatArray,
                                     projectorKetTimesVector,
                                     localVectorSize,
                                     numberVectors,
                                     XArray,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby);
            }
          else if (degree == m)
            {
              // unscale src vector with M^{1/2}
              //
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0 / alpha1Old,
                operatorMatrix.getSqrtMassVec(),
                XArray.begin());

              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0,
                operatorMatrix.getInvSqrtMassVec(),
                YArray.begin());


              dftfe::utils::deviceKernelsGeneric::axpby(
                totalVectorSize, YArray.begin(), XArray.begin(), coeff, alpha2);

              scaleFlag = true;
              //
              // call HX
              //
              operatorMatrix.HX(YArray,
                                projectorKetTimesVector,
                                localVectorSize,
                                numberVectors,
                                scaleFlag,
                                alpha1,
                                XArray);
            }
          else
            {
#if DFTFE_WITH_DEVICE_LANG_CUDA
              combinedDeviceKernel<<<
                min((totalVectorSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    30000),
                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                numberVectors,
                localVectorSize,
                dftfe::utils::makeDataTypeDeviceCompatible(YArray.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(XArray.begin()),
                coeff,
                alpha2,
                alpha1,
                alpha1Old,
                operatorMatrix.getInvSqrtMassVec(),
                operatorMatrix.getSqrtMassVec());
#elif DFTFE_WITH_DEVICE_LANG_HIP
              hipLaunchKernelGGL(
                combinedDeviceKernel,
                min((totalVectorSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    30000),
                dftfe::utils::DEVICE_BLOCK_SIZE,
                0,
                0,
                numberVectors,
                localVectorSize,
                dftfe::utils::makeDataTypeDeviceCompatible(YArray.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(XArray.begin()),
                coeff,
                alpha2,
                alpha1,
                alpha1Old,
                operatorMatrix.getInvSqrtMassVec(),
                operatorMatrix.getSqrtMassVec());
#endif

              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray,
                                     tempFloatArray,
                                     projectorKetTimesVector,
                                     localVectorSize,
                                     numberVectors,
                                     XArray,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby);
            }

          XArray.swap(YArray);


          sigma     = sigma2;
          alpha1Old = alpha1;
        }

      // copy back YArray to XArray
      dftfe::utils::deviceMemcpyD2D(
        dftfe::utils::makeDataTypeDeviceCompatible(XArray.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(YArray.begin()),
        totalVectorSize * sizeof(dataTypes::number));
    }


    //
    // Compute and comunication of two blocks (1) and (2) are overlapped during
    // chebyshev filtering.
    //
    void
    chebyshevFilter(
      operatorDFTDeviceClass &                     operatorMatrix,
      distributedDeviceVec<dataTypes::number> &    XArray1,
      distributedDeviceVec<dataTypes::number> &    YArray1,
      distributedDeviceVec<dataTypes::numberFP32> &tempFloatArray,
      distributedDeviceVec<dataTypes::number> &    projectorKetTimesVector1,
      distributedDeviceVec<dataTypes::number> &    XArray2,
      distributedDeviceVec<dataTypes::number> &    YArray2,
      distributedDeviceVec<dataTypes::number> &    projectorKetTimesVector2,
      const unsigned int                           localVectorSize,
      const unsigned int                           numberVectors,
      const unsigned int                           m,
      const double                                 a,
      const double                                 b,
      const double                                 a0,
      const bool                                   mixedPrecOverall,
      const dftParameters &                        dftParams)
    {
      double e, c, sigma, sigma1, sigma2, gamma, device_time;
      e                                  = (b - a) / 2.0;
      c                                  = (b + a) / 2.0;
      sigma                              = e / (a0 - c);
      sigma1                             = sigma;
      gamma                              = 2.0 / sigma1;
      const unsigned int totalVectorSize = localVectorSize * numberVectors;
      int                inc             = 1;

      YArray1.setValue(0);
      YArray2.setValue(0);

      const unsigned int n_ghosts  = YArray1.ghostSize();
      const unsigned int totalSize = localVectorSize + n_ghosts;

      const unsigned int localSizeNLP =
        projectorKetTimesVector1.locallyOwnedSize();
      const unsigned int n_ghosts_nlp = projectorKetTimesVector1.ghostSize();
      const unsigned int totalSizeNLP = localSizeNLP + n_ghosts_nlp;

      //
      // call HX
      //
      bool   scaleFlag = false;
      double scalar    = 1.0;

      operatorMatrix.HX(XArray1,
                        projectorKetTimesVector1,
                        localVectorSize,
                        numberVectors,
                        scaleFlag,
                        scalar,
                        YArray1);

      operatorMatrix.HX(XArray2,
                        projectorKetTimesVector2,
                        localVectorSize,
                        numberVectors,
                        scaleFlag,
                        scalar,
                        YArray2);

      double alpha1 = sigma1 / e, alpha2 = -c;
      double alpha1Old = alpha1;

      //
      // YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      dftfe::utils::deviceKernelsGeneric::axpby(
        totalVectorSize, XArray1.begin(), YArray1.begin(), alpha2, (double)1);


      dftfe::utils::deviceKernelsGeneric::ascal(totalVectorSize,
                                                YArray1.begin(),
                                                alpha1);

      dftfe::utils::deviceKernelsGeneric::axpby(
        totalVectorSize, XArray2.begin(), YArray2.begin(), alpha2, (double)1);

      dftfe::utils::deviceKernelsGeneric::ascal(totalVectorSize,
                                                YArray2.begin(),
                                                alpha1);

      bool overlap = false;
      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);


          double coeff = -c * alpha1;


          if (degree == 2)
            {
              dftfe::utils::deviceKernelsGeneric::axpby(totalVectorSize,
                                                        YArray1.begin(),
                                                        XArray1.begin(),
                                                        coeff,
                                                        alpha2);

              // scale src vector with M^{-1/2}
              //
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                alpha1,
                operatorMatrix.getInvSqrtMassVec(),
                YArray1.begin());


              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0,
                operatorMatrix.getSqrtMassVec(),
                XArray1.begin());


              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray1,
                                     tempFloatArray,
                                     projectorKetTimesVector1,
                                     localVectorSize,
                                     numberVectors,
                                     XArray1,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby);


              dftfe::utils::deviceKernelsGeneric::axpby(totalVectorSize,
                                                        YArray2.begin(),
                                                        XArray2.begin(),
                                                        coeff,
                                                        alpha2);


              // scale src vector with M^{-1/2}
              //
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                alpha1,
                operatorMatrix.getInvSqrtMassVec(),
                YArray2.begin());

              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0,
                operatorMatrix.getSqrtMassVec(),
                XArray2.begin());

              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray2,
                                     tempFloatArray,
                                     projectorKetTimesVector2,
                                     localVectorSize,
                                     numberVectors,
                                     XArray2,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby);
              overlap = false;
            }
          else if (degree == m)
            {
              // unscale src vector with M^{1/2}
              //
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0 / alpha1Old,
                operatorMatrix.getSqrtMassVec(),
                XArray1.begin());

              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0,
                operatorMatrix.getInvSqrtMassVec(),
                YArray1.begin());



              dftfe::utils::deviceKernelsGeneric::axpby(totalVectorSize,
                                                        YArray1.begin(),
                                                        XArray1.begin(),
                                                        coeff,
                                                        alpha2);

              scaleFlag = true;
              //
              // call HX
              //
              operatorMatrix.HX(YArray1,
                                projectorKetTimesVector1,
                                localVectorSize,
                                numberVectors,
                                scaleFlag,
                                alpha1,
                                XArray1);


              // unscale src vector with M^{1/2}
              //
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0 / alpha1Old,
                operatorMatrix.getSqrtMassVec(),
                XArray2.begin());

              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                numberVectors,
                localVectorSize,
                1.0,
                operatorMatrix.getInvSqrtMassVec(),
                YArray2.begin());


              dftfe::utils::deviceKernelsGeneric::axpby(totalVectorSize,
                                                        YArray2.begin(),
                                                        XArray2.begin(),
                                                        coeff,
                                                        alpha2);

              //
              // call HX
              //
              operatorMatrix.HX(YArray2,
                                projectorKetTimesVector2,
                                localVectorSize,
                                numberVectors,
                                scaleFlag,
                                alpha1,
                                XArray2);
              overlap = false;
            }
          else
            {
              /////////////PSEUDO CODE for the implementation below for
              /// Overlapping compute and communication in HX/////////////////
              //
              // In the algorithm below the communication and computation of two
              // blocks of wavefunctions: block 1 and block 2 are overlapped.
              // CM-NB and CM-B denotes non-blocking and blocking communications
              // respectively. CP denotes compute. The HX computation is divided
              // into compute part 1 and compute part 2 which are separately
              // overlapped. Note that the first and the last iterations of the
              // algorithm are edge cases and are handled a bit differently
              // (Look for step skipped remarks below).
              //
              // 1) [CM-NB] Initiate compress of nonlocal
              // projectorKetTimesVector of block 2 (skipped in first overlap
              // iteration) 2) [CP] Call combinedDeviceKernel of block 1 3)
              // [CM-B] Finish compress of nonlocal projectorKetTimesVector of
              // block 2. (skipped in first overlap iteration) 4) [CM-NB] Call
              // update_ghost_values on nonlocal projectorKetTimesVector of
              // block 2. (skipped in first overlap iteration) 5) [CM-NB]
              // Initiate update_ghost_values on wavefunctions of block 1. 6)
              // [CP] Call HX compute part 2 on block 2. (skipped in first
              // overlap iteration) 7) [CM-B] Finish update_ghost_values on
              // wavefunctions of block 1. 8) [CM-NB] Initiate compress on
              // wavefunctions of block 2. 9) [CP] Call HX compute part 1 on
              // block 1. 10)[CM-B] Finish compress on wavefunctions of block 2.
              // 11)[CM-NB] Initiate compress of nonlocal
              // projectorKetTimesVector of block 1. 12)[CP] Call
              // combinedDeviceKernel of block 2 13)[CM-B] Finish compress of
              // nonlocal projectorKetTimesVector of block 1. 14)[CM-NB]
              // Initiate update_ghost_values on wavefunctions of block 2.
              // 15)[CP] Call HX compute part 2 on block 1.
              // 16)[CM-B] Finish update_ghost_values on wavefunctions of
              // block 2. 17)[CM-NB] Initiate compress on wavefunctions of
              // block 1. 18)[CP] Call HX compute part 1 on block 2. 19)[CM-B]
              // Finish compress on wavefunctions of block 1. 20) Perform
              // chebyshev recursion related swap and scalar operations and go
              // back to step 1)
              //
              // Extra steps for second to last chebyshev filter iteration or
              // the last overlapped iteration: 21) Call compress and
              // update_ghost_values on projectorKetTimesVector of block 2 22)
              // Call HX compute part 2 on block 2. 23) Call compress on
              // wavefunctions of block 2.
              /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

              // overlap flag is used to handle the edge case for the very first
              // overlap is performed, where the previous chebyshev filtering
              // iteration did not use the overlap algorithm
              if (overlap)
                {
                  projectorKetTimesVector2.accumulateAddLocallyOwnedBegin(1);
                }

#if DFTFE_WITH_DEVICE_LANG_CUDA
              combinedDeviceKernel<<<
                min((totalVectorSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    30000),
                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                numberVectors,
                localVectorSize,
                dftfe::utils::makeDataTypeDeviceCompatible(YArray1.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(XArray1.begin()),
                coeff,
                alpha2,
                alpha1,
                alpha1Old,
                operatorMatrix.getInvSqrtMassVec(),
                operatorMatrix.getSqrtMassVec());
#elif DFTFE_WITH_DEVICE_LANG_HIP
              hipLaunchKernelGGL(
                combinedDeviceKernel,
                min((totalVectorSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    30000),
                dftfe::utils::DEVICE_BLOCK_SIZE,
                0,
                0,
                numberVectors,
                localVectorSize,
                dftfe::utils::makeDataTypeDeviceCompatible(YArray1.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(XArray1.begin()),
                coeff,
                alpha2,
                alpha1,
                alpha1Old,
                operatorMatrix.getInvSqrtMassVec(),
                operatorMatrix.getSqrtMassVec());
#endif

              if (overlap)
                {
                  projectorKetTimesVector2.accumulateAddLocallyOwnedEnd();

                  projectorKetTimesVector2.updateGhostValues(1);
                }

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  dftfe::utils::deviceKernelsGeneric::
                    copyValueType1ArrToValueType2Arr(numberVectors *
                                                       localVectorSize,
                                                     YArray1.begin(),
                                                     tempFloatArray.begin());

                  tempFloatArray.updateGhostValuesBegin();
                }
              else
                YArray1.updateGhostValuesBegin();

              // call compute part 2 of block 2
              if (overlap)
                operatorMatrix.HXCheby(YArray2,
                                       tempFloatArray,
                                       projectorKetTimesVector2,
                                       localVectorSize,
                                       numberVectors,
                                       XArray2,
                                       mixedPrecOverall &&
                                         dftParams.useMixedPrecCheby,
                                       false,
                                       true);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  tempFloatArray.updateGhostValuesEnd();
                  if (n_ghosts != 0)
                    dftfe::utils::deviceKernelsGeneric::
                      copyValueType1ArrToValueType2Arr(
                        numberVectors * n_ghosts,
                        tempFloatArray.begin() +
                          localVectorSize * numberVectors,
                        YArray1.begin() + localVectorSize * numberVectors);
                }
              else
                YArray1.updateGhostValuesEnd();

              if (overlap)
                YArray2.zeroOutGhosts();

              projectorKetTimesVector1.setValue(0);
              if (overlap)
                {
                  if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                    {
                      dftfe::utils::deviceKernelsGeneric::
                        copyValueType1ArrToValueType2Arr(
                          numberVectors * totalSize,
                          XArray2.begin(),
                          tempFloatArray.begin());

                      tempFloatArray.accumulateAddLocallyOwnedBegin();
                    }
                  else
                    XArray2.accumulateAddLocallyOwnedBegin();
                }

              // call compute part 1 of block 1
              operatorMatrix.HXCheby(YArray1,
                                     tempFloatArray,
                                     projectorKetTimesVector1,
                                     localVectorSize,
                                     numberVectors,
                                     XArray1,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby,
                                     true,
                                     false);

              if (overlap)
                {
                  if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                    {
                      tempFloatArray.accumulateAddLocallyOwnedEnd();

#if DFTFE_WITH_DEVICE_LANG_CUDA
                      copyFloatArrToDoubleArrLocallyOwned<<<
                        (numberVectors +
                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE * localVectorSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                        numberVectors,
                        localVectorSize,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          tempFloatArray.begin()),
                        (operatorMatrix
                           .getLocallyOwnedProcBoundaryNodesVectorDevice())
                          .begin(),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XArray2.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
                      hipLaunchKernelGGL(
                        copyFloatArrToDoubleArrLocallyOwned,
                        (numberVectors +
                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE * localVectorSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        0,
                        numberVectors,
                        localVectorSize,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          tempFloatArray.begin()),
                        (operatorMatrix
                           .getLocallyOwnedProcBoundaryNodesVectorDevice())
                          .begin(),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XArray2.begin()));
#endif
                      XArray2.zeroOutGhosts();
                    }
                  else
                    XArray2.accumulateAddLocallyOwnedEnd();
                  XArray2.swap(YArray2);
                }

              projectorKetTimesVector1.accumulateAddLocallyOwnedBegin(1);

#if DFTFE_WITH_DEVICE_LANG_CUDA
              combinedDeviceKernel<<<
                min((totalVectorSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    30000),
                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                numberVectors,
                localVectorSize,
                dftfe::utils::makeDataTypeDeviceCompatible(YArray2.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(XArray2.begin()),
                coeff,
                alpha2,
                alpha1,
                alpha1Old,
                operatorMatrix.getInvSqrtMassVec(),
                operatorMatrix.getSqrtMassVec());
#elif DFTFE_WITH_DEVICE_LANG_HIP
              hipLaunchKernelGGL(
                combinedDeviceKernel,
                min((totalVectorSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    30000),
                dftfe::utils::DEVICE_BLOCK_SIZE,
                0,
                0,
                numberVectors,
                localVectorSize,
                dftfe::utils::makeDataTypeDeviceCompatible(YArray2.begin()),
                dftfe::utils::makeDataTypeDeviceCompatible(XArray2.begin()),
                coeff,
                alpha2,
                alpha1,
                alpha1Old,
                operatorMatrix.getInvSqrtMassVec(),
                operatorMatrix.getSqrtMassVec());
#endif

              projectorKetTimesVector1.accumulateAddLocallyOwnedEnd();

              projectorKetTimesVector1.updateGhostValues(1);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  dftfe::utils::deviceKernelsGeneric::
                    copyValueType1ArrToValueType2Arr(numberVectors *
                                                       localVectorSize,
                                                     YArray2.begin(),
                                                     tempFloatArray.begin());

                  tempFloatArray.updateGhostValuesBegin();
                }
              else
                YArray2.updateGhostValuesBegin();

              // call compute part 2 of block 1
              operatorMatrix.HXCheby(YArray1,
                                     tempFloatArray,
                                     projectorKetTimesVector1,
                                     localVectorSize,
                                     numberVectors,
                                     XArray1,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby,
                                     false,
                                     true);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  tempFloatArray.updateGhostValuesEnd();
                  if (n_ghosts != 0)
                    dftfe::utils::deviceKernelsGeneric::
                      copyValueType1ArrToValueType2Arr(
                        numberVectors * n_ghosts,
                        tempFloatArray.begin() +
                          localVectorSize * numberVectors,
                        YArray2.begin() + localVectorSize * numberVectors);
                }
              else
                YArray2.updateGhostValuesEnd();
              YArray1.zeroOutGhosts();


              projectorKetTimesVector2.setValue(0);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  dftfe::utils::deviceKernelsGeneric::
                    copyValueType1ArrToValueType2Arr(numberVectors * totalSize,
                                                     XArray1.begin(),
                                                     tempFloatArray.begin());

                  tempFloatArray.accumulateAddLocallyOwnedBegin();
                }
              else
                XArray1.accumulateAddLocallyOwnedBegin();

              // call compute part 1 of block 2
              operatorMatrix.HXCheby(YArray2,
                                     tempFloatArray,
                                     projectorKetTimesVector2,
                                     localVectorSize,
                                     numberVectors,
                                     XArray2,
                                     mixedPrecOverall &&
                                       dftParams.useMixedPrecCheby,
                                     true,
                                     false);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  tempFloatArray.accumulateAddLocallyOwnedEnd();

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                  copyFloatArrToDoubleArrLocallyOwned<<<
                    (numberVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE * localVectorSize,
                    dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                    numberVectors,
                    localVectorSize,
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      tempFloatArray.begin()),
                    (operatorMatrix
                       .getLocallyOwnedProcBoundaryNodesVectorDevice())
                      .begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      XArray1.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
                  hipLaunchKernelGGL(
                    copyFloatArrToDoubleArrLocallyOwned,
                    (numberVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE * localVectorSize,
                    dftfe::utils::DEVICE_BLOCK_SIZE,
                    0,
                    0,
                    numberVectors,
                    localVectorSize,
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      tempFloatArray.begin()),
                    (operatorMatrix
                       .getLocallyOwnedProcBoundaryNodesVectorDevice())
                      .begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      XArray1.begin()));
#endif
                  XArray1.zeroOutGhosts();
                }
              else
                XArray1.accumulateAddLocallyOwnedEnd();

              // Handle edge case for the second to last Chebyshev filter
              // iteration as there is no overlap algorithm for the next filter
              // iteration.
              if (degree == (m - 1))
                {
                  projectorKetTimesVector2.accumulateAddLocallyOwned(1);
                  projectorKetTimesVector2.updateGhostValues(1);

                  operatorMatrix.HXCheby(YArray2,
                                         tempFloatArray,
                                         projectorKetTimesVector2,
                                         localVectorSize,
                                         numberVectors,
                                         XArray2,
                                         mixedPrecOverall &&
                                           dftParams.useMixedPrecCheby,
                                         false,
                                         true);
                  YArray2.zeroOutGhosts();
                  if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                    {
                      dftfe::utils::deviceKernelsGeneric::
                        copyValueType1ArrToValueType2Arr(
                          numberVectors * totalSize,
                          XArray2.begin(),
                          tempFloatArray.begin());

                      tempFloatArray.accumulateAddLocallyOwned();

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                      copyFloatArrToDoubleArrLocallyOwned<<<
                        (numberVectors +
                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE * localVectorSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                        numberVectors,
                        localVectorSize,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          tempFloatArray.begin()),
                        (operatorMatrix
                           .getLocallyOwnedProcBoundaryNodesVectorDevice())
                          .begin(),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XArray2.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
                      hipLaunchKernelGGL(
                        copyFloatArrToDoubleArrLocallyOwned,
                        (numberVectors +
                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE * localVectorSize,
                        dftfe::utils::DEVICE_BLOCK_SIZE,
                        0,
                        0,
                        numberVectors,
                        localVectorSize,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          tempFloatArray.begin()),
                        (operatorMatrix
                           .getLocallyOwnedProcBoundaryNodesVectorDevice())
                          .begin(),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XArray2.begin()));
#endif
                      XArray2.zeroOutGhosts();
                    }
                  else
                    XArray2.accumulateAddLocallyOwned();
                  overlap = false;
                }
              else
                overlap = true;
            }

          XArray1.swap(YArray1);
          // Handle edge cases for the first and last iteration involving
          // overlap of communication and computation
          if (!overlap)
            {
              XArray2.swap(YArray2);
            }


          sigma     = sigma2;
          alpha1Old = alpha1;
        }

      // copy back YArray to XArray
      dftfe::utils::deviceMemcpyD2D(
        dftfe::utils::makeDataTypeDeviceCompatible(XArray1.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(YArray1.begin()),
        totalVectorSize * sizeof(dataTypes::number));

      dftfe::utils::deviceMemcpyD2D(
        dftfe::utils::makeDataTypeDeviceCompatible(XArray2.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(YArray2.begin()),
        totalVectorSize * sizeof(dataTypes::number));
    }


    void
    subspaceRotationSpectrumSplitScalapack(
      const dataTypes::number *                        X,
      dataTypes::number *                              XFrac,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Nfr,
      dftfe::utils::deviceBlasHandle_t &               handle,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    dftfe::utils::DEVICEBLAS_OP_N,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      dataTypes::number *                              X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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
                      dftfe::utils::deviceBlasWrapper::gemm(
                        handle,
                        dftfe::utils::DEVICEBLAS_OP_N,
                        dftfe::utils::DEVICEBLAS_OP_N,
                        BVec,
                        BDof,
                        D,
                        &scalarCoeffAlpha,
                        rotationMatBlock.begin(),
                        BVec,
                        X + idof * N,
                        N,
                        &scalarCoeffBeta,
                        rotatedVectorsMatBlock.begin() + jvec,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      dataTypes::number *                              X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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


      dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
        N * M, X, XSP.begin());

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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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
                      dftfe::utils::deviceBlasWrapper::gemm(
                        handle,
                        dftfe::utils::DEVICEBLAS_OP_N,
                        dftfe::utils::DEVICEBLAS_OP_N,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      dataTypes::number *                              X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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


      dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
        N * M, X, XSP.begin());

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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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
                      dftfe::utils::deviceBlasWrapper::gemm(
                        handle,
                        dftfe::utils::DEVICEBLAS_OP_N,
                        dftfe::utils::DEVICEBLAS_OP_N,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      const dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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

      dftfe::utils::deviceStream_t streamDeviceCCL;
      dftfe::utils::deviceStreamCreate(&streamDeviceCCL);

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
              dftfe::utils::deviceBlasWrapper::gemm(
                handle,
                dftfe::utils::DEVICEBLAS_OP_N,
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  dftfe::utils::DEVICEBLAS_OP_C :
                  dftfe::utils::DEVICEBLAS_OP_T,
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


      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);

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
      const dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      const dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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

      dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
        N * M, X, XSP.begin());

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

      dftfe::utils::deviceStream_t streamDeviceCCL;
      dftfe::utils::deviceStreamCreate(&streamDeviceCCL);

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
              dftfe::utils::deviceBlasWrapper::gemm(
                handle,
                dftfe::utils::DEVICEBLAS_OP_N,
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  dftfe::utils::DEVICEBLAS_OP_C :
                  dftfe::utils::DEVICEBLAS_OP_T,
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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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


      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);

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
      const dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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


      dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
        N * M, X, XSP.begin());

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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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
                      dftfe::utils::deviceBlasWrapper::gemm(
                        handle,
                        dftfe::utils::DEVICEBLAS_OP_N,
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          dftfe::utils::DEVICEBLAS_OP_C :
                          dftfe::utils::DEVICEBLAS_OP_T,
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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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
                      dftfe::utils::deviceBlasWrapper::gemm(
                        handle,
                        dftfe::utils::DEVICEBLAS_OP_N,
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          dftfe::utils::DEVICEBLAS_OP_C :
                          dftfe::utils::DEVICEBLAS_OP_T,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      const dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &               handle,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, streamCompute);

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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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
                  dftfe::utils::deviceBlasWrapper::gemm(
                    handle,
                    dftfe::utils::DEVICEBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      dftfe::utils::DEVICEBLAS_OP_C :
                      dftfe::utils::DEVICEBLAS_OP_T,
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
      dftfe::utils::deviceBlasWrapper::setStream(handle, NULL);

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
      operatorDFTDeviceClass &                 operatorMatrix,
      dataTypes::number *                      X,
      distributedDeviceVec<dataTypes::number> &XBlock,
      distributedDeviceVec<dataTypes::number> &HXBlock,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                       M,
      const unsigned int                       N,
      const std::vector<double> &              eigenValues,
      const MPI_Comm &                         mpiCommDomain,
      const MPI_Comm &                         interBandGroupComm,
      dftfe::utils::deviceBlasHandle_t &       handle,
      std::vector<double> &                    residualNorm,
      const dftParameters &                    dftParams,
      const bool                               useBandParal)
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
                  dftfe::utils::deviceKernelsGeneric::
                    stridedCopyToBlockConstantStride(
                      chebyBlockSize, N, M, k, X, XBlock.begin());

                  // evaluate H times XBlock^{T} and store in HXBlock^{T}
                  HXBlock.setValue(0);
                  operatorMatrix.HX(XBlock,
                                    projectorKetTimesVector,
                                    M,
                                    chebyBlockSize,
                                    scaleFlag,
                                    scalar,
                                    HXBlock);
                  dftfe::utils::deviceKernelsGeneric::
                    stridedCopyFromBlockConstantStride(B,
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

              dftfe::utils::deviceBlasWrapper::gemm(
                handle,
                dftfe::utils::DEVICEBLAS_OP_N,
                dftfe::utils::DEVICEBLAS_OP_T,
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


      for (unsigned int iWave = 0; iWave < N; ++iWave)
        residualNorm[iWave] = std::sqrt(residualNorm[iWave]);
    }
  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
