// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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



#include <MemoryStorage.h>
#include <dftUtils.h>
#include <linearAlgebraOperationsDevice.h>
#include <linearAlgebraOperationsInternal.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>
#include "linearAlgebraOperationsDeviceKernels.h"


namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
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
      const dftfe::uInt                                                     m,
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
      operatorMatrix.overlapMatrixTimesX(X1, 1.0, 0.0, 0.0, Y1);
      operatorMatrix.overlapMatrixTimesX(X2, 1.0, 0.0, 0.0, Y2);

      //
      // call HX
      //


      double alpha1 = sigma1 / e, alpha2 = -c;
      operatorMatrix.HXCheby(Y1, alpha1, 0.0, alpha1 * alpha2, X1);
      X1.swap(Y1);
      Y2.updateGhostValues();
      operatorMatrix.HXCheby(
        Y2, alpha1, 0.0, alpha1 * alpha2, X2, false, false, true, true);
      //
      // polynomial loop
      //
      for (dftfe::uInt degree = 2; degree < m + 1; ++degree)
        {
          sigma2    = 1.0 / (gamma - sigma);
          alpha1Old = alpha1, alpha2Old = alpha2;
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);
          operatorMatrix.HXCheby(Y2,
                                 alpha1Old,
                                 degree == 2 ? 0.0 : alpha2Old,
                                 -c * alpha1Old,
                                 X2,
                                 false,
                                 true,
                                 false,
                                 true);
          Y1.updateGhostValuesBegin();
          operatorMatrix.HXCheby(Y2,
                                 alpha1Old,
                                 degree == 2 ? 0.0 : alpha2Old,
                                 -c * alpha1Old,
                                 X2,
                                 false,
                                 true,
                                 true,
                                 false);
          Y1.updateGhostValuesEnd();
          X2.accumulateAddLocallyOwnedBegin();



          //
          // call HX
          //
          operatorMatrix.HXCheby(
            Y1, alpha1, alpha2, -c * alpha1, X1, false, false, true, true);
          X2.accumulateAddLocallyOwnedEnd();
          X2.zeroOutGhosts();
          X2.swap(Y2);

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
      operatorMatrix.overlapInverseMatrixTimesX(Y1, 1.0, 0.0, 0.0, X1);
      operatorMatrix.overlapInverseMatrixTimesX(Y2, 1.0, 0.0, 0.0, X2);
    }
    template <typename T1, typename T2>
    void
    reformulatedChebyshevFilterOverlapComputeCommunication(
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                          &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<T1, dftfe::utils::MemorySpace::DEVICE>
        &X1,
      dftfe::linearAlgebra::MultiVector<T1, dftfe::utils::MemorySpace::DEVICE>
        &Y1,
      dftfe::linearAlgebra::MultiVector<T1, dftfe::utils::MemorySpace::DEVICE>
        &X2,
      dftfe::linearAlgebra::MultiVector<T1, dftfe::utils::MemorySpace::DEVICE>
        &Y2,
      dftfe::linearAlgebra::MultiVector<T2, dftfe::utils::MemorySpace::DEVICE>
        &X1_SP,
      dftfe::linearAlgebra::MultiVector<T2, dftfe::utils::MemorySpace::DEVICE>
        &Y1_SP,
      dftfe::linearAlgebra::MultiVector<T2, dftfe::utils::MemorySpace::DEVICE>
        &X2_SP,
      dftfe::linearAlgebra::MultiVector<T2, dftfe::utils::MemorySpace::DEVICE>
                         &Y2_SP,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix)
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

      operatorMatrix.overlapMatrixTimesX(
        X1, 1.0, 0.0, 0.0, Y1, approxOverlapMatrix);
      BLASWrapperPtr->rightDiagonalScale(Y1.numVectors(),
                                         Y1.locallyOwnedSize(),
                                         Y1.data(),
                                         eigenValuesFiltered.data());
      operatorMatrix.HX(X1, 1.0, -1.0, 0.0, Y1);


      operatorMatrix.overlapMatrixTimesX(
        X2, 1.0, 0.0, 0.0, Y2, approxOverlapMatrix);
      BLASWrapperPtr->rightDiagonalScale(Y1.numVectors(),
                                         Y1.locallyOwnedSize(),
                                         Y2.data(),
                                         eigenValuesFiltered.data() +
                                           X1.numVectors());
      operatorMatrix.HX(X2, 1.0, -1.0, 0.0, Y2);

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

      X1_SP.setValue(0.0);
      X2_SP.setValue(0.0);
      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
        X1.locallyOwnedSize() * X1.numVectors(), Y1.data(), Y1_SP.data());
      BLASWrapperPtr->xscal(Y1_SP.data(),
                            T2(alpha1),
                            X2.locallyOwnedSize() * X2.numVectors());

      //
      // polynomial loop
      //
      for (dftfe::uInt degree = 2; degree < m + 1; ++degree)
        {
          sigma2    = 1.0 / (gamma - sigma);
          alpha1Old = alpha1, alpha2Old = alpha2;
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          if (degree == 2)
            {
              BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                X1.locallyOwnedSize() * X1.numVectors(),
                Y2.data(),
                Y2_SP.data());
              Y1_SP.updateGhostValuesBegin();
              BLASWrapperPtr->xscal(Y2_SP.data(),
                                    T2(alpha1Old),
                                    X2.locallyOwnedSize() * X2.numVectors());
              Y1_SP.updateGhostValuesEnd();
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
          if (degree != 2)
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
      operatorMatrix.overlapInverseMatrixTimesX(Y1_SP, 1.0, 0.0, 0.0, X1_SP);
      operatorMatrix.overlapInverseMatrixTimesX(Y2_SP, 1.0, 0.0, 0.0, X2_SP);
      // copy back YArray to XArray
      BLASWrapperPtr->ApaBD(X1.locallyOwnedSize(),
                            X1.numVectors(),
                            1.0,
                            X1_SP.data(),
                            X1.data(),
                            eigenValuesFiltered2.data(),
                            X1.data());

      BLASWrapperPtr->ApaBD(X2.locallyOwnedSize(),
                            X2.numVectors(),
                            1.0,
                            X2_SP.data(),
                            X2.data(),
                            eigenValuesFiltered2.data() + X1.numVectors(),
                            X2.data());
    }



    void
    subspaceRotationScalapack(
      dataTypes::number *X,
      const dftfe::uInt  M,
      const dftfe::uInt  N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters                             &dftParams,
      const bool                                       rotationMatTranspose,
      const bool                                       isRotationMatLowerTria)
    {
      const dftfe::uInt maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt dofsBlockSize =
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
      const dftfe::uInt numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          dftfe::uInt BDof = 0;
          if (M >= idof)
            BDof = std::min(dofsBlockSize, M - idof);

          for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const dftfe::uInt BVec = std::min(vectorsBlockSize, N - jvec);

              const dftfe::uInt D = isRotationMatLowerTria ? (jvec + BVec) : N;

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
                                for (dftfe::uInt i = 0; i < D; ++i)
                                  if (globalToLocalRowIdMap.find(i) !=
                                      globalToLocalRowIdMap.end())
                                    {
                                      const dftfe::uInt localRowId =
                                        globalToLocalRowIdMap[i];
                                      for (dftfe::uInt j = 0; j < BVec; ++j)
                                        {
                                          std::unordered_map<
                                            dftfe::uInt,
                                            dftfe::uInt>::iterator it =
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
                                for (dftfe::uInt i = 0; i < D; ++i)
                                  if (globalToLocalColumnIdMap.find(i) !=
                                      globalToLocalColumnIdMap.end())
                                    {
                                      const dftfe::uInt localColumnId =
                                        globalToLocalColumnIdMap[i];
                                      for (dftfe::uInt j = 0; j < BVec; ++j)
                                        {
                                          std::unordered_map<
                                            dftfe::uInt,
                                            dftfe::uInt>::iterator it =
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
                            for (dftfe::uInt i = 0; i < D; ++i)
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  const dftfe::uInt localRowId =
                                    globalToLocalRowIdMap[i];
                                  for (dftfe::uInt j = 0; j < BVec; ++j)
                                    {
                                      std::unordered_map<dftfe::uInt,
                                                         dftfe::uInt>::iterator
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
                            for (dftfe::uInt i = 0; i < D; ++i)
                              if (globalToLocalColumnIdMap.find(i) !=
                                  globalToLocalColumnIdMap.end())
                                {
                                  const dftfe::uInt localColumnId =
                                    globalToLocalColumnIdMap[i];
                                  for (dftfe::uInt j = 0; j < BVec; ++j)
                                    {
                                      std::unordered_map<dftfe::uInt,
                                                         dftfe::uInt>::iterator
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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      const dftfe::uInt  M,
      const dftfe::uInt  N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters                             &dftParams,
      const bool                                       rotationMatTranspose)
    {
      const dftfe::uInt maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));


      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt dofsBlockSize =
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
      const dftfe::uInt           numberBlocks = (N / vectorsBlockSize);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
            for (dftfe::uInt i = 0; i < N; ++i)
              if (globalToLocalRowIdMap.find(i) != globalToLocalRowIdMap.end())
                {
                  const dftfe::uInt localRowId = globalToLocalRowIdMap[i];
                  std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator it =
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
            for (dftfe::uInt i = 0; i < N; ++i)
              if (globalToLocalColumnIdMap.find(i) !=
                  globalToLocalColumnIdMap.end())
                {
                  const dftfe::uInt localColumnId = globalToLocalColumnIdMap[i];
                  std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator it =
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
      computeDiagQTimesX(diagValues.begin(), X, N, M);

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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt BVec = std::min(vectorsBlockSize, N - jvec);

          const dftfe::uInt D = jvec + BVec;

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
                    for (dftfe::uInt i = 0; i < D; ++i)
                      if (globalToLocalRowIdMap.find(i) !=
                          globalToLocalRowIdMap.end())
                        {
                          const dftfe::uInt localRowId =
                            globalToLocalRowIdMap[i];
                          for (dftfe::uInt j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
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
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
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
                    for (dftfe::uInt i = 0; i < D; ++i)
                      if (globalToLocalColumnIdMap.find(i) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[i];
                          for (dftfe::uInt j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
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
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
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

              for (dftfe::uInt idof = 0; idof < maxNumLocalDofs;
                   idof += dofsBlockSize)
                {
                  // Correct block dimensions if block "goes off edge of" the
                  // matrix
                  dftfe::uInt BDof = 0;
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

                      addSubspaceRotatedBlockToX(
                        BDof,
                        BVec,
                        rotatedVectorsMatBlockSP.begin(),
                        X,
                        idof,
                        jvec,
                        N,
                        streamCompute);
                    }
                } // block loop over dofs
            }     // band parallalelization loop
          blockCount++;
        } // block loop over vectors

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      const dftfe::uInt  M,
      const dftfe::uInt  N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters                             &dftParams,
      const bool                                       rotationMatTranspose)
    {
      const dftfe::uInt maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      const dftfe::uInt MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));


      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt dofsBlockSize =
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
      const dftfe::uInt           numberBlocks = (N / vectorsBlockSize);
      dftfe::utils::deviceEvent_t computeEvents[numberBlocks];
      dftfe::utils::deviceEvent_t communEvents[numberBlocks];
      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
            for (dftfe::uInt i = 0; i < N; ++i)
              if (globalToLocalRowIdMap.find(i) != globalToLocalRowIdMap.end())
                {
                  const dftfe::uInt localRowId = globalToLocalRowIdMap[i];
                  std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator it =
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
            for (dftfe::uInt i = 0; i < N; ++i)
              if (globalToLocalColumnIdMap.find(i) !=
                  globalToLocalColumnIdMap.end())
                {
                  const dftfe::uInt localColumnId = globalToLocalColumnIdMap[i];
                  std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator it =
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
      computeDiagQTimesX(diagValues.begin(), X, N, M);

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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt BVec = std::min(vectorsBlockSize, N - jvec);

          const dftfe::uInt D = N;

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
                    for (dftfe::uInt i = 0; i < D; ++i)
                      if (globalToLocalRowIdMap.find(i) !=
                          globalToLocalRowIdMap.end())
                        {
                          const dftfe::uInt localRowId =
                            globalToLocalRowIdMap[i];
                          for (dftfe::uInt j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalColumnIdMap.find(j + jvec);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    j) = rotationMatPar.local_el(localRowId,
                                                                 it->second);
                                  if (i == j + jvec)
                                    *(rotationMatBlockHostSP.begin() +
                                      i * BVec + j) = dataTypes::numberFP32(0);
                                }
                            }
                        }
                }
              else
                {
                  if (processGrid->is_process_active())
                    for (dftfe::uInt i = 0; i < D; ++i)
                      if (globalToLocalColumnIdMap.find(i) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[i];
                          for (dftfe::uInt j = 0; j < BVec; ++j)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(j + jvec);
                              if (it != globalToLocalRowIdMap.end())
                                {
                                  *(rotationMatBlockHostSP.begin() + i * BVec +
                                    j) = rotationMatPar.local_el(it->second,
                                                                 localColumnId);
                                  if (i == j + jvec)
                                    *(rotationMatBlockHostSP.begin() +
                                      i * BVec + j) = dataTypes::numberFP32(0);
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

              for (dftfe::uInt idof = 0; idof < maxNumLocalDofs;
                   idof += dofsBlockSize)
                {
                  // Correct block dimensions if block "goes off edge of" the
                  // matrix
                  dftfe::uInt BDof = 0;
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
                      addSubspaceRotatedBlockToX(
                        BDof,
                        BVec,
                        rotatedVectorsMatBlockSP.begin(),
                        X,
                        idof,
                        jvec,
                        N,
                        streamCompute);
                    }
                } // block loop over dofs
            }     // band parallelization
          blockCount++;
        } // block loop over vectors


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventDestroy(computeEvents[i]);
          dftfe::utils::deviceEventDestroy(communEvents[i]);
        }

      dftfe::utils::deviceStreamDestroy(streamCompute);
      dftfe::utils::deviceStreamDestroy(streamDeviceCCL);
    }

    void
    fillParallelOverlapMatScalapack(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &OXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &overlapMatPar,
      const dftParameters                             &dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

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
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        OXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
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

      for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);


          const dftfe::uInt D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              for (dftfe::uInt k = ivec; k < ivec + B; k += chebyBlockSize)
                {
                  BLASWrapperPtr->stridedCopyToBlockConstantStride(
                    chebyBlockSize, N, M, k, X, XBlock.begin());

                  // evaluate XBlock^{T} times H^{T} and store in HXBlock
                  operatorMatrix.overlapMatrixTimesX(
                    XBlock,
                    1.0,
                    0.0,
                    0.0,
                    OXBlock,
                    dftParams.approxOverlapMatrix);

                  BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                    B,
                    chebyBlockSize,
                    M,
                    k - ivec,
                    OXBlock.begin(),
                    OXBlockFull.begin());
                }


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
                OXBlockFull.begin(),
                B,
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
                for (dftfe::uInt i = 0; i < B; ++i)
                  if (globalToLocalColumnIdMap.find(i + ivec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[i + ivec];
                      for (dftfe::uInt j = ivec + i; j < N; ++j)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(j);
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
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &OXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &overlapMatPar,
      const dftParameters                             &dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt numberBlocks     = N / vectorsBlockSize;

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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
        OXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);
          const dftfe::uInt D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock.
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  const dftfe::uInt chebyBlockSize =
                    std::min(dftParams.chebyWfcBlockSize, N);

                  for (dftfe::uInt k = ivec; k < ivec + B; k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate XBlock^{T} times H^{T} and store in HXBlock
                      operatorMatrix.overlapMatrixTimesX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        OXBlock,
                        dftParams.approxOverlapMatrix);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - ivec,
                        OXBlock.begin(),
                        OXBlockFull.begin());
                    }

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
                    OXBlockFull.begin(),
                    B,
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

              const dftfe::uInt ivecNew = ivec + vectorsBlockSize;
              const dftfe::uInt DNew    = N - ivecNew;
              const dftfe::uInt BNew = std::min(vectorsBlockSize, N - ivecNew);


              // start computations on the next block
              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  const dftfe::uInt chebyBlockSize =
                    std::min(dftParams.chebyWfcBlockSize, N);

                  for (dftfe::uInt k = ivecNew; k < ivecNew + BNew;
                       k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate XBlock^{T} times H^{T} and store in HXBlock
                      operatorMatrix.overlapMatrixTimesX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        OXBlock,
                        dftParams.approxOverlapMatrix);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - ivecNew,
                        OXBlock.begin(),
                        OXBlockFull.begin());
                    }

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
                    OXBlockFull.begin(),
                    BNew,
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
                    for (dftfe::uInt i = 0; i < B; ++i)
                      if (globalToLocalColumnIdMap.find(i + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[i + ivec];
                          for (dftfe::uInt j = ivec + i; j < N; ++j)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &OXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      const dftfe::uInt                                    Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &overlapMatPar,
      const dftParameters                             &dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSP(N * vectorsBlockSize, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDP(N * vectorsBlockSize, dataTypes::number(0));

      const dftfe::uInt MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));

      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostDP;
      overlapMatrixBlockHostDP.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostDP.begin(),
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::number));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        OXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        OXBlockFullFP32(vectorsBlockSize * M, dataTypes::numberFP32(0.0));

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

      for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);


          const dftfe::uInt D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);
              for (dftfe::uInt k = ivec; k < ivec + B; k += chebyBlockSize)
                {
                  BLASWrapperPtr->stridedCopyToBlockConstantStride(
                    chebyBlockSize, N, M, k, X, XBlock.begin());

                  // evaluate H times XBlock^{T} and store in HXBlock^{T}
                  operatorMatrix.overlapMatrixTimesX(
                    XBlock,
                    1.0,
                    0.0,
                    0.0,
                    OXBlock,
                    dftParams.approxOverlapMatrix);

                  BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                    B,
                    chebyBlockSize,
                    M,
                    k - ivec,
                    OXBlock.begin(),
                    OXBlockFull.begin());
                }
              const dftfe::uInt DRem = D - B;
              if (ivec + B > Noc)
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
                    OXBlockFull.data(),
                    B,
                    &scalarCoeffBeta,
                    overlapMatrixBlockDP.begin(),
                    D);
                }
              else
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
                    OXBlockFull.data(),
                    B,
                    &scalarCoeffBeta,
                    overlapMatrixBlockDP.begin(),
                    B);



                  if (DRem != 0)
                    {
                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        B,
                        M,
                        0,
                        OXBlockFull.begin(),
                        OXBlockFullFP32.begin());

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
                        OXBlockFullFP32.data(),
                        B,
                        &scalarCoeffBetaSP,
                        overlapMatrixBlockSP.begin(),
                        DRem);
                    }
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (ivec + B > Noc)
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          overlapMatrixBlockDP.begin(),
                          overlapMatrixBlockDP.begin(),
                          D * B,
                          tempReal.begin(),
                          tempImag.begin(),
                          streamDeviceCCL);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          overlapMatrixBlockDP.begin(),
                          overlapMatrixBlockDP.begin(),
                          D * B,
                          streamDeviceCCL);
                    }
                  else
                    {
                      if (DRem == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              overlapMatrixBlockDP.begin(),
                              overlapMatrixBlockDP.begin(),
                              B * B,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDeviceCCL);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              overlapMatrixBlockDP.begin(),
                              overlapMatrixBlockDP.begin(),
                              B * B,
                              streamDeviceCCL);
                        }
                      if (DRem != 0)
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
                    }
                }
              if (ivec + B > Noc)
                dftfe::utils::deviceMemcpyD2H(
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    overlapMatrixBlockHostDP.begin()),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    overlapMatrixBlockDP.begin()),
                  D * B * sizeof(dataTypes::number));
              else
                {
                  dftfe::utils::deviceMemcpyAsyncD2H(
                    overlapMatrixBlockHostDP.begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      overlapMatrixBlockDP.begin()),
                    B * B * sizeof(dataTypes::number));
                  if (DRem != 0)
                    dftfe::utils::deviceMemcpyAsyncD2H(
                      overlapMatrixBlockHostSP.begin(),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlockSP.begin()),
                      DRem * B * sizeof(dataTypes::numberFP32));
                }
              if (ivec + B > Noc)
                {
                  // Sum local projHamBlock across domain decomposition
                  // processors
                  if (!dftParams.useDeviceDirectAllReduce)
                    MPI_Allreduce(MPI_IN_PLACE,
                                  overlapMatrixBlockHostDP.begin(),
                                  D * B,
                                  dataTypes::mpi_type_id(
                                    overlapMatrixBlockHostDP.begin()),
                                  MPI_SUM,
                                  mpiCommDomain);

                  // Copying only the lower triangular part to the ScaLAPACK
                  // projected Hamiltonian matrix
                  if (processGrid->is_process_active())
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + ivec];
                          for (dftfe::uInt i = j + ivec; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostDP[j * D + i - ivec];
                            }
                        }
                }
              else
                {
                  // Sum local projHamBlock across domain decomposition
                  // processors
                  if (!dftParams.useDeviceDirectAllReduce)
                    {
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostDP.begin(),
                                    B * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostDP.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);
                      if (DRem != 0)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      overlapMatrixBlockHostSP.begin(),
                                      DRem * B,
                                      dataTypes::mpi_type_id(
                                        overlapMatrixBlockHostSP.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);
                    }

                  // Copying only the lower triangular part to the ScaLAPACK
                  // projected Hamiltonian matrix
                  if (processGrid->is_process_active())
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + ivec];
                          for (dftfe::uInt i = j + ivec; i < ivec + B; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostDP[j * B + i - ivec];
                            }
                          for (dftfe::uInt i = ivec + B; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostSP[j * DRem + i - ivec -
                                                           B];
                            }
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
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &OXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      const dftfe::uInt                                    Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &overlapMatPar,
      const dftParameters                             &dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt numberBlocks     = N / vectorsBlockSize;

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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSP(N * vectorsBlockSize, dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDP(N * vectorsBlockSize, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockSPNext(N * vectorsBlockSize,
                                 dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDPNext(N * vectorsBlockSize, dataTypes::number(0));

      const dftfe::uInt MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        XSP(MPadded * N, dataTypes::numberFP32(0));


      BLASWrapperPtr->copyValueType1ArrToValueType2Arr(N * M, X, XSP.begin());

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        overlapMatrixBlockHostDP;
      overlapMatrixBlockHostDP.resize(N * vectorsBlockSize, 0);
      std::memset(overlapMatrixBlockHostDP.begin(),
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::number));

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

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        OXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        OXBlockFullFP32(vectorsBlockSize * M, dataTypes::numberFP32(0.0));

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);
          const dftfe::uInt D = N - ivec;
          const dftfe::uInt chebyBlockSize =
            std::min(dftParams.chebyWfcBlockSize, N);
          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  for (dftfe::uInt k = ivec; k < ivec + B; k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      operatorMatrix.overlapMatrixTimesX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        OXBlock,
                        dftParams.approxOverlapMatrix);
                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - ivec,
                        OXBlock.begin(),
                        OXBlockFull.begin());
                    }
                  if (ivec + B > Noc)
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
                        OXBlockFull.begin(),
                        B,
                        &scalarCoeffBeta,
                        overlapMatrixBlockDP.begin(),
                        D);
                    }
                  else
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
                        OXBlockFull.begin(),
                        B,
                        &scalarCoeffBeta,
                        overlapMatrixBlockDP.begin(),
                        B);

                      const dftfe::uInt DRem = D - B;

                      if (DRem != 0)
                        {
                          BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                            B,
                            B,
                            M,
                            0,
                            OXBlockFull.begin(),
                            OXBlockFullFP32.begin());

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
                            OXBlockFullFP32.begin(),
                            B,
                            &scalarCoeffBetaSP,
                            overlapMatrixBlockSP.begin(),
                            DRem);
                        }
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

              const dftfe::uInt DRem = D - B;

              const dftfe::uInt ivecNew = ivec + vectorsBlockSize;
              const dftfe::uInt DNew    = N - ivecNew;
              const dftfe::uInt BNew = std::min(vectorsBlockSize, N - ivecNew);

              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  for (dftfe::uInt k = ivecNew; k < ivecNew + BNew;
                       k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      operatorMatrix.overlapMatrixTimesX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        OXBlock,
                        dftParams.approxOverlapMatrix);
                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        BNew,
                        chebyBlockSize,
                        M,
                        k - ivecNew,
                        OXBlock.begin(),
                        OXBlockFull.begin());
                    }

                  // evaluate X^{T} times XBlock
                  if (ivecNew + BNew > Noc)
                    {
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
                        OXBlockFull.begin(),
                        BNew,
                        &scalarCoeffBeta,
                        overlapMatrixBlockDPNext.begin(),
                        DNew);
                    }
                  else
                    {
                      BLASWrapperPtr->xgemm(
                        'N',
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
                        OXBlockFull.begin(),
                        BNew,
                        &scalarCoeffBeta,
                        overlapMatrixBlockDPNext.begin(),
                        BNew);

                      const dftfe::uInt DRemNew = DNew - BNew;

                      if (DRemNew != 0)
                        {
                          BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                            BNew,
                            BNew,
                            M,
                            0,
                            OXBlockFull.begin(),
                            OXBlockFullFP32.begin());

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
                            OXBlockFullFP32.begin(),
                            BNew,
                            &scalarCoeffBetaSP,
                            overlapMatrixBlockSPNext.begin(),
                            DRemNew);
                        }
                    }

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (ivec + B > Noc)
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          overlapMatrixBlockDP.begin(),
                          overlapMatrixBlockDP.begin(),
                          D * B,
                          tempReal.begin(),
                          tempImag.begin(),
                          streamDataMove);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          overlapMatrixBlockDP.begin(),
                          overlapMatrixBlockDP.begin(),
                          D * B,
                          streamDataMove);
                    }
                  else
                    {
                      if (DRem == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              overlapMatrixBlockDP.begin(),
                              overlapMatrixBlockDP.begin(),
                              B * B,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDataMove);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              overlapMatrixBlockDP.begin(),
                              overlapMatrixBlockDP.begin(),
                              B * B,
                              streamDataMove);
                        }
                      if (DRem != 0)
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
                    }
                }

              if (ivec + B > Noc)
                dftfe::utils::deviceMemcpyAsyncD2H(
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    overlapMatrixBlockHostDP.begin()),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    overlapMatrixBlockDP.begin()),
                  D * B * sizeof(dataTypes::number),
                  streamDataMove);
              else
                {
                  dftfe::utils::deviceMemcpyAsyncD2H(
                    overlapMatrixBlockHostDP.begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      overlapMatrixBlockDP.begin()),
                    B * B * sizeof(dataTypes::number),
                    streamDataMove);
                  if (DRem != 0)
                    dftfe::utils::deviceMemcpyAsyncD2H(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlockHostSP.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlockSP.begin()),
                      DRem * B * sizeof(dataTypes::numberFP32),
                      streamDataMove);
                }

              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  if (ivec + B > Noc)
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      overlapMatrixBlockHostDP.begin(),
                                      D * B,
                                      dataTypes::mpi_type_id(
                                        overlapMatrixBlockHostDP.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + ivec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + ivec];
                              for (dftfe::uInt i = j + ivec; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    overlapMatPar.local_el(it->second,
                                                           localColumnId) =
                                      overlapMatrixBlockHostDP[j * D + i -
                                                               ivec];
                                }
                            }
                    }
                  else
                    {
                      // Sum local overlap across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        overlapMatrixBlockHostDP.begin(),
                                        B * B,
                                        dataTypes::mpi_type_id(
                                          overlapMatrixBlockHostDP.begin()),
                                        MPI_SUM,
                                        mpiCommDomain);
                          if (DRem != 0)
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
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + ivec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + ivec];
                              for (dftfe::uInt i = j + ivec; i < ivec + B; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    overlapMatPar.local_el(it->second,
                                                           localColumnId) =
                                      overlapMatrixBlockHostDP[j * B + i -
                                                               ivec];
                                }
                              for (dftfe::uInt i = ivec + B; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    overlapMatPar.local_el(it->second,
                                                           localColumnId) =
                                      overlapMatrixBlockHostSP[j * DRem + i -
                                                               ivec - B];
                                }
                            }
                    }
                }
            } // band parallelization

          blockCount += 1;

        } // end block loop


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &OXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      const dftfe::uInt                                    Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const MPI_Comm                                  &mpiCommDomain,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &overlapMatPar,
      const dftParameters                             &dftParams)
    {
      // get global to local index maps for Scalapack matrix
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt numberBlocks     = N / vectorsBlockSize;

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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
        {
          dftfe::utils::deviceEventCreate(&computeEvents[i]);
          dftfe::utils::deviceEventCreate(&copyEvents[i]);
        }

      // create pinned memory used later to copy from Device->CPU
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projOverlapBlockHost;
      projOverlapBlockHost.resize(vectorsBlockSize * N, 0);
      std::memset(projOverlapBlockHost.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projOverlapBlockHostSP;
      projOverlapBlockHostSP.resize(N * vectorsBlockSize, 0);
      std::memset(projOverlapBlockHostSP.begin(),
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      // allocate device vectors to be used later
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        OXBlockFull(vectorsBlockSize * M, dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projOverlapMatrixBlock(N * vectorsBlockSize, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projOverlapMatrixBlockNext(N * vectorsBlockSize, dataTypes::number(0));


      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projOverlapMatrixBlockMove(vectorsBlockSize * vectorsBlockSize,
                                   dataTypes::number(0));


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        projOverlapMatrixBlockSP(N * vectorsBlockSize,
                                 dataTypes::numberFP32(0));


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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);
          const dftfe::uInt D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock.
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  const dftfe::uInt chebyBlockSize =
                    std::min(dftParams.chebyWfcBlockSize, N);

                  for (dftfe::uInt k = ivec; k < ivec + B; k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate XBlock^{T} times H^{T} and store in HXBlock
                      operatorMatrix.overlapMatrixTimesX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        OXBlock,
                        dftParams.approxOverlapMatrix);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - ivec,
                        OXBlock.begin(),
                        OXBlockFull.begin());
                    }



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
                    OXBlockFull.data(),
                    B,
                    &scalarCoeffBeta,
                    projOverlapMatrixBlock.begin(),
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
                projOverlapMatrixBlock.swap(projOverlapMatrixBlockNext);

              const dftfe::uInt ivecNew = ivec + vectorsBlockSize;
              const dftfe::uInt DNew    = N - ivecNew;
              const dftfe::uInt BNew = std::min(vectorsBlockSize, N - ivecNew);


              // start computations on the next block
              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // evaluate X^{T} times XBlock
                  const dftfe::uInt chebyBlockSize =
                    std::min(dftParams.chebyWfcBlockSize, N);

                  for (dftfe::uInt k = ivecNew; k < ivecNew + B;
                       k += chebyBlockSize)
                    {
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        chebyBlockSize, N, M, k, X, XBlock.begin());

                      // evaluate XBlock^{T} times H^{T} and store in HXBlock
                      operatorMatrix.overlapMatrixTimesX(
                        XBlock,
                        1.0,
                        0.0,
                        0.0,
                        OXBlock,
                        dftParams.approxOverlapMatrix);

                      BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                        B,
                        chebyBlockSize,
                        M,
                        k - ivecNew,
                        OXBlock.begin(),
                        OXBlockFull.begin());
                    }

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
                    OXBlockFull.begin(),
                    B,
                    &scalarCoeffBeta,
                    projOverlapMatrixBlockNext.begin(),
                    DNew);

                  // record completion of compute for next block
                  dftfe::utils::deviceEventRecord(computeEvents[blockCount + 1],
                                                  streamCompute);
                }


              const dftfe::uInt DRem = D - B;
              copyFromOverlapMatBlockToDPSPBlocks(
                B,
                D,
                projOverlapMatrixBlock.begin(),
                projOverlapMatrixBlockMove.begin(),
                projOverlapMatrixBlockSP.begin(),
                streamDataMove);


              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (ivec + B > Noc)
                    {
                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projOverlapMatrixBlock.begin(),
                          projOverlapMatrixBlock.begin(),
                          D * B,
                          tempReal.begin(),
                          tempImag.begin(),
                          streamDataMove);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          projOverlapMatrixBlock.begin(),
                          projOverlapMatrixBlock.begin(),
                          D * B,
                          streamDataMove);
                    }
                  else
                    {
                      if (DRem == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              projOverlapMatrixBlock.begin(),
                              projOverlapMatrixBlock.begin(),
                              B * B,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDataMove);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              projOverlapMatrixBlock.begin(),
                              projOverlapMatrixBlock.begin(),
                              B * B,
                              streamDataMove);
                        }
                      else
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain
                              .deviceDirectAllReduceMixedPrecGroupWrapper(
                                projOverlapMatrixBlockMove.begin(),
                                projOverlapMatrixBlockSP.begin(),
                                projOverlapMatrixBlockMove.begin(),
                                projOverlapMatrixBlockSP.begin(),
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
                                projOverlapMatrixBlockMove.begin(),
                                projOverlapMatrixBlockSP.begin(),
                                projOverlapMatrixBlockMove.begin(),
                                projOverlapMatrixBlockSP.begin(),
                                B * B,
                                DRem * B,
                                streamDataMove);
                        }
                    }
                }
              if (ivec + B > Noc)
                dftfe::utils::deviceMemcpyAsyncD2H(
                  projOverlapBlockHost.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projOverlapMatrixBlock.begin()),
                  D * B * sizeof(dataTypes::number),
                  streamDataMove);
              else
                {
                  dftfe::utils::deviceMemcpyAsyncD2H(
                    projOverlapBlockHost.begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      projOverlapMatrixBlockMove.begin()),
                    B * B * sizeof(dataTypes::number),
                    streamDataMove);
                  if (DRem != 0)
                    dftfe::utils::deviceMemcpyAsyncD2H(
                      projOverlapBlockHostSP.begin(),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        projOverlapMatrixBlockSP.begin()),
                      DRem * B * sizeof(dataTypes::numberFP32),
                      streamDataMove);
                }


              // record completion of Device->CPU copy for current block
              dftfe::utils::deviceEventRecord(copyEvents[blockCount],
                                              streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (dftfe::utils::deviceEventSynchronize(
                    copyEvents[blockCount]) == dftfe::utils::deviceSuccess)
                {
                  if (ivec + B > Noc)
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      projOverlapBlockHost.begin(),
                                      D * B,
                                      dataTypes::mpi_type_id(
                                        projOverlapBlockHost.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + ivec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + ivec];
                              for (dftfe::uInt i = j + ivec; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    overlapMatPar.local_el(it->second,
                                                           localColumnId) =
                                      projOverlapBlockHost[j * D + i - ivec];
                                }
                            }
                    }
                  else
                    {
                      // Sum local projHamBlock across domain decomposition
                      // processors
                      if (!dftParams.useDeviceDirectAllReduce)
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        projOverlapBlockHost.begin(),
                                        B * B,
                                        dataTypes::mpi_type_id(
                                          projOverlapBlockHost.begin()),
                                        MPI_SUM,
                                        mpiCommDomain);
                          if (DRem != 0)
                            MPI_Allreduce(MPI_IN_PLACE,
                                          projOverlapBlockHostSP.begin(),
                                          DRem * B,
                                          dataTypes::mpi_type_id(
                                            projOverlapBlockHostSP.begin()),
                                          MPI_SUM,
                                          mpiCommDomain);
                        }

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (dftfe::uInt i = 0; i < B; ++i)
                          if (globalToLocalColumnIdMap.find(i + ivec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[i + ivec];
                              for (dftfe::uInt j = i + ivec; j < ivec + B; ++j)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(j);
                                  if (it != globalToLocalRowIdMap.end())
                                    overlapMatPar.local_el(it->second,
                                                           localColumnId) =
                                      projOverlapBlockHost[i * B + j - ivec];
                                }
                              for (dftfe::uInt j = ivec + B; j < N; ++j)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(j);
                                  if (it != globalToLocalRowIdMap.end())
                                    overlapMatPar.local_el(it->second,
                                                           localColumnId) =
                                      projOverlapBlockHostSP[i * DRem + j -
                                                             ivec - B];
                                }
                            }
                    }
                }
            } // band parallelization

          blockCount += 1;
        } // end block loop


      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      dataTypes::number                                   *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &HXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      const std::vector<double>                           &eigenValues,
      const MPI_Comm                                      &mpiCommParent,
      const MPI_Comm                                      &mpiCommDomain,
      const MPI_Comm                                      &interBandGroupComm,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                          &BLASWrapperPtr,
      std::vector<double> &residualNorm,
      const dftParameters &dftParams,
      const bool           useBandParal)
    {
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);


      const dftfe::uInt vectorsBlockSize = dftParams.wfcBlockSize;
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

      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);


          if (((jvec + B) <=
                 bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
               (jvec + B) >
                 bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]) ||
              !useBandParal)
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              for (dftfe::uInt k = jvec; k < jvec + B; k += chebyBlockSize)
                {
                  BLASWrapperPtr->stridedCopyToBlockConstantStride(
                    chebyBlockSize, N, M, k, X, XBlock.begin());

                  // evaluate H times XBlock^{T} and store in HXBlock^{T}
                  operatorMatrix.overlapMatrixTimesX(
                    XBlock,
                    1.0,
                    0.0,
                    0.0,
                    HXBlock,
                    dftParams.approxOverlapMatrix);

                  computeDiagQTimesX(eigenValuesDevice.begin() + k,
                                     HXBlock.begin(),
                                     chebyBlockSize,
                                     M);


                  operatorMatrix.HX(XBlock, 1.0, -1.0, 0.0, HXBlock);
                  if (dftParams.approxOverlapMatrix)
                    {
                      BLASWrapperPtr->stridedBlockScale(
                        chebyBlockSize,
                        M,
                        1.0,
                        operatorMatrix.getInverseSqrtMassVector().data(),
                        HXBlock.data());
                    }
                  BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                    B,
                    chebyBlockSize,
                    M,
                    k - jvec,
                    HXBlock.begin(),
                    HXBlockFull.begin());
                }

              computeGeneralisedResidualDevice(
                B, M, N, jvec, HXBlockFull.begin(), residualSqDevice.begin());

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
      for (dftfe::uInt iWave = 0; iWave < N; ++iWave)
        residualNorm[iWave] = std::sqrt(residualNorm[iWave]);

      if (dftParams.verbosity >= 4 &&
          dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
        for (dftfe::uInt iWave = 0; iWave < N; ++iWave)
          std::cout << "eigen vector " << iWave << ": " << residualNorm[iWave]
                    << std::endl;

      if (dftParams.verbosity >= 4)
        if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
          std::cout << std::endl;
    }

    // X^{T}*HConj*XConj
    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
         const dataTypes::number                             *X,
         distributedDeviceVec<dataTypes::number>             &XBlock,
         distributedDeviceVec<dataTypes::number>             &HXBlock,
         const dftfe::uInt                                    M,
         const dftfe::uInt                                    N,
         std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                         &BLASWrapperPtr,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
         utils::DeviceCCLWrapper &devicecclMpiCommDomain,
         const MPI_Comm          &mpiCommDomain,
         const MPI_Comm          &interBandGroupComm,
         const dftParameters     &dftParams,
         const bool               onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);



      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

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

      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              for (dftfe::uInt k = jvec; k < jvec + B; k += chebyBlockSize)
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
              const dftfe::uInt D           = N - jvec;
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
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = j + jvec; i < N; ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
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
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &HXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &mpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftParameters                             &dftParams,
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


      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);



      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt numberBlocks     = N / vectorsBlockSize;

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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const dftfe::uInt D           = N - jvec;

              // handle edge case for the first block or the first block in the
              // band group in case of band parallelization
              if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // compute HXBlockFull in an inner loop over blocks of B
                  // wavefunction vectors
                  for (dftfe::uInt k = jvec; k < jvec + B; k += chebyBlockSize)
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

              const dftfe::uInt jvecNew = jvec + vectorsBlockSize;
              const dftfe::uInt DNew    = N - jvecNew;

              // start computations on the next block
              if (jvecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  for (dftfe::uInt k = jvecNew; k < jvecNew + B;
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
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + jvec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + jvec];
                          for (dftfe::uInt i = j + jvec; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &HXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      const dftfe::uInt                                    Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &mpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftParameters                             &dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);


      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

      const dftfe::uInt numberBlocks = N / vectorsBlockSize;

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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const dataTypes::numberFP32 alphaFP32 =
                                            dataTypes::numberFP32(1.0),
                                          betaFP32 = dataTypes::numberFP32(0.0);
              const dftfe::uInt D                  = N - jvec;

              // handle edge case for the first block or the first block in the
              // band group in case of band parallelization
              if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // compute HXBlockFull or HXBlockFullFP32 in an inner loop
                  // over blocks of B wavefunction vectors
                  for (dftfe::uInt k = jvec; k < jvec + B; k += chebyBlockSize)
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

                  // evaluate X^{T} times HXBlockFullConj or XFP32^{T} times
                  // HXBlockFullFP32Conj
                  const dftfe::uInt DRem = D - B;
                  if (jvec + B > Noc)
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
                        &alpha,
                        X + jvec,
                        N,
                        HXBlockFull.begin(),
                        B,
                        &beta,
                        projHamBlock.begin(),
                        D);
                    }
                  else
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
                        &alpha,
                        X + jvec,
                        N,
                        HXBlockFull.begin(),
                        B,
                        &beta,
                        projHamBlock.begin(),
                        B);
                      if (DRem != 0)
                        {
                          BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                            B,
                            B,
                            M,
                            0,
                            HXBlockFull.begin(),
                            HXBlockFullFP32.begin());

                          BLASWrapperPtr->xgemm(
                            'N',
                            std::is_same<dataTypes::numberFP32,
                                         std::complex<float>>::value ?
                              'C' :
                              'T',
                            DRem,
                            B,
                            M,
                            &alphaFP32,
                            XFP32.begin() + jvec + B,
                            N,
                            HXBlockFullFP32.begin(),
                            B,
                            &betaFP32,
                            projHamBlockFP32.begin(),
                            DRem);
                        }
                    }

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
                    {
                      projHamBlock.swap(projHamBlockNext);
                      projHamBlockFP32.swap(projHamBlockFP32Next);
                    }
                }
              const dftfe::uInt DRem    = D - B;
              const dftfe::uInt jvecNew = jvec + vectorsBlockSize;
              const dftfe::uInt DNew    = N - jvecNew;
              const dftfe::uInt BNew = std::min(vectorsBlockSize, N - jvecNew);
              if (jvecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // compute HXBlockFull or HXBlockFullFP32 in an inner loop
                  // over blocks of B wavefunction vectors
                  for (dftfe::uInt k = jvecNew; k < jvecNew + BNew;
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
                        BNew,
                        chebyBlockSize,
                        M,
                        k - jvecNew,
                        HXBlock.begin(),
                        HXBlockFull.begin());
                    }

                  // evaluate X^{T} times HXBlockFullConj or XFP32^{T} times
                  // HXBlockFullFP32Conj
                  const dftfe::uInt DRemNew = DNew - BNew;
                  if (jvecNew + BNew > Noc)
                    {
                      BLASWrapperPtr->xgemm(
                        'N',
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          'C' :
                          'T',
                        DNew,
                        BNew,
                        M,
                        &alpha,
                        X + jvecNew,
                        N,
                        HXBlockFull.begin(),
                        BNew,
                        &beta,
                        projHamBlockNext.begin(),
                        DNew);
                    }
                  else
                    {
                      BLASWrapperPtr->xgemm(
                        'N',
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          'C' :
                          'T',
                        BNew,
                        BNew,
                        M,
                        &alpha,
                        X + jvecNew,
                        N,
                        HXBlockFull.begin(),
                        BNew,
                        &beta,
                        projHamBlockNext.begin(),
                        BNew);

                      if (DRemNew != 0)
                        {
                          BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                            BNew,
                            BNew,
                            M,
                            0,
                            HXBlockFull.begin(),
                            HXBlockFullFP32.begin());

                          BLASWrapperPtr->xgemm(
                            'N',
                            std::is_same<dataTypes::numberFP32,
                                         std::complex<float>>::value ?
                              'C' :
                              'T',
                            DRemNew,
                            BNew,
                            M,
                            &alphaFP32,
                            XFP32.begin() + jvecNew + BNew,
                            N,
                            HXBlockFullFP32.begin(),
                            BNew,
                            &betaFP32,
                            projHamBlockFP32Next.begin(),
                            DRemNew);
                        }
                    }
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
                      if (DRem == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              projHamBlock.begin(),
                              projHamBlock.begin(),
                              B * B,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDataMove);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              projHamBlock.begin(),
                              projHamBlock.begin(),
                              B * B,
                              streamDataMove);
                        }
                      if (DRem != 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain
                              .deviceDirectAllReduceMixedPrecGroupWrapper(
                                projHamBlock.begin(),
                                projHamBlockFP32.begin(),
                                projHamBlock.begin(),
                                projHamBlockFP32.begin(),
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
                                projHamBlock.begin(),
                                projHamBlockFP32.begin(),
                                projHamBlock.begin(),
                                projHamBlockFP32.begin(),
                                B * B,
                                DRem * B,
                                streamDataMove);
                        }
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
                {
                  dftfe::utils::deviceMemcpyAsyncD2H(
                    projHamBlockHost.begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      projHamBlock.begin()),
                    B * B * sizeof(dataTypes::number),
                    streamDataMove);
                  if (DRem != 0)
                    dftfe::utils::deviceMemcpyAsyncD2H(
                      projHamBlockHostFP32.begin(),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        projHamBlockFP32.begin()),
                      DRem * B * sizeof(dataTypes::numberFP32),
                      streamDataMove);
                }

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
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (dftfe::uInt i = j + jvec; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
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
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        projHamBlockHost.begin(),
                                        B * B,
                                        dataTypes::mpi_type_id(
                                          projHamBlockHost.begin()),
                                        MPI_SUM,
                                        mpiCommDomain);
                          if (DRem != 0)
                            MPI_Allreduce(MPI_IN_PLACE,
                                          projHamBlockHostFP32.begin(),
                                          DRem * B,
                                          dataTypes::mpi_type_id(
                                            projHamBlockHostFP32.begin()),
                                          MPI_SUM,
                                          mpiCommDomain);
                        }

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (dftfe::uInt i = j + jvec; i < jvec + B; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHost[j * B + i - jvec];
                                }
                              for (dftfe::uInt i = jvec + B; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHostFP32[j * DRem + i - jvec -
                                                           B];
                                }
                            }
                    }
                }

            } // band parallelization
          blockCount += 1;
        }

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
      const dataTypes::number                             *X,
      distributedDeviceVec<dataTypes::number>             &XBlock,
      distributedDeviceVec<dataTypes::number>             &HXBlock,
      const dftfe::uInt                                    M,
      const dftfe::uInt                                    N,
      const dftfe::uInt                                    Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                      &BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
      utils::DeviceCCLWrapper                         &devicecclMpiCommDomain,
      const MPI_Comm                                  &mpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftParameters                             &dftParams,
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


      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);



      const dftfe::uInt vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const dftfe::uInt numberBlocks     = N / vectorsBlockSize;

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

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projHamBlockMove(vectorsBlockSize * vectorsBlockSize,
                         dataTypes::number(0.0));
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

      dftfe::uInt blockCount = 0;
      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              const dftfe::uInt chebyBlockSize =
                std::min(dftParams.chebyWfcBlockSize, N);

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              const dftfe::uInt D           = N - jvec;

              // handle edge case for the first block or the first block in the
              // band group in case of band parallelization
              if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // compute HXBlockFull in an inner loop over blocks of B
                  // wavefunction vectors
                  for (dftfe::uInt k = jvec; k < jvec + B; k += chebyBlockSize)
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

              const dftfe::uInt jvecNew = jvec + vectorsBlockSize;
              const dftfe::uInt DNew    = N - jvecNew;

              // start computations on the next block
              if (jvecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  for (dftfe::uInt k = jvecNew; k < jvecNew + B;
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
              const dftfe::uInt DRem = D - B;
              if (!(jvec + B > Noc) && DRem != 0)
                {
                  BLASWrapperPtr->setStream(streamDataMove);
                  // Add a kernel to copy the required Data
                  // BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                  //   D * B, projHamBlock.begin(), projHamBlockFP32.begin());
                  // BLASWrapperPtr
                  //   ->copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1Arr(
                  //     B,
                  //     DRem,
                  //     D,
                  //     projHamBlock.begin(),
                  //     projHamBlockMove.begin(),
                  //     projHamBlockFP32.begin());
                  copyFromOverlapMatBlockToDPSPBlocks(B,
                                                      D,
                                                      projHamBlock.begin(),
                                                      projHamBlockMove.begin(),
                                                      projHamBlockFP32.begin(),
                                                      streamDataMove);
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
                      if (DRem == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              projHamBlockMove.begin(),
                              projHamBlockMove.begin(),
                              B * B,
                              tempReal.begin(),
                              tempImag.begin(),
                              streamDataMove);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              projHamBlockMove.begin(),
                              projHamBlockMove.begin(),
                              B * B,
                              streamDataMove);
                        }
                      else
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain
                              .deviceDirectAllReduceMixedPrecGroupWrapper(
                                projHamBlockMove.begin(),
                                projHamBlockFP32.begin(),
                                projHamBlockMove.begin(),
                                projHamBlockFP32.begin(),
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
                                projHamBlockMove.begin(),
                                projHamBlockFP32.begin(),
                                projHamBlockMove.begin(),
                                projHamBlockFP32.begin(),
                                B * B,
                                DRem * B,
                                streamDataMove);
                        }
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
                {
                  dftfe::utils::deviceMemcpyAsyncD2H(
                    projHamBlockHost.begin(),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      projHamBlockMove.begin()),
                    B * B * sizeof(dataTypes::number),
                    streamDataMove);
                  if (DRem != 0)
                    dftfe::utils::deviceMemcpyAsyncD2H(
                      projHamBlockHostFP32.begin(),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        projHamBlockFP32.begin()),
                      DRem * B * sizeof(dataTypes::numberFP32),
                      streamDataMove);
                }

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
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (dftfe::uInt i = j + jvec; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
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
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        projHamBlockHost.begin(),
                                        B * B,
                                        dataTypes::mpi_type_id(
                                          projHamBlockHost.begin()),
                                        MPI_SUM,
                                        mpiCommDomain);
                          if (DRem != 0)
                            MPI_Allreduce(MPI_IN_PLACE,
                                          projHamBlockHostFP32.begin(),
                                          DRem * B,
                                          dataTypes::mpi_type_id(
                                            projHamBlockHostFP32.begin()),
                                          MPI_SUM,
                                          mpiCommDomain);
                        }

                      // Copying only the lower triangular part to the ScaLAPACK
                      // projected Hamiltonian matrix
                      if (processGrid->is_process_active())
                        for (dftfe::uInt j = 0; j < B; ++j)
                          if (globalToLocalColumnIdMap.find(j + jvec) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const dftfe::uInt localColumnId =
                                globalToLocalColumnIdMap[j + jvec];
                              for (dftfe::uInt i = j + jvec; i < jvec + B; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHost[j * B + i - jvec];
                                }
                              for (dftfe::uInt i = jvec + B; i < N; ++i)
                                {
                                  std::unordered_map<dftfe::uInt,
                                                     dftfe::uInt>::iterator it =
                                    globalToLocalRowIdMap.find(i);
                                  if (it != globalToLocalRowIdMap.end())
                                    projHamPar.local_el(it->second,
                                                        localColumnId) =
                                      projHamBlockHostFP32[j * DRem + i - jvec -
                                                           B];
                                }
                            }
                    }
                }

            } // band parallelization
          blockCount += 1;
        }

      // return deviceblas handle to default stream
      BLASWrapperPtr->setStream(NULL);

      for (dftfe::Int i = 0; i < numberBlocks; ++i)
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
    template void
    reformulatedChebyshevFilterOverlapComputeCommunication(
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                          &BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X2,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y2,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
        &X1_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
        &Y1_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
        &X2_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
                         &Y2_SP,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix);

    template void
    reformulatedChebyshevFilterOverlapComputeCommunication(
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                                          &BLASWrapperPtr,
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
                         &Y2_SP,
      std::vector<double> eigenvalues,
      const dftfe::uInt   m,
      const double        a,
      const double        b,
      const double        a0,
      const bool          approxOverlapMatrix);

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
