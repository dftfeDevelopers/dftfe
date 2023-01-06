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
// @author Sambit Das


#include <dftUtils.h>
#include <dftParameters.h>
#include <linearAlgebraOperationsDevice.h>
#include <linearAlgebraOperationsInternal.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {

    namespace
    {
      __global__ void
      setZeroKernel(const unsigned int BVec,
                    const unsigned int M,
                    const unsigned int N,
                    double *           yVec,
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
    } // namespace

    void
    pseudoGramSchmidtOrthogonalization(
      elpaScalaManager &                elpaScala,
      dataTypes::number *               X,
      const unsigned int                M,
      const unsigned int                N,
      const MPI_Comm &                  mpiCommParent,
      const MPI_Comm &                  mpiCommDomain,
      utils::DeviceCCLWrapper &         devicecclMpiCommDomain,
      const MPI_Comm &                  interBandGroupComm,
      dftfe::utils::deviceBlasHandle_t &handle,
      const dftParameters &             dftParams,
      const bool                        useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecCGS_O && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "SConj=X^{T}XConj Mixed Prec");
          else
            computing_timer.enter_subsection("SConj=X^{T}XConj");
        }


      dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatPar(N,
                                                              processGrid,
                                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  dataTypes::number(0.0));

      // SConj=X^{T}*XConj with X^{T} stored in the column
      // major format
      if (dftParams.useMixedPrecCGS_O && useMixedPrecOverall)
        {
          if (dftParams.overlapComputeCommunOrthoRR)
            linearAlgebraOperationsDevice::
              fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(
                X,
                M,
                N,
                handle,
                mpiCommDomain,
                devicecclMpiCommDomain,
                interBandGroupComm,
                processGrid,
                overlapMatPar,
                dftParams);
          else
            linearAlgebraOperationsDevice::
              fillParallelOverlapMatMixedPrecScalapack(X,
                                                       M,
                                                       N,
                                                       handle,
                                                       mpiCommDomain,
                                                       devicecclMpiCommDomain,
                                                       interBandGroupComm,
                                                       processGrid,
                                                       overlapMatPar,
                                                       dftParams);
        }
      else
        {
          if (dftParams.overlapComputeCommunOrthoRR)
            linearAlgebraOperationsDevice::
              fillParallelOverlapMatScalapackAsyncComputeCommun(
                X,
                M,
                N,
                handle,
                mpiCommDomain,
                devicecclMpiCommDomain,
                interBandGroupComm,
                processGrid,
                overlapMatPar,
                dftParams);
          else
            linearAlgebraOperationsDevice::fillParallelOverlapMatScalapack(
              X,
              M,
              N,
              handle,
              mpiCommDomain,
              devicecclMpiCommDomain,
              interBandGroupComm,
              processGrid,
              overlapMatPar,
              dftParams);
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecCGS_O && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "SConj=X^{T}XConj Mixed Prec");
          else
            computing_timer.leave_subsection("SConj=X^{T}XConj");
        }

      // SConj=LConj*L^{T}
      if (dftParams.deviceFineGrainedTimings)
        computing_timer.enter_subsection(
          "Cholesky and triangular matrix invert");      

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParams.useELPA)
        {
          // For ELPA cholesky only the upper triangular part of the hermitian
          // matrix is required
          dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatParConjTrans(
            N, processGrid, rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      dataTypes::number(0.0));

          overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

          if (processGrid->is_process_active())
            {
              int error;

              if (dftParams.useELPADeviceKernel)
                {
#ifdef DFTFE_WITH_DEVICE_NVIDIA
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "nvidia-gpu",
                                   0,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#elif DFTFE_WITH_DEVICE_AMD
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "amd-gpu",
                                   0,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
                }


              elpa_cholesky(elpaScala.getElpaHandle(),
                            &overlapMatParConjTrans.local_el(0, 0),
                            &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));

              if (dftParams.useELPADeviceKernel)
                {
#ifdef DFTFE_WITH_DEVICE_NVIDIA
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "nvidia-gpu",
                                   1,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#elif DFTFE_WITH_DEVICE_AMD
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "amd-gpu",
                                   1,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
                }
            }
          overlapMatPar.copy_conjugate_transposed(overlapMatParConjTrans);
          overlapMatPropertyPostCholesky =
            dftfe::LAPACKSupport::Property::lower_triangular;
        }
      else
        {
          overlapMatPar.compute_cholesky_factorization();

          overlapMatPropertyPostCholesky = overlapMatPar.get_property();
        }
      AssertThrow(
        overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


      // extract LConj
      dftfe::ScaLAPACKMatrix<dataTypes::number> LMatPar(
        N,
        processGrid,
        rowsBlockSize,
        dftfe::LAPACKSupport::Property::lower_triangular);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = LMatPar.global_column(i);
            for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = LMatPar.global_row(j);
                if (glob_j < glob_i)
                  LMatPar.local_el(j, i) = dataTypes::number(0);
                else
                  LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
              }
          }

      // compute LConj^{-1}
      LMatPar.invert();

      if (dftParams.deviceFineGrainedTimings)
        computing_timer.leave_subsection(
          "Cholesky and triangular matrix invert");

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecCGS_SR && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "X^{T}=Lconj^{-1}*X^{T} Mixed Prec");
          else
            computing_timer.enter_subsection(
              "X^{T}=Lconj^{-1}*X^{T}");
        }

      // X^{T}=LConj^{-1}*X^{T} with X^{T} stored in
      // the column major format
      if (dftParams.useMixedPrecCGS_SR && useMixedPrecOverall)
        subspaceRotationCGSMixedPrecScalapack(X,
                                              M,
                                              N,
                                              handle,
                                              processGrid,
                                              mpiCommDomain,
                                              devicecclMpiCommDomain,
                                              interBandGroupComm,
                                              LMatPar,
                                              dftParams,
                                              false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  devicecclMpiCommDomain,
                                  interBandGroupComm,
                                  LMatPar,
                                  dftParams,
                                  false,
                                  true);

      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


      if (numberBandGroups > 1)
        {
          // band group parallelization data structures
          const unsigned int bandGroupTaskId =
            dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
          std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
          dftUtils::createBandParallelizationIndices(
            interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

          const unsigned int vectorsBlockSize =
            std::min(dftParams.wfcBlockSize, N);
          for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

              if (!((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                {
                  // set to zero wavefunctions which are not inside a given band
                  // paral group
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                  setZeroKernel<<<(BVec +
                                   (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * M,
                                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                    BVec,
                    M,
                    N,
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    jvec);
#elif DFTFE_WITH_DEVICE_LANG_HIP
                  hipLaunchKernelGGL(
                    setZeroKernel,
                    (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE * M,
                    dftfe::utils::DEVICE_BLOCK_SIZE,
                    0,
                    0,
                    BVec,
                    M,
                    N,
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    jvec);
#endif
                }
            }



          std::vector<dataTypes::number> eigenVectorsFlattenedHost(
            M * N, dataTypes::number(0.0));

          dftfe::utils::deviceMemcpyD2H(
            dftfe::utils::makeDataTypeDeviceCompatible(
              &eigenVectorsFlattenedHost[0]),
            X,
            M * N * sizeof(dataTypes::number));

          MPI_Barrier(interBandGroupComm);


          MPI_Allreduce(MPI_IN_PLACE,
                        &eigenVectorsFlattenedHost[0],
                        M * N,
                        dataTypes::mpi_type_id(&eigenVectorsFlattenedHost[0]),
                        MPI_SUM,
                        interBandGroupComm);

          MPI_Barrier(interBandGroupComm);

          dftfe::utils::deviceMemcpyH2D(
            X,
            dftfe::utils::makeDataTypeDeviceCompatible(
              &eigenVectorsFlattenedHost[0]),
            M * N * sizeof(dataTypes::number));
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecCGS_SR && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "X^{T}=Lconj^{-1}*X^{T} Mixed Prec");
          else
            computing_timer.leave_subsection(
              "X^{T}=Lconj^{-1}*X^{T}");
        }

    }
  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
