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


#include "dftParameters.h"
#include "dftUtils.h"
#include "linearAlgebraOperationsCUDA.h"
#include "linearAlgebraOperationsInternal.h"

namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
    void
    rayleighRitz(
      operatorDFTCUDAClass &                       operatorMatrix,
      elpaScalaManager &                           elpaScala,
      dataTypes::numberGPU *                       X,
      distributedGPUVec<dataTypes::numberGPU> &    Xb,
      distributedGPUVec<dataTypes::numberFP32GPU> &floatXb,
      distributedGPUVec<dataTypes::numberGPU> &    HXb,
      distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector,
      const unsigned int                           M,
      const unsigned int                           N,
      const MPI_Comm &                             mpiCommParent,
      const MPI_Comm &                             mpiCommDomain,
      GPUCCLWrapper &                              gpucclMpiCommDomain,
      const MPI_Comm &                             interBandGroupComm,
      std::vector<double> &                        eigenValues,
      cublasHandle_t &                             handle,
      const bool                                   useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(N,
                                                           processGrid,
                                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  dataTypes::number(0.0));

      cudaDeviceSynchronize();
      computing_timer.enter_subsection("Blocked XtHX, RR step");

      if (dftParameters::overlapComputeCommunOrthoRR)
        operatorMatrix.XtHXOverlapComputeCommun(X,
                                                Xb,
                                                HXb,
                                                projectorKetTimesVector,
                                                M,
                                                N,
                                                handle,
                                                processGrid,
                                                projHamPar,
                                                gpucclMpiCommDomain);
      else
        operatorMatrix.XtHX(X,
                            Xb,
                            HXb,
                            projectorKetTimesVector,
                            M,
                            N,
                            handle,
                            processGrid,
                            projHamPar,
                            gpucclMpiCommDomain);

      cudaDeviceSynchronize();
      computing_timer.leave_subsection("Blocked XtHX, RR step");

      //
      // compute eigendecomposition of ProjHam HConjProj= QConj*D*QConj^{C} (C
      // denotes conjugate transpose LAPACK notation)
      //
      const unsigned int numberEigenValues = N;
      eigenValues.resize(numberEigenValues);
      if (dftParameters::useELPA)
        {
          cudaDeviceSynchronize();
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          dftfe::ScaLAPACKMatrix<dataTypes::number> eigenVectors(N,
                                                                 processGrid,
                                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      dataTypes::number(0.0));

          // For ELPA eigendecomposition the full matrix is required unlike
          // ScaLAPACK which can work with only the lower triangular part
          dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParConjTrans(
            N, processGrid, rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&projHamParConjTrans.local_el(0, 0),
                      &projHamParConjTrans.local_el(0, 0) +
                        projHamParConjTrans.local_m() *
                          projHamParConjTrans.local_n(),
                      dataTypes::number(0.0));


          projHamParConjTrans.copy_conjugate_transposed(projHamPar);
          projHamPar.add(projHamParConjTrans,
                         dataTypes::number(1.0),
                         dataTypes::number(1.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
              {
                const unsigned int glob_i = projHamPar.global_column(i);
                for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
                  {
                    const unsigned int glob_j = projHamPar.global_row(j);
                    if (glob_i == glob_j)
                      projHamPar.local_el(j, i) *= dataTypes::number(0.5);
                  }
              }

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandle(),
                               &projHamPar.local_el(0, 0),
                               &eigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiCommDomain);


          eigenVectors.copy_to(projHamPar);
          cudaDeviceSynchronize();
          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          cudaDeviceSynchronize();
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, N - 1), true);
          cudaDeviceSynchronize();
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }

      linearAlgebraOperations::internal::broadcastAcrossInterCommScaLAPACKMat(
        processGrid, projHamPar, interBandGroupComm, 0);

      //
      // rotate the basis in the subspace X = X*Q, implemented as
      // X^{T}=Qc^{C}*X^{T} with X^{T} stored in the column major format
      //
      cudaDeviceSynchronize();
      computing_timer.enter_subsection("Blocked subspace rotation, RR step");
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParCopy(N,
                                                               processGrid,
                                                               rowsBlockSize);
      projHamParCopy.copy_conjugate_transposed(projHamPar);

      if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
        subspaceRotationRRMixedPrecScalapack(X,
                                             M,
                                             N,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             gpucclMpiCommDomain,
                                             interBandGroupComm,
                                             projHamParCopy,
                                             false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  gpucclMpiCommDomain,
                                  interBandGroupComm,
                                  projHamParCopy,
                                  false);
      cudaDeviceSynchronize();
      computing_timer.leave_subsection("Blocked subspace rotation, RR step");
    }

    void
    rayleighRitzGEP(
      operatorDFTCUDAClass &                       operatorMatrix,
      elpaScalaManager &                           elpaScala,
      dataTypes::numberGPU *                       X,
      distributedGPUVec<dataTypes::numberGPU> &    Xb,
      distributedGPUVec<dataTypes::numberFP32GPU> &floatXb,
      distributedGPUVec<dataTypes::numberGPU> &    HXb,
      distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector,
      const unsigned int                           M,
      const unsigned int                           N,
      const MPI_Comm &                             mpiCommParent,
      const MPI_Comm &                             mpiCommDomain,
      GPUCCLWrapper &                              gpucclMpiCommDomain,
      const MPI_Comm &                             interBandGroupComm,
      std::vector<double> &                        eigenValues,
      cublasHandle_t &                             handle,
      const bool                                   useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      //
      // SConj=X^{T}*XConj.
      //

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecCGS_O && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection("SConj=X^{T}XConj, RR GEP step");
        }

      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatPar(N,
                                                              processGrid,
                                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  dataTypes::number(0.0));

      if (dftParameters::useMixedPrecCGS_O && useMixedPrecOverall)
        {
          if (dftParameters::overlapComputeCommunOrthoRR)
            linearAlgebraOperationsCUDA::
              fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(
                X,
                M,
                N,
                handle,
                mpiCommDomain,
                gpucclMpiCommDomain,
                interBandGroupComm,
                processGrid,
                overlapMatPar);
          else
            linearAlgebraOperationsCUDA::
              fillParallelOverlapMatMixedPrecScalapack(X,
                                                       M,
                                                       N,
                                                       handle,
                                                       mpiCommDomain,
                                                       gpucclMpiCommDomain,
                                                       interBandGroupComm,
                                                       processGrid,
                                                       overlapMatPar);
        }
      else
        {
          if (dftParameters::overlapComputeCommunOrthoRR)
            linearAlgebraOperationsCUDA::
              fillParallelOverlapMatScalapackAsyncComputeCommun(
                X,
                M,
                N,
                handle,
                mpiCommDomain,
                gpucclMpiCommDomain,
                interBandGroupComm,
                processGrid,
                overlapMatPar);
          else
            linearAlgebraOperationsCUDA::fillParallelOverlapMatScalapack(
              X,
              M,
              N,
              handle,
              mpiCommDomain,
              gpucclMpiCommDomain,
              interBandGroupComm,
              processGrid,
              overlapMatPar);
        }

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecCGS_O && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection("SConj=X^{T}XConj, RR GEP step");
        }

      //
      // SConj=LConj*L^{T}
      //
      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.enter_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");


      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParameters::useELPA)
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
              elpaCholesky(elpaScala.getElpaHandle(),
                           &overlapMatParConjTrans.local_el(0, 0),
                           &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));
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

      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.leave_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");


      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          computing_timer.enter_subsection(
            "HConjProj= X^{T}*HConj*XConj, RR GEP step");
        }


      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(N,
                                                           processGrid,
                                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  dataTypes::number(0.0));

      if (dftParameters::overlapComputeCommunOrthoRR)
        operatorMatrix.XtHXOverlapComputeCommun(X,
                                                Xb,
                                                HXb,
                                                projectorKetTimesVector,
                                                M,
                                                N,
                                                handle,
                                                processGrid,
                                                projHamPar,
                                                gpucclMpiCommDomain);
      else
        operatorMatrix.XtHX(X,
                            Xb,
                            HXb,
                            projectorKetTimesVector,
                            M,
                            N,
                            handle,
                            processGrid,
                            projHamPar,
                            gpucclMpiCommDomain);

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          computing_timer.leave_subsection(
            "HConjProj= X^{T}*HConj*XConj, RR GEP step");
        }

      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.enter_subsection(
          "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParConjTrans(
        N, processGrid, rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  dataTypes::number(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      projHamPar.add(projHamParConjTrans,
                     dataTypes::number(1.0),
                     dataTypes::number(1.0));

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= dataTypes::number(0.5);
              }
          }

      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParCopy(N,
                                                               processGrid,
                                                               rowsBlockSize);

      // compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
      // conjugate transpose LAPACK notation)
      LMatPar.mmult(projHamParCopy, projHamPar);
      projHamParCopy.zmCmult(projHamPar, LMatPar);

      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.leave_subsection(
          "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");
      //
      // compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      const unsigned int numberEigenValues = N;
      eigenValues.resize(numberEigenValues);
      if (dftParameters::useELPA)
        {
          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.enter_subsection("ELPA eigen decomp, RR GEP step");
          dftfe::ScaLAPACKMatrix<dataTypes::number> eigenVectors(N,
                                                                 processGrid,
                                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      dataTypes::number(0.0));

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandle(),
                               &projHamPar.local_el(0, 0),
                               &eigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiCommDomain);


          eigenVectors.copy_to(projHamPar);

          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.leave_subsection("ELPA eigen decomp, RR GEP step");
        }
      else
        {
          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.enter_subsection(
              "ScaLAPACK eigen decomp, RR GEP step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, N - 1), true);
          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.leave_subsection(
              "ScaLAPACK eigen decomp, RR GEP step");
        }

      linearAlgebraOperations::internal::broadcastAcrossInterCommScaLAPACKMat(
        processGrid, projHamPar, interBandGroupComm, 0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      //
      // rotate the basis in the subspace
      // X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, stored in the column major
      // format In the above we use Q^{T}={QConjPrime}^{C}*LConj^{-1}

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (!(dftParameters::useMixedPrecSubspaceRotRR &&
                useMixedPrecOverall))
            computing_timer.enter_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR GEP step");
          else
            computing_timer.enter_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR GEP step");
        }

      projHamParCopy.copy_conjugate_transposed(projHamPar);
      projHamParCopy.mmult(projHamPar, LMatPar);

      if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
        subspaceRotationRRMixedPrecScalapack(X,
                                             M,
                                             N,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             gpucclMpiCommDomain,
                                             interBandGroupComm,
                                             projHamPar,
                                             false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  gpucclMpiCommDomain,
                                  interBandGroupComm,
                                  projHamPar,
                                  false);

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (!(dftParameters::useMixedPrecSubspaceRotRR &&
                useMixedPrecOverall))
            computing_timer.leave_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR GEP step");
          else
            computing_timer.leave_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR GEP step");
        }
    }

    void
    rayleighRitzGEPSpectrumSplitDirect(
      operatorDFTCUDAClass &                       operatorMatrix,
      elpaScalaManager &                           elpaScala,
      dataTypes::numberGPU *                       X,
      dataTypes::numberGPU *                       XFrac,
      distributedGPUVec<dataTypes::numberGPU> &    Xb,
      distributedGPUVec<dataTypes::numberFP32GPU> &floatXb,
      distributedGPUVec<dataTypes::numberGPU> &    HXb,
      distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector,
      const unsigned int                           M,
      const unsigned int                           N,
      const unsigned int                           Noc,
      const MPI_Comm &                             mpiCommParent,     
      const MPI_Comm &                             mpiCommDomain,
      GPUCCLWrapper &                              gpucclMpiCommDomain,
      const MPI_Comm &                             interBandGroupComm,
      std::vector<double> &                        eigenValues,
      cublasHandle_t &                             handle,
      const bool                                   useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      //
      // SConj=X^{T}*XConj
      //
      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecCGS_O && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection("SConj=X^{T}XConj, RR GEP step");
        }


      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatPar(N,
                                                              processGrid,
                                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  dataTypes::number(0.0));

      if (dftParameters::useMixedPrecCGS_O && useMixedPrecOverall)
        {
          if (dftParameters::overlapComputeCommunOrthoRR)
            linearAlgebraOperationsCUDA::
              fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(
                X,
                M,
                N,
                handle,
                mpiCommDomain,
                gpucclMpiCommDomain,
                interBandGroupComm,
                processGrid,
                overlapMatPar);
          else
            linearAlgebraOperationsCUDA::
              fillParallelOverlapMatMixedPrecScalapack(X,
                                                       M,
                                                       N,
                                                       handle,
                                                       mpiCommDomain,
                                                       gpucclMpiCommDomain,
                                                       interBandGroupComm,
                                                       processGrid,
                                                       overlapMatPar);
        }
      else
        {
          if (dftParameters::overlapComputeCommunOrthoRR)
            linearAlgebraOperationsCUDA::
              fillParallelOverlapMatScalapackAsyncComputeCommun(
                X,
                M,
                N,
                handle,
                mpiCommDomain,
                gpucclMpiCommDomain,
                interBandGroupComm,
                processGrid,
                overlapMatPar);
          else
            linearAlgebraOperationsCUDA::fillParallelOverlapMatScalapack(
              X,
              M,
              N,
              handle,
              mpiCommDomain,
              gpucclMpiCommDomain,
              interBandGroupComm,
              processGrid,
              overlapMatPar);
        }

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecCGS_O && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection("SConj=X^{T}XConj, RR GEP step");
        }

      // Sc=Lc*L^{T}
      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.enter_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParameters::useELPA)
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
              elpaCholesky(elpaScala.getElpaHandle(),
                           &overlapMatParConjTrans.local_el(0, 0),
                           &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));
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
      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.leave_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");



      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecXTHXSpectrumSplit &&
              useMixedPrecOverall)
            computing_timer.enter_subsection(
              "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection(
              "HConjProj=X^{T}*HConj*XConj, RR GEP step");
        }

      //
      // compute projected Hamiltonian HConjProj=X^{T}*HConj*XConj
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(N,
                                                           processGrid,
                                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  dataTypes::number(0.0));

      if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
        {
          operatorMatrix.XtHXMixedPrecOverlapComputeCommun(
            X,
            Xb,
            floatXb,
            HXb,
            projectorKetTimesVector,
            M,
            N,
            Noc,
            handle,
            processGrid,
            projHamPar,
            gpucclMpiCommDomain);
        }
      else
        {
          if (dftParameters::overlapComputeCommunOrthoRR)
            operatorMatrix.XtHXOverlapComputeCommun(X,
                                                    Xb,
                                                    HXb,
                                                    projectorKetTimesVector,
                                                    M,
                                                    N,
                                                    handle,
                                                    processGrid,
                                                    projHamPar,
                                                    gpucclMpiCommDomain);
          else
            operatorMatrix.XtHX(X,
                                Xb,
                                HXb,
                                projectorKetTimesVector,
                                M,
                                N,
                                handle,
                                processGrid,
                                projHamPar,
                                gpucclMpiCommDomain);
        }


      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecXTHXSpectrumSplit &&
              useMixedPrecOverall)
            computing_timer.leave_subsection(
              "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection(
              "HConjProj=X^{T}*HConj*XConj, RR GEP step");
        }

      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.enter_subsection(
          "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParConjTrans(
        N, processGrid, rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  dataTypes::number(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      if (dftParameters::useELPA)
        projHamPar.add(projHamParConjTrans,
                       dataTypes::number(-1.0),
                       dataTypes::number(-1.0));
      else
        projHamPar.add(projHamParConjTrans,
                       dataTypes::number(1.0),
                       dataTypes::number(1.0));


      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= dataTypes::number(0.5);
              }
          }

      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParCopy(N,
                                                               processGrid,
                                                               rowsBlockSize);

      // compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
      // conjugate transpose LAPACK notation)
      LMatPar.mmult(projHamParCopy, projHamPar);
      projHamParCopy.zmCmult(projHamPar, LMatPar);

      if (dftParameters::gpuFineGrainedTimings)
        computing_timer.leave_subsection(
          "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");
      //
      // compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      //
      const unsigned int Nfr = N - Noc;
      eigenValues.resize(Nfr);
      if (dftParameters::useELPA)
        {
          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          std::vector<double>                       allEigenValues(N, 0.0);
          dftfe::ScaLAPACKMatrix<dataTypes::number> eigenVectors(N,
                                                                 processGrid,
                                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      dataTypes::number(0.0));

          if (processGrid->is_process_active())
            {
              int error;
              elpaEigenvectors(elpaScala.getElpaHandlePartialEigenVec(),
                               &projHamPar.local_el(0, 0),
                               &allEigenValues[0],
                               &eigenVectors.local_el(0, 0),
                               &error);
              AssertThrow(
                error == ELPA_OK,
                dealii::ExcMessage(
                  "DFT-FE Error: elpa_eigenvectors error in case spectrum splitting."));
            }

          for (unsigned int i = 0; i < Nfr; ++i)
            eigenValues[Nfr - i - 1] = -allEigenValues[i];

          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiCommDomain);


          dftfe::ScaLAPACKMatrix<dataTypes::number> permutedIdentityMat(
            N, processGrid, rowsBlockSize);
          if (processGrid->is_process_active())
            std::fill(&permutedIdentityMat.local_el(0, 0),
                      &permutedIdentityMat.local_el(0, 0) +
                        permutedIdentityMat.local_m() *
                          permutedIdentityMat.local_n(),
                      dataTypes::number(0.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < permutedIdentityMat.local_m(); ++i)
              {
                const unsigned int glob_i = permutedIdentityMat.global_row(i);
                if (glob_i < Nfr)
                  {
                    for (unsigned int j = 0; j < permutedIdentityMat.local_n();
                         ++j)
                      {
                        const unsigned int glob_j =
                          permutedIdentityMat.global_column(j);
                        if (glob_j < Nfr)
                          {
                            const unsigned int rowIndexToSetOne =
                              (Nfr - 1) - glob_j;
                            if (glob_i == rowIndexToSetOne)
                              permutedIdentityMat.local_el(i, j) =
                                dataTypes::number(1.0);
                          }
                      }
                  }
              }

          eigenVectors.mmult(projHamPar, permutedIdentityMat);

          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(Noc, N - 1), true);
          if (dftParameters::gpuFineGrainedTimings)
            computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }

      linearAlgebraOperations::internal::broadcastAcrossInterCommScaLAPACKMat(
        processGrid, projHamPar, interBandGroupComm, 0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          computing_timer.enter_subsection(
            "Xfr^{T}={QfrConjPrime}^{C}*LConj^{-1}*X^{T}, RR GEP step");
        }

      //
      // rotate the basis in the subspace
      // Xfr^{T}={QfrConjPrime}^{C}*LConj^{-1}*X^{T}
      //
      projHamParCopy.copy_conjugate_transposed(projHamPar);
      projHamParCopy.mmult(projHamPar, LMatPar);

      subspaceRotationSpectrumSplitScalapack(X,
                                             XFrac,
                                             M,
                                             N,
                                             Nfr,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             gpucclMpiCommDomain,
                                             projHamPar,
                                             false);

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          computing_timer.leave_subsection(
            "Xfr^{T}={QfrConjPrime}^{C}*LConj^{-1}*X^{T}, RR GEP step");
        }

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecCGS_SR && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "X^{T}=Lconj^{-1}*X^{T} Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection(
              "X^{T}=Lconj^{-1}*X^{T}, RR GEP step");
        }

      //
      // X^{T}=LConj^{-1}*X^{T}
      //
      if (useMixedPrecOverall && dftParameters::useMixedPrecCGS_SR)
        subspaceRotationCGSMixedPrecScalapack(X,
                                              M,
                                              N,
                                              handle,
                                              processGrid,
                                              mpiCommDomain,
                                              gpucclMpiCommDomain,
                                              interBandGroupComm,
                                              LMatPar,
                                              false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  gpucclMpiCommDomain,
                                  interBandGroupComm,
                                  LMatPar,
                                  false,
                                  true);

      if (dftParameters::gpuFineGrainedTimings)
        {
          cudaDeviceSynchronize();
          if (dftParameters::useMixedPrecCGS_SR && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "X^{T}=Lconj^{-1}*X^{T} Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection(
              "X^{T}=Lconj^{-1}*X^{T}, RR GEP step");
        }
    }


  } // namespace linearAlgebraOperationsCUDA
} // namespace dftfe
