// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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


#include <dftParameters.h>
#include <linearAlgebraOperationsCUDA.h>
#include <linearAlgebraOperationsInternal.h>

namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
    void
    pseudoGramSchmidtOrthogonalization(elpaScalaManager &    elpaScala,
                                       dataTypes::numberGPU *X,
                                       const unsigned int    M,
                                       const unsigned int    N,
                                       const MPI_Comm &      mpiCommDomain,
                                       GPUCCLWrapper & gpucclMpiCommDomain,
                                       const MPI_Comm &interBandGroupComm,
                                       cublasHandle_t &handle,
                                       const bool      useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

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

      // SConj=LConj*L^{T}
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

      // X^{T}=LConj^{-1}*X^{T} with X^{T} stored in
      // the column major format
      if (dftParameters::useMixedPrecCGS_SR && useMixedPrecOverall)
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
    }
  } // namespace linearAlgebraOperationsCUDA
} // namespace dftfe
