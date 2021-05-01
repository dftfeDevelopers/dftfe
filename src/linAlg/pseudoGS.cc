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
//
// @author Sambit Das
//


/** @file pseudoGS.cc
 *  @brief Contains linear algebra operations for Pseudo-Gram-Schimdt orthogonalization
 *
 */
namespace dftfe
{
  namespace linearAlgebraOperations
  {
    template <typename T>
    unsigned int
    pseudoGramSchmidtOrthogonalization(elpaScalaManager & elpaScala,
                                       std::vector<T> &   X,
                                       const unsigned int numberVectors,
                                       const MPI_Comm &   interBandGroupComm,
                                       const MPI_Comm &   mpiComm,
                                       const bool         useMixedPrec)

    {
      const unsigned int numLocalDofs = X.size() / numberVectors;

      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      dealii::TimerOutput computing_timer(mpiComm,
                                          pcout,
                                          dftParameters::reproducible_output ||
                                              dftParameters::verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);


      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid;
      internal::createProcessGridSquareMatrix(mpiComm,
                                              numberVectors,
                                              processGrid);

      dftfe::ScaLAPACKMatrix<T> overlapMatPar(numberVectors,
                                              processGrid,
                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  T(0.0));

      // SConj=X^{T}*XConj with X^{T} stored in the column
      // major format
      if (!(dftParameters::useMixedPrecPGS_O && useMixedPrec))
        {
          computing_timer.enter_section("Fill overlap matrix PGS");
          internal::fillParallelOverlapMatrix(&X[0],
                                              X.size(),
                                              numberVectors,
                                              processGrid,
                                              interBandGroupComm,
                                              mpiComm,
                                              overlapMatPar);
          computing_timer.exit_section("Fill overlap matrix PGS");
        }
      else
        {
          computing_timer.enter_section("Fill overlap matrix mixed prec PGS");
          if (std::is_same<T, std::complex<double>>::value)
            internal::fillParallelOverlapMatrixMixedPrec<T,
                                                         std::complex<float>>(
              &X[0],
              X.size(),
              numberVectors,
              processGrid,
              interBandGroupComm,
              mpiComm,
              overlapMatPar);
          else
            internal::fillParallelOverlapMatrixMixedPrec<T, float>(
              &X[0],
              X.size(),
              numberVectors,
              processGrid,
              interBandGroupComm,
              mpiComm,
              overlapMatPar);
          computing_timer.exit_section("Fill overlap matrix mixed prec PGS");
        }


        // SConj=LConj*L^{T}
#if (defined DFTFE_WITH_ELPA)
      computing_timer.enter_section(
        "ELPA PGS cholesky, copy, and triangular matrix invert");
#else
      computing_timer.enter_section(
        "PGS cholesky, copy, and triangular matrix invert");
#endif
#if (defined DFTFE_WITH_ELPA)
      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParameters::useELPA)
        {
          // For ELPA cholesky only the upper triangular part of the hermitian
          // matrix is required
          dftfe::ScaLAPACKMatrix<T> overlapMatParConjTrans(numberVectors,
                                                           processGrid,
                                                           rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      T(0.0));

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
#else
      overlapMatPar.compute_cholesky_factorization();

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky =
        overlapMatPar.get_property();
#endif
      AssertThrow(
        overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


      // extract LConj
      dftfe::ScaLAPACKMatrix<T> LMatPar(
        numberVectors,
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
                  LMatPar.local_el(j, i) = T(0);
                else
                  LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
              }
          }

      // Check if any of the diagonal entries of LMat are close to zero. If yes
      // break off PGS and return flag=1
      unsigned int flag = 0;
      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = LMatPar.global_column(i);
            for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = LMatPar.global_row(j);
                if (glob_i == glob_j)
                  if (std::abs(LMatPar.local_el(j, i)) < 1e-14)
                    flag = 1;
                if (flag == 1)
                  break;
              }
            if (flag == 1)
              break;
          }

      flag = dealii::Utilities::MPI::max(flag, mpiComm);
      if (dftParameters::enableSwitchToGS && flag == 1)
        return flag;

      // compute LConj^{-1}
      LMatPar.invert();

#if (defined DFTFE_WITH_ELPA)
      computing_timer.exit_section(
        "ELPA PGS cholesky, copy, and triangular matrix invert");
#else
      computing_timer.exit_section(
        "PGS cholesky, copy, and triangular matrix invert");
#endif

      // X^{T}=LConj^{-1}*X^{T} with X^{T} stored in
      // the column major format
      if (!(dftParameters::useMixedPrecPGS_SR && useMixedPrec))
        {
          computing_timer.enter_section("Subspace rotation PGS");
          internal::subspaceRotation(&X[0],
                                     X.size(),
                                     numberVectors,
                                     processGrid,
                                     interBandGroupComm,
                                     mpiComm,
                                     LMatPar,
                                     false,
                                     dftParameters::triMatPGSOpt ? true :
                                                                   false);
          computing_timer.exit_section("Subspace rotation PGS");
        }
      else
        {
          computing_timer.enter_section("Subspace rotation mixed prec PGS");
          if (std::is_same<T, std::complex<double>>::value)
            internal::subspaceRotationPGSMixedPrec<T, std::complex<float>>(
              &X[0],
              X.size(),
              numberVectors,
              processGrid,
              interBandGroupComm,
              mpiComm,
              LMatPar,
              false);
          else
            internal::subspaceRotationPGSMixedPrec<T, float>(&X[0],
                                                             X.size(),
                                                             numberVectors,
                                                             processGrid,
                                                             interBandGroupComm,
                                                             mpiComm,
                                                             LMatPar,
                                                             false);
          computing_timer.exit_section("Subspace rotation mixed prec PGS");
        }


      return 0;
    }
  } // namespace linearAlgebraOperations
} // namespace dftfe
