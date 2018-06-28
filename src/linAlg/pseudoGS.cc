// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
#if(defined DEAL_II_WITH_SCALAPACK && !USE_COMPLEX)
    template<typename T>
    unsigned int pseudoGramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
				            const unsigned int numberVectors,
					    const MPI_Comm &interBandGroupComm)
    {
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(dftParameters::orthoRROMPThreads);

      const unsigned int numLocalDofs = X.local_size()/numberVectors;

      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ||
					  dftParameters::verbosity<4 ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);


      const unsigned rowsBlockSize=std::min((unsigned int)50,numberVectors);
      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
      internal::createProcessGridSquareMatrix(X.get_mpi_communicator(),
		                           numberVectors,
					   processGrid);

      dealii::ScaLAPACKMatrix<T> overlapMatPar(numberVectors,
                                               processGrid,
                                               rowsBlockSize);

      //S=X*X^{T}. Implemented as S=X^{T}*X with X^{T} stored in the column major format
      computing_timer.enter_section("Fill overlap matrix for PGS");
      internal::fillParallelOverlapMatrix(X,
	                                  numberVectors,
		                          processGrid,
					  interBandGroupComm,
				          overlapMatPar);
      computing_timer.exit_section("Fill overlap matrix for PGS");

      //S=L*L^{T}
      computing_timer.enter_section("PGS cholesky, copy, and triangular matrix invert");
      overlapMatPar.compute_cholesky_factorization();

      dealii::LAPACKSupport::Property overlapMatPropertyPostCholesky=overlapMatPar.get_property();

      AssertThrow(overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular
		  ||overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular
	           ,dealii::ExcMessage("DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));

      dealii::ScaLAPACKMatrix<T> LMatPar(numberVectors,
                                         processGrid,
                                         rowsBlockSize,
					 overlapMatPropertyPostCholesky);

      //copy triangular part of projHamPar into LMatPar
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
           {
             const unsigned int glob_i = overlapMatPar.global_column(i);
             for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
               {
		 const unsigned int glob_j = overlapMatPar.global_row(j);
		 if (overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular)
		 {
		     if (glob_i <= glob_j)
			LMatPar.local_el(j, i)=overlapMatPar.local_el(j, i);
		     else
			LMatPar.local_el(j, i)=0;
		 }
		 else
		 {
		     if (glob_j <= glob_i)
			LMatPar.local_el(j, i)=overlapMatPar.local_el(j, i);
		     else
			LMatPar.local_el(j, i)=0;
		 }
               }
           }

      //Check if any of the diagonal entries of LMat are close to zero. If yes break off PGS and return flag=1

      unsigned int flag=0;
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
           {
             const unsigned int glob_i = overlapMatPar.global_column(i);
             for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
               {
		 const unsigned int glob_j = overlapMatPar.global_row(j);
		 if (glob_i==glob_j)
		    if (std::fabs(LMatPar.local_el(j, i))<1e-14)
			flag=1;
		 if (flag==1)
		     break;
               }
	     if (flag==1)
		 break;
           }

      flag=dealii::Utilities::MPI::max(flag,X.get_mpi_communicator());
      if (dftParameters::enableSwitchToGS && flag==1)
          return flag;

      //invert triangular matrix
      LMatPar.invert();
      computing_timer.exit_section("PGS cholesky, copy, and triangular matrix invert");

      //X=X*L^{-1}^{T} implemented as X^{T}=L^{-1}*X^{T} with X^{T} stored in the column major format
      computing_timer.enter_section("Subspace rotation PGS");
      internal::subspaceRotation(X,
		                 numberVectors,
		                 processGrid,
				 interBandGroupComm,
			         LMatPar,
				 overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false);

      computing_timer.exit_section("Subspace rotation PGS");
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(1);

      return 0;
    }
#else
    template<typename T>
    unsigned int pseudoGramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
				            const unsigned int numberVectors,
					    const MPI_Comm &interBandGroupComm)
    {
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
      return 0;
    }
#endif

  }
}
