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
    template<typename T>
    void pseudoGramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
				            const unsigned int numberVectors)
    {
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(dftParameters::orthoRROMPThreads);
#ifdef WITH_SCALAPACK
      pseudoGramSchmidtOrthogonalizationParallel(X,numberVectors);
#else
      pseudoGramSchmidtOrthogonalizationSerial(X,numberVectors);
#endif
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(1);
    }

    template<typename T>
    void pseudoGramSchmidtOrthogonalizationSerial(dealii::parallel::distributed::Vector<T> & X,
				                  const unsigned int numberVectors)
    {
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
    }


    template<typename T>
    void pseudoGramSchmidtOrthogonalizationParallel(dealii::parallel::distributed::Vector<T> & X,
				                    const unsigned int numberVectors)
    {
#if(defined WITH_SCALAPACK && !USE_COMPLEX)
      const unsigned int numLocalDofs = X.local_size()/numberVectors;

      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);


      const unsigned rowsBlockSize=std::min((unsigned int)50,numberVectors);
      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
      internal::createProcessGridSquareMatrix(X.get_mpi_communicator(),
		                           numberVectors,
		                           processGrid,
				           rowsBlockSize);

      dealii::ScaLAPACKMatrix<T> overlapMatPar(numberVectors,
                                               processGrid,
                                               rowsBlockSize);


      computing_timer.enter_section("scalapack fill overlap matrix for PGS");
      internal::fillParallelOverlapMatrix(X,
	                                  numberVectors,
		                          processGrid,
				          overlapMatPar);
      computing_timer.exit_section("scalapack fill overlap matrix for PGS");


      computing_timer.enter_section("PGS cholesky, copy, and triangular matrix invert");
      overlapMatPar.compute_cholesky_factorization();

      dealii::ScaLAPACKMatrix<T> LMatPar(numberVectors,
                                         processGrid,
                                         rowsBlockSize,
					 dealii::LAPACKSupport::Property::lower_triangular);
      //copy lower triangular part of projHamPar into LMatPar
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
           {
             const unsigned int glob_i = overlapMatPar.global_column(i);
             for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
               {
                 const unsigned int glob_j = overlapMatPar.global_row(j);
		 if (glob_i <= glob_j)
                    LMatPar.local_el(j, i)=overlapMatPar.local_el(j, i);
		 else
		    LMatPar.local_el(j, i)=0;
               }
           }
      LMatPar.invert();
      computing_timer.exit_section("PGS cholesky, copy, and triangular matrix invert");


      computing_timer.enter_section("subspace rotation PGS");

      internal::subspaceRotation(X,
		                 numberVectors,
		                 processGrid,
			         LMatPar);

      computing_timer.exit_section("subspace rotation PGS");
#else
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#endif
    }

  }
}
