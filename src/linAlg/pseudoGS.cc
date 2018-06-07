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
      const unsigned int localVectorSize = X.local_size()/numberVectors;

      std::vector<T> overlapMatrix(numberVectors*numberVectors,0.0);


      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);

      //
      //blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;
      const char uplo = 'L';
      const char trans = 'N';

      //
      //compute overlap matrix S = {(Z)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Z)^T is transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the overlap matrix as S = S^{T} = X*{X^T} here
      //

      computing_timer.enter_section("serial overlap matrix for PGS");
      dsyrk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);
      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix);
      computing_timer.exit_section("serial overlap matrix for PGS");


      const unsigned rowsBlockSize=50;
      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
      utils::createProcessGridSquareMatrix(X.get_mpi_communicator(),
		                           numberVectors,
		                           processGrid,
				           rowsBlockSize);

      dealii::ScaLAPACKMatrix<T> overlapMatPar(numberVectors,
                                               processGrid,
                                               rowsBlockSize);
      computing_timer.enter_section("scalapack copy old");
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
           {
             const unsigned int glob_i = overlapMatPar.global_column(i);
             for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
               {
                 const unsigned int glob_j = overlapMatPar.global_row(j);

		 overlapMatPar.local_el(j, i)=overlapMatrix[glob_i*numberVectors+glob_j];
               }
           }
      computing_timer.exit_section("scalapack copy old");


      computing_timer.enter_section("scalapack copy new");

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      if (processGrid->is_process_active())
      {
         for (unsigned int i = 0; i < overlapMatPar.local_m(); ++i)
             globalToLocalRowIdMap[overlapMatPar.global_row(i)]=i;

	 for (unsigned int j = 0; j < overlapMatPar.local_n(); ++j)
	     globalToLocalColumnIdMap[overlapMatPar.global_column(j)]=j;

       }

       for (unsigned int i = 0; i < numberVectors/2; ++i)
       {
	     std::vector<T> tempVec1(2*i+1);
	     std::vector<T> tempVec2(2*i+2);
	     T * beginPtrX=X.begin();

	     for (unsigned int dof = 0; dof < localVectorSize; ++dof)
	     {
	       const double temp1=*(beginPtrX+dof*numberVectors+2*i);
	       const double temp2=*(beginPtrX+dof*numberVectors+2*i+1);
               for (unsigned int j = 0; j <=2*i; ++j)
	       {
                  tempVec1[j]+=temp1* *(beginPtrX+dof*numberVectors+j);
		  tempVec2[j]+=temp2* *(beginPtrX+dof*numberVectors+j);
	       }
	       tempVec2[2*i+1]+=temp2* *(beginPtrX+dof*numberVectors+2*i+1);
	     }

              dealii::Utilities::MPI::sum(tempVec1, X.get_mpi_communicator(),tempVec1);
	      dealii::Utilities::MPI::sum(tempVec2, X.get_mpi_communicator(),tempVec2);

              for (unsigned int j = 0; j <=2*i; ++j)
		 if (globalToLocalRowIdMap.find(2*i)!=globalToLocalRowIdMap.end())
		     if(globalToLocalColumnIdMap.find(j)!=globalToLocalColumnIdMap.end())
			 overlapMatPar.local_el(globalToLocalRowIdMap[2*i], globalToLocalColumnIdMap[j])=tempVec1[j];

              for (unsigned int j = 0; j <=2*i+1; ++j)
		 if (globalToLocalRowIdMap.find(2*i+1)!=globalToLocalRowIdMap.end())
		     if(globalToLocalColumnIdMap.find(j)!=globalToLocalColumnIdMap.end())
			 overlapMatPar.local_el(globalToLocalRowIdMap[2*i+1], globalToLocalColumnIdMap[j])=tempVec2[j];

       }
       computing_timer.exit_section("scalapack copy new");


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

      computing_timer.enter_section("scalapack copy");
      std::fill(overlapMatrix.begin(),overlapMatrix.end(),T(0));
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
           {
             const unsigned int glob_i = LMatPar.global_column(i);
             for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
               {
                 const unsigned int glob_j = LMatPar.global_row(j);
                 overlapMatrix[glob_i*numberVectors+glob_j]=LMatPar.local_el(j, i);
               }
           }
      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix);
      computing_timer.exit_section("scalapack copy");

      computing_timer.enter_section("subspace rotation in PGS");
      //
      //Rotate the given vectors using L^{-1}^{T} i.e Y = X*L^{-1}^{T} but implemented as Yt = L^{-1}*Xt
      //using the column major format of blas
      //
      const char transA2  = 'N', transB2  = 'N';
      dealii::parallel::distributed::Vector<T> orthoNormalizedBasis;
      orthoNormalizedBasis.reinit(X);
      dgemm_(&transA2,
	     &transB2,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &overlapMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     orthoNormalizedBasis.begin(),
	     &numberEigenValues);
      X = orthoNormalizedBasis;
      computing_timer.exit_section("subspace rotation in PGS");

#else
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#endif
    }

  }
}
