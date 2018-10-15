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
    unsigned int pseudoGramSchmidtOrthogonalization(std::vector<T> & X,
						    const unsigned int numberVectors,
						    const MPI_Comm &interBandGroupComm,
						    const MPI_Comm & mpiComm,
						    const bool useMixedPrec)

    {
      const unsigned int numLocalDofs = X.size()/numberVectors;

      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      dealii::TimerOutput computing_timer(mpiComm,
	                                  pcout,
					  dftParameters::reproducible_output ||
					  dftParameters::verbosity<4 ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);


      const unsigned rowsBlockSize=std::min((unsigned int)50,numberVectors);
      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
      internal::createProcessGridSquareMatrix(mpiComm,
					      numberVectors,
					      processGrid);

      dealii::ScaLAPACKMatrix<T> overlapMatPar(numberVectors,
                                               processGrid,
                                               rowsBlockSize);

      //S=X*X^{T}. Implemented as S=X^{T}*X with X^{T} stored in the column major format
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
	  internal::fillParallelOverlapMatrixMixedPrec(&X[0],
					               X.size(),
						       numberVectors,
						       processGrid,
						       interBandGroupComm,
						       mpiComm,
						       overlapMatPar);
	  computing_timer.exit_section("Fill overlap matrix mixed prec PGS");
      }


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
		    if (std::abs(LMatPar.local_el(j, i))<1e-14)
			flag=1;
		 if (flag==1)
		     break;
               }
	     if (flag==1)
		 break;
           }

      flag=dealii::Utilities::MPI::max(flag,mpiComm);
      if (dftParameters::enableSwitchToGS && flag==1)
          return flag;

      //invert triangular matrix
      LMatPar.invert();
      computing_timer.exit_section("PGS cholesky, copy, and triangular matrix invert");

      //X=X*L^{-1}^{T} implemented as X^{T}=L^{-1}*X^{T} with X^{T} stored in the column major format
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
				     overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false,
				     dftParameters::triMatPGSOpt?true:false);
	  computing_timer.exit_section("Subspace rotation PGS");
      }
      else
      {
	  computing_timer.enter_section("Subspace rotation mixed prec PGS");
	  internal::subspaceRotationPGSMixedPrec(&X[0],
				     X.size(),
				     numberVectors,
				     processGrid,
				     interBandGroupComm,
				     mpiComm,
				     LMatPar,
				     overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false);
	  computing_timer.exit_section("Subspace rotation mixed prec PGS");
      }


      return 0;
    }

    void pseudoGramSchmidtOrthogonalization(dealii::ScaLAPACKMatrix<dataTypes::number> & X,
					    std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid> & processGrid)
    {

      //
      //create overlap matrix allocation
      //
      const unsigned int blockSize=std::min((unsigned int)50,X.n());
      dealii::ScaLAPACKMatrix<dataTypes::number> overlapMatPar(X.n(),
							       processGrid,
							       blockSize);


      //
      //compute overlap matrix S = X^{T}*X
      //
      X.mult(1.0,X,0.0,overlapMatPar,true);

      //
      //compute Cholesky factorization S = L*L^{T}
      //
      overlapMatPar.compute_cholesky_factorization();

      dealii::LAPACKSupport::Property overlapMatPropertyPostCholesky=overlapMatPar.get_property();

      AssertThrow(overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular
		  ||overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular
	           ,dealii::ExcMessage("DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));

      
      dealii::ScaLAPACKMatrix<dataTypes::number> LMatPar(X.n(),
							 processGrid,
							 blockSize,
							 overlapMatPropertyPostCholesky);

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

      //
      //invert triangular matrix
      //
      LMatPar.invert();

      //
      //do a subspace rotation to get orthogonalized matrix Y = X*L^{-1}^{T}
      //


      //
      //allocate storage for Y
      //
      const unsigned int rowsBlockSize=std::min((unsigned int)50,X.m());
      const unsigned int columnsBlockSize=std::min((unsigned int)50,X.n());
      dealii::ScaLAPACKMatrix<dataTypes::number> Y(X.m(),
						   X.n(),
						   processGrid,
						   rowsBlockSize,
						   columnsBlockSize);
      //
      //compute Y = X*L^{-1}^{T}
      //
      X.mult(1.0,LMatPar,0.0,Y,false,true);

      //
      //copy X = Y
      //
      Y.copy_to(X);

      
      
    }

#else
    template<typename T>
    unsigned int pseudoGramSchmidtOrthogonalization(std::vector<T> & X,
						    const unsigned int numberVectors,
						    const MPI_Comm &interBandGroupComm,
						    const MPI_Comm & mpiComm,
						    const bool useMixedPrec)
    {
       const unsigned int localVectorSize = X.size()/numberVectors;

       std::vector<T> overlapMatrix(numberVectors*numberVectors,0.0);


       dealii::ConditionalOStream   pcout(std::cout,
	                                 (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

       dealii::TimerOutput computing_timer(mpiComm,
	                                   pcout,
					  dftParameters::reproducible_output ||
					  dftParameters::verbosity<4? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);

       //
       //compute overlap matrix S = {(Zc)^T}*Z on local proc
       //where Z is a matrix with size number of degrees of freedom times number of column vectors
       //and (Zc)^T is conjugate transpose of Z

       computing_timer.enter_section("local overlap matrix for pgs");

       //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
       //corresponding to column-major format required for blas, we compute
       //the transpose of overlap matrix i.e Sc = X^{T}*(Xc) here
       const char uplo1 = 'L';
       const char trans1 = 'N';
       const double alpha1 = 1.0, beta1 = 0.0;
#ifdef USE_COMPLEX
       zherk_(&uplo1,
	     &trans1,
	     &numberVectors,
	     &localVectorSize,
	     &alpha1,
	     &X[0],
	     &numberVectors,
	     &beta1,
	     &overlapMatrix[0],
	     &numberVectors);
#else
       dsyrk_(&uplo1,
	     &trans1,
	     &numberVectors,
	     &localVectorSize,
	     &alpha1,
	     &X[0],
	     &numberVectors,
	     &beta1,
	     &overlapMatrix[0],
	     &numberVectors);
#endif
       dealii::Utilities::MPI::sum(overlapMatrix, mpiComm, overlapMatrix);
       computing_timer.exit_section("local overlap matrix for pgs");

       computing_timer.enter_section("PGS cholesky and triangular matrix invert");

       //Sc=Lc*L^{T}, Lc is L conjugate
       int info2;
       const char uplo2 = 'L';
#ifdef USE_COMPLEX
       zpotrf_(&uplo2,
	       &numberVectors,
	       &overlapMatrix[0],
	       &numberVectors,
	       &info2);

#else
       dpotrf_(&uplo2,
	       &numberVectors,
	       &overlapMatrix[0],
	       &numberVectors,
	       &info2);
#endif
       unsigned int flag=0;
       if (info2!=0)
	  flag=1;
       else
       {
	  for (unsigned int i = 0; i < numberVectors; ++i)
	    if (std::abs(overlapMatrix[i*numberVectors+i])<1e-14)
	    {
		flag=1;
		break;
	    }
       }

       if (dftParameters::enableSwitchToGS && flag==1)
           return flag;

       AssertThrow(info2==0,dealii::ExcMessage("Error in dpotrf/zpotrf"));

       //Compute Lc^{-1}
       int info3;
       const char uplo3 = 'L';
       const char diag3 = 'N';
#ifdef USE_COMPLEX
       ztrtri_(&uplo3,
	       &diag3,
	       &numberVectors,
	       &overlapMatrix[0],
	       &numberVectors,
	       &info3);

#else
       dtrtri_(&uplo3,
	       &diag3,
	       &numberVectors,
	       &overlapMatrix[0],
	       &numberVectors,
	       &info3);
#endif
       AssertThrow(info3==0,dealii::ExcMessage("Error in dtrtri/ztrtri"));
       computing_timer.exit_section("PGS cholesky and triangular matrix invert");

       //X=X*Lc^{-1}^{T} implemented as X^{T}=Lc^{-1}*X^{T} with X^{T} stored in the column major format

       computing_timer.enter_section("subspace rotation in pgs");
       //dealii::parallel::distributed::Vector<T> orthoNormalizedBasis;
       //orthoNormalizedBasis.reinit(X);
       std::vector<T> orthoNormalizedBasis(X.size(),0.0);
       const char transA4  = 'N', transB4  = 'N';
       const T alpha4 = 1.0, beta4 = 0.0;
#ifdef USE_COMPLEX
       zgemm_(&transA4,
	     &transB4,
	     &numberVectors,
             &localVectorSize,
	     &numberVectors,
	     &alpha4,
	     &overlapMatrix[0],
	     &numberVectors,
	     &X[0],
	     &numberVectors,
	     &beta4,
	     &orthoNormalizedBasis[0],
	     &numberVectors);
#else
       dgemm_(&transA4,
	      &transB4,
	      &numberVectors,
	      &localVectorSize,
	      &numberVectors,
	      &alpha4,
	      &overlapMatrix[0],
	      &numberVectors,
	      &X[0],
	      &numberVectors,
	      &beta4,
	      &orthoNormalizedBasis[0],
	      &numberVectors);
#endif
       computing_timer.exit_section("subspace rotation in pgs");


       X = orthoNormalizedBasis;

       return 0;
    }
#endif

  }
}
