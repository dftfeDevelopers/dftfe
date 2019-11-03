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
// @author Sambit Das



#include<linearAlgebraOperationsInternalCUDA.h>
#include<linearAlgebraOperationsCUDA.h>
#include<dftParameters.h>
#include<dftUtils.h>

namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
 

    void rayleighRitzSpectrumSplitDirect(operatorDFTCUDAClass & operatorMatrix,
		      const double* X,
                      double* XFrac,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
                      const unsigned int N,
                      const unsigned int Noc,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
		      double* eigenValues,
                      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const bool useMixedPrecOverall)
    {

          int this_process;
          MPI_Comm_rank(mpiComm, &this_process);

          const unsigned int Nfr = N-Noc;

          double gpu_time;

          if (!isElpaStep2)
          {
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime();
                      }

		      if (processGrid->is_process_active())
			  std::fill(&projHamPar.local_el(0,0),
				    &projHamPar.local_el(0,0)+projHamPar.local_m()*projHamPar.local_n(),
				    0.0);

                      if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
			      operatorMatrix.XtHXMixedPrec(X,
						 Xb,
						 HXb,
						 projectorKetTimesVector,
						 M,
						 N,
                                                 Noc,
						 handle,
						 processGrid,
						 projHamPar);
                      else
                      {
                              if (dftParameters::overlapComputeCommunXtHX)
				      operatorMatrix.XtHXOverlapComputeCommun(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                              else
				      operatorMatrix.XtHX(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                      }

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
                        {
                          if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
                            std::cout<<"Time for Blocked XtHX Mixed Prec, RR step: "<<gpu_time<<std::endl;
                          else
  			     std::cout<<"Time for Blocked XtHX, RR step: "<<gpu_time<<std::endl;
                        }
		      }
          }
               
          if (isElpaStep1)
               return;

          if (!isElpaStep2)
          {
		      //
		      //compute eigendecomposition of ProjHam
		      //
		      std::vector<double> eigenValuesStdVec(Nfr,0.0);
		      
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                         cudaDeviceSynchronize();
                         MPI_Barrier(MPI_COMM_WORLD);
			 gpu_time = MPI_Wtime();
                      }

		      eigenValuesStdVec=projHamPar.eigenpairs_symmetric_by_index_MRRR(std::make_pair(Noc,N-1),true);
		      std::copy(eigenValuesStdVec.begin(),eigenValuesStdVec.end(),eigenValues);

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
			  std::cout<<"Time for ScaLAPACK eigen decomp, RR step: "<<gpu_time<<std::endl;
		      }
          }
             
	  //
	  //rotate the basis in the subspace Xfr = Xfr*Q, implemented as Xfr^{T}=Q^{T}*Xfr^{T} with Xfr^{T}
	  //stored in the column major format
	  //
          if (dftParameters::gpuFineGrainedTimings)
          {
              cudaDeviceSynchronize();
              MPI_Barrier(MPI_COMM_WORLD);
	      gpu_time = MPI_Wtime();
          }

          subspaceRotationSpectrumSplitScalapack(X,
                    XFrac,
		    M,
		    N,
                    Nfr,
		    handle,
		    processGrid,
		    mpiComm,
		    projHamPar,
		    true);

          if (dftParameters::gpuFineGrainedTimings)
          {
	      cudaDeviceSynchronize();
              MPI_Barrier(MPI_COMM_WORLD);
	      gpu_time = MPI_Wtime() - gpu_time;

	      if (this_process==0)
	        std::cout<<"Time for Blocked subspace rotation, RR step: "<<gpu_time<<std::endl;
          }
    }


    void rayleighRitz(operatorDFTCUDAClass & operatorMatrix,
		      double* X,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
                      const unsigned int N,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
                      const MPI_Comm &interBandGroupComm,
		      double* eigenValues,
                      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const bool useMixedPrecOverall)
    {

              int this_process;
              MPI_Comm_rank(mpiComm, &this_process);

              double gpu_time;

              if (!isElpaStep2)
              {
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime();
                      }

		      if (processGrid->is_process_active())
			  std::fill(&projHamPar.local_el(0,0),
				    &projHamPar.local_el(0,0)+projHamPar.local_m()*projHamPar.local_n(),
				    0.0);

                      if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
			      operatorMatrix.XtHXMixedPrec(X,
						 Xb,
						 HXb,
						 projectorKetTimesVector,
						 M,
						 N,
                                                 N-dftParameters::mixedPrecXtHXFracStates,
						 handle,
						 processGrid,
						 projHamPar);
                      else
                      {
                              if (dftParameters::overlapComputeCommunXtHX)
				      operatorMatrix.XtHXOverlapComputeCommun(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                              else
				      operatorMatrix.XtHX(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                      }

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
                        {
                          if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
                            std::cout<<"Time for Blocked XtHX Mixed Prec, RR step: "<<gpu_time<<std::endl;
                          else
  			     std::cout<<"Time for Blocked XtHX, RR step: "<<gpu_time<<std::endl;
                        }
		      }
              }
               
              if (isElpaStep1)
                 return;

              if (!isElpaStep2)
              {
		      //
		      //compute eigendecomposition of ProjHam
		      //
		      const unsigned int numberEigenValues = N;
		      std::vector<double> eigenValuesStdVec(numberEigenValues,0.0);
		      
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                         cudaDeviceSynchronize();
                         MPI_Barrier(MPI_COMM_WORLD);
			 gpu_time = MPI_Wtime();
                      }

		      eigenValuesStdVec=projHamPar.eigenpairs_symmetric_by_index_MRRR(std::make_pair(0,numberEigenValues-1),true);
		      std::copy(eigenValuesStdVec.begin(),eigenValuesStdVec.end(),eigenValues);

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
			  std::cout<<"Time for ScaLAPACK eigen decomp, RR step: "<<gpu_time<<std::endl;
		      }
              }
             
	      //
	      //rotate the basis in the subspace X = X*Q, implemented as X^{T}=Q^{T}*X^{T} with X^{T}
	      //stored in the column major format
	      //
              if (dftParameters::gpuFineGrainedTimings)
              {
                 cudaDeviceSynchronize();
                 MPI_Barrier(MPI_COMM_WORLD);
	         gpu_time = MPI_Wtime();
              }

              if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
                 subspaceRotationRRMixedPrecScalapack(X,
                            M,
                            N,
                            handle,
                            processGrid,
                            mpiComm,
                            interBandGroupComm,
                            projHamPar,
                            true);
              else
                 subspaceRotationScalapack(X,
                            M,
                            N,
                            handle,
                            processGrid,
                            mpiComm,
                            interBandGroupComm,
                            projHamPar,
                            true);

              if (dftParameters::gpuFineGrainedTimings)
              {
                 cudaDeviceSynchronize();
                 MPI_Barrier(MPI_COMM_WORLD);                 
                 gpu_time = MPI_Wtime() - gpu_time;

                 if (this_process==0)
                  if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
                     std::cout<<"Time for Blocked subspace rotation Mixed Prec, RR step: "<<gpu_time<<std::endl;
                  else
                     std::cout<<"Time for Blocked subspace rotation, RR step: "<<gpu_time<<std::endl;
              }

    }

    void rayleighRitzGEP(operatorDFTCUDAClass & operatorMatrix,
		      double* X,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
                      const unsigned int N,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
                      const MPI_Comm &interBandGroupComm,
		      double* eigenValues,
                      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      dealii::ScaLAPACKMatrix<double> & overlapMatPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const bool useMixedPrecOverall)
    {

              int this_process;
              MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

              double gpu_time;

	      const unsigned int rowsBlockSize=operatorMatrix.getScalapackBlockSize();

              if (!isElpaStep2)
              {
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                         cudaDeviceSynchronize();
                         MPI_Barrier(MPI_COMM_WORLD);
		         gpu_time = MPI_Wtime();
                      }

		      //S=X*X^{T}. Implemented as S=X^{T}*X with X^{T} stored in the column major format
		      if (dftParameters::useMixedPrecPGS_O && useMixedPrecOverall)
				     linearAlgebraOperationsCUDA::
					     fillParallelOverlapMatMixedPrecScalapack
							      (X,
							       M,
							       N,
							       handle,
							       mpiComm,
                                                               interBandGroupComm,
							       processGrid,
							       overlapMatPar);

		      else
				     linearAlgebraOperationsCUDA::
					     fillParallelOverlapMatScalapack
							      (X,
							       M,
							       N,
							       handle,
							       mpiComm,
                                                               interBandGroupComm,
							       processGrid,
							       overlapMatPar); 
			    
		      if (dftParameters::gpuFineGrainedTimings)
		      { 
                            cudaDeviceSynchronize();
                            MPI_Barrier(MPI_COMM_WORLD);
			    gpu_time = MPI_Wtime() - gpu_time;
			    if (this_process==0)
			    {
			      if (dftParameters::useMixedPrecPGS_O && useMixedPrecOverall)
				  std::cout<<"Time for X^{T}X Mixed Prec, RR GEP step: "<<gpu_time<<std::endl;
			      else
				  std::cout<<"Time for X^{T}X, RR GEP step: "<<gpu_time<<std::endl;
			    }
		      }

		      if (dftParameters::gpuFineGrainedTimings)
                      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime();
                      }

		      if (processGrid->is_process_active())
			  std::fill(&projHamPar.local_el(0,0),
				    &projHamPar.local_el(0,0)+projHamPar.local_m()*projHamPar.local_n(),
				    0.0);

                      if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
			      operatorMatrix.XtHXMixedPrec(X,
						 Xb,
						 HXb,
						 projectorKetTimesVector,
						 M,
						 N,
                                                 N-dftParameters::mixedPrecXtHXFracStates,
						 handle,
						 processGrid,
						 projHamPar);
                      else
                      {
                              if (dftParameters::overlapComputeCommunXtHX)
				      operatorMatrix.XtHXOverlapComputeCommun(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                              else
				      operatorMatrix.XtHX(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                      }

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
                        {
                          if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
                            std::cout<<"Time for X^{T}HX Mixed Prec, RR GEP step: "<<gpu_time<<std::endl;
                          else
  			     std::cout<<"Time for X^{T}HX, RR GEP step: "<<gpu_time<<std::endl;
                        }
		      }
              }
               
              if (isElpaStep1)
                 return;

              if (!isElpaStep2)
              {

                      
		      //S=L*L^{T}
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                         cudaDeviceSynchronize();
                         MPI_Barrier(MPI_COMM_WORLD);
			 gpu_time = MPI_Wtime();
                      }

                      overlapMatPar.compute_cholesky_factorization();
                      dealii::LAPACKSupport::Property overlapMatPropertyPostCholesky=overlapMatPar.get_property();

		      AssertThrow(overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular
				  ||overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular
				   ,dealii::ExcMessage("DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));
		      dealii::ScaLAPACKMatrix<double> LMatPar(N,
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

                      // invert triangular matrix
                      LMatPar.invert();

		      //
		      //compute projected Hamiltonian
		      //
		      dealii::ScaLAPACKMatrix<double> projHamParTrans(N,
							    processGrid,
							    rowsBlockSize);

		      if (processGrid->is_process_active())
			  std::fill(&projHamParTrans.local_el(0,0),
				    &projHamParTrans.local_el(0,0)+projHamParTrans.local_m()*projHamParTrans.local_n(),
				    0.0);


		      projHamParTrans.copy_transposed(projHamPar);
		      projHamPar.add(projHamParTrans,1.0,1.0);

		      if (processGrid->is_process_active())
			 for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
			   {
			     const unsigned int glob_i = projHamPar.global_column(i);
			     for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
			       {
				 const unsigned int glob_j = projHamPar.global_row(j);
				 if (glob_i==glob_j)
				    projHamPar.local_el(j, i)*=0.5;
			       }
			   }

		      dealii::ScaLAPACKMatrix<double> projHamParCopy(N,
							    processGrid,
							    rowsBlockSize);

		      if (overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular)
		      {
			  LMatPar.mmult(projHamParCopy,projHamPar);
			  projHamParCopy.mTmult(projHamPar,LMatPar);
		      }
		      else
		      {
			  LMatPar.Tmmult(projHamParCopy,projHamPar);
			  projHamParCopy.mmult(projHamPar,LMatPar);
		      }

		      //
		      //compute eigendecomposition of ProjHam
		      //
		      const unsigned int numberEigenValues = N;
		      std::vector<double> eigenValuesStdVec(numberEigenValues,0.0);
		      
		      eigenValuesStdVec=projHamPar.eigenpairs_symmetric_by_index_MRRR(std::make_pair(0,numberEigenValues-1),true);
		      std::copy(eigenValuesStdVec.begin(),eigenValuesStdVec.end(),eigenValues);

		      projHamPar.copy_to(projHamParCopy);
		      if (overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular)
			LMatPar.Tmmult(projHamPar,projHamParCopy);
		      else
			LMatPar.mmult(projHamPar,projHamParCopy);

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
			  std::cout<<"Time for ScaLAPACK GEP eigen decomp, RR GEP step: "<<gpu_time<<std::endl;
		      }
              }
             
	      //
	      //rotate the basis in the subspace X = X*L_{inv}^{T}*Q implemented as X^{T}=Q^{T}*L^{-1}*X^{T}
	      //stored in the column major format
	      //
              if (dftParameters::gpuFineGrainedTimings)
              {
                 cudaDeviceSynchronize();
                 MPI_Barrier(MPI_COMM_WORLD);
	         gpu_time = MPI_Wtime();
              }

              if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
                 subspaceRotationRRMixedPrecScalapack(X,
                            M,
                            N,
                            handle,
                            processGrid,
                            mpiComm,
                            interBandGroupComm,
                            projHamPar,
                            true);
              else
                 subspaceRotationScalapack(X,
                            M,
                            N,
                            handle,
                            processGrid,
                            mpiComm,
                            interBandGroupComm,
                            projHamPar,
                            true);

              if (dftParameters::gpuFineGrainedTimings)
              {
                 cudaDeviceSynchronize();
                 gpu_time = MPI_Wtime() - gpu_time;

                 if (this_process==0)
                  if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
                     std::cout<<"Time for X = X*L_{inv}^{T}*Q mixed prec, RR GEP step: "<<gpu_time<<std::endl;
                  else
                     std::cout<<"Time for X = X*L_{inv}^{T}*Q, RR GEP step: "<<gpu_time<<std::endl;
              }

    }

    void rayleighRitzGEPSpectrumSplitDirect(operatorDFTCUDAClass & operatorMatrix,
		      double* X,
                      double* XFrac,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
                      const unsigned int N,
                      const unsigned int Noc,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
                      const MPI_Comm &interBandGroupComm,
		      double* eigenValues,
                      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      dealii::ScaLAPACKMatrix<double> & overlapMatPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const bool useMixedPrecOverall)
    {

              int this_process;
              MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

              const unsigned int Nfr=N-Noc;

              double gpu_time;

	      const unsigned int rowsBlockSize=operatorMatrix.getScalapackBlockSize();

              if (!isElpaStep2)
              {
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                         cudaDeviceSynchronize();
                         MPI_Barrier(MPI_COMM_WORLD);
		         gpu_time = MPI_Wtime();
                      }

		      //S=X*X^{T}. Implemented as S=X^{T}*X with X^{T} stored in the column major format
		      if (dftParameters::useMixedPrecPGS_O && useMixedPrecOverall)
				     linearAlgebraOperationsCUDA::
					     fillParallelOverlapMatMixedPrecScalapack
							      (X,
							       M,
							       N,
							       handle,
							       mpiComm,
                                                               interBandGroupComm,
							       processGrid,
							       overlapMatPar);

		      else
				     linearAlgebraOperationsCUDA::
					     fillParallelOverlapMatScalapack
							      (X,
							       M,
							       N,
							       handle,
							       mpiComm,
                                                               interBandGroupComm,
							       processGrid,
							       overlapMatPar); 
			    
		      if (dftParameters::gpuFineGrainedTimings)
		      { 
                            cudaDeviceSynchronize();
                            MPI_Barrier(MPI_COMM_WORLD);
			    gpu_time = MPI_Wtime() - gpu_time;
			    if (this_process==0)
			    {
			      if (dftParameters::useMixedPrecPGS_O && useMixedPrecOverall)
				  std::cout<<"Time for X^{T}X Mixed Prec, RR GEP step: "<<gpu_time<<std::endl;
			      else
				  std::cout<<"Time for X^{T}X, RR GEP step: "<<gpu_time<<std::endl;
			    }
		      }

		      if (dftParameters::gpuFineGrainedTimings)
                      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime();
                      }

		      if (processGrid->is_process_active())
			  std::fill(&projHamPar.local_el(0,0),
				    &projHamPar.local_el(0,0)+projHamPar.local_m()*projHamPar.local_n(),
				    0.0);

                      if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
			      operatorMatrix.XtHXMixedPrec(X,
						 Xb,
						 HXb,
						 projectorKetTimesVector,
						 M,
						 N,
                                                 Noc,
						 handle,
						 processGrid,
						 projHamPar);
                      else
                      {
                              if (dftParameters::overlapComputeCommunXtHX)
				      operatorMatrix.XtHXOverlapComputeCommun(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                              else
				      operatorMatrix.XtHX(X,
							 Xb,
							 HXb,
							 projectorKetTimesVector,
							 M,
							 N,
							 handle,
							 processGrid,
							 projHamPar);
                      }

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
                        {
                          if (useMixedPrecOverall && dftParameters::useMixedPrecXTHXSpectrumSplit)
                            std::cout<<"Time for X^{T}HX Mixed Prec, RR GEP step: "<<gpu_time<<std::endl;
                          else
  			     std::cout<<"Time for X^{T}HX, RR GEP step: "<<gpu_time<<std::endl;
                        }
		      }
              }
               
              if (isElpaStep1)
                 return;

              
	      dealii::ScaLAPACKMatrix<double> LMatPar(N,
						 processGrid,
						 rowsBlockSize);
              overlapMatPar.copy_to(LMatPar);
              dealii::LAPACKSupport::Property overlapMatPropertyPostCholesky=overlapMatPar.get_property(); 
              if (!isElpaStep2)
              {

                      
		      //
		      //compute eigendecomposition
		      //
		      if (dftParameters::gpuFineGrainedTimings)
                      {
                         cudaDeviceSynchronize();
                         MPI_Barrier(MPI_COMM_WORLD);
			 gpu_time = MPI_Wtime();
                      }

                      overlapMatPar.compute_cholesky_factorization();
                      overlapMatPropertyPostCholesky=overlapMatPar.get_property();

		      AssertThrow(overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular
				  ||overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular
				   ,dealii::ExcMessage("DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));
		      LMatPar.set_property(overlapMatPropertyPostCholesky); 

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
                      LMatPar.invert();

		      dealii::ScaLAPACKMatrix<double> projHamParTrans(N,
							    processGrid,
							    rowsBlockSize);

		      if (processGrid->is_process_active())
			  std::fill(&projHamParTrans.local_el(0,0),
				    &projHamParTrans.local_el(0,0)+projHamParTrans.local_m()*projHamParTrans.local_n(),
				    0.0);


		      projHamParTrans.copy_transposed(projHamPar);
		      projHamPar.add(projHamParTrans,1.0,1.0);

		      if (processGrid->is_process_active())
			 for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
			   {
			     const unsigned int glob_i = projHamPar.global_column(i);
			     for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
			       {
				 const unsigned int glob_j = projHamPar.global_row(j);
				 if (glob_i==glob_j)
				    projHamPar.local_el(j, i)*=0.5;
			       }
			   }

		      dealii::ScaLAPACKMatrix<double> projHamParCopy(N,
							    processGrid,
							    rowsBlockSize);

		      if (overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular)
		      {
			  LMatPar.mmult(projHamParCopy,projHamPar);
			  projHamParCopy.mTmult(projHamPar,LMatPar);
		      }
		      else
		      {
			  LMatPar.Tmmult(projHamParCopy,projHamPar);
			  projHamParCopy.mmult(projHamPar,LMatPar);
		      }

		      std::vector<double> eigenValuesStdVec(Nfr,0.0);
		      
		      eigenValuesStdVec=projHamPar.eigenpairs_symmetric_by_index_MRRR(std::make_pair(Noc,N-1),true);
		      std::copy(eigenValuesStdVec.begin(),eigenValuesStdVec.end(),eigenValues);

		      projHamPar.copy_to(projHamParCopy);
		      if (overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular)
			LMatPar.Tmmult(projHamPar,projHamParCopy);
		      else
			LMatPar.mmult(projHamPar,projHamParCopy);

		      if (dftParameters::gpuFineGrainedTimings)
		      {
                        cudaDeviceSynchronize();
                        MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0)
			  std::cout<<"Time for ScaLAPACK eigen decomp, RR GEP step: "<<gpu_time<<std::endl;
		      }
              }
             

              //
              //rotate the basis in the subspace X_{fr}=X*(L^{-1}^{T}*Q_{fr}
              //
	      if (dftParameters::gpuFineGrainedTimings)
	      {
		 cudaDeviceSynchronize();
		 MPI_Barrier(MPI_COMM_WORLD);
		 gpu_time = MPI_Wtime();
	      }

              subspaceRotationSpectrumSplitScalapack(X,
			    XFrac,
			    M,
			    N,
			    Nfr,
			    handle,
			    processGrid,
			    mpiComm,
			    projHamPar,
			    true);

	      if (dftParameters::gpuFineGrainedTimings)
	      {
		 cudaDeviceSynchronize();
		 MPI_Barrier(MPI_COMM_WORLD);
		 gpu_time = MPI_Wtime() - gpu_time;

		 if (this_process==0)
		      std::cout<<"Time for X_{fr}=X*(L^{-1}^{T}*Q_{fr}), RR GEP step: "<<gpu_time<<std::endl;
	      }

              //
              //X=X*L^{-1}^{T} implemented as X^{T}=L^{-1}*X^{T} with X^{T} stored in the column major format
              //
              if (dftParameters::gpuFineGrainedTimings)
              {
                 cudaDeviceSynchronize();
                 MPI_Barrier(MPI_COMM_WORLD);
	         gpu_time = MPI_Wtime();
              }

              if (useMixedPrecOverall && dftParameters::useMixedPrecPGS_SR)
	         subspaceRotationPGSMixedPrecScalapack(X,
			    M,
			    N,
			    handle,
			    processGrid,
			    mpiComm,
			    interBandGroupComm,
			    LMatPar,
			    overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false);
              else
	         subspaceRotationScalapack(X,
			    M,
			    N,
			    handle,
			    processGrid,
			    mpiComm,
			    interBandGroupComm,
			    LMatPar,
			    overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false,
			    dftParameters::triMatPGSOpt?true:false);

              if (dftParameters::gpuFineGrainedTimings)
              {
                 cudaDeviceSynchronize();
                 gpu_time = MPI_Wtime() - gpu_time;

                 if (this_process==0)
                  if (useMixedPrecOverall && dftParameters::useMixedPrecSubspaceRotRR)
                     std::cout<<"Time for X=X*L^{-1}^{T} mixed prec, RR GEP step: "<<gpu_time<<std::endl;
                  else
                     std::cout<<"Time for X=X*L^{-1}^{T}, RR GEP step: "<<gpu_time<<std::endl;
              }



    }


  }
}
