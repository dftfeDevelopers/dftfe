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

#include <linearAlgebraOperationsInternal.h>
#include <linearAlgebraOperations.h>
#include <dftParameters.h>
#include <dftUtils.h>

/** @file linearAlgebraOperationsInternal.cc
 *  @brief Contains small internal functions used in linearAlgebraOperations
 *
 */
namespace dftfe
{

  namespace linearAlgebraOperations
  {

    namespace internal
    {
#ifdef DEAL_II_WITH_SCALAPACK
	void createProcessGridSquareMatrix(const MPI_Comm & mpi_communicator,
		                           const unsigned size,
				           std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid)
	{
	      const unsigned int numberProcs = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

	      //Rule of thumb from http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
	      const unsigned int rowProcs=dftParameters::scalapackParalProcs==0?
		                std::min(std::floor(std::sqrt(numberProcs)),
				std::ceil((double)size/(double)(1000))):
				std::min((unsigned int)std::floor(std::sqrt(numberProcs)),
				         dftParameters::scalapackParalProcs);
	      if(dftParameters::verbosity>=4)
	      {
		 dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
		 pcout<<"Scalapack Matrix created, row procs: "<< rowProcs<<std::endl;
	      }

	      processGrid=std::make_shared<const dealii::Utilities::MPI::ProcessGrid>(mpi_communicator,
										      rowProcs,
										      rowProcs);
	}


	template<typename T>
	void createGlobalToLocalIdMapsScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                                   const dealii::ScaLAPACKMatrix<T> & mat,
				                   std::map<unsigned int, unsigned int> & globalToLocalRowIdMap,
					           std::map<unsigned int, unsigned int> & globalToLocalColumnIdMap)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
	  globalToLocalRowIdMap.clear();
          globalToLocalColumnIdMap.clear();
	  if (processGrid->is_process_active())
	  {
	     for (unsigned int i = 0; i < mat.local_m(); ++i)
		 globalToLocalRowIdMap[mat.global_row(i)]=i;

	     for (unsigned int j = 0; j < mat.local_n(); ++j)
		 globalToLocalColumnIdMap[mat.global_column(j)]=j;

	  }
#endif
	}


	template<typename T>
        void sumAcrossInterCommScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                            dealii::ScaLAPACKMatrix<T> & mat,
				            const MPI_Comm &interComm)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
  	  //sum across all inter communicator groups
	  if (processGrid->is_process_active() &&
	      dealii::Utilities::MPI::n_mpi_processes(interComm)>1)
	  {

                MPI_Allreduce(MPI_IN_PLACE,
                              &mat.local_el(0,0),
                              mat.local_m()*mat.local_n(),
                              MPI_DOUBLE,
                              MPI_SUM,
                              interComm);

	   }
#endif
	}

	template<typename T>
        void broadcastAcrossInterCommScaLAPACKMat
	                                   (const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                            dealii::ScaLAPACKMatrix<T> & mat,
				            const MPI_Comm &interComm,
					    const unsigned int broadcastRoot)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
  	  //sum across all inter communicator groups
	  if (processGrid->is_process_active() &&
	      dealii::Utilities::MPI::n_mpi_processes(interComm)>1)
	  {
                MPI_Bcast(&mat.local_el(0,0),
                           mat.local_m()*mat.local_n(),
                           MPI_DOUBLE,
                           broadcastRoot,
                           interComm);

	   }
#endif
	}

	template<typename T>
	void fillParallelOverlapMatrix(const dealii::parallel::distributed::Vector<T> & subspaceVectorsArray,
		                       const unsigned int N,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       const MPI_Comm &interBandGroupComm,
				       dealii::ScaLAPACKMatrix<T> & overlapMatPar)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = subspaceVectorsArray.local_size()/N;

          //band group parallelization data structures
          const unsigned int numberBandGroups=
	     dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
          const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
          std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
          dftUtils::createBandParallelizationIndices(interBandGroupComm,
						     N,
						     bandGroupLowHighPlusOneIndices);

          //get global to local index maps for Scalapack matrix
	  std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	  std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	  internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
		                                          overlapMatPar,
				                          globalToLocalRowIdMap,
					                  globalToLocalColumnIdMap);


          /*
	   * Sc=X^{T}*Xc is done in a blocked approach for memory optimization:
           * Sum_{blocks} X^{T}*XcBlock. The result of each X^{T}*XBlock
           * has a much smaller memory compared to X^{T}*Xc.
	   * X^{T} is a matrix with size number of wavefunctions times
	   * number of local degrees of freedom (N x MLoc).
	   * MLoc is denoted by numLocalDofs.
	   * Xc denotes complex conjugate of X.
	   * A further optimization is done to reduce floating point operations:
	   * As X^{T}*Xc is a Hermitian matrix, it suffices to compute only the lower
	   * triangular part. To exploit this, we do
	   * X^{T}*Xc=Sum_{blocks} XTrunc^{T}*XcBlock
	   * where XTrunc^{T} is a (D x MLoc) sub matrix of X^{T} with the row indices
	   * ranging fromt the lowest global index of XcBlock (denoted by ivec in the code)
	   * to N. D=N-ivec.
	   * The parallel ScaLapack overlap matrix is directly filled from
	   * the XTrunc^{T}*XcBlock result
	   */
	  const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                               bandGroupLowHighPlusOneIndices[1]);

	  std::vector<T> overlapMatrixBlock(N*vectorsBlockSize,0.0);
	  std::vector<T> blockVectorsMatrix(numLocalDofs*vectorsBlockSize,0.0);

	  for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
	  {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      const unsigned int B = std::min(vectorsBlockSize, N-ivec);

	      // If one plus the ending index of a block lies within a band parallelization group
	      // do computations for that block within the band group, otherwise skip that
	      // block. This is only activated if NPBAND>1
	      if ((ivec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (ivec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	      {
		  const char transA = 'N',transB = 'N';
		  const T scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		  std::fill(overlapMatrixBlock.begin(),overlapMatrixBlock.end(),0.);

	          // Extract XcBlock from X^{T}.
		  for (unsigned int i = 0; i <numLocalDofs; ++i)
		      for (unsigned int j = 0; j <B; ++j)
		      {
#ifdef USE_COMPLEX
			  blockVectorsMatrix[j*numLocalDofs+i]=
			      std::conj(subspaceVectorsArray.local_element(N*i+j+ivec));
#else
			  blockVectorsMatrix[j*numLocalDofs+i]=
			      subspaceVectorsArray.local_element(N*i+j+ivec);
#endif
		      }

		  const unsigned int D=N-ivec;


		  // Comptute local XTrunc^{T}*XcBlock.
		  dgemm_(&transA,
			 &transB,
			 &D,
			 &B,
			 &numLocalDofs,
			 &scalarCoeffAlpha,
			 subspaceVectorsArray.begin()+ivec,
			 &N,
			 &blockVectorsMatrix[0],
			 &numLocalDofs,
			 &scalarCoeffBeta,
			 &overlapMatrixBlock[0],
			 &D);


		  // Sum local XTrunc^{T}*XcBlock across domain decomposition processors
#ifdef USE_COMPLEX
		  MPI_Allreduce(MPI_IN_PLACE,
				&overlapMatrixBlock[0],
				D*B,
				MPI_C_DOUBLE_COMPLEX,
				MPI_SUM,
				subspaceVectorsArray.get_mpi_communicator());
#else
		  MPI_Allreduce(MPI_IN_PLACE,
			        &overlapMatrixBlock[0],
				D*B,
				MPI_DOUBLE,
				MPI_SUM,
			        subspaceVectorsArray.get_mpi_communicator());
#endif

		  //Copying only the lower triangular part to the ScaLAPACK overlap matrix
		  if (processGrid->is_process_active())
		      for(unsigned int i = 0; i <B; ++i)
			  if (globalToLocalColumnIdMap.find(i+ivec)!=globalToLocalColumnIdMap.end())
			  {
			      const unsigned int localColumnId=globalToLocalColumnIdMap[i+ivec];
			      for (unsigned int j = ivec; j <N; ++j)
			      {
				 std::map<unsigned int, unsigned int>::iterator it=
					      globalToLocalRowIdMap.find(j);
				 if(it!=globalToLocalRowIdMap.end())
				     overlapMatPar.local_el(it->second,
							    localColumnId)
							    =overlapMatrixBlock[i*D+j-ivec];
			      }
			  }
		}//band parallelization
	  }//block loop

	  //accumulate contribution from all band parallelization groups
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat
	                                          (processGrid,
						   overlapMatPar,
						   interBandGroupComm);
#endif
	}


	template<typename T>
	void subspaceRotation(dealii::parallel::distributed::Vector<T> & subspaceVectorsArray,
		              const unsigned int N,
			      const unsigned int numberCoreVectors,
			      dealii::parallel::distributed::Vector<T> & nonCoreVectorsArray,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const MPI_Comm &interBandGroupComm,
			      const dealii::ScaLAPACKMatrix<T> & rotationMatPar,
			      const bool rotationMatTranspose,
			      const bool isRotationMatLowerTria)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = subspaceVectorsArray.local_size()/N;

	  const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
						  subspaceVectorsArray.get_mpi_communicator());

          //band group parallelization data structures
          const unsigned int numberBandGroups=
	     dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
          const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
          std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
          dftUtils::createBandParallelizationIndices(interBandGroupComm,
						     N,
						     bandGroupLowHighPlusOneIndices);

	  std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	  std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	  internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
		                                          rotationMatPar,
				                          globalToLocalRowIdMap,
					                  globalToLocalColumnIdMap);

          /*
	   * Q*X^{T} is done in a blocked approach for memory optimization:
           * Sum_{dof_blocks} Sum_{vector_blocks} QBvec*XBdof^{T}.
	   * The result of each QBvec*XBdof^{T}
           * has a much smaller memory compared to Q*X^{T}.
	   * X^{T} (denoted by subspaceVectorsArray in the code with column major format storage)
	   * is a matrix with size (N x MLoc).
	   * N is denoted by numberWaveFunctions in the code.
	   * MLoc, which is number of local dofs is denoted by numLocalDofs in the code.
	   * QBvec is a matrix of size (BVec x N)
	   * XBdof is a matrix of size (N x BDof)
	   * A further optimization is done to reduce floating point operations when
	   * Q is a lower triangular matrix in the subspace rotation step of PGS:
	   * Then it suffices to compute only the multiplication of lower
	   * triangular part of Q with X^{T}. To exploit this, we do
	   * Sum_{dof_blocks} Sum_{vector_blocks} QBvecTrunc*XBdofTrunc^{T}.
	   * where QBvecTrunc is a (BVec x D) sub matrix of QBvec with the column indices
	   * ranging from O to D-1, where D=jvec(lowest global index of QBvec) + BVec.
	   * XBdofTrunc is a (D x BDof) sub matrix of XBdof with the row indices
	   * ranging from 0 to D-1.
	   * X^{T} is directly updated from
	   * the Sum_{vector_blocks} QBvecTrunc*XBdofTrunc^{T} result
	   * for each {dof_block}.
	   */
	  const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                               bandGroupLowHighPlusOneIndices[1]);
	  const unsigned int dofsBlockSize=dftParameters::subspaceRotDofsBlockSize;

	  std::vector<T> rotationMatBlock(vectorsBlockSize*N,0.0);
	  std::vector<T> rotatedVectorsMatBlock(N*dofsBlockSize,0.0);
          std::vector<T> rotatedVectorsMatBlockTemp(vectorsBlockSize*dofsBlockSize,0.0);

          if (dftParameters::verbosity>=4)
                   dftUtils::printCurrentMemoryUsage(subspaceVectorsArray.get_mpi_communicator(),
	                      "Inside Blocked susbpace rotation");

	  for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	  {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      unsigned int BDof=0;
	      if (numLocalDofs>=idof)
                 BDof = std::min(dofsBlockSize, numLocalDofs-idof);

	      std::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
	      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
	      {
		  // Correct block dimensions if block "goes off edge of" the matrix
		  const unsigned int BVec = std::min(vectorsBlockSize, N-jvec);

		  const unsigned int D=isRotationMatLowerTria?
		                                         (jvec+BVec)
		                                         :N;

		  // If one plus the ending index of a block lies within a band parallelization group
		  // do computations for that block within the band group, otherwise skip that
		  // block. This is only activated if NPBAND>1
		  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
		  (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		  {
		      const char transA = 'N',transB = 'N';
		      const T scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		      std::fill(rotationMatBlock.begin(),rotationMatBlock.end(),0.);

		      //Extract QBVec from parallel ScaLAPACK matrix Q
		      if (rotationMatTranspose)
		      {
			  if (processGrid->is_process_active())
			      for (unsigned int i = 0; i <D; ++i)
				  if (globalToLocalRowIdMap.find(i)
					  !=globalToLocalRowIdMap.end())
				  {
				     const unsigned int localRowId=globalToLocalRowIdMap[i];
				     for (unsigned int j = 0; j <BVec; ++j)
				     {
					std::map<unsigned int, unsigned int>::iterator it=
					      globalToLocalColumnIdMap.find(j+jvec);
					if(it!=globalToLocalColumnIdMap.end())
						 rotationMatBlock[i*BVec+j]=
						     rotationMatPar.local_el(localRowId,
									     it->second);
				     }
				  }
		      }
		      else
		      {
			  if (processGrid->is_process_active())
			      for (unsigned int i = 0; i <D; ++i)
				  if(globalToLocalColumnIdMap.find(i)
					  !=globalToLocalColumnIdMap.end())
				  {
				      const unsigned int localColumnId=globalToLocalColumnIdMap[i];
				      for (unsigned int j = 0; j <BVec; ++j)
				      {
					 std::map<unsigned int, unsigned int>::iterator it=
					       globalToLocalRowIdMap.find(j+jvec);
					 if (it!=globalToLocalRowIdMap.end())
					       rotationMatBlock[i*BVec+j]=
						   rotationMatPar.local_el(it->second,
									   localColumnId);
				      }
				  }
		      }

#ifdef USE_COMPLEX
		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatBlock[0],
				    vectorsBlockSize*D,
				    MPI_C_DOUBLE_COMPLEX,
				    MPI_SUM,
				    subspaceVectorsArray.get_mpi_communicator());
#else
		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatBlock[0],
				    vectorsBlockSize*D,
				    MPI_DOUBLE,
				    MPI_SUM,
				    subspaceVectorsArray.get_mpi_communicator());
#endif

		      if (BDof!=0)
		      {

			  dgemm_(&transA,
				 &transB,
				 &BVec,
				 &BDof,
				 &D,
				 &scalarCoeffAlpha,
				 &rotationMatBlock[0],
				 &BVec,
				 subspaceVectorsArray.begin()+idof*N,
				 &N,
				 &scalarCoeffBeta,
				 &rotatedVectorsMatBlockTemp[0],
				 &BVec);

			  for (unsigned int i = 0; i <BDof; ++i)
			      for (unsigned int j = 0; j <BVec; ++j)
				  rotatedVectorsMatBlock[N*i+j+jvec]
				      =rotatedVectorsMatBlockTemp[i*BVec+j];
		      }

		  }// band parallelization
	      }//block loop over vectors

	      if (BDof!=0)
	      {
		  for (unsigned int i = 0; i <BDof; ++i)
		      for (unsigned int j = 0; j <N; ++j)
			  subspaceVectorsArray.local_element(N*(i+idof)+j)
			      =rotatedVectorsMatBlock[i*N+j];
	      }
	  }//block loop over dofs

	  // In case of spectrum splitting and band parallelization
	  // only communicate the valence wavefunctions
	  if (numberBandGroups>1)
  	  {
	    if (numberCoreVectors!=0)
	    {

		const unsigned int numberNonCoreVectors=N-numberCoreVectors;
		for(unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
		    for(unsigned int iWave = 0; iWave < numberNonCoreVectors; ++iWave)
			nonCoreVectorsArray.local_element(iNode*numberNonCoreVectors +iWave)
			     =subspaceVectorsArray.local_element(iNode*N
								  +numberCoreVectors
								  +iWave);

		MPI_Allreduce(MPI_IN_PLACE,
			      nonCoreVectorsArray.begin(),
			      numberNonCoreVectors*numLocalDofs,
			      MPI_DOUBLE,
			      MPI_SUM,
			      interBandGroupComm);

		for(unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
		    for(unsigned int iWave = 0; iWave < numberNonCoreVectors; ++iWave)
		        subspaceVectorsArray.local_element
			                      (iNode*N
					       +numberCoreVectors
					       +iWave)
		                                =nonCoreVectorsArray.local_element(iNode*numberNonCoreVectors+iWave);

	    }
	    else
	    {
		MPI_Allreduce(MPI_IN_PLACE,
			      subspaceVectorsArray.begin(),
			      N*numLocalDofs,
			      MPI_DOUBLE,
			      MPI_SUM,
			      interBandGroupComm);
	    }
	  }
#endif
	}
#endif

#ifdef DEAL_II_WITH_SCALAPACK
	template
	void createGlobalToLocalIdMapsScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                                   const dealii::ScaLAPACKMatrix<dataTypes::number> & mat,
				                   std::map<unsigned int, unsigned int> & globalToLocalRowIdMap,
					           std::map<unsigned int, unsigned int> & globalToLocalColumnIdMap);

        template
	void fillParallelOverlapMatrix(const dealii::parallel::distributed::Vector<dataTypes::number> & X,
		                       const unsigned int N,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       const MPI_Comm &interBandGroupComm,
				       dealii::ScaLAPACKMatrix<dataTypes::number> & overlapMatPar);

	template
	void subspaceRotation(dealii::parallel::distributed::Vector<dataTypes::number> & subspaceVectorsArray,
		              const unsigned int N,
			      const unsigned int numberCoreVectors,
			      dealii::parallel::distributed::Vector<dataTypes::number> & nonCoreVectorsArray,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const MPI_Comm &interBandGroupComm,
			      const dealii::ScaLAPACKMatrix<dataTypes::number> & rotationMatPar,
			      const bool rotationMatTranpose,
			      const bool isRotationMatLowerTria);

        template
	void sumAcrossInterCommScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                            dealii::ScaLAPACKMatrix<dataTypes::number> & mat,
				            const MPI_Comm &interComm);
	template
        void broadcastAcrossInterCommScaLAPACKMat
	                                   (const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                            dealii::ScaLAPACKMatrix<dataTypes::number> & mat,
				            const MPI_Comm &interComm,
					    const unsigned int broadcastRoot);
#endif
    }
  }
}
