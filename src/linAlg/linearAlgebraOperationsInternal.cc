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


      void createProcessGridRectangularMatrix(const MPI_Comm & mpi_communicator,
					      const unsigned sizeRows,
					      const unsigned sizeColumns,
					      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid)
      {
	const unsigned int numberProcs = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

	//Rule of thumb from http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
	const unsigned int rowProcs=  std::min(std::floor(std::sqrt(numberProcs)),std::ceil((double)sizeRows/(double)(1000)));
	const unsigned int columnProcs = std::min(std::floor(std::sqrt(numberProcs)),std::ceil((double)sizeColumns/(double)(1000)));

	if(dftParameters::verbosity>=4)
	  {
	    dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
	    pcout<<"Scalapack Matrix created, row procs x column procs: "<< rowProcs<<" x "<<columnProcs<<std::endl;
	  }

	processGrid=std::make_shared<const dealii::Utilities::MPI::ProcessGrid>(mpi_communicator,
										rowProcs,
										columnProcs);
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
      void scaleScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			     dealii::ScaLAPACKMatrix<T> & mat,
			     const T scalar)
      {
#ifdef USE_COMPLEX
	AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
	if(processGrid->is_process_active())
	  {
	    const unsigned int numberComponents =  mat.local_m()*mat.local_n();
	    const unsigned int inc = 1;
	    dscal_(&numberComponents,
		   &scalar,
		   &mat.local_el(0,0),
		   &inc);
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

      void fillParallelOverlapMatrixMixedPrec(const dataTypes::number* subspaceVectorsArray,
	                             const unsigned int subspaceVectorsArrayLocalSize,
				     const unsigned int N,
				     const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				     const MPI_Comm &interBandGroupComm,
				     const MPI_Comm &mpiComm,
				     dealii::ScaLAPACKMatrix<dataTypes::number> & overlapMatPar)
      {

#ifdef USE_COMPLEX
	AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize/N;

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
	   * XcBlock is a matrix with size (MLoc x B). B is the block size.
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

	  std::vector<dataTypes::number> overlapMatrixBlock(N*vectorsBlockSize,0.0);
	  std::vector<dataTypes::number> blockVectorsMatrix(numLocalDofs*vectorsBlockSize,0.0);
	  std::vector<dataTypes::numberLowPrec> blockVectorsMatrixLowPrec(numLocalDofs*vectorsBlockSize,0.0);
	  std::vector<dataTypes::numberLowPrec> overlapMatrixBlockLowPrec(N*vectorsBlockSize,0.0);

	  std::vector<dataTypes::numberLowPrec> subspaceVectorsArrayLowPrec(subspaceVectorsArray,
		                                                             subspaceVectorsArray+
									     subspaceVectorsArrayLocalSize);
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
		  const dataTypes::number scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
		  const dataTypes::numberLowPrec scalarCoeffAlphaLowPrec = 1.0,scalarCoeffBetaLowPrec = 0.0;

		  std::fill(overlapMatrixBlock.begin(),overlapMatrixBlock.end(),0.);
                  std::fill(overlapMatrixBlockLowPrec.begin(),overlapMatrixBlockLowPrec.end(),0.);

	          // Extract XcBlock from X^{T}.
		  for (unsigned int i = 0; i <numLocalDofs; ++i)
		      for (unsigned int j = 0; j <B; ++j)
		      {
#ifdef USE_COMPLEX
			  blockVectorsMatrix[j*numLocalDofs+i]=
			      std::conj(subspaceVectorsArray[N*i+j+ivec]);

		          blockVectorsMatrixLowPrec[j*numLocalDofs+i]=
			      std::conj((dataTypes::numberLowPrec)subspaceVectorsArray[N*i+j+ivec]);
#else
			  blockVectorsMatrix[j*numLocalDofs+i]=
			      subspaceVectorsArray[N*i+j+ivec];

			  blockVectorsMatrixLowPrec[j*numLocalDofs+i]=
			      (dataTypes::numberLowPrec)subspaceVectorsArray[N*i+j+ivec];
#endif
		      }

		  const unsigned int diagBlockSize=B;
		  const unsigned int D=N-ivec;

		  dgemm_(&transA,
			 &transB,
			 &diagBlockSize,
			 &B,
			 &numLocalDofs,
			 &scalarCoeffAlpha,
			 subspaceVectorsArray+ivec,
			 &N,
			 &blockVectorsMatrix[0],
			 &numLocalDofs,
			 &scalarCoeffBeta,
			 &overlapMatrixBlock[0],
			 &D);

		  const unsigned int DRem=D-diagBlockSize;
		  sgemm_(&transA,
			 &transB,
			 &DRem,
			 &B,
			 &numLocalDofs,
			 &scalarCoeffAlphaLowPrec,
			 &subspaceVectorsArrayLowPrec[0]+ivec+diagBlockSize,
			 &N,
			 &blockVectorsMatrixLowPrec[0],
			 &numLocalDofs,
			 &scalarCoeffBetaLowPrec,
			 &overlapMatrixBlockLowPrec[diagBlockSize],
			 &D);

		  for(unsigned int i = 0; i <B; ++i)
		      for (unsigned int j = ivec+diagBlockSize; j <N; ++j)
			  overlapMatrixBlock[i*D+j-ivec]
			      =overlapMatrixBlockLowPrec[i*D+j-ivec];

		  MPI_Barrier(mpiComm);
		  // Sum local XTrunc^{T}*XcBlock across domain decomposition processors
		  MPI_Allreduce(MPI_IN_PLACE,
				&overlapMatrixBlock[0],
				D*B,
				dataTypes::mpi_type_id(&overlapMatrixBlock[0]),
				MPI_SUM,
				mpiComm);

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
      void fillParallelOverlapMatrix(const T* subspaceVectorsArray,
	                             const unsigned int subspaceVectorsArrayLocalSize,
				     const unsigned int N,
				     const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				     const MPI_Comm &interBandGroupComm,
				     const MPI_Comm &mpiComm,
				     dealii::ScaLAPACKMatrix<T> & overlapMatPar)
      {

#ifdef USE_COMPLEX
	AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize/N;

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
	   * XcBlock is a matrix with size (MLoc x B). B is the block size.
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
			      std::conj(subspaceVectorsArray[N*i+j+ivec]);
#else
			  blockVectorsMatrix[j*numLocalDofs+i]=
			      subspaceVectorsArray[N*i+j+ivec];
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
			 subspaceVectorsArray+ivec,
			 &N,
			 &blockVectorsMatrix[0],
			 &numLocalDofs,
			 &scalarCoeffBeta,
			 &overlapMatrixBlock[0],
			 &D);

                  MPI_Barrier(mpiComm);
		  // Sum local XTrunc^{T}*XcBlock across domain decomposition processors
		  MPI_Allreduce(MPI_IN_PLACE,
				&overlapMatrixBlock[0],
				D*B,
				dataTypes::mpi_type_id(&overlapMatrixBlock[0]),
				MPI_SUM,
				mpiComm);

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
      void subspaceRotation(T* subspaceVectorsArray,
	                    const unsigned int subspaceVectorsArrayLocalSize,
			    const unsigned int N,
			    const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			    const MPI_Comm &interBandGroupComm,
			    const MPI_Comm &mpiComm,
			    const dealii::ScaLAPACKMatrix<T> & rotationMatPar,
			    const bool rotationMatTranspose,
			    const bool isRotationMatLowerTria)
      {
#ifdef USE_COMPLEX
	AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
	const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize/N;

	const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
								       mpiComm);

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
	   * QBvec is a matrix of size (BVec x N). BVec is the vectors block size.
	   * XBdof is a matrix of size (N x BDof). BDof is the block size of dofs.
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
	const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
		                                    dftParameters::subspaceRotDofsBlockSize);

	std::vector<T> rotationMatBlock(vectorsBlockSize*N,0.0);
	std::vector<T> rotatedVectorsMatBlock(N*dofsBlockSize,0.0);

	if (dftParameters::verbosity>=4)
	  dftUtils::printCurrentMemoryUsage(mpiComm,
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


		      MPI_Barrier(mpiComm);
		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatBlock[0],
				    BVec*D,
				    dataTypes::mpi_type_id(&rotationMatBlock[0]),
				    MPI_SUM,
				    mpiComm);

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
				 subspaceVectorsArray+idof*N,
				 &N,
				 &scalarCoeffBeta,
				 &rotatedVectorsMatBlock[0]+jvec,
				 &N);
		      }

		  }// band parallelization
	      }//block loop over vectors

	    if (BDof!=0)
	      {
		for (unsigned int i = 0; i <BDof; ++i)
		  for (unsigned int j = 0; j <N; ++j)
		    *(subspaceVectorsArray+N*(i+idof)+j)
		      =rotatedVectorsMatBlock[i*N+j];
	      }
	  }//block loop over dofs

	  if (numberBandGroups>1)
  	  {
	        MPI_Barrier(interBandGroupComm);
		const unsigned int blockSize=dftParameters::mpiAllReduceMessageBlockSizeMB*1e+6/sizeof(T);

		for (unsigned int i=0; i<N*numLocalDofs;i+=blockSize)
		{
		   const unsigned int currentBlockSize=std::min(blockSize,N*numLocalDofs-i);

		   MPI_Allreduce(MPI_IN_PLACE,
				 subspaceVectorsArray+i,
				 currentBlockSize,
				 dataTypes::mpi_type_id(subspaceVectorsArray),
				 MPI_SUM,
				 interBandGroupComm);
		}
	  }
#endif
      }



	template<typename T>
	void subspaceRotationSpectrumSplit(const T* X,
		              T* Y,
		              const unsigned int subspaceVectorsArrayLocalSize,
		              const unsigned int N,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const unsigned int numberTopVectors,
			      const MPI_Comm &interBandGroupComm,
			      const MPI_Comm &mpiComm,
			      const dealii::ScaLAPACKMatrix<T> & QMat,
			      const bool QMatTranspose)
	{
#ifdef USE_COMPLEX
	AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
	const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize/N;

	const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
								       mpiComm);

	//band group parallelization data structures
	const unsigned int numberBandGroups=
	  dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
	const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
	std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
	dftUtils::createBandParallelizationIndices(interBandGroupComm,
						   numberTopVectors,
						   bandGroupLowHighPlusOneIndices);

	std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
							QMat,
							globalToLocalRowIdMap,
							globalToLocalColumnIdMap);


	const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                               bandGroupLowHighPlusOneIndices[1]);
	const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
		                                    dftParameters::subspaceRotDofsBlockSize);

	std::vector<T> rotationMatBlock(vectorsBlockSize*N,0.0);
	std::vector<T> rotatedVectorsMatBlock(numberTopVectors*dofsBlockSize,0.0);

	if (dftParameters::verbosity>=4)
	  dftUtils::printCurrentMemoryUsage(mpiComm,
					    "Inside Blocked susbpace rotation");

	for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	  {
	    // Correct block dimensions if block "goes off edge of" the matrix
	    unsigned int BDof=0;
	    if (numLocalDofs>=idof)
	      BDof = std::min(dofsBlockSize, numLocalDofs-idof);

	    std::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
	    for (unsigned int jvec = 0; jvec < numberTopVectors; jvec += vectorsBlockSize)
	    {
		  // Correct block dimensions if block "goes off edge of" the matrix
		  const unsigned int BVec = std::min(vectorsBlockSize, numberTopVectors-jvec);

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
		    if (QMatTranspose)
		      {
			if (processGrid->is_process_active())
			  for (unsigned int i = 0; i <N; ++i)
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
					QMat.local_el(localRowId,
								it->second);
				  }
			      }
		      }
		    else
		      {
			if (processGrid->is_process_active())
			  for (unsigned int i = 0; i <N; ++i)
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
					QMat.local_el(it->second,
								localColumnId);
				  }
			      }
		      }

                      MPI_Barrier(mpiComm);
		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatBlock[0],
				    BVec*N,
				    dataTypes::mpi_type_id(&rotationMatBlock[0]),
				    MPI_SUM,
				    mpiComm);

		      if (BDof!=0)
		      {

			  dgemm_(&transA,
				 &transB,
				 &BVec,
				 &BDof,
				 &N,
				 &scalarCoeffAlpha,
				 &rotationMatBlock[0],
				 &BVec,
				 X+idof*N,
				 &N,
				 &scalarCoeffBeta,
				 &rotatedVectorsMatBlock[0]+jvec,
				 &numberTopVectors);
		      }

		  }// band parallelization
	    }//block loop over vectors


	    if (BDof!=0)
	      {
		for (unsigned int i = 0; i <BDof; ++i)
		  for (unsigned int j = 0; j <numberTopVectors; ++j)
		    *(Y+numberTopVectors*(i+idof)+j)
		      =rotatedVectorsMatBlock[i*numberTopVectors+j];
	      }
	  }//block loop over dofs

	  if (numberBandGroups>1)
	  {
		const unsigned int blockSize=dftParameters::mpiAllReduceMessageBlockSizeMB*1e+6/sizeof(T);
                MPI_Barrier(interBandGroupComm);
		for (unsigned int i=0; i<numberTopVectors*numLocalDofs;i+=blockSize)
		{
		   const unsigned int currentBlockSize=std::min(blockSize,numberTopVectors*numLocalDofs-i);

		   MPI_Allreduce(MPI_IN_PLACE,
				 Y+i,
				 currentBlockSize,
				 dataTypes::mpi_type_id(Y),
				 MPI_SUM,
				 interBandGroupComm);
		}
	  }
#endif
      }


	void subspaceRotationSpectrumSplitMixedPrec(const dataTypes::number* X,
		              dataTypes::number* Y,
		              const unsigned int subspaceVectorsArrayLocalSize,
		              const unsigned int N,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const unsigned int numberTopVectors,
			      const MPI_Comm &interBandGroupComm,
			      const MPI_Comm &mpiComm,
			      const dealii::ScaLAPACKMatrix<dataTypes::number> & QMat,
			      const bool QMatTranspose)
	{
#ifdef USE_COMPLEX
	AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
	const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize/N;

	const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
								       mpiComm);

	//band group parallelization data structures
	const unsigned int numberBandGroups=
	  dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
	const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
	std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
	dftUtils::createBandParallelizationIndices(interBandGroupComm,
						   numberTopVectors,
						   bandGroupLowHighPlusOneIndices);

	std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
							QMat,
							globalToLocalRowIdMap,
							globalToLocalColumnIdMap);


	const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                               bandGroupLowHighPlusOneIndices[1]);
	const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
		                                    dftParameters::subspaceRotDofsBlockSize);

	const unsigned int Ncore=N-numberTopVectors;
	std::vector<dataTypes::number> rotationMatTopCompBlock(vectorsBlockSize*numberTopVectors,0.0);
	std::vector<dataTypes::number> rotatedVectorsMatBlock(numberTopVectors*dofsBlockSize,0.0);

	std::vector<dataTypes::numberLowPrec> rotationMatCoreCompBlock(vectorsBlockSize*Ncore,0.0);
        std::vector<dataTypes::numberLowPrec> rotatedVectorsMatCoreContrBlockTemp(vectorsBlockSize*dofsBlockSize,0.0);

	std::vector<dataTypes::numberLowPrec> XSinglePrec(X,
		                                          X+
					                  subspaceVectorsArrayLocalSize);
	if (dftParameters::verbosity>=4)
	  dftUtils::printCurrentMemoryUsage(mpiComm,
					    "Inside Blocked susbpace rotation");

	for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	  {
	    // Correct block dimensions if block "goes off edge of" the matrix
	    unsigned int BDof=0;
	    if (numLocalDofs>=idof)
	      BDof = std::min(dofsBlockSize, numLocalDofs-idof);

	    std::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
	    for (unsigned int jvec = 0; jvec < numberTopVectors; jvec += vectorsBlockSize)
	    {
		  // Correct block dimensions if block "goes off edge of" the matrix
		  const unsigned int BVec = std::min(vectorsBlockSize, numberTopVectors-jvec);

		  // If one plus the ending index of a block lies within a band parallelization group
		  // do computations for that block within the band group, otherwise skip that
		  // block. This is only activated if NPBAND>1
		  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
		  (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		  {
		    const char transA = 'N',transB = 'N';
		    const dataTypes::number scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
		    const dataTypes::numberLowPrec scalarCoeffAlphaSinglePrec = 1.0,scalarCoeffBetaSinglePrec = 0.0;

		    std::fill(rotationMatCoreCompBlock.begin(),rotationMatCoreCompBlock.end(),0.);
		    std::fill(rotationMatTopCompBlock.begin(),rotationMatTopCompBlock.end(),0.);

		    //Extract QBVec from parallel ScaLAPACK matrix Q
		    if (QMatTranspose)
		      {
			if (processGrid->is_process_active())
			  for (unsigned int i = 0; i <N; ++i)
			    if (globalToLocalRowIdMap.find(i)
				!=globalToLocalRowIdMap.end())
			      {
				const unsigned int localRowId=globalToLocalRowIdMap[i];
				for (unsigned int j = 0; j <BVec; ++j)

				  {
				    std::map<unsigned int, unsigned int>::iterator it=
				      globalToLocalColumnIdMap.find(j+jvec);
				    if(it!=globalToLocalColumnIdMap.end())
				    {
				        const dataTypes::number val=
					          QMat.local_el(localRowId,
								it->second);
					if (i<Ncore)
					  rotationMatCoreCompBlock[i*BVec+j]=val;
					else
					  rotationMatTopCompBlock[(i-Ncore)*BVec+j]=val;
				    }
				  }
			      }
		      }
		    else
		      {
			if (processGrid->is_process_active())
			  for (unsigned int i = 0; i <N; ++i)
			    if(globalToLocalColumnIdMap.find(i)
			       !=globalToLocalColumnIdMap.end())
			      {
				const unsigned int localColumnId=globalToLocalColumnIdMap[i];
				for (unsigned int j = 0; j <BVec; ++j)
				  {
				    std::map<unsigned int, unsigned int>::iterator it=
				      globalToLocalRowIdMap.find(j+jvec);
				    if (it!=globalToLocalRowIdMap.end())
				    {
				        const dataTypes::number val=
				 	          QMat.local_el(it->second,
								localColumnId);
					if (i<Ncore)
					  rotationMatCoreCompBlock[i*BVec+j]=val;
					else
					  rotationMatTopCompBlock[(i-Ncore)*BVec+j]=val;

				    }
				  }
			      }
		      }

                      MPI_Barrier(mpiComm);
		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatCoreCompBlock[0],
				    BVec*Ncore,
				    dataTypes::mpi_type_id(&rotationMatCoreCompBlock[0]),
				    MPI_SUM,
				    mpiComm);

		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatTopCompBlock[0],
				    BVec*numberTopVectors,
				    dataTypes::mpi_type_id(&rotationMatTopCompBlock[0]),
				    MPI_SUM,
				    mpiComm);

		      if (BDof!=0)
		      {
			  sgemm_(&transA,
				 &transB,
				 &BVec,
				 &BDof,
				 &Ncore,
				 &scalarCoeffAlphaSinglePrec,
				 &rotationMatCoreCompBlock[0],
				 &BVec,
				 &XSinglePrec[0]+idof*N,
				 &N,
				 &scalarCoeffBetaSinglePrec,
				 &rotatedVectorsMatCoreContrBlockTemp[0],
				 &BVec);

			  dgemm_(&transA,
				 &transB,
				 &BVec,
				 &BDof,
				 &numberTopVectors,
				 &scalarCoeffAlpha,
				 &rotationMatTopCompBlock[0],
				 &BVec,
				 X+idof*N+Ncore,
				 &N,
				 &scalarCoeffBeta,
				 &rotatedVectorsMatBlock[0]+jvec,
				 &numberTopVectors);

		           for (unsigned int i = 0; i <BDof; ++i)
			      for (unsigned int j = 0; j <BVec; ++j)
			          rotatedVectorsMatBlock[i*numberTopVectors+j+jvec]
				      +=rotatedVectorsMatCoreContrBlockTemp[i*BVec+j];
		      }

		  }// band parallelization
	    }//block loop over vectors


	    if (BDof!=0)
	      {
		for (unsigned int i = 0; i <BDof; ++i)
		  for (unsigned int j = 0; j <numberTopVectors; ++j)
		    *(Y+numberTopVectors*(i+idof)+j)
		      =rotatedVectorsMatBlock[i*numberTopVectors+j];
	      }
	  }//block loop over dofs

	  if (numberBandGroups>1)
	  {
		const unsigned int blockSize=dftParameters::mpiAllReduceMessageBlockSizeMB*1e+6/sizeof(dataTypes::number);
                MPI_Barrier(interBandGroupComm);
		for (unsigned int i=0; i<numberTopVectors*numLocalDofs;i+=blockSize)
		{
		   const unsigned int currentBlockSize=std::min(blockSize,numberTopVectors*numLocalDofs-i);

		   MPI_Allreduce(MPI_IN_PLACE,
				 Y+i,
				 currentBlockSize,
				 dataTypes::mpi_type_id(Y),
				 MPI_SUM,
				 interBandGroupComm);
		}
	  }
#endif
      }

      void subspaceRotationPGSMixedPrec
			   (dataTypes::number* subspaceVectorsArray,
			    const unsigned int subspaceVectorsArrayLocalSize,
			    const unsigned int N,
			    const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			    const MPI_Comm &interBandGroupComm,
			    const MPI_Comm &mpiComm,
			    const dealii::ScaLAPACKMatrix<dataTypes::number> & rotationMatPar,
			    const bool rotationMatTranspose)
      {
#ifdef USE_COMPLEX
        AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
	const unsigned int numLocalDofs = subspaceVectorsArrayLocalSize/N;

	const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
								       mpiComm);

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
	   * QBvec is a matrix of size (BVec x N). BVec is the vectors block size.
	   * XBdof is a matrix of size (N x BDof). BDof is the block size of dofs.
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
	const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
		                                    dftParameters::subspaceRotDofsBlockSize);

	std::vector<dataTypes::numberLowPrec> rotationMatBlock(vectorsBlockSize*N,0.0);
        std::vector<dataTypes::numberLowPrec> rotatedVectorsMatBlockTemp(vectorsBlockSize*dofsBlockSize,0.0);

	std::vector<dataTypes::numberLowPrec> subspaceVectorsArraySinglePrec(subspaceVectorsArray,
		                                                             subspaceVectorsArray+
									     subspaceVectorsArrayLocalSize);
        std::vector<dataTypes::number> diagValuesBlock(vectorsBlockSize,0.0);
	if (dftParameters::verbosity>=4)
	  dftUtils::printCurrentMemoryUsage(mpiComm,
					    "Inside Blocked susbpace rotation");

	for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	  {
	    // Correct block dimensions if block "goes off edge of" the matrix
	    unsigned int BDof=0;
	    if (numLocalDofs>=idof)
	      BDof = std::min(dofsBlockSize, numLocalDofs-idof);

	      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
	      {
		  // Correct block dimensions if block "goes off edge of" the matrix
		  const unsigned int BVec = std::min(vectorsBlockSize, N-jvec);

		  const unsigned int D=jvec+BVec;

		  // If one plus the ending index of a block lies within a band parallelization group
		  // do computations for that block within the band group, otherwise skip that
		  // block. This is only activated if NPBAND>1
		  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
		  (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		  {
		    const char transA = 'N',transB = 'N';
		    const dataTypes::numberLowPrec scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		    std::fill(rotationMatBlock.begin(),rotationMatBlock.end(),0.);
                    std::fill(diagValuesBlock.begin(),diagValuesBlock.end(),0.);
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
				    {
				      rotationMatBlock[i*BVec+j]=
					rotationMatPar.local_el(localRowId,
								it->second);

				    }
				}

				if (i>=jvec && i<(jvec+BVec))
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				      globalToLocalColumnIdMap.find(i);
				  if (it!=globalToLocalColumnIdMap.end())
				  {
                                    rotationMatBlock[i*BVec+i-jvec]=0.0;
				    diagValuesBlock[i-jvec]=rotationMatPar.local_el(localRowId,
								                    it->second);
				  }
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
				    {
				      rotationMatBlock[i*BVec+j]=
					rotationMatPar.local_el(it->second,
								localColumnId);
				    }
				  }

				  if (i>=jvec && i<(jvec+BVec))
				  {
				    std::map<unsigned int, unsigned int>::iterator it=
				      globalToLocalRowIdMap.find(i);
				    if (globalToLocalRowIdMap.find(i)!=globalToLocalRowIdMap.end())
				    {
                                      rotationMatBlock[i*BVec+i-jvec]=0.0;
				      diagValuesBlock[i-jvec]
					=rotationMatPar.local_el(it->second,
								 localColumnId);
				    }
				  }
			      }
		      }

		      MPI_Barrier(mpiComm);
		      MPI_Allreduce(MPI_IN_PLACE,
				    &rotationMatBlock[0],
				    BVec*D,
				    dataTypes::mpi_type_id(&rotationMatBlock[0]),
				    MPI_SUM,
				    mpiComm);

		      MPI_Allreduce(MPI_IN_PLACE,
				    &diagValuesBlock[0],
				    BVec,
				    dataTypes::mpi_type_id(&diagValuesBlock[0]),
				    MPI_SUM,
				    mpiComm);

		      if (BDof!=0)
		      {

			  sgemm_(&transA,
				 &transB,
				 &BVec,
				 &BDof,
				 &D,
				 &scalarCoeffAlpha,
				 &rotationMatBlock[0],
				 &BVec,
				 &subspaceVectorsArraySinglePrec[0]+idof*N,
				 &N,
				 &scalarCoeffBeta,
				 &rotatedVectorsMatBlockTemp[0],
				 &BVec);

			  for (unsigned int i = 0; i <BDof; ++i)
			      for (unsigned int j = 0; j <BVec; ++j)
				  *(subspaceVectorsArray+N*(idof+i)+j+jvec)
				     = *(subspaceVectorsArray+N*(idof+i)+j+jvec)*diagValuesBlock[j]
				       +rotatedVectorsMatBlockTemp[i*BVec+j];
		      }

		  }// band parallelization
		  else
		  {
			  for (unsigned int i = 0; i <BDof; ++i)
			      for (unsigned int j = 0; j <BVec; ++j)
				  *(subspaceVectorsArray+N*(idof+i)+j+jvec)
				     =0.0;
		  }
	      }//block loop over vectors
	  }//block loop over dofs

	  if (numberBandGroups>1)
  	  {
                MPI_Barrier(interBandGroupComm);
		const unsigned int blockSize=dftParameters::mpiAllReduceMessageBlockSizeMB*1e+6
		                                /sizeof(dataTypes::number);

		for (unsigned int i=0; i<N*numLocalDofs;i+=blockSize)
		{
		   const unsigned int currentBlockSize=std::min(blockSize,N*numLocalDofs-i);

		   MPI_Allreduce(MPI_IN_PLACE,
				 subspaceVectorsArray+i,
				 currentBlockSize,
				 dataTypes::mpi_type_id(subspaceVectorsArray),
				 MPI_SUM,
				 interBandGroupComm);
		}
	  }
#endif
      }

      template
      void createGlobalToLocalIdMapsScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
						 const dealii::ScaLAPACKMatrix<dataTypes::number> & mat,
						 std::map<unsigned int, unsigned int> & globalToLocalRowIdMap,
						 std::map<unsigned int, unsigned int> & globalToLocalColumnIdMap);

      template
      void fillParallelOverlapMatrix(const dataTypes::number* X,
	                             const unsigned int XLocalSize,
				     const unsigned int numberVectors,
				     const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				     const MPI_Comm &interBandGroupComm,
				     const MPI_Comm &mpiComm,
				     dealii::ScaLAPACKMatrix<dataTypes::number> & overlapMatPar);

      template
      void subspaceRotation(dataTypes::number* subspaceVectorsArray,
	                    const unsigned int subspaceVectorsArrayLocalSize,
			    const unsigned int N,
			    const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			    const MPI_Comm &interBandGroupComm,
			    const MPI_Comm &mpiComm,
			    const dealii::ScaLAPACKMatrix<dataTypes::number> & rotationMatPar,
			    const bool rotationMatTranpose,
			    const bool isRotationMatLowerTria);


      template
      void scaleScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			     dealii::ScaLAPACKMatrix<dataTypes::number> & mat,
			     const dataTypes::number scalar);

      template
      void sumAcrossInterCommScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
					  dealii::ScaLAPACKMatrix<dataTypes::number> & mat,
					  const MPI_Comm &interComm);

      template
      void subspaceRotationSpectrumSplit(const dataTypes::number* X,
	                      dataTypes::number* Y,
		              const unsigned int subspaceVectorsArrayLocalSize,
		              const unsigned int N,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const unsigned int numberTopVectors,
			      const MPI_Comm &interBandGroupComm,
			      const MPI_Comm &mpiComm,
			      const dealii::ScaLAPACKMatrix<dataTypes::number> & QMat,
			      const bool QMatTranspose);


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
