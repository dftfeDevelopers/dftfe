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
	void fillParallelOverlapMatrix(const dealii::parallel::distributed::Vector<T> & X,
		                       const unsigned int numberVectors,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       const MPI_Comm &interBandGroupComm,
				       dealii::ScaLAPACKMatrix<T> & overlapMatPar)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = X.local_size()/numberVectors;

          //band group parallelization data structures
          const unsigned int numberBandGroups=
	     dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
          const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
          std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
          dftUtils::createBandParallelizationIndices(interBandGroupComm,
						     numberVectors,
						     bandGroupLowHighPlusOneIndices);

          //get global to local index maps for Scalapack matrix
	  std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	  std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	  internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
		                                          overlapMatPar,
				                          globalToLocalRowIdMap,
					                  globalToLocalColumnIdMap);

	  const unsigned int vectorsBlockSize=std::min(dftParameters::orthoRRWaveFuncBlockSize,
	                                               bandGroupLowHighPlusOneIndices[1]);

	  std::vector<T> overlapMatrixBlock(numberVectors*vectorsBlockSize,0.0);
	  std::vector<T> blockVectorsMatrix(numLocalDofs*vectorsBlockSize,0.0);

	  for (unsigned int ivec = 0; ivec < numberVectors; ivec += vectorsBlockSize)
	  {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      const unsigned int B = std::min(vectorsBlockSize, numberVectors-ivec);

	      if ((ivec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (ivec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	      {
		  const char transA = 'N',transB = 'N';
		  const T scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		  std::fill(overlapMatrixBlock.begin(),overlapMatrixBlock.end(),0.);

		  for (unsigned int i = 0; i <numLocalDofs; ++i)
		      for (unsigned int j = 0; j <B; ++j)
			  blockVectorsMatrix[j*numLocalDofs+i]=X.local_element(numberVectors*i+j+ivec);

		  dgemm_(&transA,
			 &transB,
			 &numberVectors,
			 &B,
			 &numLocalDofs,
			 &scalarCoeffAlpha,
			 X.begin(),
			 &numberVectors,
			 &blockVectorsMatrix[0],
			 &numLocalDofs,
			 &scalarCoeffBeta,
			 &overlapMatrixBlock[0],
			 &numberVectors);

		  dealii::Utilities::MPI::sum(overlapMatrixBlock,
					      X.get_mpi_communicator(),
					      overlapMatrixBlock);


		  if (processGrid->is_process_active())
		      for(unsigned int i = 0; i <B; ++i)
			  if (globalToLocalColumnIdMap.find(i+ivec)!=globalToLocalColumnIdMap.end())
			  {
			      const unsigned int localColumnId=globalToLocalColumnIdMap[i+ivec];
			      for (unsigned int j = 0; j <numberVectors; ++j)
			      {
				 std::map<unsigned int, unsigned int>::iterator it=
					      globalToLocalRowIdMap.find(j);
				 if(it!=globalToLocalRowIdMap.end())
				     overlapMatPar.local_el(it->second,
							    localColumnId)
							    =overlapMatrixBlock[i*numberVectors+j];
			      }
			  }
		}//band parallelization
	  }//block loop

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat
	                                          (processGrid,
						   overlapMatPar,
						   interBandGroupComm);
#endif
	}


	template<typename T>
	void subspaceRotation(dealii::parallel::distributed::Vector<T> & subspaceVectorsArray,
		              const unsigned int numberSubspaceVectors,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const MPI_Comm &interBandGroupComm,
			      const dealii::ScaLAPACKMatrix<T> & rotationMatPar,
			      const bool rotationMatTranspose)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = subspaceVectorsArray.local_size()/numberSubspaceVectors;

	  const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
						  subspaceVectorsArray.get_mpi_communicator());

          //band group parallelization data structures
          const unsigned int numberBandGroups=
	     dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
          const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
          std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
          dftUtils::createBandParallelizationIndices(interBandGroupComm,
						     numberSubspaceVectors,
						     bandGroupLowHighPlusOneIndices);

	  std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	  std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	  internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
		                                          rotationMatPar,
				                          globalToLocalRowIdMap,
					                  globalToLocalColumnIdMap);


	  const unsigned int vectorsBlockSize=std::min(dftParameters::orthoRRWaveFuncBlockSize,
	                                               bandGroupLowHighPlusOneIndices[1]);
	  const unsigned int dofsBlockSize=dftParameters::subspaceRotDofsBlockSize;

	  std::vector<T> rotationMatBlock(vectorsBlockSize*numberSubspaceVectors,0.0);
	  std::vector<T> rotatedVectorsMatBlock(numberSubspaceVectors*dofsBlockSize,0.0);
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
	      for (unsigned int jvec = 0; jvec < numberSubspaceVectors; jvec += vectorsBlockSize)
	      {
		  // Correct block dimensions if block "goes off edge of" the matrix
		  const unsigned int BVec = std::min(vectorsBlockSize, numberSubspaceVectors-jvec);


		  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
		  (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		  {
		      const char transA = 'N',transB = 'N';
		      const T scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		      std::fill(rotationMatBlock.begin(),rotationMatBlock.end(),0.);

		      if (rotationMatTranspose)
		      {
			  if (processGrid->is_process_active())
			      for (unsigned int i = 0; i <numberSubspaceVectors; ++i)
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
			      for (unsigned int i = 0; i <numberSubspaceVectors; ++i)
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

		      dealii::Utilities::MPI::sum(rotationMatBlock,
						  subspaceVectorsArray.get_mpi_communicator(),
						  rotationMatBlock);
		      if (BDof!=0)
		      {

			  dgemm_(&transA,
				 &transB,
				 &BVec,
				 &BDof,
				 &numberSubspaceVectors,
				 &scalarCoeffAlpha,
				 &rotationMatBlock[0],
				 &BVec,
				 subspaceVectorsArray.begin()+idof*numberSubspaceVectors,
				 &numberSubspaceVectors,
				 &scalarCoeffBeta,
				 &rotatedVectorsMatBlockTemp[0],
				 &BVec);

			  for (unsigned int i = 0; i <BDof; ++i)
			      for (unsigned int j = 0; j <BVec; ++j)
				  rotatedVectorsMatBlock[numberSubspaceVectors*i+j+jvec]
				      =rotatedVectorsMatBlockTemp[i*BVec+j];
		      }

		  }// band parallelization
	      }//block loop over vectors

	      if (BDof!=0)
	      {
		  for (unsigned int i = 0; i <BDof; ++i)
		      for (unsigned int j = 0; j <numberSubspaceVectors; ++j)
			  subspaceVectorsArray.local_element(numberSubspaceVectors*(i+idof)+j)
			      =rotatedVectorsMatBlock[i*numberSubspaceVectors+j];
	      }
	  }//block loop over dofs

	  if (numberBandGroups>1)
  	  {
	    MPI_Allreduce(MPI_IN_PLACE,
			  subspaceVectorsArray.begin(),
			  numberSubspaceVectors*numLocalDofs,
			  MPI_DOUBLE,
			  MPI_SUM,
			  interBandGroupComm);
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
		                       const unsigned int numberVectors,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       const MPI_Comm &interBandGroupComm,
				       dealii::ScaLAPACKMatrix<dataTypes::number> & overlapMatPar);

	template
	void subspaceRotation(dealii::parallel::distributed::Vector<dataTypes::number> & subspaceVectorsArray,
		              const unsigned int numberSubspaceVectors,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const MPI_Comm &interBandGroupComm,
			      const dealii::ScaLAPACKMatrix<dataTypes::number> & rotationMatPar,
			      const bool rotationMatTranpose);

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
