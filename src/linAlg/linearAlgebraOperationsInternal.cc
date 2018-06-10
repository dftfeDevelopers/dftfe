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
#ifdef WITH_SCALAPACK
	void createProcessGridSquareMatrix(const MPI_Comm & mpi_communicator,
		                      const unsigned size,
		                      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				      const unsigned int rowsBlockSize)
	{
	      const unsigned int numberProcs = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
	      const unsigned int blocksPerProc=4;
	      const unsigned int rowProcs=std::min(std::floor(std::sqrt(numberProcs)),
				std::ceil((double)size/(double)(blocksPerProc*rowsBlockSize)));
	      if(dftParameters::verbosity>=2)
	      {
		 dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
		 pcout<<"Scalapack Matrix created, "<<"rowsBlockSize: "<<rowsBlockSize<<", blocksPerProc: "<<blocksPerProc<<", row procs: "<< rowProcs<<std::endl;
	      }

	      processGrid=std::make_shared<const dealii::Utilities::MPI::ProcessGrid>(mpi_communicator,
										      rowProcs,
										      rowProcs);
	}

	template<typename T>
	void fillParallelOverlapMatrix(const dealii::parallel::distributed::Vector<T> & X,
		                       const unsigned int numberVectors,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       dealii::ScaLAPACKMatrix<T> & overlapMatPar)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = X.local_size()/numberVectors;

	  std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	  std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	  if (processGrid->is_process_active())
	  {
	     for (unsigned int i = 0; i < overlapMatPar.local_m(); ++i)
		 globalToLocalRowIdMap[overlapMatPar.global_row(i)]=i;

	     for (unsigned int j = 0; j < overlapMatPar.local_n(); ++j)
		 globalToLocalColumnIdMap[overlapMatPar.global_column(j)]=j;

	  }

	  const unsigned int vectorsBlockSize=200;

	  std::vector<T> overlapMatrixBlock(numberVectors*vectorsBlockSize,0.0);
	  std::vector<T> blockVectorsMatrix(numLocalDofs*vectorsBlockSize,0.0);

	  for (unsigned int ivec = 0; ivec < numberVectors; ivec += vectorsBlockSize)
	  {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      const unsigned int B = std::min(vectorsBlockSize, numberVectors-ivec);
	      const char transA = 'N',transB = 'N';
	      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

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

	      for (unsigned int i = 0; i <numberVectors; ++i)
		  for (unsigned int j = 0; j <B; ++j)
		     if (globalToLocalRowIdMap.find(i)!=globalToLocalRowIdMap.end())
			 if(globalToLocalColumnIdMap.find(j+ivec)!=globalToLocalColumnIdMap.end())
			     overlapMatPar.local_el(globalToLocalRowIdMap[i], globalToLocalColumnIdMap[j+ivec])=overlapMatrixBlock[j*numberVectors+i];
	  }
#endif
	}


	template<typename T>
	void subspaceRotation(dealii::parallel::distributed::Vector<T> & subspaceVectorsArray,
		              const unsigned int numberSubspaceVectors,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const dealii::ScaLAPACKMatrix<T> & rotationMatPar)
	{
#ifdef USE_COMPLEX
	  AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
          const unsigned int numLocalDofs = subspaceVectorsArray.local_size()/numberSubspaceVectors;

	  const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(numLocalDofs,
						  subspaceVectorsArray.get_mpi_communicator());

	  std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	  std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	  if (processGrid->is_process_active())
	  {
	     for (unsigned int i = 0; i < rotationMatPar.local_m(); ++i)
		 globalToLocalRowIdMap[rotationMatPar.global_row(i)]=i;

	     for (unsigned int j = 0; j < rotationMatPar.local_n(); ++j)
		 globalToLocalColumnIdMap[rotationMatPar.global_column(j)]=j;

	  }


	  const unsigned int vectorsBlockSize=200;
	  const unsigned int dofsBlockSize=200;

	  std::vector<T> rotationMatBlock(vectorsBlockSize*numberSubspaceVectors,0.0);
	  std::vector<T> rotatedVectorsMatBlock(numberSubspaceVectors*dofsBlockSize,0.0);
          std::vector<T> rotatedVectorsMatBlockTemp(vectorsBlockSize*dofsBlockSize,0.0);

	  for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	  {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      unsigned int BDof=0;
	      if (numLocalDofs>=idof)
                 BDof = std::min(dofsBlockSize, numLocalDofs-idof);

	      for (unsigned int jvec = 0; jvec < numberSubspaceVectors; jvec += vectorsBlockSize)
	      {
		  // Correct block dimensions if block "goes off edge of" the matrix
		  const unsigned int BVec = std::min(vectorsBlockSize, numberSubspaceVectors-jvec);

		  const char transA = 'N',transB = 'N';
		  const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		  std::fill(rotationMatBlock.begin(),rotationMatBlock.end(),0.);

		  for (unsigned int i = 0; i <numberSubspaceVectors; ++i)
		      for (unsigned int j = 0; j <BVec; ++j)
			 if (globalToLocalRowIdMap.find(j+jvec)!=globalToLocalRowIdMap.end())
			     if(globalToLocalColumnIdMap.find(i)!=globalToLocalColumnIdMap.end())
				 rotationMatBlock[i*BVec+j]=rotationMatPar.local_el(globalToLocalRowIdMap[j+jvec], globalToLocalColumnIdMap[i]);

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

	      }// block loop over vectors

	      if (BDof!=0)
		  for (unsigned int i = 0; i <BDof; ++i)
		      for (unsigned int j = 0; j <numberSubspaceVectors; ++j)
			  subspaceVectorsArray.local_element(numberSubspaceVectors*(i+idof)+j)
			      =rotatedVectorsMatBlock[i*numberSubspaceVectors+j];
	  }//block loop over dofs
#endif
	}
#endif

#ifdef WITH_SCALAPACK
        template
	void fillParallelOverlapMatrix(const dealii::parallel::distributed::Vector<dataTypes::number> & X,
		                       const unsigned int numberVectors,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       dealii::ScaLAPACKMatrix<dataTypes::number> & overlapMatPar);

	template
	void subspaceRotation(dealii::parallel::distributed::Vector<dataTypes::number> & subspaceVectorsArray,
		              const unsigned int numberSubspaceVectors,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const dealii::ScaLAPACKMatrix<dataTypes::number> & rotationMatPar);

#endif
    }
  }
}
