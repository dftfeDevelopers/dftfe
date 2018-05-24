// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri

#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>


namespace dftfe{

  //
  // Constructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolver::chebyshevOrthogonalizedSubspaceIterationSolver(double lowerBoundWantedSpectrum,
												 double lowerBoundUnWantedSpectrum):
    d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum),
    d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    computing_timer(pcout,
		    dftParameters::reproducible_output ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
		    dealii::TimerOutput::wall_times)
  {

  }

  //
  // Destructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolver::~chebyshevOrthogonalizedSubspaceIterationSolver()
  {

    //
    //
    //
    return;

  }

  //
  //reinitialize spectrum bounds
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolver::reinitSpectrumBounds(double lowerBoundWantedSpectrum,
								       double lowerBoundUnWantedSpectrum)
  {
    d_lowerBoundWantedSpectrum = lowerBoundWantedSpectrum;
    d_lowerBoundUnWantedSpectrum = lowerBoundUnWantedSpectrum;
  }

  //
  // initialize direction
  //
  eigenSolverClass::ReturnValueType
  chebyshevOrthogonalizedSubspaceIterationSolver::solve(operatorDFTClass           & operatorMatrix,
							std::vector<vectorType>    & eigenVectors,
							std::vector<double>        & eigenValues,
							std::vector<double>        & residualNorms)
  {


    computing_timer.enter_section("Lanczos k-step Upper Bound");
    operatorMatrix.reinit(1);
    double upperBoundUnwantedSpectrum = linearAlgebraOperations::lanczosUpperBoundEigenSpectrum(operatorMatrix,
												eigenVectors[0]);

    computing_timer.exit_section("Lanczos k-step Upper Bound");

    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

    const unsigned int totalNumberWaveFunctions = eigenVectors.size();

    //
    //set Chebyshev order
    //
    if(chebyshevOrder == 0)
      {
	if(upperBoundUnwantedSpectrum <= 500)
	  chebyshevOrder = 40;
	else if(upperBoundUnwantedSpectrum > 500  && upperBoundUnwantedSpectrum <= 1000)
	  chebyshevOrder = 50;
	else if(upperBoundUnwantedSpectrum > 1000 && upperBoundUnwantedSpectrum <= 2000)
	  chebyshevOrder = 80;
	else if(upperBoundUnwantedSpectrum > 2000 && upperBoundUnwantedSpectrum <= 5000)
	  chebyshevOrder = 150;
	else if(upperBoundUnwantedSpectrum > 5000 && upperBoundUnwantedSpectrum <= 9000)
	  chebyshevOrder = 200;
	else if(upperBoundUnwantedSpectrum > 9000 && upperBoundUnwantedSpectrum <= 14000)
	  chebyshevOrder = 250;
	else if(upperBoundUnwantedSpectrum > 14000 && upperBoundUnwantedSpectrum <= 20000)
	  chebyshevOrder = 300;
	else if(upperBoundUnwantedSpectrum > 20000 && upperBoundUnwantedSpectrum <= 30000)
	  chebyshevOrder = 350;
	else if(upperBoundUnwantedSpectrum > 30000 && upperBoundUnwantedSpectrum <= 50000)
	  chebyshevOrder = 450;
	else if(upperBoundUnwantedSpectrum > 50000 && upperBoundUnwantedSpectrum <= 80000)
	  chebyshevOrder = 550;
	else if(upperBoundUnwantedSpectrum > 80000 && upperBoundUnwantedSpectrum <= 1e5)
	  chebyshevOrder = 800;
	else if(upperBoundUnwantedSpectrum > 1e5 && upperBoundUnwantedSpectrum <= 2e5)
	  chebyshevOrder = 1000;
	else if(upperBoundUnwantedSpectrum > 2e5 && upperBoundUnwantedSpectrum <= 5e5)
	  chebyshevOrder = 1250;
	else if(upperBoundUnwantedSpectrum > 5e5)
	  chebyshevOrder = 1500;
      }


    //
    //output statements
    //
    if (dftParameters::verbosity>=2)
      {
	char buffer[100];

	sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", upperBoundUnwantedSpectrum);
	pcout << buffer;
	sprintf(buffer, "%s:%18.10e\n", "lower bound of unwanted spectrum", d_lowerBoundUnWantedSpectrum);
	pcout << buffer;
	sprintf(buffer, "%s: %u\n\n", "Chebyshev polynomial degree", chebyshevOrder);
	pcout << buffer;
      }


    //
    //Set the constraints to zero
    //
    for(unsigned int i = 0; i < totalNumberWaveFunctions; ++i)
      operatorMatrix.getConstraintMatrixEigen()->set_zero(eigenVectors[i]);


     if(dftParameters::verbosity >= 4)
       {
	 PetscLogDouble bytes;
	 PetscMemoryGetCurrentUsage(&bytes);
	 FILE *dummy;
	 unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());
	 PetscSynchronizedPrintf(operatorMatrix.getMPICommunicator(),"[%d] Memory Usage before starting eigen solution  %e\n",this_mpi_process,bytes);
	 PetscSynchronizedFlush(operatorMatrix.getMPICommunicator(),dummy);
       }

    if((dftParameters::nkx*dftParameters::nky*dftParameters::nkz) == 1 && (dftParameters::dkx*dftParameters::dky*dftParameters::dkz) <= 1e-10)
      {
	//
	//create custom partitioned dealii array by storing wave functions at a given node contiguously
	//
	computing_timer.enter_section("Custom Partitioned Array Creation");
#ifdef USE_COMPLEX
	const unsigned int localVectorSize = eigenVectors[0].local_size()/2;
	dealii::parallel::distributed::Vector<std::complex<double> > eigenVectorsFlattenedArray;

	vectorTools::createDealiiVector<std::complex<double> >(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
							       totalNumberWaveFunctions,
							       eigenVectorsFlattenedArray);
#else
	const unsigned int localVectorSize = eigenVectors[0].local_size();
	dealii::parallel::distributed::Vector<double> eigenVectorsFlattenedArray;
	vectorTools::createDealiiVector<double>(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
						totalNumberWaveFunctions,
						eigenVectorsFlattenedArray);
#endif
	computing_timer.exit_section("Custom Partitioned Array Creation");

	if(dftParameters::verbosity >= 3)
	  {
	    pcout<<"Custom Partioned Array Creation Done: "<<std::endl;
	    pcout<<std::endl;
	  }

	//
	//copy the data from eigenVectors to eigenVectorsFlattened
	//
	computing_timer.enter_section("Copy to flattened array");
	for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	  {
	    for (unsigned int iWave = 0; iWave < totalNumberWaveFunctions; ++iWave)
	      {
		//unsigned int flattenedArrayGlobalIndex = (totalNumberWaveFunctions*(iNode + (operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->local_range()).first) + iWave);
		//unsigned int flattenedArrayLocalIndex = flattenedArrayGlobalIndex - eigenVectorsFlattenedArray.get_partitioner()->local_range().first;
		unsigned int flattenedArrayLocalIndex = totalNumberWaveFunctions*iNode + iWave;
#ifdef USE_COMPLEX
		eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).real(eigenVectors[iWave].local_element((*operatorMatrix.getLocalProcDofIndicesReal())[iNode]));
		eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).imag(eigenVectors[iWave].local_element((*operatorMatrix.getLocalProcDofIndicesImag())[iNode]));
#else
		eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex) = eigenVectors[iWave].local_element(iNode);
#endif
	      }
	  }
	computing_timer.exit_section("Copy to flattened array");

	//
	//Free the memory of eigenVectors array
	//
	vectorType d_tempDealiiVector;
	d_tempDealiiVector.reinit(eigenVectors[0]);
	for(unsigned int iWave = 0; iWave < totalNumberWaveFunctions; ++iWave)
	  eigenVectors[iWave].reinit(0);


	if(dftParameters::verbosity >= 4)
	  {
	    PetscLogDouble bytes;
	    PetscMemoryGetCurrentUsage(&bytes);
	    FILE *dummy;
	    unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());
	    PetscSynchronizedPrintf(operatorMatrix.getMPICommunicator(),"[%d] Memory after creating eigen vector flattened and freeing STL memory   %e\n",this_mpi_process,bytes);
	    PetscSynchronizedFlush(operatorMatrix.getMPICommunicator(),dummy);
	  }



	//
	//Split the complete wavefunctions into multiple blocks.
	//Create the size of vectors in each block
	//
	const unsigned int equalNumberWaveFunctionsPerBlock = std::min(totalNumberWaveFunctions,dftParameters::chebyshevBlockSize);
	const double temp = (double)totalNumberWaveFunctions/(double)equalNumberWaveFunctionsPerBlock;
	const unsigned int totalNumberBlocks = std::ceil(temp);
	const unsigned int numberWaveFunctionsLastBlock = totalNumberWaveFunctions - equalNumberWaveFunctionsPerBlock*(totalNumberBlocks-1);

	std::vector<unsigned int> d_numberWaveFunctionsBlock(totalNumberBlocks,equalNumberWaveFunctionsPerBlock);

	if(totalNumberBlocks > 1)
	  d_numberWaveFunctionsBlock[totalNumberBlocks - 1] = numberWaveFunctionsLastBlock;


	std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayCellLocalProcIndexIdMap,flattenedArrayMacroCellLocalProcIndexIdMap;
#ifdef USE_COMPLEX
	dealii::parallel::distributed::Vector<std::complex<double> > eigenVectorsFlattenedArrayBlock;
#else
	dealii::parallel::distributed::Vector<double> eigenVectorsFlattenedArrayBlock;
#endif
	for(unsigned int nBlock = 0; nBlock < totalNumberBlocks; ++nBlock)
	  {
	    //
	    //Get the current block data
	    //
	    const unsigned int numberWaveFunctionsPerCurrentBlock = d_numberWaveFunctionsBlock[nBlock];
	    const unsigned int lowIndex = equalNumberWaveFunctionsPerBlock*nBlock;
	    const unsigned int highIndexPlusOne = lowIndex + numberWaveFunctionsPerCurrentBlock;

	    //
	    //create custom partitioned dealii array by storing wavefunctions(no need to createDealii vector if block size does not change)
	    //

	    if(totalNumberBlocks > 1)
	      {

		if (nBlock==0 || nBlock==totalNumberBlocks-1)
		{
		    operatorMatrix.reinit(numberWaveFunctionsPerCurrentBlock);
#ifdef USE_COMPLEX
		    vectorTools::createDealiiVector<std::complex<double> >(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
									   numberWaveFunctionsPerCurrentBlock,
									   eigenVectorsFlattenedArrayBlock);
#else
		    vectorTools::createDealiiVector<double>(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
							    numberWaveFunctionsPerCurrentBlock,
							    eigenVectorsFlattenedArrayBlock);

#endif


		    //
		    //precompute certain maps
		    //
		    vectorTools::computeCellLocalIndexSetMap(eigenVectorsFlattenedArrayBlock.get_partitioner(),
							     operatorMatrix.getMatrixFreeData(),
							     numberWaveFunctionsPerCurrentBlock,
							     flattenedArrayMacroCellLocalProcIndexIdMap,
							     flattenedArrayCellLocalProcIndexIdMap);


		    operatorMatrix.getOverloadedConstraintMatrix()
			->precomputeMaps(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
			  	         eigenVectorsFlattenedArrayBlock.get_partitioner(),
			 	         numberWaveFunctionsPerCurrentBlock);
		}


		//
		//fill the eigenVectorsFlattenedArrayBlock from eigenVectorsFlattenedArray(to be coded)
		//

		computing_timer.enter_section("Copy from full to block flattened array");
		for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		    for(unsigned int iWave = 0; iWave < numberWaveFunctionsPerCurrentBlock; ++iWave)
                        eigenVectorsFlattenedArrayBlock.local_element(iNode*numberWaveFunctionsPerCurrentBlock
				 +iWave)
			     =eigenVectorsFlattenedArray.local_element(iNode*totalNumberWaveFunctions+lowIndex+iWave);

	        computing_timer.exit_section("Copy from full to block flattened array");

		//
		//call Chebyshev filtering function only for the current block to be filtered
		//and does in-place filtering
		computing_timer.enter_section("Chebyshev filtering opt");
		linearAlgebraOperations::chebyshevFilter(operatorMatrix,
							 eigenVectorsFlattenedArrayBlock,
							 numberWaveFunctionsPerCurrentBlock,
							 flattenedArrayMacroCellLocalProcIndexIdMap,
							 flattenedArrayCellLocalProcIndexIdMap,
							 chebyshevOrder,
							 d_lowerBoundUnWantedSpectrum,
							 upperBoundUnwantedSpectrum,
							 d_lowerBoundWantedSpectrum);
		computing_timer.exit_section("Chebyshev filtering opt");

		//
		//copy the eigenVectorsFlattenedArrayBlock into eigenVectorsFlattenedArray after filtering(to be coded)
		//

		computing_timer.enter_section("Copy from block to full flattened array");
		for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		    for(unsigned int iWave = 0; iWave < numberWaveFunctionsPerCurrentBlock; ++iWave)
			  eigenVectorsFlattenedArray.local_element(iNode*totalNumberWaveFunctions+lowIndex+iWave)
			  = eigenVectorsFlattenedArrayBlock.local_element(iNode*numberWaveFunctionsPerCurrentBlock
				 +iWave);

	        computing_timer.exit_section("Copy from block to full flattened array");
	      }
	    else
	      {
		operatorMatrix.reinit(numberWaveFunctionsPerCurrentBlock);
		//
		//precompute certain maps
		//
		vectorTools::computeCellLocalIndexSetMap(eigenVectorsFlattenedArray.get_partitioner(),
							 operatorMatrix.getMatrixFreeData(),
							 numberWaveFunctionsPerCurrentBlock,
							 flattenedArrayMacroCellLocalProcIndexIdMap,
							 flattenedArrayCellLocalProcIndexIdMap);


		if(dftParameters::verbosity >= 3)
		  {
		    pcout<<"Cell Local Index Set Map Done: "<<std::endl;
		    pcout<<std::endl;
		  }


		operatorMatrix.getOverloadedConstraintMatrix()->precomputeMaps(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
									       eigenVectorsFlattenedArray.get_partitioner(),
									       numberWaveFunctionsPerCurrentBlock);



		if(dftParameters::verbosity >= 3)
		  {
		    pcout<<"Precomputing Maps for Constraint Matrix Distribute functions: "<<std::endl;
		    pcout<<std::endl;
		  }


		//
		//call Chebyshev filtering function only for the current block to be filtered
		//and does in-place filtering
		computing_timer.enter_section("Chebyshev filtering opt");
		linearAlgebraOperations::chebyshevFilter(operatorMatrix,
							 eigenVectorsFlattenedArray,
							 numberWaveFunctionsPerCurrentBlock,
							 flattenedArrayMacroCellLocalProcIndexIdMap,
							 flattenedArrayCellLocalProcIndexIdMap,
							 chebyshevOrder,
							 d_lowerBoundUnWantedSpectrum,
							 upperBoundUnwantedSpectrum,
							 d_lowerBoundWantedSpectrum);
		computing_timer.exit_section("Chebyshev filtering opt");

	      }

	  }//block loop

	eigenVectorsFlattenedArrayBlock.reinit(0);
	if(dftParameters::verbosity >= 2)
	  pcout<<"ChebyShev Filtering Done: "<<std::endl;

	if(dftParameters::orthogType.compare("lowden") == 0)
	  {
	    computing_timer.enter_section("Lowden Orthogn Opt");
	    linearAlgebraOperations::lowdenOrthogonalization(eigenVectorsFlattenedArray,
							     totalNumberWaveFunctions);
	    computing_timer.exit_section("Lowden Orthogn Opt");
	  }
	else
	  {
	    computing_timer.enter_section("Gram-Schmidt Orthogn Opt");
	    linearAlgebraOperations::gramSchmidtOrthogonalization(eigenVectorsFlattenedArray,
								  totalNumberWaveFunctions);
	    computing_timer.exit_section("Gram-Schmidt Orthogn Opt");
	  }

	if(dftParameters::verbosity >= 2)
	  pcout<<"Orthogonalization Done: "<<std::endl;

	computing_timer.enter_section("Rayleigh-Ritz proj Opt");
	operatorMatrix.reinit(totalNumberWaveFunctions);

	//
	//precompute certain maps
	//
	vectorTools::computeCellLocalIndexSetMap(eigenVectorsFlattenedArray.get_partitioner(),
						 operatorMatrix.getMatrixFreeData(),
						 totalNumberWaveFunctions,
						 flattenedArrayMacroCellLocalProcIndexIdMap,
						 flattenedArrayCellLocalProcIndexIdMap);


	operatorMatrix.getOverloadedConstraintMatrix()
	    ->precomputeMaps(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
	  	             eigenVectorsFlattenedArray.get_partitioner(),
			     totalNumberWaveFunctions);

	linearAlgebraOperations::rayleighRitz(operatorMatrix,
					      eigenVectorsFlattenedArray,
					      totalNumberWaveFunctions,
					      flattenedArrayMacroCellLocalProcIndexIdMap,
					      flattenedArrayCellLocalProcIndexIdMap,
					      eigenValues);
	computing_timer.exit_section("Rayleigh-Ritz proj Opt");

	if(dftParameters::verbosity >= 2)
	  {
	    pcout<<"Rayleigh-Ritz Done: "<<std::endl;
	    pcout<<std::endl;
	  }

	computing_timer.enter_section("eigen vectors residuals opt");
	linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
							  eigenVectorsFlattenedArray,
							  eigenValues,
							  flattenedArrayMacroCellLocalProcIndexIdMap,
							  flattenedArrayCellLocalProcIndexIdMap,
							  residualNorms);
	computing_timer.exit_section("eigen vectors residuals opt");

	if(dftParameters::verbosity >= 2)
	  {
	    pcout<<"EigenVector Residual Computation Done: "<<std::endl;
	    pcout<<std::endl;
	  }

	if(dftParameters::verbosity >= 4)
	  {
	    PetscLogDouble bytes;
	    PetscMemoryGetCurrentUsage(&bytes);
	    FILE *dummy;
	    unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());
	    PetscSynchronizedPrintf(operatorMatrix.getMPICommunicator(),"[%d] Memory after all steps of subspace iteration before recreating STL vector  %e\n",this_mpi_process,bytes);
	    PetscSynchronizedFlush(operatorMatrix.getMPICommunicator(),dummy);
	  }


	//
        //allocate back the memory of eigenVectors array
        //
	for(unsigned int iWave = 0; iWave < totalNumberWaveFunctions; ++iWave)
	  eigenVectors[iWave].reinit(d_tempDealiiVector);

	//
	//copy back to eigenVectors array from eigenVectors Flattened Array
	//
	computing_timer.enter_section("Copy to eigen vectors array");
	for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	  {
	    for(unsigned int iWave = 0; iWave < totalNumberWaveFunctions; ++iWave)
	      {
		//unsigned int flattenedArrayGlobalIndex = (totalNumberWaveFunctions*(iNode + (operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->local_range()).first) + iWave);
		//unsigned int flattenedArrayLocalIndex = flattenedArrayGlobalIndex - eigenVectorsFlattenedArray.get_partitioner()->local_range().first;
		unsigned int flattenedArrayLocalIndex = totalNumberWaveFunctions*iNode + iWave;
#ifdef USE_COMPLEX
		eigenVectors[iWave].local_element((*operatorMatrix.getLocalProcDofIndicesReal())[iNode]) = eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).real();
		eigenVectors[iWave].local_element((*operatorMatrix.getLocalProcDofIndicesImag())[iNode]) = eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).imag();
#else
		eigenVectors[iWave].local_element(iNode) = eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex);

#endif
	      }
	  }
	computing_timer.exit_section("Copy to eigen vectors array");





      }
    else
      {
	operatorMatrix.reinit(totalNumberWaveFunctions);
	//
	//call chebyshev filtering routine
	//
	computing_timer.enter_section("Chebyshev filtering");

	linearAlgebraOperations::chebyshevFilter(operatorMatrix,
						 eigenVectors,
						 chebyshevOrder,
						 d_lowerBoundUnWantedSpectrum,
						 upperBoundUnwantedSpectrum,
						 d_lowerBoundWantedSpectrum);

	computing_timer.exit_section("Chebyshev filtering");


	computing_timer.enter_section("Gram-Schmidt Orthogonalization");

	linearAlgebraOperations::gramSchmidtOrthogonalization(operatorMatrix,
							      eigenVectors);


	computing_timer.exit_section("Gram-Schmidt Orthogonalization");


	computing_timer.enter_section("Rayleigh Ritz Projection");

	linearAlgebraOperations::rayleighRitz(operatorMatrix,
					      eigenVectors,
					      eigenValues);

	computing_timer.exit_section("Rayleigh Ritz Projection");


	computing_timer.enter_section("compute eigen vectors residuals");
	linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
							  eigenVectors,
							  eigenValues,
							  residualNorms);
	computing_timer.exit_section("compute eigen vectors residuals");

      }
    //
    //
    return;

  }

}
