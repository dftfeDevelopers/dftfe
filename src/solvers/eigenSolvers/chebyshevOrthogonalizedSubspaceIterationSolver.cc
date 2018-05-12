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
							std::vector<double>        & eigenValues)
  {


    computing_timer.enter_section("Lanczos k-step Upper Bound");
    double upperBoundUnwantedSpectrum = linearAlgebraOperations::lanczosUpperBoundEigenSpectrum(operatorMatrix,
												eigenVectors[0]);

    computing_timer.exit_section("Lanczos k-step Upper Bound");

    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

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
    if (dftParameters::verbosity>=1)
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
    for(unsigned int i = 0; i < eigenVectors.size(); ++i)
      operatorMatrix.getConstraintMatrixEigen()->set_zero(eigenVectors[i]);

    //
    //Split the complete wavefunctions into multiple blocks.
    //Create the size of vectors in each block
    //
    const unsigned int totalNumberWaveFunctions = eigenVectors.size();
    const unsigned int equalNumberWaveFunctionsPerBlock = totalNumberWaveFunctions; //1000;
    const double temp = (double)totalNumberWaveFunctions/(double)equalNumberWaveFunctionsPerBlock;
    const unsigned int totalNumberBlocks = std::ceil(temp);
    const unsigned int numberWaveFunctionsLastBlock = totalNumberWaveFunctions - equalNumberWaveFunctionsPerBlock*(totalNumberBlocks-1);

    std::vector<unsigned int> d_numberWaveFunctionsBlock(totalNumberBlocks,equalNumberWaveFunctionsPerBlock);

    if(totalNumberBlocks > 1)
      d_numberWaveFunctionsBlock[totalNumberBlocks - 1] = numberWaveFunctionsLastBlock;

    if((dftParameters::nkx*dftParameters::nky*dftParameters::nkz) == 1 && (dftParameters::dkx*dftParameters::dky*dftParameters::dkz) <= 1e-10)
      {
	for(unsigned int nBlock = 0; nBlock < totalNumberBlocks; ++nBlock)
	  {
	    //
	    //Get the current block data
	    //
	    unsigned int numberWaveFunctionsPerCurrentBlock = d_numberWaveFunctionsBlock[nBlock];
	    unsigned int lowIndex = equalNumberWaveFunctionsPerBlock*nBlock;
	    unsigned int highIndexPlusOne = lowIndex + numberWaveFunctionsPerCurrentBlock;

	    //
	    //create custom partitioned dealii array by storing wavefunctions
	    //
#ifdef USE_COMPLEX
	    unsigned int localVectorSize = eigenVectors[0].local_size()/2;
	    dealii::parallel::distributed::Vector<std::complex<double> > eigenVectorsFlattenedArray;

	    vectorTools::createDealiiVector<std::complex<double> >(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
								   operatorMatrix.getMPICommunicator(),
								   operatorMatrix.getMatrixFreeData()->get_dof_handler().n_dofs(),
								   numberWaveFunctionsPerCurrentBlock,
								   eigenVectorsFlattenedArray);
#else
	    unsigned int localVectorSize = eigenVectors[0].local_size();
	    dealii::parallel::distributed::Vector<double> eigenVectorsFlattenedArray;
	    vectorTools::createDealiiVector<double>(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
						    operatorMatrix.getMPICommunicator(),
						    operatorMatrix.getMatrixFreeData()->get_dof_handler().n_dofs(),
						    numberWaveFunctionsPerCurrentBlock,
						    eigenVectorsFlattenedArray);
#endif

	    //
	    //precompute certain maps
	    //
	    std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayCellLocalProcIndexIdMap,flattenedArrayMacroCellLocalProcIndexIdMap;
	    vectorTools::computeCellLocalIndexSetMap(eigenVectorsFlattenedArray.get_partitioner(),
						     operatorMatrix.getMatrixFreeData(),
						     numberWaveFunctionsPerCurrentBlock,
						     flattenedArrayMacroCellLocalProcIndexIdMap,
						     flattenedArrayCellLocalProcIndexIdMap);


	    operatorMatrix.getOverloadedConstraintMatrix()->precomputeMaps(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
									   eigenVectorsFlattenedArray.get_partitioner(),
									   numberWaveFunctionsPerCurrentBlock);

	    //
	    //copy the data from eigenVectors to eigenVectorsFlattened (this may have to be changed from flattened
	    //to flattened array containing only block vectors eventually)
	    //
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	      {
		for (unsigned int iWave = 0; iWave < numberWaveFunctionsPerCurrentBlock; ++iWave)
		  {
		    unsigned int flattenedArrayGlobalIndex = (numberWaveFunctionsPerCurrentBlock*(iNode + (operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->local_range()).first) + iWave);
		    unsigned int flattenedArrayLocalIndex = flattenedArrayGlobalIndex - eigenVectorsFlattenedArray.get_partitioner()->local_range().first;
#ifdef USE_COMPLEX
		    eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).real(eigenVectors[iWave+lowIndex].local_element((*operatorMatrix.getLocalProcDofIndicesReal())[iNode]));
		    eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).imag(eigenVectors[iWave+lowIndex].local_element((*operatorMatrix.getLocalProcDofIndicesImag())[iNode]));
#else
		    eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex) = eigenVectors[iWave+lowIndex].local_element(iNode);
#endif
		  }
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


	    //
	    //copy back to eigenVectors array
	    //
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	      {
		for(unsigned int iWave = 0; iWave < numberWaveFunctionsPerCurrentBlock; ++iWave)
		  {
		    unsigned int flattenedArrayGlobalIndex = (numberWaveFunctionsPerCurrentBlock*(iNode + (operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->local_range()).first) + iWave);
		    unsigned int flattenedArrayLocalIndex = flattenedArrayGlobalIndex - eigenVectorsFlattenedArray.get_partitioner()->local_range().first;
#ifdef USE_COMPLEX
		    eigenVectors[iWave+lowIndex].local_element((*operatorMatrix.getLocalProcDofIndicesReal())[iNode]) = eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).real();
		    eigenVectors[iWave+lowIndex].local_element((*operatorMatrix.getLocalProcDofIndicesImag())[iNode]) = eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex).imag();
#else
		    eigenVectors[iWave+lowIndex].local_element(iNode) = eigenVectorsFlattenedArray.local_element(flattenedArrayLocalIndex);
#endif
		  }
	      }

	  }//block loop
      }
    else
      {
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
      }



    computing_timer.enter_section("Gram-Schmidt Orthogonalization");

    linearAlgebraOperations::gramSchmidtOrthogonalization(operatorMatrix,
							  eigenVectors);


    computing_timer.exit_section("Gram-Schmidt Orthogonalization");


    computing_timer.enter_section("Rayleigh Ritz Projection");

    linearAlgebraOperations::rayleighRitz(operatorMatrix,
					  eigenVectors,
					  eigenValues);


    computing_timer.exit_section("Rayleigh Ritz Projection");



    //
    //
    return;

  }

}
