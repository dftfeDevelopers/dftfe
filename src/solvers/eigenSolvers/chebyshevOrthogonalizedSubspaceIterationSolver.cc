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
// @author Phani Motamarri, Sambit Das

#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>
#include <dftUtils.h>


namespace dftfe{

  namespace internal
  {

      unsigned int setChebyshevOrder(const unsigned int upperBoundUnwantedSpectrum)
      {
	unsigned int chebyshevOrder;
	if(upperBoundUnwantedSpectrum <= 500)
	  chebyshevOrder = 30;
	else if(upperBoundUnwantedSpectrum > 500  && upperBoundUnwantedSpectrum <= 1000)
	  chebyshevOrder = 40;
	else if(upperBoundUnwantedSpectrum > 1000 && upperBoundUnwantedSpectrum <= 2000)
	  chebyshevOrder = 50;
	else if(upperBoundUnwantedSpectrum > 2000 && upperBoundUnwantedSpectrum <= 3000)
	  chebyshevOrder = 60;
	else if(upperBoundUnwantedSpectrum > 3000 && upperBoundUnwantedSpectrum <= 5000)
	  chebyshevOrder = 75;
	else if(upperBoundUnwantedSpectrum > 5000 && upperBoundUnwantedSpectrum <= 9000)
	  chebyshevOrder = 90;
	else if(upperBoundUnwantedSpectrum > 9000 && upperBoundUnwantedSpectrum <= 14000)
	  chebyshevOrder = 140;
	else if(upperBoundUnwantedSpectrum > 14000 && upperBoundUnwantedSpectrum <= 20000)
	  chebyshevOrder = 160;
	else if(upperBoundUnwantedSpectrum > 20000 && upperBoundUnwantedSpectrum <= 30000)
	  chebyshevOrder = 210;
	else if(upperBoundUnwantedSpectrum > 30000 && upperBoundUnwantedSpectrum <= 50000)
	  chebyshevOrder = 300;
	else if(upperBoundUnwantedSpectrum > 50000 && upperBoundUnwantedSpectrum <= 80000)
	  chebyshevOrder = 450;
	else if(upperBoundUnwantedSpectrum > 80000 && upperBoundUnwantedSpectrum <= 1e5)
	  chebyshevOrder = 550;
	else if(upperBoundUnwantedSpectrum > 1e5 && upperBoundUnwantedSpectrum <= 2e5)
	  chebyshevOrder = 700;
	else if(upperBoundUnwantedSpectrum > 2e5 && upperBoundUnwantedSpectrum <= 5e5)
	  chebyshevOrder = 1000;
	else if(upperBoundUnwantedSpectrum > 5e5)
	  chebyshevOrder = 1250;

	return chebyshevOrder;
      }
  }

  //
  // Constructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolver::chebyshevOrthogonalizedSubspaceIterationSolver(double lowerBoundWantedSpectrum,
												 double lowerBoundUnWantedSpectrum):
    d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum),
    d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    computing_timer(pcout,
		    dftParameters::reproducible_output ||
		    dftParameters::verbosity<4? dealii::TimerOutput::never : dealii::TimerOutput::summary,
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
  // solve
  //
  eigenSolverClass::ReturnValueType
  chebyshevOrthogonalizedSubspaceIterationSolver::solve(operatorDFTClass  & operatorMatrix,
							std::vector<dataTypes::number> & eigenVectorsFlattened,
							vectorType  & tempEigenVec,
							const unsigned int totalNumberWaveFunctions,
							std::vector<double>        & eigenValues,
							std::vector<double>        & residualNorms,
							const MPI_Comm &interBandGroupComm,
							const bool useMixedPrec)
  {


    if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(operatorMatrix.getMPICommunicator(),
	                      "Before Lanczos k-step upper Bound");

    computing_timer.enter_section("Lanczos k-step Upper Bound");
    operatorMatrix.reinit(1);
    const double upperBoundUnwantedSpectrum
	=linearAlgebraOperations::lanczosUpperBoundEigenSpectrum(operatorMatrix,
								 tempEigenVec);
    computing_timer.exit_section("Lanczos k-step Upper Bound");

    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;
    //set Chebyshev order
    if(chebyshevOrder == 0)
      chebyshevOrder=internal::setChebyshevOrder(upperBoundUnwantedSpectrum);

    if (dftParameters::lowerBoundUnwantedFracUpper>1e-6)
       d_lowerBoundUnWantedSpectrum=dftParameters::lowerBoundUnwantedFracUpper*upperBoundUnwantedSpectrum;
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
    //operatorMatrix.getOverloadedConstraintMatrix()->set_zero(eigenVectorsFlattened,
    //	                                                    totalNumberWaveFunctions);


    if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(operatorMatrix.getMPICommunicator(),
	                      "Before starting chebyshev filtering");

    const unsigned int localVectorSize = eigenVectorsFlattened.size()/totalNumberWaveFunctions;


    //band group parallelization data structures
    const unsigned int numberBandGroups=
    dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
					       totalNumberWaveFunctions,
					       bandGroupLowHighPlusOneIndices);
    const unsigned int totalNumberBlocks=std::ceil((double)totalNumberWaveFunctions/(double)dftParameters::wfcBlockSize);
    const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                         bandGroupLowHighPlusOneIndices[1]);


    //
    //allocate storage for eigenVectorsFlattenedArray for multiple blocks
    //
    dealii::parallel::distributed::Vector<dataTypes::number> eigenVectorsFlattenedArrayBlock;
    operatorMatrix.reinit(vectorsBlockSize,
			  eigenVectorsFlattenedArrayBlock,
			  true);

    for (unsigned int jvec = 0; jvec < totalNumberWaveFunctions; jvec += vectorsBlockSize)
    {

	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int BVec = std::min(vectorsBlockSize, totalNumberWaveFunctions-jvec);

	  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	       (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	  {
	    //create custom partitioned dealii array
	    if (BVec!=vectorsBlockSize)
		operatorMatrix.reinit(BVec,
				      eigenVectorsFlattenedArrayBlock,
				      true);

	    //fill the eigenVectorsFlattenedArrayBlock from eigenVectorsFlattenedArray
	    computing_timer.enter_section("Copy from full to block flattened array");
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		for(unsigned int iWave = 0; iWave < BVec; ++iWave)
		    eigenVectorsFlattenedArrayBlock.local_element(iNode*BVec+iWave)
			 =eigenVectorsFlattened[iNode*totalNumberWaveFunctions+jvec+iWave];

	    computing_timer.exit_section("Copy from full to block flattened array");

	    //
	    //call Chebyshev filtering function only for the current block to be filtered
	    //and does in-place filtering
	    computing_timer.enter_section("Chebyshev filtering opt");
	    if (jvec+BVec<dftParameters::numAdaptiveFilterStates)
	    {
	       const double chebyshevOrd=(double)chebyshevOrder;
	       const double adaptiveOrder=0.5*chebyshevOrd
		                          +jvec*0.3*chebyshevOrd/dftParameters::numAdaptiveFilterStates;
	       linearAlgebraOperations::chebyshevFilter(operatorMatrix,
						        eigenVectorsFlattenedArrayBlock,
						        BVec,
						        std::ceil(adaptiveOrder),
						        d_lowerBoundUnWantedSpectrum,
						        upperBoundUnwantedSpectrum,
						        d_lowerBoundWantedSpectrum,
							dftParameters::useMixedPrecCheby && useMixedPrec?
							true:false);
	    else
	       linearAlgebraOperations::chebyshevFilter(operatorMatrix,
						     eigenVectorsFlattenedArrayBlock,
						     BVec,
						     chebyshevOrder,
						     d_lowerBoundUnWantedSpectrum,
						     upperBoundUnwantedSpectrum,
						     d_lowerBoundWantedSpectrum,
						     dftParameters::useMixedPrecCheby && useMixedPrec?
						     true:false);
	    computing_timer.exit_section("Chebyshev filtering opt");

	    if (dftParameters::verbosity>=4)
	      dftUtils::printCurrentMemoryUsage(operatorMatrix.getMPICommunicator(),
						"During blocked chebyshev filtering");

	    //copy the eigenVectorsFlattenedArrayBlock into eigenVectorsFlattenedArray after filtering
	    computing_timer.enter_section("Copy from block to full flattened array");
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		for(unsigned int iWave = 0; iWave < BVec; ++iWave)
		      eigenVectorsFlattened[iNode*totalNumberWaveFunctions+jvec+iWave]
		      = eigenVectorsFlattenedArrayBlock.local_element(iNode*BVec+iWave);

	    computing_timer.exit_section("Copy from block to full flattened array");
	}
	else
	{
	    //set to zero wavefunctions which wont go through chebyshev filtering inside a given band group
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		for(unsigned int iWave = 0; iWave < BVec; ++iWave)
		      eigenVectorsFlattened[iNode*totalNumberWaveFunctions+jvec+iWave]
		      = dataTypes::number(0.0);

	}
    }//block loop

    eigenVectorsFlattenedArrayBlock.reinit(0);

    if (numberBandGroups>1)
    {
	computing_timer.enter_section("MPI All Reduce wavefunctions across all band groups");

	const unsigned int blockSize=dftParameters::mpiAllReduceMessageBlockSizeMB*1e+6/sizeof(dataTypes::number);
        for (unsigned int i=0; i<totalNumberWaveFunctions*localVectorSize;i+=blockSize)
        {
            const unsigned int currentBlockSize=std::min(blockSize,totalNumberWaveFunctions*localVectorSize-i);
	    MPI_Allreduce(MPI_IN_PLACE,
			  &eigenVectorsFlattened[0]+i,
			  currentBlockSize,
			  dataTypes::mpi_type_id(&eigenVectorsFlattened[0]),
			  MPI_SUM,
			  interBandGroupComm);
	}

	computing_timer.exit_section("MPI All Reduce wavefunctions across all band groups");
    }


    if(dftParameters::verbosity >= 4)
      pcout<<"ChebyShev Filtering Done: "<<std::endl;

    std::vector<dataTypes::number> eigenVectorsFlattenedRR;
    if (eigenValues.size()!=totalNumberWaveFunctions)
    {

	eigenVectorsFlattenedRR.resize(eigenValues.size()*localVectorSize,dataTypes::number(0.0));
        //operatorMatrix.reinit(eigenValues.size(),
        //                      eigenVectorsFlattenedRR,
        //                      true);
    }

    if(dftParameters::orthogType.compare("LW") == 0)
      {
	computing_timer.enter_section("Lowden Orthogn Opt");
	const unsigned int flag = linearAlgebraOperations::lowdenOrthogonalization(eigenVectorsFlattened,
										   totalNumberWaveFunctions,
										   operatorMatrix.getMPICommunicator());

	if (flag==1)
	{
	    if(dftParameters::verbosity >= 1)
		pcout<<"Switching to Gram-Schimdt orthogonalization as Lowden orthogonalization was not successful"<<std::endl;

	    computing_timer.enter_section("Gram-Schmidt Orthogn Opt");
	    linearAlgebraOperations::gramSchmidtOrthogonalization(eigenVectorsFlattened,
								  totalNumberWaveFunctions,
								  operatorMatrix.getMPICommunicator());
	    computing_timer.exit_section("Gram-Schmidt Orthogn Opt");
	}
	computing_timer.exit_section("Lowden Orthogn Opt");
      }
    else if (dftParameters::orthogType.compare("PGS") == 0)
    {
	computing_timer.enter_section("Pseudo-Gram-Schmidt");
	const unsigned int flag=linearAlgebraOperations::pseudoGramSchmidtOrthogonalization
	                         (eigenVectorsFlattened,
				  totalNumberWaveFunctions,
				  interBandGroupComm,
				  totalNumberWaveFunctions-eigenValues.size(),
				  operatorMatrix.getMPICommunicator(),
				  useMixedPrec,
				  eigenVectorsFlattenedRR);

	if (flag==1)
	{
	    if(dftParameters::verbosity >= 1)
		pcout<<"Switching to Gram-Schimdt orthogonalization as Pseudo-Gram-Schimdt orthogonalization was not successful"<<std::endl;

	    computing_timer.enter_section("Gram-Schmidt Orthogn Opt");
	    linearAlgebraOperations::gramSchmidtOrthogonalization(eigenVectorsFlattened,
								  totalNumberWaveFunctions,
								  operatorMatrix.getMPICommunicator());
	    computing_timer.exit_section("Gram-Schmidt Orthogn Opt");
	}
	computing_timer.exit_section("Pseudo-Gram-Schmidt");
    }
    else if (dftParameters::orthogType.compare("GS") == 0)
      {
	computing_timer.enter_section("Gram-Schmidt Orthogn Opt");
	linearAlgebraOperations::gramSchmidtOrthogonalization(eigenVectorsFlattened,
							      totalNumberWaveFunctions,
							      operatorMatrix.getMPICommunicator());
	computing_timer.exit_section("Gram-Schmidt Orthogn Opt");
      }

    if(dftParameters::verbosity >= 4)
      pcout<<"Orthogonalization Done: "<<std::endl;

    computing_timer.enter_section("Rayleigh-Ritz proj Opt");

    if (eigenValues.size()!=totalNumberWaveFunctions)
    {

	for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    for(unsigned int iWave = 0; iWave < eigenValues.size(); ++iWave)
		eigenVectorsFlattenedRR[iNode*eigenValues.size()
			 +iWave]
		     =eigenVectorsFlattened[iNode*totalNumberWaveFunctions+(totalNumberWaveFunctions-eigenValues.size())+iWave];

        linearAlgebraOperations::rayleighRitz(operatorMatrix,
					     eigenVectorsFlattenedRR,
					     eigenValues.size(),
					     interBandGroupComm,
					     operatorMatrix.getMPICommunicator(),
					     eigenValues);

	for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    for(unsigned int iWave = 0; iWave < eigenValues.size(); ++iWave)
		  eigenVectorsFlattened[iNode*totalNumberWaveFunctions+(totalNumberWaveFunctions-eigenValues.size())+iWave]
		  = eigenVectorsFlattenedRR[iNode*eigenValues.size()+iWave];
    }
    else
      {


	linearAlgebraOperations::rayleighRitz(operatorMatrix,
					      eigenVectorsFlattened,
					      totalNumberWaveFunctions,
					      interBandGroupComm,
					      operatorMatrix.getMPICommunicator(),
					      eigenValues);
      }


    computing_timer.exit_section("Rayleigh-Ritz proj Opt");

    if(dftParameters::verbosity >= 4)
      {
	pcout<<"Rayleigh-Ritz Done: "<<std::endl;
	pcout<<std::endl;
      }

    computing_timer.enter_section("eigen vectors residuals opt");
    if (eigenValues.size()!=totalNumberWaveFunctions)
      linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
							eigenVectorsFlattenedRR,
							eigenValues,
							operatorMatrix.getMPICommunicator(),
							residualNorms);
    else
      linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
      						        eigenVectorsFlattened,
      						        eigenValues,
							operatorMatrix.getMPICommunicator(),
      						        residualNorms);
    computing_timer.exit_section("eigen vectors residuals opt");

    if(dftParameters::verbosity >= 4)
      {
	pcout<<"EigenVector Residual Computation Done: "<<std::endl;
	pcout<<std::endl;
      }

    if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(operatorMatrix.getMPICommunicator(),
					"After all steps of subspace iteration");

    return;

  }


  //
  // solve
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

    //set Chebyshev order
    if(chebyshevOrder == 0)
      chebyshevOrder=internal::setChebyshevOrder(upperBoundUnwantedSpectrum);

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
     //linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
     //						      eigenVectors,
     //						      eigenValues,
     //						      residualNorms);
     computing_timer.exit_section("compute eigen vectors residuals");

     return;

  }

}
