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

static const unsigned int order_lookup[][2] = {
	{500, 24}, // <= 500 ~> chebyshevOrder = 24
	{750, 30},
	{1000, 39},
	{1500, 50},
	{2000, 53},
	{3000, 57},
	{4000, 62},
	{5000, 69},
	{9000, 77},
	{14000, 104},
	{20000, 119},
	{30000, 162},
	{50000, 300},
	{80000, 450},
	{1e5, 550},
	{2e5, 700},
	{5e5, 1000}
};

namespace dftfe{

	namespace internal
	{
		unsigned int setChebyshevOrder(const unsigned int upperBoundUnwantedSpectrum)
		{
			for(int i=0; i<sizeof(order_lookup)/sizeof(order_lookup[0]); i++) {
				if(upperBoundUnwantedSpectrum <= order_lookup[i][0])
					return order_lookup[i][1];
			}
			return 1250;
		}
	}

	//
	// Constructor.
	//
	chebyshevOrthogonalizedSubspaceIterationSolver::chebyshevOrthogonalizedSubspaceIterationSolver
		(const MPI_Comm &mpi_comm,
		 double lowerBoundWantedSpectrum,
		 double lowerBoundUnWantedSpectrum):
			d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum),
			d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum),
			pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
			computing_timer(mpi_comm,
					pcout,
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
	void
		chebyshevOrthogonalizedSubspaceIterationSolver::solve(operatorDFTClass  & operatorMatrix,
				std::vector<dataTypes::number> & eigenVectorsFlattened,
				std::vector<dataTypes::number> & eigenVectorsRotFracDensityFlattened,
				distributedCPUVec<double>  & tempEigenVec,
				const unsigned int totalNumberWaveFunctions,
				std::vector<double>        & eigenValues,
				std::vector<double>        & residualNorms,
				const MPI_Comm &interBandGroupComm,
				const bool useMixedPrec,
				const bool isFirstScf,
				const bool useFullMassMatrixGEP)
		{


			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(operatorMatrix.getMPICommunicator(),
						"Before Lanczos k-step upper Bound");

			computing_timer.enter_section("Lanczos k-step Upper Bound");
			distributedCPUVec<dataTypes::number> vecForLanczos;
			operatorMatrix.reinit(1,
					vecForLanczos,
					true);			
			const double upperBoundUnwantedSpectrum =linearAlgebraOperations::lanczosUpperBoundEigenSpectrum(operatorMatrix,
					vecForLanczos);
			computing_timer.exit_section("Lanczos k-step Upper Bound");

			unsigned int chebyshevOrder = dftParameters::chebyshevOrder;


			//
			//set Chebyshev order
			//
			if(chebyshevOrder == 0)
				chebyshevOrder=internal::setChebyshevOrder(upperBoundUnwantedSpectrum);

			chebyshevOrder=(isFirstScf && dftParameters::isPseudopotential)?chebyshevOrder*dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor:chebyshevOrder;

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

			int this_process;
			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

			MPI_Barrier(MPI_COMM_WORLD);
			double cheby_time = MPI_Wtime();

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
			const unsigned int totalNumberBlocks=std::ceil((double)totalNumberWaveFunctions/(double)dftParameters::chebyWfcBlockSize);
			const unsigned int vectorsBlockSize=std::min(dftParameters::chebyWfcBlockSize,
					bandGroupLowHighPlusOneIndices[1]);


			//
			//allocate storage for eigenVectorsFlattenedArray for multiple blocks
			//
			distributedCPUVec<dataTypes::number> eigenVectorsFlattenedArrayBlock;
			operatorMatrix.reinit(vectorsBlockSize,
			       		      eigenVectorsFlattenedArrayBlock,
					      true);

			///storage for cell wavefunction matrix
			std::vector<dataTypes::number>  cellWaveFunctionMatrix;

                        //// Debug check lines///////////////////////////////////////////////
                        /*distributedCPUVec<dataTypes::number> productArray;
                        productArray.reinit(eigenVectorsFlattenedArrayBlock);
                        productArray = 0.0;

                        std::vector<std::vector<dataTypes::number> > productWaveFunctionMatrix;
		        pcout<<"Vectors Block Size: "<<vectorsBlockSize<<std::endl;
                        pcout<<"Total WaveFunctions: "<<totalNumberWaveFunctions<<std::endl;
                        for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
                           for(unsigned int iWave = 0; iWave < vectorsBlockSize; ++iWave)
                                 eigenVectorsFlattenedArrayBlock.local_element(iNode*vectorsBlockSize+iWave)
                                                                = eigenVectorsFlattened[iNode*totalNumberWaveFunctions+iWave];	
                       
                        
                        pcout<<"Src Eigen-Vectors Norm before old HX: "<<eigenVectorsFlattenedArrayBlock.l2_norm()<<std::endl;
                        pcout<<"Product Eigen-Vectors Norm before old HX: "<<productArray.l2_norm()<<std::endl;
                        

                        operatorMatrix.HX(eigenVectorsFlattenedArrayBlock,
                                          vectorsBlockSize,
                                          false,
                                          1.0,
                                          productArray);

                        pcout<<"Output Vectors Norm after old HX: "<<productArray.l2_norm()<<std::endl;

                        double alpha2 = 25.0,alpha1=10.0;
                        productArray.add(alpha2,eigenVectorsFlattenedArrayBlock);
                        productArray *= alpha1;

                        pcout<<"Output Vectors Norm after old HX and recursion: "<<productArray.l2_norm()<<std::endl; 

                        operatorMatrix.initCellWaveFunctionMatrix(vectorsBlockSize,
                                                                  eigenVectorsFlattenedArrayBlock,
                                                                  cellWaveFunctionMatrix);

                        productArray = 0.0;

                        pcout<<"Src Eigen-Vectors Norm before new HX: "<<eigenVectorsFlattenedArrayBlock.l2_norm()<<std::endl;
                        pcout<<"Product Eigen-Vectors Norm before new HX: "<<productArray.l2_norm()<<std::endl;


                        std::vector<unsigned int> globalArrayClassificationMap;
			operatorMatrix.getInteriorSurfaceNodesMapFromGlobalArray(globalArrayClassificationMap);

                        operatorMatrix.initWithScalar(vectorsBlockSize,
						      0.0,
						      productWaveFunctionMatrix);


                        operatorMatrix.HX(eigenVectorsFlattenedArrayBlock,
	                                  cellWaveFunctionMatrix,
			                  vectorsBlockSize,
			                  false,
			                  1.0,
			                  productArray,
		                          productWaveFunctionMatrix);

                        const unsigned int numberDofs = eigenVectorsFlattenedArrayBlock.local_size()/vectorsBlockSize;
				
		        for(unsigned int iDof = 0; iDof < numberDofs; ++iDof)
		         {
		           if(globalArrayClassificationMap[iDof] == 1)
			     {
			        for(unsigned int iWave = 0; iWave < vectorsBlockSize; ++iWave)
		                 {
	                           productArray.local_element(iDof*vectorsBlockSize+iWave) += alpha2*eigenVectorsFlattenedArrayBlock.local_element(iDof*vectorsBlockSize+iWave);
			           productArray.local_element(iDof*vectorsBlockSize+iWave) *= alpha1;
		                 }
		             }
		        }

  
                        operatorMatrix.axpy(alpha2,
				            vectorsBlockSize,
					    cellWaveFunctionMatrix,
				            productWaveFunctionMatrix);


                        operatorMatrix.scale(alpha1,
				             vectorsBlockSize,
				             productWaveFunctionMatrix); 

                        operatorMatrix.fillGlobalArrayFromCellWaveFunctionMatrix(vectorsBlockSize,
								                 productWaveFunctionMatrix,
										 productArray);

                        pcout<<"Output Vectors Norm after new HX: "<<productArray.l2_norm()<<std::endl;

                 
                        exit(0);*/

			int startIndexBandParal=totalNumberWaveFunctions;
			int numVectorsBandParal=0;
			for (unsigned int jvec = 0; jvec < totalNumberWaveFunctions; jvec += vectorsBlockSize)
			{

				// Correct block dimensions if block "goes off edge of" the matrix
				const unsigned int BVec = std::min(vectorsBlockSize, totalNumberWaveFunctions-jvec);

				if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
						(jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
				{
					if (jvec<startIndexBandParal)
						startIndexBandParal=jvec;
					numVectorsBandParal= jvec+BVec-startIndexBandParal;

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


					computing_timer.enter_section("Copy from global-vectors to cellwavefunction array");
					operatorMatrix.initCellWaveFunctionMatrix(BVec,
										  eigenVectorsFlattenedArrayBlock,
										  cellWaveFunctionMatrix);

					computing_timer.exit_section("Copy from global-vectors to cellwavefunction array");
					

					//
					//call Chebyshev filtering function only for the current block to be filtered
					//and does in-place filtering
					computing_timer.enter_section("Chebyshev filtering opt");
					if(jvec+BVec<dftParameters::numAdaptiveFilterStates)
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
								                         d_lowerBoundWantedSpectrum);
						/*linearAlgebraOperations::chebyshevFilterOpt(operatorMatrix,
											    eigenVectorsFlattenedArrayBlock,
											    cellWaveFunctionMatrix,
											    BVec,
											    std::ceil(adaptiveOrder),
											    d_lowerBoundUnWantedSpectrum,
											    upperBoundUnwantedSpectrum,
											    d_lowerBoundWantedSpectrum);*/

						//copy back cell wavefunction data interior nodes also into global dealii vectors
						
					}
					else
					  {
					    linearAlgebraOperations::chebyshevFilter(operatorMatrix,
					                                             eigenVectorsFlattenedArrayBlock,
					                                             BVec,
					                                             chebyshevOrder,
					                                             d_lowerBoundUnWantedSpectrum,
					                                             upperBoundUnwantedSpectrum,
					                                             d_lowerBoundWantedSpectrum);
					    
					    /*linearAlgebraOperations::chebyshevFilterOpt(operatorMatrix,
											eigenVectorsFlattenedArrayBlock,
											cellWaveFunctionMatrix,
											BVec,
											chebyshevOrder,
											d_lowerBoundUnWantedSpectrum,
											upperBoundUnwantedSpectrum,
											d_lowerBoundWantedSpectrum);*/
					  }
					  
					computing_timer.exit_section("Chebyshev filtering opt");


					computing_timer.enter_section("Copy from cellwavefunction array to global array");
					/*operatorMatrix.fillGlobalArrayFromCellWaveFunctionMatrix(BVec,
												 cellWaveFunctionMatrix,
												 eigenVectorsFlattenedArrayBlock);*/
												 

					computing_timer.exit_section("Copy from cellwavefunction array to global array");

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

			MPI_Barrier(MPI_COMM_WORLD);
			cheby_time = MPI_Wtime() - cheby_time;
			if (this_process==0 && dftParameters::verbosity>=2)
					std::cout<<"Time for chebyshev filtering on CPU: "<<cheby_time<<std::endl;

			if (numberBandGroups>1)
			{
				if (!dftParameters::bandParalOpt)
				{
					computing_timer.enter_section("MPI All Reduce wavefunctions across all band groups");
					MPI_Barrier(interBandGroupComm);
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
				else
				{
					computing_timer.enter_section("MPI_Allgatherv across band groups");
					MPI_Barrier(interBandGroupComm);
					std::vector<dataTypes::number> eigenVectorsBandGroup(numVectorsBandParal*localVectorSize,0);
					std::vector<dataTypes::number> eigenVectorsBandGroupTransposed(numVectorsBandParal*localVectorSize,0);
					std::vector<dataTypes::number> eigenVectorsTransposed(totalNumberWaveFunctions*localVectorSize,0);

					for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
						for(unsigned int iWave = 0; iWave < numVectorsBandParal; ++iWave)
							eigenVectorsBandGroup[iNode*numVectorsBandParal+iWave]
								= eigenVectorsFlattened[iNode*totalNumberWaveFunctions+startIndexBandParal+iWave];


					for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
						for(unsigned int iWave = 0; iWave < numVectorsBandParal; ++iWave)
							eigenVectorsBandGroupTransposed[iWave*localVectorSize+iNode]
								= eigenVectorsBandGroup[iNode*numVectorsBandParal+iWave];

					std::vector<int> recvcounts(numberBandGroups,0);
					std::vector<int> displs(numberBandGroups,0);

					int recvcount=numVectorsBandParal*localVectorSize;
					MPI_Allgather(&recvcount,
							1,
							MPI_INT,
							&recvcounts[0],
							1,
							MPI_INT,
							interBandGroupComm);

					int displ=startIndexBandParal*localVectorSize;
					MPI_Allgather(&displ,
							1,
							MPI_INT,
							&displs[0],
							1,
							MPI_INT,
							interBandGroupComm);

					MPI_Allgatherv(&eigenVectorsBandGroupTransposed[0],
							numVectorsBandParal*localVectorSize,
							dataTypes::mpi_type_id(&eigenVectorsBandGroupTransposed[0]),
							&eigenVectorsTransposed[0],
							&recvcounts[0],
							&displs[0],
							dataTypes::mpi_type_id(&eigenVectorsTransposed[0]),
							interBandGroupComm);


					for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
						for(unsigned int iWave = 0; iWave < totalNumberWaveFunctions; ++iWave)
							eigenVectorsFlattened[iNode*totalNumberWaveFunctions+iWave]
								= eigenVectorsTransposed[iWave*localVectorSize+iNode];
					MPI_Barrier(interBandGroupComm);
					computing_timer.exit_section("MPI_Allgatherv across band groups");
				}
			}


			if(dftParameters::verbosity >= 4)
				pcout<<"ChebyShev Filtering Done: "<<std::endl;

			if (dftParameters::rrGEP)
			{
				computing_timer.enter_section("Rayleigh-Ritz GEP");
				if (eigenValues.size()!=totalNumberWaveFunctions)
				{
					linearAlgebraOperations::rayleighRitzGEPSpectrumSplitDirect(operatorMatrix,
							eigenVectorsFlattened,
							eigenVectorsRotFracDensityFlattened,
							totalNumberWaveFunctions,
							totalNumberWaveFunctions-eigenValues.size(),
							interBandGroupComm,
							operatorMatrix.getMPICommunicator(),
							useMixedPrec,
							eigenValues);
				}
				else
				{

					if(useFullMassMatrixGEP)
					{
						linearAlgebraOperations::rayleighRitzGEPFullMassMatrix(operatorMatrix,
								eigenVectorsFlattened,
								totalNumberWaveFunctions,
								interBandGroupComm,
								operatorMatrix.getMPICommunicator(),
								eigenValues,
								useMixedPrec);
					}
					else
					{
						linearAlgebraOperations::rayleighRitzGEP(operatorMatrix,
								eigenVectorsFlattened,
								totalNumberWaveFunctions,
								interBandGroupComm,
								operatorMatrix.getMPICommunicator(),
								eigenValues,
								useMixedPrec);
					}
				}
				computing_timer.exit_section("Rayleigh-Ritz GEP");

				computing_timer.enter_section("eigen vectors residuals opt");
				if (eigenValues.size()!=totalNumberWaveFunctions)
				{
					linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
							eigenVectorsRotFracDensityFlattened,
							eigenValues,
							operatorMatrix.getMPICommunicator(),
							interBandGroupComm,
							residualNorms);
				}
				else
				{
					if(!useFullMassMatrixGEP)
						linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
								eigenVectorsFlattened,
								eigenValues,
								operatorMatrix.getMPICommunicator(),
								interBandGroupComm,
								residualNorms);
				}
				computing_timer.exit_section("eigen vectors residuals opt");
			}
			else
			{
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
						(operatorMatrix,
						 eigenVectorsFlattened,
						 totalNumberWaveFunctions,
						 interBandGroupComm,
						 operatorMatrix.getMPICommunicator(),
						 useMixedPrec);

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
					linearAlgebraOperations::rayleighRitzSpectrumSplitDirect(operatorMatrix,
							eigenVectorsFlattened,
							eigenVectorsRotFracDensityFlattened,
							totalNumberWaveFunctions,
							totalNumberWaveFunctions-eigenValues.size(),
							interBandGroupComm,
							operatorMatrix.getMPICommunicator(),
							useMixedPrec,
							eigenValues);
				}
				else
				{


					linearAlgebraOperations::rayleighRitz(operatorMatrix,
							eigenVectorsFlattened,
							totalNumberWaveFunctions,
							interBandGroupComm,
							operatorMatrix.getMPICommunicator(),
							eigenValues,
							false);
				}


				computing_timer.exit_section("Rayleigh-Ritz proj Opt");

				if(dftParameters::verbosity >= 4)
				{
					pcout<<"Rayleigh-Ritz Done: "<<std::endl;
					pcout<<std::endl;
				}

				computing_timer.enter_section("eigen vectors residuals opt");

				if (eigenValues.size()!=totalNumberWaveFunctions)
				{
					linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
							eigenVectorsRotFracDensityFlattened,
							eigenValues,
							operatorMatrix.getMPICommunicator(),
							interBandGroupComm,
							residualNorms);
				}
				else
					linearAlgebraOperations::computeEigenResidualNorm(operatorMatrix,
							eigenVectorsFlattened,
							eigenValues,
							operatorMatrix.getMPICommunicator(),
							interBandGroupComm,
							residualNorms);
				computing_timer.exit_section("eigen vectors residuals opt");
			}



			if(dftParameters::verbosity >= 4)
			{
				pcout<<"EigenVector Residual Computation Done: "<<std::endl;
				pcout<<std::endl;
			}

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(operatorMatrix.getMPICommunicator(),
						"After all steps of subspace iteration");

		}


	//
	// solve
	//
	void
		chebyshevOrthogonalizedSubspaceIterationSolver::solve(operatorDFTClass           & operatorMatrix,
				std::vector<distributedCPUVec<double>>    & eigenVectors,
				std::vector<double>        & eigenValues,
				std::vector<double>        & residualNorms)
		{

			/*
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
#ifdef USE_PETSC
PetscLogDouble bytes;
PetscMemoryGetCurrentUsage(&bytes);
FILE *dummy;
unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());
PetscSynchronizedPrintf(operatorMatrix.getMPICommunicator(),"[%d] Memory Usage before starting eigen solution  %e\n",this_mpi_process,bytes);
PetscSynchronizedFlush(operatorMatrix.getMPICommunicator(),dummy);
#endif
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
*/
}

}
