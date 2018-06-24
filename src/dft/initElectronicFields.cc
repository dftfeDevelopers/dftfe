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
// @author  Phani Motamarri, Sambit Das
//

template<unsigned int FEOrder>
void dftClass<FEOrder>::initPsiAndRhoFromPreviousGroundStatePsi(std::vector<std::vector<vectorType>> eigenVectors)
{
     const unsigned int totalNumEigenVectors=(1+dftParameters::spinPolarized)*d_kPointWeights.size()*eigenVectors[0].size();
     std::vector<vectorType> eigenVectorsPrevious(totalNumEigenVectors);
     std::vector<vectorType* > eigenVectorsPreviousPtrs(totalNumEigenVectors);
     std::vector<vectorType* > eigenVectorsCurrentPtrs(totalNumEigenVectors);

     for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
        for(unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  eigenVectorsPrevious[kPoint* eigenVectors[0].size()+i]=eigenVectors[kPoint][i];
	  eigenVectorsPreviousPtrs[kPoint* eigenVectors[0].size()+i]=&(eigenVectorsPrevious[kPoint* eigenVectors[0].size()+i]);
	  eigenVectors[kPoint][i].reinit(d_tempEigenVec);
	  eigenVectorsCurrentPtrs[kPoint* eigenVectors[0].size()+i]=&(eigenVectors[kPoint][i]);
	}

     if (dftParameters::verbosity>=2)
       pcout<<"L2 Norm Value of previous eigenvector 0: "<<eigenVectorsPreviousPtrs[0]->l2_norm()<<std::endl;

     computing_timer.enter_section("interpolate previous PSI");

     pcout <<std::endl<< "Interpolating previous groundstate PSI into the new finite element mesh...."<<std::endl;
     vectorTools::interpolateFieldsFromPreviousMesh interpolateEigenVecPrev(mpi_communicator);
     interpolateEigenVecPrev.interpolate(d_mesh.getSerialMeshUnmovedPrevious(),
	                         d_mesh.getParallelMeshUnmovedPrevious(),
				 d_mesh.getParallelMeshUnmoved(),
				 FEEigen,
				 FEEigen,
				 eigenVectorsPreviousPtrs,
				 eigenVectorsCurrentPtrs);

     computing_timer.exit_section("interpolate previous PSI");

     for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
        for(unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  constraintsNoneEigenDataInfo.distribute(eigenVectors[kPoint][i]);
	  eigenVectors[kPoint][i].update_ghost_values();
	}

     if (dftParameters::verbosity>=2)
      pcout<<"L2 Norm Value of interpolated eigenvector 0: "<<eigenVectorsCurrentPtrs[0]->l2_norm()<<std::endl;

     pcout <<std::endl<< "Computing rho initial guess from previous ground state PSI...."<<std::endl;
     computeRhoInitialGuessFromPSI(eigenVectors);
}

//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initElectronicFields(const unsigned int usePreviousGroundStateFields){
  TimerOutput::Scope scope (computing_timer,"init electronic fields");

  //reading data from pseudopotential files and fitting splines
  //if(dftParameters::pseudoProjector == 2)
    initNonLocalPseudoPotential_OV();
    //else
    //initNonLocalPseudoPotential();

  //initialize electrostatics fields
  matrix_free_data.initialize_dof_vector(d_phiTotRhoIn,phiTotDofHandlerIndex);
  d_phiTotRhoOut.reinit(d_phiTotRhoIn);
  matrix_free_data.initialize_dof_vector(d_phiExt,phiExtDofHandlerIndex);

  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(d_tempEigenVec,eigenDofHandlerIndex);


  //
  //initialize density and PSI/ interpolate from previous ground state solution
  //
  if (usePreviousGroundStateFields==0)
  {
     for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  vectorTools::createDealiiVector<dataTypes::number>
		     (matrix_free_data.get_vector_partitioner(),
		      numEigenValues,
		      d_eigenVectorsFlattened[kPoint]);

     pcout <<std::endl<< "Setting initial guess for wavefunctions...."<<std::endl;
     readPSI();

     if(dftParameters::verbosity >= 4)
       {
	 PetscLogDouble bytes;
	 PetscMemoryGetCurrentUsage(&bytes);
	 FILE *dummy;
	 unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
	 PetscSynchronizedPrintf(mpi_communicator,"[%d] Memory Usage after creating flattened eigenvectors %e\n",this_mpi_process,bytes);
	 PetscSynchronizedFlush(mpi_communicator,dummy);
       }

     if (!(dftParameters::chkType==2 && dftParameters::restartFromChk))
	initRho();

  }
  else if (usePreviousGroundStateFields==1)
  {
     for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  vectorTools::createDealiiVector<dataTypes::number>
		     (matrix_free_data.get_vector_partitioner(),
		      numEigenValues,
		      d_eigenVectorsFlattened[kPoint]);

     pcout <<std::endl<< "Reading initial guess for PSI...."<<std::endl;
     readPSI();

     initRhoFromPreviousGroundStateRho();
  }
  else if (usePreviousGroundStateFields==2)
  {
      //initPsiAndRhoFromPreviousGroundStatePsi(eigenVectors);
  }

   if (dftParameters::verbosity>=2)
       if (dftParameters::spinPolarized==1)
	pcout<< std::endl<<"net magnetization: "<< totalMagnetization(rhoInValuesSpinPolarized) <<std::endl;

  //
  //update serial and parallel unmoved previous mesh
  //
  d_mesh.generateSerialAndParallelUnmovedPreviousMesh(atomLocations,
				                      d_imagePositions,
				                      d_domainBoundingVectors);

  //
  //store constraintEigen Matrix entries into STL vector
  //
  constraintsNoneEigenDataInfo.initialize(d_tempEigenVec.get_partitioner(),
					  constraintsNoneEigen);

  constraintsNoneDataInfo.initialize(matrix_free_data.get_vector_partitioner(),
				     constraintsNone);
}
