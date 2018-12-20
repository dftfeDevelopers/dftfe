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

     initRhoFromPreviousGroundStateRho();
     //pcout <<std::endl<< "Computing rho initial guess from previous ground state PSI...."<<std::endl;
     //computeRhoInitialGuessFromPSI(eigenVectors);
}

//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initElectronicFields(const unsigned int usePreviousGroundStateFields){
  TimerOutput::Scope scope (computing_timer,"init electronic fields");

  //reading data from pseudopotential files and fitting splines
  if(dftParameters::isPseudopotential)
    initNonLocalPseudoPotential_OV();
    //else
    //initNonLocalPseudoPotential();

  if (dftParameters::verbosity>=4)
     dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Call to initNonLocalPseudoPotential");

  //initialize electrostatics fields
  matrix_free_data.initialize_dof_vector(d_phiTotRhoIn,phiTotDofHandlerIndex);
  d_phiTotRhoOut.reinit(d_phiTotRhoIn);
  matrix_free_data.initialize_dof_vector(d_phiExt,phiExtDofHandlerIndex);

  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(d_tempEigenVec,eigenDofHandlerIndex);

  //
  //store constraintEigen Matrix entries into STL vector
  //
  constraintsNoneEigenDataInfo.initialize(d_tempEigenVec.get_partitioner(),
					  constraintsNoneEigen);

  constraintsNoneDataInfo.initialize(matrix_free_data.get_vector_partitioner(),
				     constraintsNone);

  constraintsNoneDataInfo2.initialize(matrix_free_data.get_vector_partitioner(),
				     constraintsNone);

 if (dftParameters::verbosity>=4)
   dftUtils::printCurrentMemoryUsage(mpi_communicator,
			  "Overloaded constraint matrices initialized");

  //
  //initialize density and PSI/ interpolate from previous ground state solution
  //
  if (usePreviousGroundStateFields==0)
  {
     for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
       {

	 d_eigenVectorsFlattenedSTL[kPoint].resize(d_numEigenValues*matrix_free_data.get_vector_partitioner()->local_size(),dataTypes::number(0.0));

	 if (d_numEigenValuesRR!=d_numEigenValues)
	 {
	    d_eigenVectorsRotFracDensityFlattenedSTL[kPoint].resize(d_numEigenValuesRR*matrix_free_data.get_vector_partitioner()->local_size(),dataTypes::number(0.0));
	 }
       }

     pcout <<std::endl<< "Setting initial guess for wavefunctions...."<<std::endl;

     if (dftParameters::verbosity>=4)
       dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Created flattened array eigenvectors before update ghost values");

     readPSI();

     if (dftParameters::verbosity>=4)
       dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Created flattened array eigenvectors");

     if (!(dftParameters::chkType==2 && dftParameters::restartFromChk))
	initRho();

     if (dftParameters::verbosity>=4)
       dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "initRho called");

  }
  else if (usePreviousGroundStateFields==1)
  {
     for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
     {
	 d_eigenVectorsFlattenedSTL[kPoint].resize(d_numEigenValues*matrix_free_data.get_vector_partitioner()->local_size(),dataTypes::number(0.0));

	 if (d_numEigenValuesRR!=d_numEigenValues)
	 {
	    d_eigenVectorsRotFracDensityFlattenedSTL[kPoint].resize(d_numEigenValuesRR*matrix_free_data.get_vector_partitioner()->local_size(),dataTypes::number(0.0));
	 }
     }

     pcout <<std::endl<< "Reading initial guess for PSI...."<<std::endl;
     readPSI();

     initRhoFromPreviousGroundStateRho();
  }
  else if (usePreviousGroundStateFields==2)
  {
      std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*d_kPointWeights.size(),
	                                                 std::vector<vectorType>(d_numEigenValues));

      for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  for(unsigned int i= 0; i < d_numEigenValues; ++i)
	       eigenVectors[kPoint][i].reinit(d_tempEigenVecPrev);
      for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
      {
#ifdef USE_COMPLEX
	 vectorTools::copyFlattenedSTLVecToSingleCompVec
		 (d_eigenVectorsFlattenedSTL[kPoint],
		  d_numEigenValues,
		  std::make_pair(0,d_numEigenValues),
		  localProc_dof_indicesReal,
		  localProc_dof_indicesImag,
		  eigenVectors[kPoint]);
#else
	 vectorTools::copyFlattenedSTLVecToSingleCompVec
		 (d_eigenVectorsFlattenedSTL[kPoint],
		  d_numEigenValues,
		  std::make_pair(0,d_numEigenValues),
		  eigenVectors[kPoint]);

#endif
	for(unsigned int i= 0; i < d_numEigenValues; ++i)
	{
	  constraintsNoneEigenDataInfoPrev.distribute(eigenVectors[kPoint][i]);
	  eigenVectors[kPoint][i].update_ghost_values();
	}

	d_eigenVectorsFlattenedSTL[kPoint].clear();
	std::vector<dataTypes::number>().swap(d_eigenVectorsFlattenedSTL[kPoint]);
      }

      initPsiAndRhoFromPreviousGroundStatePsi(eigenVectors);

      //Create the full STL array
      for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
      {
	  d_eigenVectorsFlattenedSTL[kPoint].resize
	      (d_numEigenValues*matrix_free_data.get_vector_partitioner()->local_size(),
	       dataTypes::number(0.0));

	 if (d_numEigenValuesRR!=d_numEigenValues)
	    d_eigenVectorsRotFracDensityFlattenedSTL[kPoint].resize
		(d_numEigenValuesRR*matrix_free_data.get_vector_partitioner()->local_size()
		 ,dataTypes::number(0.0));
      }

      for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
      {
#ifdef USE_COMPLEX
	 vectorTools::copySingleCompVecToFlattenedSTLVec
		 (d_eigenVectorsFlattenedSTL[kPoint],
		  d_numEigenValues,
		  std::make_pair(0,d_numEigenValues),
		  localProc_dof_indicesReal,
		  localProc_dof_indicesImag,
		  eigenVectors[kPoint]);
#else
	 vectorTools::copySingleCompVecToFlattenedSTLVec
		 (d_eigenVectorsFlattenedSTL[kPoint],
		  d_numEigenValues,
		  std::make_pair(0,d_numEigenValues),
		  eigenVectors[kPoint]);

#endif
      }
  }

  if  (dftParameters::isIonOpt)
    updatePrevMeshDataStructures();

  if (dftParameters::verbosity>=2)
       if (dftParameters::spinPolarized==1)
	pcout<< std::endl<<"net magnetization: "<< totalMagnetization(rhoInValuesSpinPolarized) <<std::endl;
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::updatePrevMeshDataStructures()
{
  matrix_free_data.initialize_dof_vector(d_tempEigenVecPrev,eigenDofHandlerIndex);


  constraintsNoneEigenDataInfoPrev.initialize(d_tempEigenVecPrev.get_partitioner(),
					      constraintsNoneEigen);

  //
  //update serial and parallel unmoved previous mesh
  //
  d_mesh.generateSerialAndParallelUnmovedPreviousMesh(atomLocations,
						      d_imagePositions,
						      d_domainBoundingVectors);
 if (dftParameters::verbosity>=4)
   dftUtils::printCurrentMemoryUsage(mpi_communicator,
			  "Serial and parallel prev mesh generated");
}
