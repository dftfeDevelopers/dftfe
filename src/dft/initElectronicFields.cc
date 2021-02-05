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

//init
template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::initElectronicFields()
{
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
	d_matrixFreeDataPRefined.initialize_dof_vector(d_phiTotRhoIn,d_phiTotDofHandlerIndexElectro);
	d_phiTotRhoOut.reinit(d_phiTotRhoIn);
	d_matrixFreeDataPRefined.initialize_dof_vector(d_phiExt,d_phiExtDofHandlerIndexElectro);

	d_matrixFreeDataPRefined.initialize_dof_vector(d_rhoInNodalValues,d_densityDofHandlerIndexElectro);
	d_rhoOutNodalValues.reinit(d_rhoInNodalValues);
	d_rhoOutNodalValuesSplit.reinit(d_rhoInNodalValues);
	//d_atomicRho.reinit(d_rhoInNodalValues);

	if (dftParameters::isIonOpt || dftParameters::isCellOpt)
	{
		initAtomicRho();
	} 

	//
	//initialize eigen vectors
	//
	matrix_free_data.initialize_dof_vector(d_tempEigenVec,d_eigenDofHandlerIndex);

	//
	//store constraintEigen Matrix entries into STL vector
	//
	constraintsNoneEigenDataInfo.initialize(d_tempEigenVec.get_partitioner(),
			constraintsNoneEigen);

	constraintsNoneDataInfo.initialize(matrix_free_data.get_vector_partitioner(),
			constraintsNone);

#ifdef DFTFE_WITH_GPU
	if (dftParameters::useGPU)
		d_constraintsNoneDataInfoCUDA.initialize(matrix_free_data.get_vector_partitioner(),
				constraintsNone);
#endif

	if (dftParameters::verbosity>=4)
		dftUtils::printCurrentMemoryUsage(mpi_communicator,
				"Overloaded constraint matrices initialized");

	//
	//initialize density and PSI/ interpolate from previous ground state solution
	//
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

  if(!(dftParameters::chkType==2 && dftParameters::restartFromChk))
  {
    initRho();
    //d_rhoOutNodalValues.reinit(d_rhoInNodalValues);
  }

  if (dftParameters::verbosity>=4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator,
        "initRho called");

#ifdef DFTFE_WITH_GPU
	if (dftParameters::useGPU)
	{
		d_eigenVectorsFlattenedCUDA.resize(d_eigenVectorsFlattenedSTL[0].size()*(1+dftParameters::spinPolarized)*d_kPointWeights.size());

		if (d_numEigenValuesRR!=d_numEigenValues)
			d_eigenVectorsRotFracFlattenedCUDA.resize(d_eigenVectorsRotFracDensityFlattenedSTL[0].size()*(1+dftParameters::spinPolarized)*d_kPointWeights.size());
		else
			d_eigenVectorsRotFracFlattenedCUDA.resize(1);

		for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
		{
			vectorToolsCUDA::copyHostVecToCUDAVec(&d_eigenVectorsFlattenedSTL[kPoint][0],
					d_eigenVectorsFlattenedCUDA.begin()+kPoint*d_eigenVectorsFlattenedSTL[0].size(),
					d_eigenVectorsFlattenedSTL[0].size());

		}
	}
#endif

	if (dftParameters::verbosity>=2)
		if (dftParameters::spinPolarized==1)
			pcout<< std::endl<<"net magnetization: "<< totalMagnetization(rhoInValuesSpinPolarized) <<std::endl;
}
