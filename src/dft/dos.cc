// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020x The Regents of the University of Michigan and DFT-FE authors.
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
//



//compute fermi energy
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_tdos(const std::vector<std::vector<double>> & eigenValuesInput,
				     const std::string & dosFileName)
{
  computing_timer.enter_section("DOS computation");
  std::vector<double> eigenValuesAllkPoints;
  for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
    {
      for(int statesIter = 0; statesIter < eigenValuesInput[0].size(); ++statesIter)
	{
	  eigenValuesAllkPoints.push_back(eigenValuesInput[kPoint][statesIter]);
	}
    }

  std::sort(eigenValuesAllkPoints.begin(),eigenValuesAllkPoints.end());

  double totalEigenValues = eigenValuesAllkPoints.size();
  double intervalSize = 0.001;
  double sigma =  C_kb*dftParameters::TVal;
  double lowerBoundEpsilon=1.5*eigenValuesAllkPoints[0];
  double upperBoundEpsilon=eigenValuesAllkPoints[totalEigenValues-1]*1.5;
  unsigned int numberIntervals = std::ceil((upperBoundEpsilon - lowerBoundEpsilon)/intervalSize);

  std::vector<double> densityOfStates,densityOfStatesUp,densityOfStatesDown;


  if(dftParameters::spinPolarized == 1)
    {
      densityOfStatesUp.resize(numberIntervals,0.0);
      densityOfStatesDown.resize(numberIntervals,0.0);
      for(int epsInt = 0; epsInt < numberIntervals; ++epsInt)
	{
	  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
	  for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	    {
	      for(unsigned int spinType = 0; spinType < 1 + dftParameters::spinPolarized; ++spinType)
		{
		  for(unsigned int statesIter = 0; statesIter < d_numEigenValues; ++statesIter)
		    {
		      double term1 = (epsValue - eigenValuesInput[kPoint][spinType*d_numEigenValues + statesIter]);
		      double denom = term1*term1+sigma*sigma;
		      if(spinType == 0)
			densityOfStatesUp[epsInt] += (sigma/M_PI)*(1.0/denom);
		      else
			densityOfStatesDown[epsInt] += (sigma/M_PI)*(1.0/denom);
		    }
		}
	    }
	}
    }
  else
    {
      densityOfStates.resize(numberIntervals,0.0);
      for(int epsInt = 0; epsInt < numberIntervals; ++epsInt)
	{
	  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
	  for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	    {
	      for(unsigned int statesIter = 0; statesIter < d_numEigenValues; ++statesIter)
		{
		  double term1 = (epsValue - eigenValuesInput[kPoint][statesIter]);
		  double denom = term1*term1+sigma*sigma;
		  densityOfStates[epsInt] += 2.0*(sigma/M_PI)*(1.0/denom);
		}
	    }
	}

    }

  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream outFile(dosFileName.c_str());
      outFile.setf(std::ios_base::fixed);

      if(outFile.is_open())
	{
	  if(dftParameters::spinPolarized == 1)
	    {
	      for(unsigned int epsInt = 0; epsInt < numberIntervals; ++epsInt)
		{
		  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
		  outFile << std::setprecision(18) << epsValue*27.21138602<< "  " << densityOfStatesUp[epsInt]<< " " << densityOfStatesDown[epsInt]<<std::endl;
		}
	    }
	  else
	    {
	      for(unsigned int epsInt = 0; epsInt < numberIntervals; ++epsInt)
		{
		  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
		  outFile << std::setprecision(18) << epsValue*27.21138602<< "  " << densityOfStates[epsInt]<<std::endl;
		}
	    }
	}
    }
    computing_timer.exit_section("DOS computation");
}


//compute fermi energy
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_ldos(const std::vector<std::vector<double>> & eigenValuesInput,
				     const std::string & ldosFileName)
{
  computing_timer.enter_section("LDOS computation");
  //
  //create a map of cellId and atomId
  //

  //loop over elements
  std::vector<double> eigenValuesAllkPoints;
  for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
    {
      for(int statesIter = 0; statesIter < eigenValuesInput[0].size(); ++statesIter)
	{
	  eigenValuesAllkPoints.push_back(eigenValuesInput[kPoint][statesIter]);
	}
    }

  std::sort(eigenValuesAllkPoints.begin(),eigenValuesAllkPoints.end());

  double totalEigenValues = eigenValuesAllkPoints.size();
  double intervalSize = 0.001;
  double sigma = C_kb*dftParameters::TVal;
  double lowerBoundEpsilon=1.5*eigenValuesAllkPoints[0];
  double upperBoundEpsilon=eigenValuesAllkPoints[totalEigenValues-1]*1.5;
  unsigned int numberIntervals = std::ceil((upperBoundEpsilon - lowerBoundEpsilon)/intervalSize);
  unsigned int numberGlobalAtoms = atomLocations.size();

  // map each cell to an atom based on closest atom to the centroid of each cell
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  std::map<dealii::CellId,unsigned int> cellToAtomIdMap;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  const dealii::Point<3> center(cell->center());

	  //loop over all atoms
	  double distanceToClosestAtom = 1e8;
	  Point<3> closestAtom;
	  unsigned int closestAtomId;
	  for (unsigned int n=0; n<atomLocations.size(); n++)
	    {
	      Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
	      if(center.distance(atom) < distanceToClosestAtom)
		{
		  distanceToClosestAtom = center.distance(atom);
		  closestAtom = atom;
		  closestAtomId = n;
		}
	    }
	  cellToAtomIdMap[cell->id()] = closestAtomId;
	}
    }

  std::vector<double> localDensityOfStates,localDensityOfStatesUp,localDensityOfStatesDown;
  localDensityOfStates.resize(numberGlobalAtoms*numberIntervals,0.0);
  if (dftParameters::spinPolarized==1)
  {
    localDensityOfStatesUp.resize(numberGlobalAtoms*numberIntervals,0.0);
    localDensityOfStatesDown.resize(numberGlobalAtoms*numberIntervals,0.0);
  }

  //compute density of states
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (dofHandler.get_fe(), quadrature_formula, update_values|update_JxW_values);
  const unsigned int dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();


  const unsigned int blockSize=std::min(dftParameters::wfcBlockSize,
	                                d_numEigenValues);

  std::vector<double> tempContribution(blockSize,0.0);
  std::vector<double> tempQuadPointValues(n_q_points);

  const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL[0].size()/d_numEigenValues;
  std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*d_kPointWeights.size());
  std::vector<dealii::parallel::distributed::Vector<dataTypes::number> > eigenVectorsFlattenedBlock((1+dftParameters::spinPolarized)*d_kPointWeights.size());

   for(unsigned int ivec = 0; ivec < d_numEigenValues; ivec+=blockSize)
   {
      const unsigned int currentBlockSize=std::min(blockSize,d_numEigenValues-ivec);

      if (currentBlockSize!=blockSize || ivec==0)
      {
	   for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	   {
	      eigenVectors[kPoint].resize(currentBlockSize);
	      for(unsigned int i= 0; i < currentBlockSize; ++i)
		  eigenVectors[kPoint][i].reinit(d_tempEigenVec);


	      vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							         currentBlockSize,
							         eigenVectorsFlattenedBlock[kPoint]);
	      eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
	   }

	   constraintsNoneDataInfo.precomputeMaps(matrix_free_data.get_vector_partitioner(),
					          eigenVectorsFlattenedBlock[0].get_partitioner(),
					          currentBlockSize);
      }


      std::vector<std::vector<double>> blockedEigenValues(d_kPointWeights.size(),std::vector<double>((1+dftParameters::spinPolarized)*currentBlockSize,0.0));
      for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	 for (unsigned int iWave=0; iWave<currentBlockSize;++iWave)
	 {
	     blockedEigenValues[kPoint][iWave]=eigenValues[kPoint][ivec+iWave];
	     if (dftParameters::spinPolarized==1)
		 blockedEigenValues[kPoint][currentBlockSize+iWave]
		     =eigenValues[kPoint][d_numEigenValues+ivec+iWave];
	 }

      for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
      {
	     for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		for(unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
		    eigenVectorsFlattenedBlock[kPoint].local_element(iNode*currentBlockSize+iWave)
		      = d_eigenVectorsFlattenedSTL[kPoint][iNode*d_numEigenValues+ivec+iWave];

	     constraintsNoneDataInfo.distribute(eigenVectorsFlattenedBlock[kPoint],
						currentBlockSize);
	     eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
	     vectorTools::copyFlattenedDealiiVecToSingleCompVec
		     (eigenVectorsFlattenedBlock[kPoint],
		      currentBlockSize,
		      std::make_pair(0,currentBlockSize),
		      localProc_dof_indicesReal,
		      localProc_dof_indicesImag,
		      eigenVectors[kPoint],
		      false);

	     //FIXME: The underlying call to update_ghost_values
	     //is required because currently localProc_dof_indicesReal
	     //and localProc_dof_indicesImag are only available for
	     //locally owned nodes. Once they are also made available
	     //for ghost nodes- use true for the last argument in
	     //copyFlattenedDealiiVecToSingleCompVec(..) above and supress
	     //underlying call.
	     for(unsigned int i= 0; i < currentBlockSize; ++i)
		 eigenVectors[kPoint][i].update_ghost_values();
#else
	     vectorTools::copyFlattenedDealiiVecToSingleCompVec
		     (eigenVectorsFlattenedBlock[kPoint],
		      currentBlockSize,
		      std::make_pair(0,currentBlockSize),
		      eigenVectors[kPoint],
		      true);

#endif
      }

      if(dftParameters::spinPolarized == 1)
	{
	  for(unsigned int spinType = 0; spinType < 2;++spinType)
	    {
	       typename DoFHandler<3>::active_cell_iterator cellN = dofHandler.begin_active(), endcN = dofHandler.end();

	       for(; cellN!=endcN; ++cellN)
		 {
		   if(cellN->is_locally_owned())
		     {
		       fe_values.reinit(cellN);
		       unsigned int globalAtomId = cellToAtomIdMap[cellN->id()];

		       for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
		       {
			 fe_values.get_function_values(eigenVectors[spinType][iEigenVec],
						    tempQuadPointValues);

			 tempContribution[iEigenVec]=0.0;
			 for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			 {
			  tempContribution[iEigenVec]+= tempQuadPointValues[q_point]*tempQuadPointValues[q_point]*fe_values.JxW(q_point);
			 }
		       }

		       for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			   for(unsigned int epsInt = 0; epsInt < numberIntervals; ++epsInt)
			     {
			       double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
			       double term1 = (epsValue - blockedEigenValues[0][spinType*currentBlockSize+iEigenVec]);
			       double smearedEnergyLevel = (sigma/M_PI)*(1.0/(term1*term1+sigma*sigma));

			       if(spinType == 0)
				 localDensityOfStatesUp[numberIntervals*globalAtomId + epsInt] += tempContribution[iEigenVec]*smearedEnergyLevel;
			       else
				 localDensityOfStatesDown[numberIntervals*globalAtomId + epsInt] +=tempContribution[iEigenVec]*smearedEnergyLevel;
			     }
		     }
		 }
	    }
	}
      else
	{
	  typename DoFHandler<3>::active_cell_iterator cellN = dofHandler.begin_active(), endcN = dofHandler.end();

	  for(; cellN!=endcN; ++cellN)
	    {
	      if(cellN->is_locally_owned())
		{
		  fe_values.reinit(cellN);
		  unsigned int globalAtomId = cellToAtomIdMap[cellN->id()];

		  for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
		  {
		     fe_values.get_function_values(eigenVectors[0][iEigenVec],
						tempQuadPointValues);

		     tempContribution[iEigenVec]=0.0;
		     for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		     {
		      tempContribution[iEigenVec]+= tempQuadPointValues[q_point]*tempQuadPointValues[q_point]*fe_values.JxW(q_point);
		     }
		  }

                  for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
		      for(unsigned int epsInt = 0; epsInt < numberIntervals; ++epsInt)
			{
			  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
			  double term1 = (epsValue - blockedEigenValues[0][iEigenVec]);
			  double smearedEnergyLevel = (sigma/M_PI)*(1.0/(term1*term1+sigma*sigma));
			  localDensityOfStates[numberIntervals*globalAtomId + epsInt]
			      += 2.0*tempContribution[iEigenVec]*smearedEnergyLevel;
			}
		}
	    }

	}
   }

  if(dftParameters::spinPolarized == 1)
    {

      dealii::Utilities::MPI::sum(localDensityOfStatesUp,
				  mpi_communicator,
				  localDensityOfStatesUp);

      dealii::Utilities::MPI::sum(localDensityOfStatesDown,
				  mpi_communicator,
				  localDensityOfStatesDown);
    }
  else
    {
      dealii::Utilities::MPI::sum(localDensityOfStates,
				  mpi_communicator,
				  localDensityOfStates);

    }

  double checkSum=0;
  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream outFile(ldosFileName.c_str());
      outFile.setf(std::ios_base::fixed);

      if(outFile.is_open())
	{

	  if(dftParameters::spinPolarized == 1)
	    {
	      for(unsigned int epsInt = 0; epsInt < numberIntervals; ++epsInt)
		{
		  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
		  outFile << std::setprecision(18) << epsValue*27.21138602 << " ";
		  for(unsigned int iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
		    {
		      outFile << std::setprecision(18) << localDensityOfStatesUp[numberIntervals*iAtom + epsInt]<<" "<<localDensityOfStatesDown[numberIntervals*iAtom + epsInt] << " ";;
		      checkSum+=std::fabs(localDensityOfStatesUp[numberIntervals*iAtom + epsInt])
			       +std::fabs(localDensityOfStatesDown[numberIntervals*iAtom + epsInt]);
		    }
		  outFile<<std::endl;
		}
	    }
	  else
	    {
	      for(unsigned int epsInt = 0; epsInt < numberIntervals; ++epsInt)
		{
		  double epsValue = lowerBoundEpsilon+epsInt*intervalSize;
		  outFile << std::setprecision(18) << epsValue*27.21138602 << " ";
		  for(unsigned int iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
		    {
		      outFile << std::setprecision(18) << localDensityOfStates[numberIntervals*iAtom + epsInt]<<" ";
		      checkSum+=std::fabs(localDensityOfStates[numberIntervals*iAtom + epsInt]);
		    }
		  outFile<<std::endl;
		}

	    }
	}

    }
    if (dftParameters::verbosity>=4)
       pcout<<"Absolute sum of all ldos values: "<< checkSum<< std::endl;

    computing_timer.exit_section("LDOS computation");
}









