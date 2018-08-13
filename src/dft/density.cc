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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh, Sambit Das
//

//source file for electron density related computations

//calculate electron density
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_rhoOut()
{
  const unsigned int numEigenVectors=numEigenValues;
  const unsigned int numKPoints=d_kPointWeights.size();

#ifdef USE_COMPLEX
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),2> psiEval(matrix_free_data,eigenDofHandlerIndex , 0);
#else
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),1> psiEval(matrix_free_data,eigenDofHandlerIndex , 0);
#endif
  const unsigned int numQuadPoints=psiEval.n_q_points;

  Tensor<1,2,VectorizedArray<double> > zeroTensor1;
  zeroTensor1[0]=make_vectorized_array(0.0);
  zeroTensor1[1]=make_vectorized_array(0.0);
  Tensor<1,2, Tensor<1,3,VectorizedArray<double> > > zeroTensor2;
  Tensor<1,3,VectorizedArray<double> > zeroTensor3;
  for (unsigned int idim=0; idim<3; idim++)
  {
    zeroTensor2[0][idim]=make_vectorized_array(0.0);
    zeroTensor2[1][idim]=make_vectorized_array(0.0);
    zeroTensor3[idim]=make_vectorized_array(0.0);
  }

  resizeAndAllocateRhoTableStorage
		    (rhoOutVals,
		     gradRhoOutVals,
		     rhoOutValsSpinPolarized,
		     gradRhoOutValsSpinPolarized);

  rhoOutValues = &(rhoOutVals.back());
  if (dftParameters::spinPolarized==1)
    rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());

  if(dftParameters::xc_id == 4)
    {
      gradRhoOutValues = &(gradRhoOutVals.back());
      if (dftParameters::spinPolarized==1)
         gradRhoOutValuesSpinPolarized = &(gradRhoOutValsSpinPolarized.back());
    }

   //temp arrays
   std::vector<double> rhoTemp(numQuadPoints), rhoTempSpinPolarized(2*numQuadPoints), rhoOut(numQuadPoints), rhoOutSpinPolarized(2*numQuadPoints);
   std::vector<double> gradRhoTemp(3*numQuadPoints), gradRhoTempSpinPolarized(6*numQuadPoints),gradRhoOut(3*numQuadPoints), gradRhoOutSpinPolarized(6*numQuadPoints);


   //band group parallelization data structures
   const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
   const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
   std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
   dftUtils::createBandParallelizationIndices(interBandGroupComm,
					      numEigenValues,
					      bandGroupLowHighPlusOneIndices);

   const unsigned int eigenVectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                             bandGroupLowHighPlusOneIndices[1]);

   const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL.size()/numEigenValues;

   std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*d_kPointWeights.size());



   for(unsigned int ivec = 0; ivec < numEigenValues; ivec+=eigenVectorsBlockSize)
   {
      const unsigned int currentBlockSize=std::min(eigenVectorsBlockSize,numEigenValues-ivec);

      if (currentBlockSize!=eigenVectorsBlockSize || ivec==0)
      {
	   for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	   {
	      eigenVectors[kPoint].resize(currentBlockSize);
	      for(unsigned int i= 0; i < currentBlockSize; ++i)
		  eigenVectors[kPoint][i].reinit(d_tempEigenVec);
	   }
      }

      if ((ivec+currentBlockSize)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	  (ivec+currentBlockSize)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
      {

	dealii::parallel::distributed::Vector<dataTypes::number> eigenVectorsFlattenedArrayBlock;
	    vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							       currentBlockSize,
							       eigenVectorsFlattenedArrayBlock);

	constraintsNoneDataInfo.precomputeMaps(matrix_free_data.get_partitioner(),
					       eigenVectorsFlattenedArrayBlock.get_partitioner(),
					       currentBlockSize);
	

	  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	      for(unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
		eigenVectorsFlattenedArrayBlock.local_element(iNode*currentBlockSize+iWave)
		  = d_eigenVectorsFlattenedSTL[kPoint][iNode*numEigenValues+ivec+iWave];

	    constraintsNoneDataInfo.distribute(eigenVectorsFlattenedArrayBlock,
					       currentBlockSize);
	  

#ifdef USE_COMPLEX
		 vectorTools::copyFlattenedDealiiVecToSingleCompVec
			 (eigenVectorsFlattenedArrayBlock,
			  currentBlockSize,
			  std::make_pair(0,currentBlockSize),
			  localProc_dof_indicesReal,
			  localProc_dof_indicesImag,
			  eigenVectors[kPoint]);
#else
		 vectorTools::copyFlattenedDealiiVecToSingleCompVec
			 (eigenVectorsFlattenedArrayBlock,
			  currentBlockSize,
			  std::make_pair(0,currentBlockSize),
			  eigenVectors[kPoint]);

#endif
	  }

#ifdef USE_COMPLEX
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads2(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiQuads2(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
#else
	  std::vector< VectorizedArray<double> > psiQuads(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	  std::vector< VectorizedArray<double> > psiQuads2(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*currentBlockSize,zeroTensor3);
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiQuads2(numQuadPoints*currentBlockSize,zeroTensor3);
#endif

	  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell)
	  {
		  psiEval.reinit(cell);

		  const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);

		  for(unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
		      for(unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			{

			   psiEval.read_dof_values_plain
			       (eigenVectors[(1+dftParameters::spinPolarized)*kPoint][iEigenVec]);

			   if(dftParameters::xc_id == 4)
			      psiEval.evaluate(true,true);
			   else
			      psiEval.evaluate(true,false);

			   for (unsigned int q=0; q<numQuadPoints; ++q)
			   {
			     psiQuads[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_value(q);
			     if(dftParameters::xc_id == 4)
				gradPsiQuads[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_gradient(q);
			   }

			   if(dftParameters::spinPolarized==1)
			   {

			       psiEval.read_dof_values_plain
				   (eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][iEigenVec]);

			       if(dftParameters::xc_id == 4)
				  psiEval.evaluate(true,true);
			       else
				  psiEval.evaluate(true,false);

			       for (unsigned int q=0; q<numQuadPoints; ++q)
			       {
				 psiQuads2[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_value(q);
				 if(dftParameters::xc_id == 4)
				    gradPsiQuads2[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_gradient(q);
			       }
			   }
			}//eigenvector per k point

		  for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		  {
			const dealii::CellId subCellId=matrix_free_data.get_cell_iterator(cell,iSubCell)->id();

			std::fill(rhoTemp.begin(),rhoTemp.end(),0.0); std::fill(rhoOut.begin(),rhoOut.end(),0.0);

			if (dftParameters::spinPolarized==1)
			    std::fill(rhoTempSpinPolarized.begin(),rhoTempSpinPolarized.end(),0.0);

			if(dftParameters::xc_id == 4)
			{
			  std::fill(gradRhoTemp.begin(),gradRhoTemp.end(),0.0);
			  if (dftParameters::spinPolarized==1)
			      std::fill(gradRhoTempSpinPolarized.begin(),gradRhoTempSpinPolarized.end(),0.0);
			}

			for(unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
			  for(unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			    {

			      const double partialOccupancy=dftUtils::getPartialOccupancy
							    (eigenValues[kPoint][ivec+iEigenVec],
							     fermiEnergy,
							     C_kb,
							     dftParameters::TVal);

			      const double partialOccupancy2=dftUtils::getPartialOccupancy
							    (eigenValues[kPoint][ivec+iEigenVec+dftParameters::spinPolarized*numEigenVectors],
							     fermiEnergy,
							     C_kb,
							     dftParameters::TVal);

			      for(unsigned int q=0; q<numQuadPoints; ++q)
				{
				  const unsigned int id=q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec;
#ifdef USE_COMPLEX
				  Vector<double> psi, psi2;
				  psi.reinit(2); psi2.reinit(2);

				  psi(0)= psiQuads[id][0][iSubCell];
				  psi(1)=psiQuads[id][1][iSubCell];

				  if(dftParameters::spinPolarized==1)
				  {
				    psi2(0)=psiQuads2[id][0][iSubCell];
				    psi2(1)=psiQuads2[id][1][iSubCell];
				  }

				  std::vector<Tensor<1,3,double> > gradPsi(2),gradPsi2(2);

				  if(dftParameters::xc_id == 4)
				      for(unsigned int idim=0; idim<3; ++idim)
				      {
					 gradPsi[0][idim]=gradPsiQuads[id][0][idim][iSubCell];
					 gradPsi[1][idim]=gradPsiQuads[id][1][idim][iSubCell];

					 if(dftParameters::spinPolarized==1)
					 {
					     gradPsi2[0][idim]=gradPsiQuads2[id][0][idim][iSubCell];
					     gradPsi2[1][idim]=gradPsiQuads2[id][1][idim][iSubCell];
					 }
				      }
#else
				  double psi, psi2;
				  psi=psiQuads[id][iSubCell];
				  if (dftParameters::spinPolarized==1)
				      psi2=psiQuads2[id][iSubCell];

				  Tensor<1,3,double> gradPsi,gradPsi2;
				  if(dftParameters::xc_id == 4)
				      for(unsigned int idim=0; idim<3; ++idim)
				      {
					 gradPsi[idim]=gradPsiQuads[id][idim][iSubCell];
					 if(dftParameters::spinPolarized==1)
					     gradPsi2[idim]=gradPsiQuads2[id][idim][iSubCell];
				      }

#endif

#ifdef USE_COMPLEX
				  if(dftParameters::spinPolarized==1)
				    {
				      rhoTempSpinPolarized[2*q] += partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*psi(0) + psi(1)*psi(1));
				      rhoTempSpinPolarized[2*q+1] += partialOccupancy2*d_kPointWeights[kPoint]*(psi2(0)*psi2(0) + psi2(1)*psi2(1));
				      //
				      if(dftParameters::xc_id == 4)
					  for(unsigned int idim=0; idim<3; ++idim)
					  {
					      gradRhoTempSpinPolarized[6*q + idim] +=
					      2.0*partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*gradPsi[0][idim] + psi(1)*gradPsi[1][idim]);
					      gradRhoTempSpinPolarized[6*q + 3+idim] +=
					      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(psi2(0)*gradPsi2[0][idim] + psi2(1)*gradPsi2[1][idim]);
					  }
				    }
				  else
				    {
				      rhoTemp[q] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*psi(0) + psi(1)*psi(1));
				      if(dftParameters::xc_id == 4)
					for(unsigned int idim=0; idim<3; ++idim)
					   gradRhoTemp[3*q + idim] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*gradPsi[0][idim] + psi(1)*gradPsi[1][idim]);
				    }
#else
				  if(dftParameters::spinPolarized==1)
				    {
				      rhoTempSpinPolarized[2*q] += partialOccupancy*psi*psi;
				      rhoTempSpinPolarized[2*q+1] += partialOccupancy2*psi2*psi2;

				      if(dftParameters::xc_id == 4)
					  for(unsigned int idim=0; idim<3; ++idim)
					  {
					      gradRhoTempSpinPolarized[6*q + idim] += 2.0*partialOccupancy*(psi*gradPsi[idim]);
					      gradRhoTempSpinPolarized[6*q + 3+idim] +=  2.0*partialOccupancy2*(psi2*gradPsi2[idim]);
					  }
				    }
				  else
				    {
				      rhoTemp[q] += 2.0*partialOccupancy*psi*psi;

				      if(dftParameters::xc_id == 4)
					for(unsigned int idim=0; idim<3; ++idim)
					   gradRhoTemp[3*q + idim] += 2.0*2.0*partialOccupancy*psi*gradPsi[idim];
				    }

#endif
				}//quad point loop
			    }//block eigenvectors per k point

			for (unsigned int q=0; q<numQuadPoints; ++q)
			{
			    if(dftParameters::spinPolarized==1)
			    {
				    (*rhoOutValuesSpinPolarized)[subCellId][2*q]+=rhoTempSpinPolarized[2*q];
				    (*rhoOutValuesSpinPolarized)[subCellId][2*q+1]+=rhoTempSpinPolarized[2*q+1];

				    if(dftParameters::xc_id == 4)
					for(unsigned int idim=0; idim<3; ++idim)
					{
					  (*gradRhoOutValuesSpinPolarized)[subCellId][6*q+idim]
					      +=gradRhoTempSpinPolarized[6*q + idim];
					  (*gradRhoOutValuesSpinPolarized)[subCellId][6*q+3+idim]
					      +=gradRhoTempSpinPolarized[6*q + 3+idim];
				       }

				    (*rhoOutValues)[subCellId][q]+= rhoTempSpinPolarized[2*q] + rhoTempSpinPolarized[2*q+1];

				    if(dftParameters::xc_id == 4)
				      for(unsigned int idim=0; idim<3; ++idim)
					(*gradRhoOutValues)[subCellId][3*q + idim]
					    += gradRhoTempSpinPolarized[6*q + idim]
					       + gradRhoTempSpinPolarized[6*q + 3+idim];
			     }
			     else
			     {
				    (*rhoOutValues)[subCellId][q] += rhoTemp[q];

				     if(dftParameters::xc_id == 4)
					 for(unsigned int idim=0; idim<3; ++idim)
					    (*gradRhoOutValues)[subCellId][3*q+idim]+= gradRhoTemp[3*q+idim];
			     }
			}
		  }//subcell loop
	   }//macro cell loop
	}//band parallelization
   }//eigenvectors block loop

   //gather density from all inter communicators
   sumRhoData(rhoOutValues,
	      gradRhoOutValues,
	      rhoOutValuesSpinPolarized,
	      gradRhoOutValuesSpinPolarized,
	      interBandGroupComm);

   sumRhoData(rhoOutValues,
	      gradRhoOutValues,
	      rhoOutValuesSpinPolarized,
	      gradRhoOutValuesSpinPolarized,
	      interpoolcomm);

  //pop out rhoInVals and rhoOutVals if their size exceeds mixing history size
  if(rhoInVals.size() == dftParameters::mixingHistory)
    {
      rhoInVals.pop_front();
      rhoOutVals.pop_front();

      if(dftParameters::spinPolarized==1)
      {
	  rhoInValsSpinPolarized.pop_front();
	  rhoOutValsSpinPolarized.pop_front();
      }

      if(dftParameters::xc_id == 4)//GGA
      {
	  gradRhoInVals.pop_front();
	  gradRhoOutVals.pop_front();
      }

      if(dftParameters::spinPolarized==1 && dftParameters::xc_id==4)
      {
	  gradRhoInValsSpinPolarized.pop_front();
	  gradRhoOutValsSpinPolarized.pop_front();
      }
    }

}

template<unsigned int FEOrder>
void dftClass<FEOrder>::resizeAndAllocateRhoTableStorage
		    (std::deque<std::map<dealii::CellId,std::vector<double> >> & rhoVals,
		     std::deque<std::map<dealii::CellId,std::vector<double> >> & gradRhoVals,
		     std::deque<std::map<dealii::CellId,std::vector<double> >> & rhoValsSpinPolarized,
		     std::deque<std::map<dealii::CellId,std::vector<double> >> & gradRhoValsSpinPolarized)
{
  const unsigned int numQuadPoints = matrix_free_data.get_n_q_points(0);;

  //create new rhoValue tables
  rhoVals.push_back(std::map<dealii::CellId,std::vector<double> > ());
  if (dftParameters::spinPolarized==1)
	rhoValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> > ());

  if(dftParameters::xc_id == 4)
    {
      gradRhoVals.push_back(std::map<dealii::CellId, std::vector<double> >());
      if (dftParameters::spinPolarized==1)
         gradRhoValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
    }


   typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
   for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
	{
	    const dealii::CellId cellId=cell->id();
	    rhoVals.back()[cellId] = std::vector<double>(numQuadPoints,0.0);
	    if(dftParameters::xc_id == 4)
		gradRhoVals.back()[cellId] = std::vector<double>(3*numQuadPoints,0.0);

	    if (dftParameters::spinPolarized==1)
	    {
		 rhoValsSpinPolarized.back()[cellId] = std::vector<double>(2*numQuadPoints,0.0);
		 if(dftParameters::xc_id == 4)
		    gradRhoValsSpinPolarized.back()[cellId]
		     = std::vector<double>(6*numQuadPoints,0.0);
	    }
	}
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::sumRhoData(std::map<dealii::CellId, std::vector<double> > * rhoValues,
	              std::map<dealii::CellId, std::vector<double> > * gradRhoValues,
	              std::map<dealii::CellId, std::vector<double> > * rhoValuesSpinPolarized,
		      std::map<dealii::CellId, std::vector<double> > * gradRhoValuesSpinPolarized,
		      const MPI_Comm &interComm)
{
   typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

   //gather density from inter communicator
   if (dealii::Utilities::MPI::n_mpi_processes(interComm)>1)
      for (; cell!=endc; ++cell)
	  if (cell->is_locally_owned())
	    {
		    const dealii::CellId cellId=cell->id();

		    dealii::Utilities::MPI::sum((*rhoValues)[cellId],
						interComm,
						(*rhoValues)[cellId]);
		    if(dftParameters::xc_id == 4)
		       dealii::Utilities::MPI::sum((*gradRhoValues)[cellId],
						   interComm,
						   (*gradRhoValues)[cellId]);

		    if (dftParameters::spinPolarized==1)
		    {
			dealii::Utilities::MPI::sum((*rhoValuesSpinPolarized)[cellId],
						    interComm,
						    (*rhoValuesSpinPolarized)[cellId]);
			if(dftParameters::xc_id == 4)
			   dealii::Utilities::MPI::sum((*gradRhoValuesSpinPolarized)[cellId],
						       interComm,
						       (*gradRhoValuesSpinPolarized)[cellId]);
		    }
	      }
}

//rho data reinitilization without remeshing. The rho out of last ground state solve is made the rho in of the new solve
template<unsigned int FEOrder>
void dftClass<FEOrder>::noRemeshRhoDataInit()
{
  //create temporary copies of rho Out data
  std::map<dealii::CellId, std::vector<double> > rhoOutValuesCopy=*(rhoOutValues);

  std::map<dealii::CellId, std::vector<double> > gradRhoOutValuesCopy;
  if (dftParameters::xc_id==4)
  {
     gradRhoOutValuesCopy=*(gradRhoOutValues);
  }

  std::map<dealii::CellId, std::vector<double> > rhoOutValuesSpinPolarizedCopy;
  if(dftParameters::spinPolarized==1)
  {
     rhoOutValuesSpinPolarizedCopy=*(rhoOutValuesSpinPolarized);

  }

  std::map<dealii::CellId, std::vector<double> > gradRhoOutValuesSpinPolarizedCopy;
  if(dftParameters::spinPolarized==1 && dftParameters::xc_id==4)
  {
     gradRhoOutValuesSpinPolarizedCopy=*(gradRhoOutValuesSpinPolarized);

  }
  //cleanup of existing rho Out and rho In data
  clearRhoData();

  ///copy back temporary rho out to rho in data
  rhoInVals.push_back(rhoOutValuesCopy);
  rhoInValues=&(rhoInVals.back());

  if (dftParameters::xc_id==4)
  {
    gradRhoInVals.push_back(gradRhoOutValuesCopy);
    gradRhoInValues=&(gradRhoInVals.back());
  }

  if(dftParameters::spinPolarized==1)
  {
    rhoInValsSpinPolarized.push_back(rhoOutValuesSpinPolarizedCopy);
    rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
  }

  if (dftParameters::xc_id==4 && dftParameters::spinPolarized==1)
  {
    gradRhoInValsSpinPolarized.push_back(gradRhoOutValuesSpinPolarizedCopy);
    gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
  }

  normalizeRho();

}
