#include "../../include2/eigen.h"

//constructor
eigenClass::eigenClass(dftClass* _dftPtr):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
}


//initialize eigenClass object
void eigenClass::init(){
  computing_timer.enter_section("eigenClass setup");
  //init vectors
  for (unsigned int i=0; i<dftPtr->numEigenValues; ++i){
    vectorType* temp=new vectorType;
    HXvalue.push_back(temp);
  } 
  dftPtr->matrix_free_data.initialize_dof_vector(massVector,dftPtr->eigenDofHandlerIndex);
  //compute mass vector
  computeMassVector();
  //XHX size
  XHXValue.resize(dftPtr->eigenVectors[0].size()*dftPtr->eigenVectors[0].size(),0.0);
  //HX
  for (unsigned int i=0; i<dftPtr->numEigenValues; ++i){
    HXvalue[i]->reinit(dftPtr->vChebyshev);
  }
  computing_timer.exit_section("eigenClass setup"); 
} 

void eigenClass::computeMassVector(){
  computing_timer.enter_section("eigenClass Mass assembly"); 
  
  
#ifdef ENABLE_PERIODIC_BC
  Tensor<1,2,VectorizedArray<double> > one;
  one[0] =  make_vectorized_array (1.0);
  one[1] =  make_vectorized_array (1.0);
  FEEvaluation<3,FEOrder,FEOrder+1,2,double>  fe_eval(dftPtr->matrix_free_data, dftPtr->eigenDofHandlerIndex, 1);
#else
  VectorizedArray<double>  one = make_vectorized_array (1.0);
  FEEvaluation<3,FEOrder,FEOrder+1,1,double>  fe_eval(dftPtr->matrix_free_data, dftPtr->eigenDofHandlerIndex, 1); //Selecting GL quadrature points
#endif
  const unsigned int       n_q_points = fe_eval.n_q_points;
  for (unsigned int cell=0; cell<dftPtr->matrix_free_data.n_macro_cells(); ++cell){
    fe_eval.reinit(cell);
    for (unsigned int q=0; q<n_q_points; ++q) fe_eval.submit_value(one,q);
    fe_eval.integrate (true,false);
    fe_eval.distribute_local_to_global (massVector);
  }
  massVector.compress(VectorOperation::add);
  //compute inverse
  for (unsigned int i=0; i<massVector.local_size(); i++){
    if (std::abs(massVector.local_element(i))>1.0e-15){
      massVector.local_element(i)=1.0/std::sqrt(massVector.local_element(i));
    }
    else{
      massVector.local_element(i)=0.0;
    }
  }
  //pcout << "massVector norm: " << massVector.l2_norm() << "\n";
  computing_timer.exit_section("eigenClass Mass assembly");
}

#ifdef xc_id
#if xc_id < 4
void eigenClass::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
			     const vectorType & phi,
			     const vectorType & phiExt,
			     std::map<dealii::CellId,std::vector<double> >* pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder> fe_eval_phiExt(dftPtr->matrix_free_data, 0 ,0);
  int numberQuadraturePoints = fe_eval_phi.n_q_points;
  vEff.reinit (n_cells, numberQuadraturePoints);
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  //loop over cell block
  //
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {

      //
      //extract total potential
      //
      fe_eval_phi.reinit(cell);
      fe_eval_phi.read_dof_values(phi);
      fe_eval_phi.evaluate(true, false, false);

      //
      //extract phiExt
      //
      fe_eval_phiExt.reinit(cell);
      fe_eval_phiExt.read_dof_values(phiExt);
      fe_eval_phiExt.evaluate(true, false, false);



      for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
	{
	  //
	  //loop over each cell
	  //
	  unsigned int n_sub_cells=dftPtr->matrix_free_data.n_components_filled(cell);
	  std::vector<double> densityValue(n_sub_cells), exchangePotentialVal(n_sub_cells), corrPotentialVal(n_sub_cells);
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      densityValue[v] = ((*rhoValues)[cellPtr->id()][q]);
	    }

	  xc_lda_vxc(&funcX,n_sub_cells,&densityValue[0],&exchangePotentialVal[0]);
	  xc_lda_vxc(&funcC,n_sub_cells,&densityValue[0],&corrPotentialVal[0]);

	  VectorizedArray<double>  exchangePotential, corrPotential;
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      exchangePotential[v]=exchangePotentialVal[v];
	      corrPotential[v]=corrPotentialVal[v];
	    }

	  //
	  //sum all to vEffective
	  //
	  if(isPseudopotential)
	    {
	      VectorizedArray<double>  pseudoPotential;
	      for (unsigned int v = 0; v < n_sub_cells; ++v)
		{
		  cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
		  pseudoPotential[v]=((*pseudoValues)[cellPtr->id()][q]);
		}
	      vEff(cell,q)=fe_eval_phi.get_value(q)+exchangePotential+corrPotential+(pseudoPotential-fe_eval_phiExt.get_value(q));
	    }
	  else
	    {  
	      vEff(cell,q)=fe_eval_phi.get_value(q)+exchangePotential+corrPotential;
	    }
	}
    }
}
#elif xc_id == 4
void eigenClass::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
			     std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
			     const vectorType & phi,
			     const vectorType & phiExt,
			     std::map<dealii::CellId,std::vector<double> >* pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder> fe_eval_phiExt(dftPtr->matrix_free_data, 0 ,0);
  int numberQuadraturePoints = fe_eval_phi.n_q_points;
  vEff.reinit (n_cells, numberQuadraturePoints);
  derExcWithSigmaTimesGradRho(n_cells, numberQuadraturePoints, 3);
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  //loop over cell block
  //
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {

      //
      //extract total potential
      //
      fe_eval_phi.reinit(cell);
      fe_eval_phi.read_dof_values(phi);
      fe_eval_phi.evaluate(true, false, false);

      //
      //extract phiExt
      //
      fe_eval_phiExt.reinit(cell);
      fe_eval_phiExt.read_dof_values(phiExt);
      fe_eval_phiExt.evaluate(true, false, false);



      for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
	{
	  //
	  //loop over each cell
	  //
	  unsigned int n_sub_cells=dftPtr->matrix_free_data.n_components_filled(cell);
	  std::vector<double> densityValue(n_sub_cells), derExchEnergyWithDensityVal(n_sub_cells), derCorrEnergyWithDensityVal(n_sub_cells), derExchEnergyWithSigma(n_sub_cells), derCorrEnergyWithSigma(n_sub_cells), sigmaValue(n_sub_cells);
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      densityValue[v] = ((*rhoValues)[cellPtr->id()][q]);
	      double gradRhoX = ((*gradRhoValues)[cellPtr->id()][3*q + 0]);
	      double gradRhoY = ((*gradRhoValues)[cellPtr->id()][3*q + 1]);
	      double gradRhoZ = ((*gradRhoValues)[cellPtr->id()][3*q + 2]);
	      sigmaValue[v] = gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ;
	    }
	
	  xc_gga_vxc(&funcX,n_sub_cells,&densityValue[0],&sigmaValue[0],&derExchEnergyWithDensityVal[0],&derExchEnergyWithSigma[0]);
	  xc_gga_vxc(&funcC,n_sub_cells,&densityValue[0],&sigmaValue[0],&derCorrEnergyWithDensityVal[0],&derCorrEnergyWithSigma[0]);


	  VectorizedArray<double>  derExchEnergyWithDensity, derCorrEnergyWithDensity, derExcWithSigmaTimesGradRhoX, derExcWithSigmaTimesGradRhoY, derExcWithSigmaTimesGradRhoZ;
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      derExchEnergyWithDensity[v]=derExchEnergyWithDensityVal[v];
	      derCorrEnergyWithDensity[v]=derCorrEnergyWithDensityVal[v];
	      double gradRhoX = ((*gradRhoValues)[cellPtr->id()][3*q + 0]);
	      double gradRhoY = ((*gradRhoValues)[cellPtr->id()][3*q + 1]);
	      double gradRhoZ = ((*gradRhoValues)[cellPtr->id()][3*q + 2]);
	      derExcWithSigmaTimesGradRhoX[v] = (derExchEnergyWithSigma[v]+derCorrEnergyWithSigma[v])*gradRhoX;
	      derExcWithSigmaTimesGradRhoY[v] = (derExchEnergyWithSigma[v]+derCorrEnergyWithSigma[v])*gradRhoY;
	      derExcWithSigmaTimesGradRhoZ[v] = (derExchEnergyWithSigma[v]+derCorrEnergyWithSigma[v])*gradRhoZ;
	    }

	  //
	  //sum all to vEffective
	  //
	  if(isPseudopotential)
	    {
	      VectorizedArray<double>  pseudoPotential;
	      for (unsigned int v = 0; v < n_sub_cells; ++v)
		{
		  cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
		  pseudoPotential[v]=((*pseudoValues)[cellPtr->id()][q]);
		}
	      vEff(cell,q)=fe_eval_phi.get_value(q)+derExchEnergyWithDensity+derCorrEnergyWithDensity+(pseudoPotential-fe_eval_phiExt.get_value(q));
	      derExcWithSigmaTimesGradRho(cell,q,0) = derExcWithSigmaTimesGradRhoX;
	      derExcWithSigmaTimesGradRho(cell,q,1) = derExcWithSigmaTimesGradRhoY;
	      derExcWithSigmaTimesGradRho(cell,q,2) = derExcWithSigmaTimesGradRhoZ;
	    }
	  else
	    {  
	      vEff(cell,q)=fe_eval_phi.get_value(q)+derExchEnergyWithDensity+derCorrEnergyWithDensity;
	      derExcWithSigmaTimesGradRho(cell,q,0) = derExcWithSigmaTimesGradRhoX;
	      derExcWithSigmaTimesGradRho(cell,q,1) = derExcWithSigmaTimesGradRhoY;
	      derExcWithSigmaTimesGradRho(cell,q,2) = derExcWithSigmaTimesGradRhoZ;
	    }
	}
    }
}
#endif
#endif

void eigenClass::computeNonLocalHamiltonianTimesX(const std::vector<vectorType*> &src,
						  std::vector<vectorType*>       &dst)
{

  const int kPointIndex = dftPtr->d_kPointIndex;
  const unsigned int numberElements  = dftPtr->triangulation.n_locally_owned_active_cells();

  int numberNodesPerElement  = GeometryInfo<3>::vertices_per_cell;

   //compute nonlocal projector ket times x i.e C^{T}*X 
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::vector<std::complex<double> > > projectorKetTimesVector;
  std::vector<dealii::types::global_dof_index> local_dof_indices(2*numberNodesPerElement);
#else
  std::vector<std::vector<double> > projectorKetTimesVector;
  std::vector<dealii::types::global_dof_index> local_dof_indices(numberNodesPerElement);
#endif


  //get number of Nonlocal atoms
  const int numberNonLocalAtoms = dftPtr->d_nonLocalAtomGlobalChargeIds.size();
  int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();

  //allocate memory for matrix-vector product
  projectorKetTimesVector.resize(numberNonLocalAtoms);
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      projectorKetTimesVector[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }
  
  //
  //some useful vectors
  //
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif
  
  //
  //parallel loop over all elements to compute nonlocal projector ket times x i.e C^{T}*X 
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandlerEigen.begin_active(), endc = dftPtr->dofHandlerEigen.end();
  int iElem = -1;
  for(; cell!=endc; ++cell) 
    {
      if(cell->is_locally_owned())
	{
	  iElem += 1;
	  cell->get_dof_indices(local_dof_indices);

	  unsigned int index=0;
#ifdef ENABLE_PERIODIC_BC
	  std::vector<double> temp(2*numberNodesPerElement,0.0);
	  for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
	      (*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
	      for(int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		{
		  inputVectors[numberNodesPerElement*index + iNode].real(temp[2*iNode]);
		  inputVectors[numberNodesPerElement*index + iNode].imag(temp[2*iNode+1]);
		}
	      index++;
	    }
	  /* if(iElem == 0)
            {
              std::cout<<"Input Vectors for wave 0: "<<std::endl;  
              for(int i = 0; i < numberNodesPerElement; ++i)
		{
		  std::cout<<inputVectors[numberNodesPerElement*0 + i].real()<<std::endl;
		  std::cout<<inputVectors[numberNodesPerElement*0 + i].imag()<<std::endl;
		}

	      std::cout<<"Input Vectors for wave 3: "<<std::endl;  
              for(int i = 0; i < numberNodesPerElement; ++i)
		{
		  std::cout<<inputVectors[numberNodesPerElement*3 + i].real()<<std::endl;
		  std::cout<<inputVectors[numberNodesPerElement*3 + i].imag()<<std::endl;
		}

		}*/

#else
	  for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
	      (*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), inputVectors.begin()+numberNodesPerElement*index);
	      index++;
	    }
#endif

	  for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];
#ifdef ENABLE_PERIODIC_BC
	      char transA = 'C';
	      char transB = 'N';
	      std::complex<double> alpha = 1.0;
	      std::complex<double> beta = 1.0;
	      zgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberPseudoWaveFunctions);
#else
	      char transA = 'T';
	      char transB = 'N';
	      double alpha = 1.0;
	      double beta = 1.0;
	      dgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberPseudoWaveFunctions);
#endif
	    }

	}

    }//element loop

#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > tempVectorloc;
  std::vector<std::complex<double> > tempVector;
#else
  std::vector<double> tempVector;
#endif

  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      for(int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
#ifdef ENABLE_PERIODIC_BC
	      tempVectorloc.push_back(projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave]);
#else
	      tempVector.push_back(projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave]);
#endif
	    }

	}

    }



#ifdef ENABLE_PERIODIC_BC
  int size = tempVectorloc.size();
  tempVector.resize(size);
  MPI_Allreduce(&tempVectorloc[0],
		&tempVector[0],
		size,
		MPI_C_DOUBLE_COMPLEX,
		MPI_SUM,
		mpi_communicator);
#else
  Utilities::MPI::sum(tempVector,
  		      mpi_communicator,
  		      tempVector);
#endif

  int count = 0;
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      for(int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	      projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] = tempVector[count];
	      count += 1;
	    }

	}
    }

  //
  //compute V*C^{T}*X
  //
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      for(int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[iAtom][iPseudoAtomicWave];
	}
    }
  
  
  char transA1 = 'N';
  char transB1 = 'N';
 	  
  //
  //access elementIdsInAtomCompactSupport
  //
  
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > outputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> outputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif

  //
  //compute C*V*C^{T}*x
  //
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      for(int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[iAtom].size(); ++iElemComp)
	{

	  DoFHandler<3>::active_cell_iterator cell = dftPtr->d_elementIteratorsInAtomCompactSupport[iAtom][iElemComp];

#ifdef PETSC_USE_COMPLEX
	  
	  std::complex<double> alpha1 = 1.0;
	  std::complex<double> beta1 = 0.0;

	  zgemm_(&transA1,
		 &transB1,
		 &numberNodesPerElement,
		 &numberWaveFunctions,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &dftPtr->d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPointIndex][0],
		 &numberNodesPerElement,
		 &projectorKetTimesVector[iAtom][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#else
	  double alpha1 = 1.0;
	  double beta1 = 0.0;

	  dgemm_(&transA1,
		 &transB1,
		 &numberNodesPerElement,
		 &numberWaveFunctions,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &dftPtr->d_nonLocalProjectorElementMatrices[iAtom][iElemComp][kPointIndex][0],
		 &numberNodesPerElement,
		 &projectorKetTimesVector[iAtom][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#endif


	  cell->get_dof_indices(local_dof_indices);

	  /*if(iElemComp == 100)
            {
              std::cout<<"Output Vectors for wave 0: "<<std::endl;  
              for(int i = 0; i < numberNodesPerElement; ++i)
		{
		  std::cout<<outputVectors[numberNodesPerElement*0 + i].real()<<std::endl;
		  std::cout<<outputVectors[numberNodesPerElement*0 + i].imag()<<std::endl;
		}

	      std::cout<<"Output Vectors for wave 3: "<<std::endl;  
              for(int i = 0; i < numberNodesPerElement; ++i)
		{
		  std::cout<<outputVectors[numberNodesPerElement*3 + i].real()<<std::endl;
		  std::cout<<outputVectors[numberNodesPerElement*3 + i].imag()<<std::endl;
		}

		}*/

#ifdef ENABLE_PERIODIC_BC
	  unsigned int index = 0;
	  std::vector<double> temp(2*numberNodesPerElement,0.0);
	  for(std::vector<vectorType*>::iterator it = dst.begin(); it != dst.end(); ++it)
	    {
	      for(int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		{
		  temp[2*iNode]   = outputVectors[numberNodesPerElement*index + iNode].real();
		  temp[2*iNode+1] = outputVectors[numberNodesPerElement*index + iNode].imag();
		}
	      dftPtr->constraintsNoneEigen.distribute_local_to_global(temp.begin(), temp.end(),local_dof_indices.begin(), **it);
	      index++;
	    }
#else
	  std::vector<double>::iterator iter = outputVectors.begin();
	  for (std::vector<vectorType*>::iterator it=dst.begin(); it!=dst.end(); ++it)
	    {
	      dftPtr->constraintsNoneEigen.distribute_local_to_global(iter, iter+numberNodesPerElement,local_dof_indices.begin(), **it);
	      iter+=numberNodesPerElement;
	    }
#endif

	}

    }


  for (std::vector<vectorType*>::iterator it=dst.begin(); it!=dst.end(); it++)
    {
      (*it)->compress(VectorOperation::add);
    }

}
						  

//HX
/*void eigenClass::implementHX (const dealii::MatrixFree<3,double>  &data,
			      std::vector<vectorType*>  &dst, 
			      const std::vector<vectorType*>  &src,
			      const std::pair<unsigned int,unsigned int> &cell_range) const
{
  VectorizedArray<double>  half = make_vectorized_array(0.5);
  FEEvaluation<3,FEOrder>  fe_eval(data, 0, 0);
#ifdef ENABLE_PERIODIC_BC
  int kPointIndex = dftPtr->d_kPointIndex;
  Tensor<1,3,VectorizedArray<std::complex<double> > > kPointValues;
  kPointValues[0].real() = 0.0; kPointValues[0].imag() = -dftPtr->d_kPointCoordinates[3*kPointIndex+0];
  kPointValues[1].real() = 0.0; kPointValues[1].imag() = -dftPtr->d_kPointCoordinates[3*kPointIndex+1];
  kPointValues[2].real() = 0.0; kPointValues[2].imag() = -dftPtr->d_kPointCoordinates[3*kPointIndex+2];
  Tensor<1,1,VectorizedArray<double> > kSquare;
  kSquare = 0.5*(dftPtr->d_kPointCoordinates[3*kPointIndex+0]*dftPtr->d_kPointCoordinates[3*kPointIndex+0] + dftPtr->d_kPointCoordinates[3*kPointIndex+1]*dftPtr->d_kPointCoordinates[3*kPointIndex+1] + dftPtr->d_kPointCoordinates[3*kPointIndex+2]*dftPtr->d_kPointCoordinates[3*kPointIndex+2]);
#endif
  for(unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell); 
      for(unsigned int i = 0; i < dst.size(); i++)
	{
	  fe_eval.read_dof_values(*src[i]);
	  fe_eval.evaluate (true,true,false);
	  for(unsigned int q=0; q<fe_eval.n_q_points; ++q)
	    {
	      fe_eval.submit_gradient (fe_eval.get_gradient(q)*half, q);
	      fe_eval.submit_value    (fe_eval.get_value(q)*vEff(cell,q), q);
#ifdef ENABLE_PERIODIC_BC
	      Tensor<1,3,VectorizedArray<std::complex<double> > > temp = fe_eval.get_gradient(q);
	      VectorizedArray<std::complex<double> > kDotGradPsi = temp[0]*kPointValues[0] + temp[1]*kPointValues[1] + temp[2]*kPointValues[2];
	      fe_eval.submit_value(kDotGradPsi,q);
	      fe_eval.submit_value(fe_eval.get_value(q)*kSquare,q);
#endif
	    }
	  fe_eval.integrate (true, true);
	  fe_eval.distribute_local_to_global (*dst[i]);
	}
    }
    }*/
void eigenClass::implementHX (const dealii::MatrixFree<3,double>  &data,
			      std::vector<vectorType*>  &dst, 
			      const std::vector<vectorType*>  &src,
			      const std::pair<unsigned int,unsigned int> &cell_range) const
{
  VectorizedArray<double>  half = make_vectorized_array(0.5);

#ifdef ENABLE_PERIODIC_BC
  int kPointIndex = dftPtr->d_kPointIndex;
  FEEvaluation<3,FEOrder, FEOrder+1, 2, double>  fe_eval(data, dftPtr->eigenDofHandlerIndex, 0);
  Tensor<1,2,VectorizedArray<double> > psiVal, vEffTerm, kSquareTerm, kDotGradientPsiTerm, derExchWithSigmaTimesGradRhoDotGradientPsiTerm;
  Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > gradientPsiVal, gradientPsiTerm, derExchWithSigmaTimesGradRhoTimesPsi; 

  Tensor<1,3,VectorizedArray<double> > kPointCoors;
  kPointCoors[0] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+0]);
  kPointCoors[1] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+1]);
  kPointCoors[2] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+2]);

  double kSquareTimesHalf =  0.5*(dftPtr->d_kPointCoordinates[3*kPointIndex+0]*dftPtr->d_kPointCoordinates[3*kPointIndex+0] + dftPtr->d_kPointCoordinates[3*kPointIndex+1]*dftPtr->d_kPointCoordinates[3*kPointIndex+1] + dftPtr->d_kPointCoordinates[3*kPointIndex+2]*dftPtr->d_kPointCoordinates[3*kPointIndex+2]);
  VectorizedArray<double> halfkSquare = make_vectorized_array(kSquareTimesHalf);

  for(unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell); 
      for(unsigned int i = 0; i < dst.size(); ++i)
	{
	  fe_eval.read_dof_values(*src[i]);
	  fe_eval.evaluate (true,true,false);
	  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
	    {
	      //
	      //get the quadrature point values of psi and gradPsi which are complex
	      //
	      psiVal = fe_eval.get_value(q);
	      gradientPsiVal = fe_eval.get_gradient(q);

	      //
	      //compute gradientPsiTerm of the stiffnessMatrix times vector (0.5*gradientPsi)
	      //
	      gradientPsiTerm[0] = gradientPsiVal[0]*half;
	      gradientPsiTerm[1] = gradientPsiVal[1]*half;

	      //
	      //compute Veff part of the stiffness matrix times vector (Veff*psi)
	      //
	      vEffTerm[0] = psiVal[0]*vEff(cell,q);
	      vEffTerm[1] = psiVal[1]*vEff(cell,q);

	      //
	      //compute term involving dot product of k-vector and gradientPsi in stiffnessmatrix times vector
	      //
	      kDotGradientPsiTerm[0] = kPointCoors[0]*gradientPsiVal[1][0] + kPointCoors[1]*gradientPsiVal[1][1] + kPointCoors[2]*gradientPsiVal[1][2];
	      kDotGradientPsiTerm[1] = -(kPointCoors[0]*gradientPsiVal[0][0] + kPointCoors[1]*gradientPsiVal[0][1] + kPointCoors[2]*gradientPsiVal[0][2]);

#ifdef xc_id
#if xc_id == 4
	      derExchWithSigmaTimesGradRhoDotGradientPsiTerm[0] = derExcWithSigmaTimesGradRho(cell,q,0)*gradientPsiVal[0][0] + derExcWithSigmaTimesGradRho(cell,q,1)*gradientPsiVal[0][1] + derExcWithSigmaTimesGradRho(cell,q,2)*gradientPsiVal[0][2];
	      derExchWithSigmaTimesGradRhoDotGradientPsiTerm[1] = derExcWithSigmaTimesGradRho(cell,q,0)*gradientPsiVal[1][0] + derExcWithSigmaTimesGradRho(cell,q,1)*gradientPsiVal[1][1] + derExcWithSigmaTimesGradRho(cell,q,2)*gradientPsiVal[1][2];
	      //
	      //see if you can make this shorter
	      //
	      derExchWithSigmaTimesGradRhoTimesPsi[0][0] = derExcWithSigmaTimesGradRho(cell,q,0)*psiVal[0];
	      derExchWithSigmaTimesGradRhoTimesPsi[0][1] = derExcWithSigmaTimesGradRho(cell,q,1)*psiVal[0];
	      derExchWithSigmaTimesGradRhoTimesPsi[0][2] = derExcWithSigmaTimesGradRho(cell,q,2)*psiVal[0];
	      derExchWithSigmaTimesGradRhoTimesPsi[1][0] = derExcWithSigmaTimesGradRho(cell,q,0)*psiVal[1];
	      derExchWithSigmaTimesGradRhoTimesPsi[1][1] = derExcWithSigmaTimesGradRho(cell,q,1)*psiVal[1];
	      derExchWithSigmaTimesGradRhoTimesPsi[1][2] = derExcWithSigmaTimesGradRho(cell,q,2)*psiVal[1];	
#endif
#endif	      

	      //
	      //compute kSquareTerm
	      //
	      kSquareTerm[0] = halfkSquare*psiVal[0];
	      kSquareTerm[1] = halfkSquare*psiVal[1];

	      //
	      //submit gradients and values
	      //
	      
#ifdef xc_id
#if xc_id == 4
	      fe_eval.submit_gradient(gradientPsiTerm+derExchWithSigmaTimesGradRhoTimesPsi,q);
	      fe_eval.submit_value(vEffTerm+kDotGradientPsiTerm+kSquareTerm+derExchWithSigmaTimesGradRhoDotGradientPsiTerm,q);
#else
	      fe_eval.submit_gradient(gradientPsiTerm,q);
	      fe_eval.submit_value(vEffTerm+kDotGradientPsiTerm+kSquareTerm,q);
#endif
#endif
	    }

	  fe_eval.integrate (true, true);
	  fe_eval.distribute_local_to_global (*dst[i]);

	}
    }
#else
  FEEvaluation<3,FEOrder, FEOrder+1, 1, double>  fe_eval(data, dftPtr->eigenDofHandlerIndex, 0);
  Tensor<1,3,VectorizedArray<double> > derExchWithSigmaTimesGradRhoTimesPsi,gradientPsiVal;
  VectorizedArray<double> psiVal,derExchWithSigmaTimesGradRhoDotGradientPsiTerm;
  for(unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell); 
      for(unsigned int i = 0; i < dst.size(); i++)
	{
	  fe_eval.read_dof_values(*src[i]);
	  fe_eval.evaluate (true,true,false);
	  for(unsigned int q=0; q<fe_eval.n_q_points; ++q)
	    {
#ifdef xc_id
#if xc_id == 4
	      psiVal = fe_eval.get_value(q);
	      gradientPsiVal = fe_eval.get_gradient(q);
	      derExchWithSigmaTimesGradRhoTimesPsi[0] = derExcWithSigmaTimesGradRho(cell,q,0)*psiVal;
	      derExchWithSigmaTimesGradRhoTimesPsi[1] = derExcWithSigmaTimesGradRho(cell,q,1)*psiVal;
	      derExchWithSigmaTimesGradRhoTimesPsi[2] = derExcWithSigmaTimesGradRho(cell,q,2)*psiVal;
	      derExchWithSigmaTimesGradRhoDotGradientPsiTerm = derExcWithSigmaTimesGradRho(cell,q,0)*gradientPsiVal[0] + derExcWithSigmaTimesGradRho(cell,q,1)*gradientPsiVal[1] + derExcWithSigmaTimesGradRho(cell,q,2)*gradientPsiVal[2];

	      //submit gradient and value
	      fe_eval.submit_gradient(gradientPsiVal*half + derExchWithSigmaTimesGradRhoTimesPsi,q); 
	      fe_eval.submit_value(vEff(cell,q)*psiVal + derExchWithSigmaTimesGradRhoDotGradientPsiTerm,q);
	      
#else
	      fe_eval.submit_gradient (fe_eval.get_gradient(q)*half, q);
	      fe_eval.submit_value    (fe_eval.get_value(q)*vEff(cell,q), q);
#endif
#endif
	    }

	  fe_eval.integrate (true, true);
	  fe_eval.distribute_local_to_global (*dst[i]);
	}
    }
#endif
}


//HX
void eigenClass::HX(const std::vector<vectorType*> &src, 
		          std::vector<vectorType*> &dst) 
{
  computing_timer.enter_section("eigenClass HX");
  for (unsigned int i = 0; i < src.size(); i++)
    {
      *(dftPtr->tempPSI2[i]) = *src[i];
      dftPtr->tempPSI2[i]->scale(massVector); //MX
      dftPtr->constraintsNoneEigen.distribute(*(dftPtr->tempPSI2[i]));
      *dst[i] = 0.0;
      *dftPtr->tempPSI4[i] = 0.0;
    }

  dftPtr->matrix_free_data.cell_loop(&eigenClass::implementHX, this, dst, dftPtr->tempPSI2); //HMX
  
  //
  //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
  //
  if(isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
    {

      for (unsigned int i = 0; i < src.size(); i++)
	{
	  dftPtr->tempPSI2[i]->update_ghost_values();
	}

      for (unsigned int i = 0; i < src.size(); i++)
	{
	  *dftPtr->tempPSI4[i] = 0.0;
	}
          
      computeNonLocalHamiltonianTimesX(dftPtr->tempPSI2,
				       dftPtr->tempPSI4);

      for(unsigned int i = 0; i < src.size(); ++i)
	{
	  *dst[i]+=*dftPtr->tempPSI4[i];
	}

    }

  for (std::vector<vectorType*>::iterator it=dst.begin(); it!=dst.end(); it++)
    {
      (*it)->scale(massVector); //MHMX  
      (*it)->compress(VectorOperation::add);  
    }
  computing_timer.exit_section("eigenClass HX");
}

//XHX
void eigenClass::XHX(const std::vector<vectorType*> &src){
  computing_timer.enter_section("eigenClass XHX");

  //HX
  HX(src, dftPtr->tempPSI3);
  for (unsigned int i = 0; i < src.size(); i++)
    {
      dftPtr->tempPSI3[i]->update_ghost_values();
    }

#ifdef ENABLE_PERIODIC_BC
  unsigned int dofs_per_proc=src[0]->local_size()/2; 
#else
  unsigned int dofs_per_proc=src[0]->local_size(); 
#endif

  //
  //required for lapack functions
  //
  int k = dofs_per_proc, n = src.size(); 
  int vectorSize = k*n;
  int lda=k, ldb=k, ldc=n;

#ifdef ENABLE_PERIODIC_BC
  std::vector<double> hxReal(vectorSize), xReal(vectorSize);
  std::vector<double> hxImag(vectorSize), xImag(vectorSize);

  //
  //extract vectors at the processor level(too much memory expensive)
  //
  unsigned int index = 0;
  for (std::vector<vectorType*>::const_iterator it = src.begin(); it != src.end(); it++)
    {
      (*it)->extract_subvector_to(dftPtr->local_dof_indicesReal.begin(), 
				  dftPtr->local_dof_indicesReal.end(), 
				  xReal.begin()+dofs_per_proc*index); 

      (*it)->extract_subvector_to(dftPtr->local_dof_indicesImag.begin(), 
				  dftPtr->local_dof_indicesImag.end(), 
				  xImag.begin()+dofs_per_proc*index);

      dftPtr->tempPSI3[index]->extract_subvector_to(dftPtr->local_dof_indicesReal.begin(),
						    dftPtr->local_dof_indicesReal.end(),
						    hxReal.begin()+dofs_per_proc*index);

      dftPtr->tempPSI3[index]->extract_subvector_to(dftPtr->local_dof_indicesImag.begin(),
						    dftPtr->local_dof_indicesImag.end(),
						    hxImag.begin()+dofs_per_proc*index);
 
      index++;
    }

  //
  //create complex vectors
  //
  std::vector<std::complex<double> > hx(vectorSize,0.0);
  std::vector<std::complex<double> >  x(vectorSize,0.0);
  for(int i = 0; i < vectorSize; ++i)
    {
      hx[i].real(hxReal[i]);
      hx[i].imag(hxImag[i]);
       x[i].real(xReal[i]);
       x[i].imag(xImag[i]);
    }
  char transA  = 'C', transB  = 'N';
  std::complex<double> alpha = 1.0, beta  = 0.0;
  int sizeXtHX = n*n;
  std::vector<std::complex<double> > XtHXValuelocal(sizeXtHX,0.0);
  zgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &XtHXValuelocal[0], &ldc);

  MPI_Allreduce(&XtHXValuelocal[0],
		&XHXValue[0],
		sizeXtHX,
		MPI_C_DOUBLE_COMPLEX,
		MPI_SUM,
		mpi_communicator);
#else
   std::vector<double> hx(dofs_per_proc*src.size()), x(dofs_per_proc*src.size());

   //
   //extract vectors at the processor level
   //
   std::vector<unsigned int> local_dof_indices(dofs_per_proc);
   src[0]->locally_owned_elements().fill_index_vector(local_dof_indices);

   unsigned int index=0;
   for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++)
     {
       (*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), x.begin()+dofs_per_proc*index);
       dftPtr->tempPSI3[index]->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), hx.begin()+dofs_per_proc*index);
       index++;
       }
   char transA  = 'T', transB  = 'N';
   double alpha = 1.0, beta  = 0.0;
   dgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &XHXValue[0], &ldc);
   Utilities::MPI::sum(XHXValue, mpi_communicator, XHXValue); 
#endif

  //all reduce XHXValue(check in parallel)
  
  computing_timer.exit_section("eigenClass XHX");
}





