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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

#include "../../include/eigen.h"
#include "../../include/dft.h"
#include "../../include/dftParameters.h"

using namespace dftParameters ;

//
//constructor
//
template<unsigned int FEOrder>
eigenClass<FEOrder>::eigenClass(dftClass<FEOrder>* _dftPtr):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(C_num1DQuad<FEOrder>())),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times)
{

}

//
//initialize eigenClass object
//
template<unsigned int FEOrder>
void eigenClass<FEOrder>::init()
{
  computing_timer.enter_section("eigenClass setup");
  //
  //init vectors
  //
  for (unsigned int i=0; i<dftPtr->numEigenValues; ++i)
    {
      vectorType* temp=new vectorType;
      HXvalue.push_back(temp);
    } 

  dftPtr->matrix_free_data.initialize_dof_vector(massVector,dftPtr->eigenDofHandlerIndex);

  //
  //compute mass vector
  //
  computeMassVector();

  //
  //XHX size
  //
  XHXValue.resize(dftPtr->numEigenValues*dftPtr->numEigenValues,0.0);

  //
  //HX
  //
  for (unsigned int i=0; i<dftPtr->numEigenValues; ++i)
    {
      HXvalue[i]->reinit(dftPtr->vChebyshev);
    }
  computing_timer.exit_section("eigenClass setup"); 
} 

//
//compute mass Vector
//
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeMassVector()
{
  computing_timer.enter_section("eigenClass Mass assembly"); 
  massVector = 0.0;
  
#ifdef ENABLE_PERIODIC_BC
  Tensor<1,2,VectorizedArray<double> > one;
  one[0] =  make_vectorized_array (1.0);
  one[1] =  make_vectorized_array (1.0);
  FEEvaluation<3,FEOrder,FEOrder+1,2,double>  fe_eval(dftPtr->matrix_free_data, dftPtr->eigenDofHandlerIndex, 1);
#else
  VectorizedArray<double>  one = make_vectorized_array (1.0);
  FEEvaluation<3,FEOrder,FEOrder+1,1,double>  fe_eval(dftPtr->matrix_free_data, dftPtr->eigenDofHandlerIndex, 1); //Selecting GL quadrature points
#endif
  const unsigned int n_q_points = fe_eval.n_q_points;
  for(unsigned int cell=0; cell<dftPtr->matrix_free_data.n_macro_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      for (unsigned int q = 0; q < n_q_points; ++q) 
	fe_eval.submit_value(one,q);
      fe_eval.integrate (true,false);
      fe_eval.distribute_local_to_global (massVector);
    }

  massVector.compress(VectorOperation::add);
  
  //compute inverse
  /*for(unsigned int i=0; i<massVector.local_size(); i++)
    {
      if(std::abs(massVector.local_element(i)) > 1.0e-15)
	{
	  double temp = massVector.local_element(i);
	  massVector.local_element(i) = 1.0/std::sqrt(massVector.local_element(i));
	  if(std::isnan(massVector.local_element(i)))
	    {
	      std::cout<<"Value of temp is: "<<temp<<std::endl;
	    }
	}
      else
	{
	  massVector.local_element(i) = 0.0;
	}
	}*/

  for(types::global_dof_index i = 0; i < massVector.size(); ++i)
    {
      if(massVector.in_local_range(i))
	{
	  if(!dftPtr->constraintsNoneEigen.is_constrained(i))
	    {

	      double temp = massVector(i);

	      if(std::abs(massVector(i)) > 1.0e-15)
		massVector(i) = 1.0/std::sqrt(massVector(i));

	      if(std::isnan(massVector(i)))
		{
		  std::cout<<"Value of inverse square root of mass matrix on the unconstrained node is: "<<temp<<std::endl;
		  exit(-1);
		}
	    }
	}
    }

  massVector.compress(VectorOperation::insert);
  computing_timer.exit_section("eigenClass Mass assembly");
}


template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
				      const vectorType & phi,
				      const vectorType & phiExt,
				      std::map<dealii::CellId,std::vector<double> >* pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex, 0);
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

	  xc_lda_vxc(&(dftPtr->funcX),n_sub_cells,&densityValue[0],&exchangePotentialVal[0]);
	  xc_lda_vxc(&(dftPtr->funcC),n_sub_cells,&densityValue[0],&corrPotentialVal[0]);

	  VectorizedArray<double>  exchangePotential, corrPotential;
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      exchangePotential[v]=exchangePotentialVal[v];
	      corrPotential[v]=corrPotentialVal[v];
	    }

	  //
	  //sum all to vEffective
	  //
	  if(dftParameters::isPseudopotential)
	    {
	      VectorizedArray<double>  pseudoPotential;
	      for (unsigned int v = 0; v < n_sub_cells; ++v)
		{
		  cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
		  pseudoPotential[v]=((*pseudoValues)[cellPtr->id()][q]);
		}
	      vEff(cell,q) = fe_eval_phi.get_value(q)+exchangePotential+corrPotential+(pseudoPotential-fe_eval_phiExt.get_value(q));
	    }
	  else
	    {  
	      vEff(cell,q) = fe_eval_phi.get_value(q)+exchangePotential+corrPotential;
	    }
	}
    }
}

template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
				      std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				      const vectorType & phi,
				      const vectorType & phiExt,
				      std::map<dealii::CellId,std::vector<double> >* pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex ,0);
  int numberQuadraturePoints = fe_eval_phi.n_q_points;
  vEff.reinit (n_cells, numberQuadraturePoints);
  derExcWithSigmaTimesGradRho.reinit(TableIndices<3>(n_cells, numberQuadraturePoints, 3));
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
	
	  xc_gga_vxc(&(dftPtr->funcX),n_sub_cells,&densityValue[0],&sigmaValue[0],&derExchEnergyWithDensityVal[0],&derExchEnergyWithSigma[0]);
	  xc_gga_vxc(&(dftPtr->funcC),n_sub_cells,&densityValue[0],&sigmaValue[0],&derCorrEnergyWithDensityVal[0],&derCorrEnergyWithSigma[0]);


	  VectorizedArray<double>  derExchEnergyWithDensity, derCorrEnergyWithDensity, derExcWithSigmaTimesGradRhoX, derExcWithSigmaTimesGradRhoY, derExcWithSigmaTimesGradRhoZ;
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      derExchEnergyWithDensity[v]=derExchEnergyWithDensityVal[v];
	      derCorrEnergyWithDensity[v]=derCorrEnergyWithDensityVal[v];
	      double gradRhoX = ((*gradRhoValues)[cellPtr->id()][3*q + 0]);
	      double gradRhoY = ((*gradRhoValues)[cellPtr->id()][3*q + 1]);
	      double gradRhoZ = ((*gradRhoValues)[cellPtr->id()][3*q + 2]);
	      double term = derExchEnergyWithSigma[v]+derCorrEnergyWithSigma[v];
	      derExcWithSigmaTimesGradRhoX[v] = term*gradRhoX;
	      derExcWithSigmaTimesGradRhoY[v] = term*gradRhoY;
	      derExcWithSigmaTimesGradRhoZ[v] = term*gradRhoZ;
	    }

	  //
	  //sum all to vEffective
	  //
	  if(dftParameters::isPseudopotential)
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


template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeNonLocalHamiltonianTimesX(const std::vector<vectorType*> &src,
							   std::vector<vectorType*>       &dst)
{
  //
  //get FE data
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (dftPtr->FEEigen, quadrature_formula, update_values);

  //
  //get access to triangulation objects from meshGenerator class
  //
  const int kPointIndex = dftPtr->d_kPointIndex;
  const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;

  int numberNodesPerElement;
#ifdef ENABLE_PERIODIC_BC
  numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;//GeometryInfo<3>::vertices_per_cell;
#else
  numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  //
  //compute nonlocal projector ket times x i.e C^{T}*X 
  //
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::vector<std::complex<double> > > projectorKetTimesVector;
#else
  std::vector<std::vector<double> > projectorKetTimesVector;
#endif

  //
  //get number of Nonlocal atoms
  //
  const int numberNonLocalAtoms = dftPtr->d_nonLocalAtomGlobalChargeIds.size();
  int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();

  //
  //allocate memory for matrix-vector product
  //
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
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
	      (*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
	      for(int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //
		  //This is the component index 0(real) or 1(imag).
		  //
		  const unsigned int ck = fe_values.get_fe().system_to_component_index(idof).first; 
		  const unsigned int iNode = fe_values.get_fe().system_to_component_index(idof).second;
		  if(ck == 0)
		    inputVectors[numberNodesPerElement*index + iNode].real(temp[idof]);
		  else
		    inputVectors[numberNodesPerElement*index + iNode].imag(temp[idof]);
		}
	      index++;
	    }
	 

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

  //std::cout<<"Finished Element Loop"<<std::endl;

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
  
  //std::cout<<"Scaling V*C^{T} "<<std::endl;

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

#ifdef ENABLE_PERIODIC_BC
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

#ifdef ENABLE_PERIODIC_BC
	  unsigned int index = 0;
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for(std::vector<vectorType*>::iterator it = dst.begin(); it != dst.end(); ++it)
	    {
	      for(int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //temp[2*iNode]   = outputVectors[numberNodesPerElement*index + iNode].real();
		  //temp[2*iNode+1] = outputVectors[numberNodesPerElement*index + iNode].imag();
		  const unsigned int ck = fe_values.get_fe().system_to_component_index(idof).first;
		  const unsigned int iNode = fe_values.get_fe().system_to_component_index(idof).second;
		  
		  if(ck == 0)
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].real();
		  else
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].imag();

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
						  

template<unsigned int FEOrder>
void eigenClass<FEOrder>::implementHX (const dealii::MatrixFree<3,double>  &data,
				       std::vector<vectorType*>  &dst, 
				       const std::vector<vectorType*>  &src,
				       const std::pair<unsigned int,unsigned int> &cell_range) const
{
  VectorizedArray<double>  half = make_vectorized_array(0.5);
  VectorizedArray<double>  two = make_vectorized_array(2.0);

#ifdef ENABLE_PERIODIC_BC
  int kPointIndex = dftPtr->d_kPointIndex;
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(), 2, double>  fe_eval(data, dftPtr->eigenDofHandlerIndex, 0);
  Tensor<1,2,VectorizedArray<double> > psiVal, vEffTerm, kSquareTerm, kDotGradientPsiTerm, derExchWithSigmaTimesGradRhoDotGradientPsiTerm;
  Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > gradientPsiVal, gradientPsiTerm, derExchWithSigmaTimesGradRhoTimesPsi,sumGradientTerms; 

  Tensor<1,3,VectorizedArray<double> > kPointCoors;
  kPointCoors[0] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+0]);
  kPointCoors[1] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+1]);
  kPointCoors[2] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+2]);

  double kSquareTimesHalf =  0.5*(dftPtr->d_kPointCoordinates[3*kPointIndex+0]*dftPtr->d_kPointCoordinates[3*kPointIndex+0] + dftPtr->d_kPointCoordinates[3*kPointIndex+1]*dftPtr->d_kPointCoordinates[3*kPointIndex+1] + dftPtr->d_kPointCoordinates[3*kPointIndex+2]*dftPtr->d_kPointCoordinates[3*kPointIndex+2]);
  VectorizedArray<double> halfkSquare = make_vectorized_array(kSquareTimesHalf);

  if(dftParameters::xc_id == 4)
    {
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

	     
		  derExchWithSigmaTimesGradRhoDotGradientPsiTerm[0] = two*(derExcWithSigmaTimesGradRho(cell,q,0)*gradientPsiVal[0][0] + derExcWithSigmaTimesGradRho(cell,q,1)*gradientPsiVal[0][1] + derExcWithSigmaTimesGradRho(cell,q,2)*gradientPsiVal[0][2]);
		  derExchWithSigmaTimesGradRhoDotGradientPsiTerm[1] = two*(derExcWithSigmaTimesGradRho(cell,q,0)*gradientPsiVal[1][0] + derExcWithSigmaTimesGradRho(cell,q,1)*gradientPsiVal[1][1] + derExcWithSigmaTimesGradRho(cell,q,2)*gradientPsiVal[1][2]);
		  //
		  //see if you can make this shorter
		  //
		  derExchWithSigmaTimesGradRhoTimesPsi[0][0] = two*derExcWithSigmaTimesGradRho(cell,q,0)*psiVal[0];
		  derExchWithSigmaTimesGradRhoTimesPsi[0][1] = two*derExcWithSigmaTimesGradRho(cell,q,1)*psiVal[0];
		  derExchWithSigmaTimesGradRhoTimesPsi[0][2] = two*derExcWithSigmaTimesGradRho(cell,q,2)*psiVal[0];
		  derExchWithSigmaTimesGradRhoTimesPsi[1][0] = two*derExcWithSigmaTimesGradRho(cell,q,0)*psiVal[1];
		  derExchWithSigmaTimesGradRhoTimesPsi[1][1] = two*derExcWithSigmaTimesGradRho(cell,q,1)*psiVal[1];
		  derExchWithSigmaTimesGradRhoTimesPsi[1][2] = two*derExcWithSigmaTimesGradRho(cell,q,2)*psiVal[1];	


		  //
		  //compute kSquareTerm
		  //
		  kSquareTerm[0] = halfkSquare*psiVal[0];
		  kSquareTerm[1] = halfkSquare*psiVal[1];

		  //
		  //submit gradients and values
		  //
	      
		  for(int i = 0; i < 3; ++i)
		    {
		      sumGradientTerms[0][i] = gradientPsiTerm[0][i] + derExchWithSigmaTimesGradRhoTimesPsi[0][i];
		      sumGradientTerms[1][i] = gradientPsiTerm[1][i] + derExchWithSigmaTimesGradRhoTimesPsi[1][i];
		    }

		  fe_eval.submit_gradient(sumGradientTerms,q);
		  fe_eval.submit_value(vEffTerm+kDotGradientPsiTerm+kSquareTerm+derExchWithSigmaTimesGradRhoDotGradientPsiTerm,q);
		    
		}

	      fe_eval.integrate (true, true);
	      fe_eval.distribute_local_to_global (*dst[i]);

	    }
	}
    }
  else
    {
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

		  //
		  //compute kSquareTerm
		  //
		  kSquareTerm[0] = halfkSquare*psiVal[0];
		  kSquareTerm[1] = halfkSquare*psiVal[1];

		  //
		  //submit gradients and values
		  //
		  fe_eval.submit_gradient(gradientPsiTerm,q);
		  fe_eval.submit_value(vEffTerm+kDotGradientPsiTerm+kSquareTerm,q);
		}

	      fe_eval.integrate (true, true);
	      fe_eval.distribute_local_to_global (*dst[i]);

	    }
	}

    }
#else
  FEEvaluation<3,FEOrder, C_num1DQuad<FEOrder>(), 1, double>  fe_eval(data, dftPtr->eigenDofHandlerIndex, 0);
  Tensor<1,3,VectorizedArray<double> > derExchWithSigmaTimesGradRhoTimesPsi,gradientPsiVal;
  VectorizedArray<double> psiVal,derExchWithSigmaTimesGradRhoDotGradientPsiTerm;
  if(dftParameters::xc_id == 4)
    {
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
	{
	  fe_eval.reinit (cell); 
	  for(unsigned int i = 0; i < dst.size(); i++)
	    {
	      fe_eval.read_dof_values(*src[i]);
	      fe_eval.evaluate (true,true,false);
	      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
		{
		  psiVal = fe_eval.get_value(q);
		  gradientPsiVal = fe_eval.get_gradient(q);
		  derExchWithSigmaTimesGradRhoTimesPsi[0] = derExcWithSigmaTimesGradRho(cell,q,0)*psiVal;
		  derExchWithSigmaTimesGradRhoTimesPsi[1] = derExcWithSigmaTimesGradRho(cell,q,1)*psiVal;
		  derExchWithSigmaTimesGradRhoTimesPsi[2] = derExcWithSigmaTimesGradRho(cell,q,2)*psiVal;
		  derExchWithSigmaTimesGradRhoDotGradientPsiTerm = derExcWithSigmaTimesGradRho(cell,q,0)*gradientPsiVal[0] + derExcWithSigmaTimesGradRho(cell,q,1)*gradientPsiVal[1] + derExcWithSigmaTimesGradRho(cell,q,2)*gradientPsiVal[2];

		  //
		  //submit gradient and value
		  //
		  fe_eval.submit_gradient(gradientPsiVal*half + two*derExchWithSigmaTimesGradRhoTimesPsi,q); 
		  fe_eval.submit_value(vEff(cell,q)*psiVal + two*derExchWithSigmaTimesGradRhoDotGradientPsiTerm,q);
		}

	      fe_eval.integrate (true, true);
	      fe_eval.distribute_local_to_global (*dst[i]);
	    }
	}
    }
  else
    {
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
	{
	  fe_eval.reinit (cell); 
	  for(unsigned int i = 0; i < dst.size(); i++)
	    {
	      fe_eval.read_dof_values(*src[i]);
	      fe_eval.evaluate (true,true,false);
	      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
		{
		  fe_eval.submit_gradient(fe_eval.get_gradient(q)*half, q);
		  fe_eval.submit_value(fe_eval.get_value(q)*vEff(cell,q), q);
		}

	      fe_eval.integrate (true, true);
	      fe_eval.distribute_local_to_global (*dst[i]);
	    }
	}

    }
#endif
}


//HX
template<unsigned int FEOrder>
void eigenClass<FEOrder>::HX(const std::vector<vectorType*> &src, 
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

  dftPtr->matrix_free_data.cell_loop(&eigenClass<FEOrder>::implementHX, this, dst, dftPtr->tempPSI2); //HMX
  
  //
  //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
  //
  if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
    {

      for (unsigned int i = 0; i < src.size(); i++)
	{
	  dftPtr->tempPSI2[i]->compress(VectorOperation::insert);  
	  dftPtr->tempPSI2[i]->update_ghost_values();
	}

      for (unsigned int i = 0; i < src.size(); i++)
	{
	  *dftPtr->tempPSI4[i] = 0.0;
	}
        computeNonLocalHamiltonianTimesX(dftPtr->tempPSI2,
      				       dftPtr->tempPSI4);
      //
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
template<unsigned int FEOrder>
void eigenClass<FEOrder>::XHX(const std::vector<vectorType*> &src)
{
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

template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues, 
				      const vectorType & phi,
				      const vectorType & phiExt,
				      const unsigned int spinIndex,
				      std::map<dealii::CellId,std::vector<double> >* pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex, 0);
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
	  std::vector<double> densityValue(2*n_sub_cells), exchangePotentialVal(2*n_sub_cells), corrPotentialVal(2*n_sub_cells);
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      densityValue[2*v+1] = ((*rhoValues)[cellPtr->id()][2*q+1]);
	      densityValue[2*v] = ((*rhoValues)[cellPtr->id()][2*q]);
	    }

	  xc_lda_vxc(&(dftPtr->funcX),n_sub_cells,&densityValue[0],&exchangePotentialVal[0]);
	  xc_lda_vxc(&(dftPtr->funcC),n_sub_cells,&densityValue[0],&corrPotentialVal[0]);

	  VectorizedArray<double>  exchangePotential, corrPotential;
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      exchangePotential[v]=exchangePotentialVal[2*v+spinIndex];
	      corrPotential[v]=corrPotentialVal[2*v+spinIndex];
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

template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues,
				      std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				      const vectorType & phi,
				      const vectorType & phiExt,
				      const unsigned int spinIndex,
				      std::map<dealii::CellId,std::vector<double> >* pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex ,0);
  int numberQuadraturePoints = fe_eval_phi.n_q_points;
  vEff.reinit (n_cells, numberQuadraturePoints);
  derExcWithSigmaTimesGradRho.reinit(TableIndices<3>(n_cells, numberQuadraturePoints, 3));
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
	  std::vector<double> densityValue(2*n_sub_cells), derExchEnergyWithDensityVal(2*n_sub_cells), derCorrEnergyWithDensityVal(2*n_sub_cells), 
				derExchEnergyWithSigma(3*n_sub_cells), derCorrEnergyWithSigma(3*n_sub_cells), sigmaValue(3*n_sub_cells);
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      densityValue[2*v+1] = ((*rhoValues)[cellPtr->id()][2*q+1]);
	      densityValue[2*v] = ((*rhoValues)[cellPtr->id()][2*q]);
	      double gradRhoX1 = ((*gradRhoValues)[cellPtr->id()][6*q + 0]);
	      double gradRhoY1 = ((*gradRhoValues)[cellPtr->id()][6*q + 1]);
	      double gradRhoZ1 = ((*gradRhoValues)[cellPtr->id()][6*q + 2]);
	      double gradRhoX2 = ((*gradRhoValues)[cellPtr->id()][6*q + 3]);
	      double gradRhoY2 = ((*gradRhoValues)[cellPtr->id()][6*q + 4]);
	      double gradRhoZ2 = ((*gradRhoValues)[cellPtr->id()][6*q + 5]);
	      //
	      sigmaValue[3*v+0] = gradRhoX1*gradRhoX1 + gradRhoY1*gradRhoY1 + gradRhoZ1*gradRhoZ1;
	      sigmaValue[3*v+1] = gradRhoX1*gradRhoX2 + gradRhoY1*gradRhoY2 + gradRhoZ1*gradRhoZ2;
	      sigmaValue[3*v+2] = gradRhoX2*gradRhoX2 + gradRhoY2*gradRhoY2 + gradRhoZ2*gradRhoZ2;
	    }
	
	  xc_gga_vxc(&(dftPtr->funcX),n_sub_cells,&densityValue[0],&sigmaValue[0],&derExchEnergyWithDensityVal[0],&derExchEnergyWithSigma[0]);
	  xc_gga_vxc(&(dftPtr->funcC),n_sub_cells,&densityValue[0],&sigmaValue[0],&derCorrEnergyWithDensityVal[0],&derCorrEnergyWithSigma[0]);


	  VectorizedArray<double>  derExchEnergyWithDensity, derCorrEnergyWithDensity, derExcWithSigmaTimesGradRhoX, derExcWithSigmaTimesGradRhoY, derExcWithSigmaTimesGradRhoZ;
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      derExchEnergyWithDensity[v]=derExchEnergyWithDensityVal[2*v+spinIndex];
	      derCorrEnergyWithDensity[v]=derCorrEnergyWithDensityVal[2*v+spinIndex];
	      double gradRhoX = ((*gradRhoValues)[cellPtr->id()][6*q + 0 + 3*spinIndex]);
	      double gradRhoY = ((*gradRhoValues)[cellPtr->id()][6*q + 1 + 3*spinIndex]);
	      double gradRhoZ = ((*gradRhoValues)[cellPtr->id()][6*q + 2 + 3*spinIndex]);
	      double gradRhoOtherX = ((*gradRhoValues)[cellPtr->id()][6*q + 0 + 3*(1-spinIndex)]);
	      double gradRhoOtherY = ((*gradRhoValues)[cellPtr->id()][6*q + 1 + 3*(1-spinIndex)]);
	      double gradRhoOtherZ = ((*gradRhoValues)[cellPtr->id()][6*q + 2 + 3*(1-spinIndex)]);
	      double term = derExchEnergyWithSigma[3*v+2*spinIndex]+derCorrEnergyWithSigma[3*v+2*spinIndex];
	      double termOff = derExchEnergyWithSigma[3*v+1]+derCorrEnergyWithSigma[3*v+1];
	      derExcWithSigmaTimesGradRhoX[v] = term*gradRhoX + 0.5*termOff*gradRhoOtherX; 
	      derExcWithSigmaTimesGradRhoY[v] = term*gradRhoY + 0.5*termOff*gradRhoOtherY;
	      derExcWithSigmaTimesGradRhoZ[v] = term*gradRhoZ + 0.5*termOff*gradRhoOtherZ;
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
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeNonLocalHamiltonianTimesX_OV(const std::vector<vectorType*> &src,
							   std::vector<vectorType*>       &dst)
{
  //
  //get FE data
  //
  QGauss<3>  quadrature_formula(FEOrder+1);
  FEValues<3> fe_values (dftPtr->FEEigen, quadrature_formula, update_values);

  const int kPointIndex = dftPtr->d_kPointIndex;
  const unsigned int numberElements  = dftPtr->triangulation.n_locally_owned_active_cells();
  const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;

  int numberNodesPerElement;
#ifdef ENABLE_PERIODIC_BC
  numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;//GeometryInfo<3>::vertices_per_cell;
#else
  numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  //
  //compute nonlocal projector ket times x i.e C^{T}*X 
  //
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::vector<std::complex<double> > > projectorKetTimesVector, projectorKetTimesVector_OV;
#else
  std::vector<std::vector<double> > projectorKetTimesVector, projectorKetTimesVector_OV;
#endif

  //
  //get number of Nonlocal atoms
  //
  const int numberNonLocalAtoms = dftPtr->d_nonLocalAtomGlobalChargeIds.size();
  int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();
  projectorKetTimesVector_OV.clear();

  //
  //allocate memory for matrix-vector product
  //
  projectorKetTimesVector.resize(numberNonLocalAtoms);
  projectorKetTimesVector_OV.resize(numberNonLocalAtoms);
  for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
    {
      int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
      projectorKetTimesVector[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
	  projectorKetTimesVector_OV[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
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
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
	      (*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
	      for(int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //
		  //This is the component index 0(real) or 1(imag).
		  //
		  const unsigned int ck = fe_values.get_fe().system_to_component_index(idof).first; 
		  const unsigned int iNode = fe_values.get_fe().system_to_component_index(idof).second;
		  if(ck == 0)
		    inputVectors[numberNodesPerElement*index + iNode].real(temp[idof]);
		  else
		    inputVectors[numberNodesPerElement*index + iNode].imag(temp[idof]);
		}
	      index++;
	    }
	 

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

  //std::cout<<"Finished Element Loop"<<std::endl;

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
	  for(int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave){
		  for(int jPseudoAtomicWave = 0; jPseudoAtomicWave < numberPseudoWaveFunctions; ++jPseudoAtomicWave) {
	          projectorKetTimesVector_OV[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] += 
			   ( projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + jPseudoAtomicWave] *
			   dftPtr->d_nonLocalPseudoPotentialConstants_OV[iAtom][iPseudoAtomicWave][jPseudoAtomicWave] );		  
			}
		       
	  }
	}
    }
  
  //std::cout<<"Scaling V*C^{T} "<<std::endl;

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

#ifdef ENABLE_PERIODIC_BC
	  
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
		 &projectorKetTimesVector_OV[iAtom][0],
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
		 &projectorKetTimesVector_OV[iAtom][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#endif

	  cell->get_dof_indices(local_dof_indices);

#ifdef ENABLE_PERIODIC_BC
	  unsigned int index = 0;
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for(std::vector<vectorType*>::iterator it = dst.begin(); it != dst.end(); ++it)
	    {
	      for(int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //temp[2*iNode]   = outputVectors[numberNodesPerElement*index + iNode].real();
		  //temp[2*iNode+1] = outputVectors[numberNodesPerElement*index + iNode].imag();
		  const unsigned int ck = fe_values.get_fe().system_to_component_index(idof).first;
		  const unsigned int iNode = fe_values.get_fe().system_to_component_index(idof).second;
		  
		  if(ck == 0)
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].real();
		  else
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].imag();

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

template class eigenClass<1>;
template class eigenClass<2>;
template class eigenClass<3>;
template class eigenClass<4>;
template class eigenClass<5>;
template class eigenClass<6>;
template class eigenClass<7>;
template class eigenClass<8>;
template class eigenClass<9>;
template class eigenClass<10>;
template class eigenClass<11>;
template class eigenClass<12>;


