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
// @author Shiva Rudraraju, Phani Motamarri
//

#include <kohnShamDFTOperator.h>
#include <dft.h>
#include <dftParameters.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <vectorUtilities.h>
#include <dftUtils.h>


namespace dftfe {

#include "computeNonLocalHamiltonianTimesXMemoryOpt.cc"
#include "computeNonLocalHamiltonianTimesXMemoryOptBatchGEMM.cc"
#include "matrixVectorProductImplementations.cc"
#include "shapeFunctionDataCalculator.cc"
#include "hamiltonianMatrixCalculator.cc"


  //
  //constructor
  //
  template<unsigned int FEOrder>
  kohnShamDFTOperatorClass<FEOrder>::kohnShamDFTOperatorClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
    dftPtr(_dftPtr),
    d_kPointIndex(0),
    d_numberNodesPerElement(_dftPtr->matrix_free_data.get_dofs_per_cell()),
    d_numberMacroCells(_dftPtr->matrix_free_data.n_macro_cells()),
    mpi_communicator (mpi_comm_replica),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
    pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times),
    operatorDFTClass(mpi_comm_replica,
		     _dftPtr->getMatrixFreeData(),
		     _dftPtr->getLocalDofIndicesReal(),
		     _dftPtr->getLocalDofIndicesImag(),
		     _dftPtr->getLocalProcDofIndicesReal(),
		     _dftPtr->getLocalProcDofIndicesImag(),
		     _dftPtr->getConstraintMatrixEigen(),
		     _dftPtr->constraintsNoneDataInfo)
  {

  }


  //
  //initialize kohnShamDFTOperatorClass object
  //
  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::init()
  {
    computing_timer.enter_section("kohnShamDFTOperatorClass setup");


    dftPtr->matrix_free_data.initialize_dof_vector(d_invSqrtMassVector,dftPtr->eigenDofHandlerIndex);
    d_sqrtMassVector.reinit(d_invSqrtMassVector);


    //
    //create macro cell map to subcells
    //
    d_macroCellSubCellMap.resize(d_numberMacroCells);
    for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
      {
	const  unsigned int n_sub_cells = dftPtr->matrix_free_data.n_components_filled(iMacroCell);
	d_macroCellSubCellMap[iMacroCell] = n_sub_cells;
      }

    //
    //compute mass vector
    //
    computeMassVector(dftPtr->dofHandlerEigen,
		      dftPtr->constraintsNoneEigen,
		      d_sqrtMassVector,
		      d_invSqrtMassVector);

    computing_timer.exit_section("kohnShamDFTOperatorClass setup");
  }

  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::reinit(const unsigned int numberWaveFunctions,
				   dealii::parallel::distributed::Vector<dataTypes::number> & flattenedArray,
				   bool flag)
  {

    if(flag)
	vectorTools::createDealiiVector<dataTypes::number>(dftPtr->matrix_free_data.get_vector_partitioner(),
							   numberWaveFunctions,
							   flattenedArray);

    if(dftParameters::isPseudopotential)
    {
      vectorTools::createDealiiVector<dataTypes::number>(dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
							 numberWaveFunctions,
							 dftPtr->d_projectorKetTimesVectorParFlattened);

      vectorTools::createDealiiVector<dataTypes::numberLowPrec>(dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
							 numberWaveFunctions,
							 dftPtr->d_projectorKetTimesVectorParFlattenedLowPrec);
    }



    vectorTools::computeCellLocalIndexSetMap(flattenedArray.get_partitioner(),
					     dftPtr->matrix_free_data,
					     numberWaveFunctions,
					     d_flattenedArrayMacroCellLocalProcIndexIdMap,
					     d_flattenedArrayCellLocalProcIndexIdMap);

    getOverloadedConstraintMatrix()->precomputeMaps(dftPtr->matrix_free_data.get_vector_partitioner(),
						    flattenedArray.get_partitioner(),
						    numberWaveFunctions);
  }

template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::reinit(const unsigned int numberWaveFunctions)
{

  if(dftParameters::isPseudopotential)
  {
    vectorTools::createDealiiVector<dataTypes::number>(dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
						       numberWaveFunctions,
						       dftPtr->d_projectorKetTimesVectorParFlattened);
    vectorTools::createDealiiVector<dataTypes::numberLowPrec>
	  (dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
	   numberWaveFunctions,
	   dftPtr->d_projectorKetTimesVectorParFlattenedLowPrec);
  }

}


//
//compute mass Vector
//
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeMassVector(const dealii::DoFHandler<3> & dofHandler,
	                                    const dealii::ConstraintMatrix & constraintMatrix,
			                    vectorType & sqrtMassVec,
			                    vectorType & invSqrtMassVec)
{
  computing_timer.enter_section("kohnShamDFTOperatorClass Mass assembly");
  invSqrtMassVec = 0.0;
  sqrtMassVec = 0.0;

  QGaussLobatto<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (dofHandler.get_fe(), quadrature, update_values | update_JxW_values);
  const unsigned int   dofs_per_cell = (dofHandler.get_fe()).dofs_per_cell;
  const unsigned int   num_quad_points = quadrature.size();
  Vector<double>       massVectorLocal (dofs_per_cell) ;
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


  //
  //parallel loop over all elements
  //
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for(; cell!=endc; ++cell)
      if (cell->is_locally_owned())
	{
	  //compute values for the current element
	  fe_values.reinit (cell);
	  massVectorLocal=0.0;
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		      massVectorLocal(i) += fe_values.shape_value(i, q_point)*fe_values.shape_value(i, q_point)*fe_values.JxW (q_point);

	  cell->get_dof_indices (local_dof_indices);
	  constraintMatrix.distribute_local_to_global(massVectorLocal, local_dof_indices, invSqrtMassVec);
	}

   invSqrtMassVec.compress(VectorOperation::add);


   for(types::global_dof_index i = 0; i < invSqrtMassVec.size(); ++i)
       if(invSqrtMassVec.in_local_range(i) && !constraintMatrix.is_constrained(i))
	 {
	   if(std::abs(invSqrtMassVec(i)) > 1.0e-15)
	     {
	       sqrtMassVec(i) = std::sqrt(invSqrtMassVec(i));
	       invSqrtMassVec(i) = 1.0/std::sqrt(invSqrtMassVec(i));
	     }
	   AssertThrow(!std::isnan(invSqrtMassVec(i)),ExcMessage("Value of inverse square root of mass matrix on the unconstrained node is undefined"));
	 }

   invSqrtMassVec.compress(VectorOperation::insert);
   sqrtMassVec.compress(VectorOperation::insert);
   computing_timer.exit_section("kohnShamDFTOperatorClass Mass assembly");
}


template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::reinitkPointIndex(unsigned int & kPointIndex)
{
  d_kPointIndex = kPointIndex;
}


template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
				      const vectorType & phi,
				      const vectorType & phiExt,
				      const std::map<dealii::CellId,std::vector<double> > & pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex, 0);
  const int numberQuadraturePoints = fe_eval_phi.n_q_points;
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
	      densityValue[v] = (*rhoValues).find(cellPtr->id())->second[q];
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
		  pseudoPotential[v]=pseudoValues.find(cellPtr->id())->second[q];
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
void kohnShamDFTOperatorClass<FEOrder>::computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
				      const std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
				      const vectorType & phi,
				      const vectorType & phiExt,
				      const std::map<dealii::CellId,std::vector<double> > & pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex ,0);
  int numberQuadraturePoints = fe_eval_phi.n_q_points;
  vEff.reinit (n_cells, numberQuadraturePoints);
  derExcWithSigmaTimesGradRho.reinit(TableIndices<2>(n_cells, numberQuadraturePoints));
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
	      densityValue[v] = (*rhoValues).find(cellPtr->id())->second[q];
	      double gradRhoX = (*gradRhoValues).find(cellPtr->id())->second[3*q + 0];
	      double gradRhoY = (*gradRhoValues).find(cellPtr->id())->second[3*q + 1];
	      double gradRhoZ = (*gradRhoValues).find(cellPtr->id())->second[3*q + 2];
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
	      double gradRhoX = (*gradRhoValues).find(cellPtr->id())->second[3*q + 0];
	      double gradRhoY = (*gradRhoValues).find(cellPtr->id())->second[3*q + 1];
	      double gradRhoZ = (*gradRhoValues).find(cellPtr->id())->second[3*q + 2];
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
		  pseudoPotential[v]=pseudoValues.find(cellPtr->id())->second[q];
		}
	      vEff(cell,q)=fe_eval_phi.get_value(q)+derExchEnergyWithDensity+derCorrEnergyWithDensity+(pseudoPotential-fe_eval_phiExt.get_value(q));
	      derExcWithSigmaTimesGradRho(cell,q)[0] = derExcWithSigmaTimesGradRhoX;
	      derExcWithSigmaTimesGradRho(cell,q)[1] = derExcWithSigmaTimesGradRhoY;
	      derExcWithSigmaTimesGradRho(cell,q)[2] = derExcWithSigmaTimesGradRhoZ;
	    }
	  else
	    {
	      vEff(cell,q)=fe_eval_phi.get_value(q)+derExchEnergyWithDensity+derCorrEnergyWithDensity;
	      derExcWithSigmaTimesGradRho(cell,q)[0] = derExcWithSigmaTimesGradRhoX;
	      derExcWithSigmaTimesGradRho(cell,q)[1] = derExcWithSigmaTimesGradRhoY;
	      derExcWithSigmaTimesGradRho(cell,q)[2] = derExcWithSigmaTimesGradRhoZ;
	    }
	}
    }
}


#ifdef USE_COMPLEX
  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::HX(dealii::parallel::distributed::Vector<std::complex<double> > & src,
			       const unsigned int numberWaveFunctions,
			       const bool scaleFlag,
			       const double scalar,
			       const bool useSinglePrec,
			       dealii::parallel::distributed::Vector<std::complex<double> > & dst)


  {
    const unsigned int numberDofs = src.local_size()/numberWaveFunctions;
    const unsigned int inc = 1;

    //
    //scale src vector with M^{-1/2}
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	const double scalingCoeff = d_invSqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i])*scalar;
	zdscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctions,
	       &inc);
      }

    //const std::complex<double> zeroValue = 0.0;
    //dst = zeroValue;

    if(scaleFlag)
      {
	for(int i = 0; i < numberDofs; ++i)
	  {
	    const double scalingCoeff = d_sqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i]);
	    zdscal_(&numberWaveFunctions,
		   &scalingCoeff,
		   dst.begin()+i*numberWaveFunctions,
		   &inc);

	  }
      }

    //
    //update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src,
    					       numberWaveFunctions);


    src.update_ghost_values();


    //
    //Hloc*M^{-1/2}*X
    //
#ifdef WITH_MKL
    if (dftParameters::useBatchGEMM && numberWaveFunctions<1000)
       computeLocalHamiltonianTimesXBatchGEMM(src,
				              numberWaveFunctions,
				              dst);
    else
       computeLocalHamiltonianTimesX(src,
				     numberWaveFunctions,
 				     dst);
#else
       computeLocalHamiltonianTimesX(src,
				     numberWaveFunctions,
 				     dst);
#endif

    //
    //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
    //H^{nloc}*M^{-1/2}*X
    if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
    {
#ifdef WITH_MKL
      if (dftParameters::useBatchGEMM && numberWaveFunctions<1000)
        computeNonLocalHamiltonianTimesXBatchGEMM(src,
				                  numberWaveFunctions,
				                  dst);
      else
        computeNonLocalHamiltonianTimesX(src,
				         numberWaveFunctions,
				         dst);
#else
        computeNonLocalHamiltonianTimesX(src,
				         numberWaveFunctions,
				         dst);
#endif
    }


    //
    //update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(dst,
    							      numberWaveFunctions);



    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    //
    //M^{-1/2}*H*M^{-1/2}*X
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	const double scalingCoeff = d_invSqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i]);
	zdscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       dst.begin()+i*numberWaveFunctions,
	       &inc);
      }


    //
    //unscale src M^{1/2}*X
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	const double scalingCoeff = d_sqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i])*(1.0/scalar);
	zdscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctions,
	       &inc);
      }

  }
#else
 template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::HX(dealii::parallel::distributed::Vector<double> & src,
			       const unsigned int numberWaveFunctions,
			       const bool scaleFlag,
			       const double scalar,
			       const bool useSinglePrec,
			       dealii::parallel::distributed::Vector<double> & dst)


  {
    const unsigned int numberDofs = src.local_size()/numberWaveFunctions;
    const unsigned int inc = 1;


    //
    //scale src vector with M^{-1/2}
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	const double scalingCoeff = d_invSqrtMassVector.local_element(i)*scalar;
	dscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctions,
	       &inc);
      }


    if(scaleFlag)
      {
	for(int i = 0; i < numberDofs; ++i)
	  {
	    const double scalingCoeff = d_sqrtMassVector.local_element(i);
	    dscal_(&numberWaveFunctions,
		   &scalingCoeff,
		   dst.begin()+i*numberWaveFunctions,
		   &inc);

	  }
      }

    //
    //update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src,
					      numberWaveFunctions);

    src.update_ghost_values();

    //
    //Hloc*M^{-1/2}*X
    //
#ifdef WITH_MKL
    if (dftParameters::useBatchGEMM && numberWaveFunctions<1000)
    {
       if (useSinglePrec)
         computeLocalHamiltonianTimesXBatchGEMMSinglePrec(src,
				              numberWaveFunctions,
				              dst);
       else
	  computeLocalHamiltonianTimesXBatchGEMM(src,
				  numberWaveFunctions,
				  dst);
    }
    else
       computeLocalHamiltonianTimesX(src,
				     numberWaveFunctions,
 				     dst);
#else
       computeLocalHamiltonianTimesX(src,
				     numberWaveFunctions,
 				     dst);
#endif

    //
    //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
    //H^{nloc}*M^{-1/2}*X
    if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
    {
#ifdef WITH_MKL
      if (dftParameters::useBatchGEMM && numberWaveFunctions<1000)
      {
	if (useSinglePrec)
            computeNonLocalHamiltonianTimesXBatchGEMMSinglePrec(src,
				                  numberWaveFunctions,
				                  dst);
	else
            computeNonLocalHamiltonianTimesXBatchGEMM(src,
				                  numberWaveFunctions,
				                  dst);
      }
      else
        computeNonLocalHamiltonianTimesX(src,
				         numberWaveFunctions,
				         dst);
#else
        computeNonLocalHamiltonianTimesX(src,
				         numberWaveFunctions,
				         dst);
#endif
    }



    //
    //update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(dst,
							      numberWaveFunctions);


    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    //
    //M^{-1/2}*H*M^{-1/2}*X
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	dscal_(&numberWaveFunctions,
	       &d_invSqrtMassVector.local_element(i),
	       dst.begin()+i*numberWaveFunctions,
	       &inc);
      }


    //
    //unscale src M^{1/2}*X
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	double scalingCoeff = d_sqrtMassVector.local_element(i)*(1.0/scalar);
	dscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctions,
	       &inc);
      }


  }
#endif


  //HX
  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::HX(std::vector<vectorType> &src,
			       std::vector<vectorType> &dst)
  {

    for (unsigned int i = 0; i < src.size(); i++)
      {
	src[i].scale(d_invSqrtMassVector); //M^{-1/2}*X
	//dftPtr->constraintsNoneEigen.distribute(src[i]);
	dftPtr->getConstraintMatrixEigenDataInfo().distribute(src[i]);
	src[i].update_ghost_values();
	dst[i] = 0.0;
      }


    //
    //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
    //H^{nloc}*M^{-1/2}*X
    if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      computeNonLocalHamiltonianTimesX(src,dst);

    //
    //First evaluate H^{loc}*M^{-1/2}*X and then add to H^{nloc}*M^{-1/2}*X
    //
    dftPtr->matrix_free_data.cell_loop(&kohnShamDFTOperatorClass<FEOrder>::computeLocalHamiltonianTimesXMF, this, dst, src); //HMX

    //
    //Finally evaluate M^{-1/2}*H*M^{-1/2}*X
    //
    for (std::vector<vectorType>::iterator it=dst.begin(); it!=dst.end(); it++)
      {
	(*it).scale(d_invSqrtMassVector);
      }

    //
    //unscale src back
    //
    for (std::vector<vectorType>::iterator it=src.begin(); it!=src.end(); it++)
      {
	(*it).scale(d_sqrtMassVector); //MHMX
      }

  }

  //XHX


  //XHX
#ifdef USE_COMPLEX
  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::XtHX(std::vector<vectorType> & src,
				 std::vector<std::complex<double> > & ProjHam)
  {

    //Resize ProjHam
    ProjHam.resize(src.size()*src.size(),0.0);

    std::vector<vectorType> tempPSI3(src.size());

    for(unsigned int i = 0; i < src.size(); ++i)
      tempPSI3[i].reinit(src[0]);


    //HX
    HX(src, tempPSI3);
    for (unsigned int i = 0; i < src.size(); i++)
      tempPSI3[i].update_ghost_values();

    unsigned int dofs_per_proc=src[0].local_size()/2;

    //
    //required for lapack functions
    //
    const unsigned int k = dofs_per_proc, n = src.size();
    const unsigned int vectorSize = k*n;
    const unsigned int lda=k, ldb=k, ldc=n;


    std::vector<double> hxReal(vectorSize), xReal(vectorSize);
    std::vector<double> hxImag(vectorSize), xImag(vectorSize);

    //
    //extract vectors at the processor level(too much memory expensive)
    //
    unsigned int index = 0;
    for (std::vector<vectorType>::const_iterator it = src.begin(); it != src.end(); it++)
      {
	(*it).extract_subvector_to(dftPtr->getLocalDofIndicesReal().begin(),
				   dftPtr->getLocalDofIndicesReal().end(),
				   xReal.begin()+dofs_per_proc*index);

	(*it).extract_subvector_to(dftPtr->getLocalDofIndicesImag().begin(),
				   dftPtr->getLocalDofIndicesImag().end(),
				   xImag.begin()+dofs_per_proc*index);

	tempPSI3[index].extract_subvector_to(dftPtr->getLocalDofIndicesReal().begin(),
					     dftPtr->getLocalDofIndicesReal().end(),
					     hxReal.begin()+dofs_per_proc*index);

	tempPSI3[index].extract_subvector_to(dftPtr->getLocalDofIndicesImag().begin(),
					     dftPtr->getLocalDofIndicesImag().end(),
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
    const std::complex<double> alpha = 1.0, beta  = 0.0;
    const unsigned int sizeXtHX = n*n;
    std::vector<std::complex<double> > XtHXValuelocal(sizeXtHX,0.0);
    zgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &XtHXValuelocal[0], &ldc);

    MPI_Allreduce(&XtHXValuelocal[0],
		  &ProjHam[0],
		  sizeXtHX,
		  MPI_C_DOUBLE_COMPLEX,
		  MPI_SUM,
		  mpi_communicator);

  }

  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::XtHX(const std::vector<std::complex<double> > & X,
				 const unsigned int numberWaveFunctions,
				 std::vector<std::complex<double> > & ProjHam)
  {
    //
    //Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size()/numberWaveFunctions;

    //
    //Resize ProjHam
    //
    ProjHam.clear();
    ProjHam.resize(numberWaveFunctions*numberWaveFunctions,0.0);
    //
    //create temporary array XTemp
    //
    dealii::parallel::distributed::Vector<std::complex<double>> XTemp;
    reinit(numberWaveFunctions,
	   XTemp,
	   true);
    for(unsigned int iNode = 0; iNode<numberDofs; ++iNode)
       for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    XTemp.local_element(iNode*numberWaveFunctions
		     +iWave)
		 =X[iNode*numberWaveFunctions+iWave];
    //
    //create temporary array Y
    //
    dealii::parallel::distributed::Vector<std::complex<double> > Y;
    reinit(numberWaveFunctions,
	   Y,
	   true);
    std::complex<double> zeroValue = 0.0;
    Y = zeroValue;


    //
    //evaluate H times XTemp and store in Y
    //
    const bool scaleFlag = false;
    const double scalar = 1.0;
    HX(XTemp,
       numberWaveFunctions,
       scaleFlag,
       scalar,
       false,
       Y);

    for(unsigned int i = 0; i < Y.local_size(); ++i)
      Y.local_element(i) = std::conj(Y.local_element(i));


    char transA = 'N';
    char transB = 'T';
    const std::complex<double> alpha = 1.0, beta = 0.0;

    std::vector<std::complex<double> > XtHXValuelocal(numberWaveFunctions*numberWaveFunctions,0.0);

    //
    //evaluates Z = Yc*Xt
    //
    zgemm_(&transA,
	   &transB,
	   &numberWaveFunctions,
	   &numberWaveFunctions,
	   &numberDofs,
	   &alpha,
	   Y.begin(),
	   &numberWaveFunctions,
           &X[0],
	   &numberWaveFunctions,
	   &beta,
	   &XtHXValuelocal[0],
	   &numberWaveFunctions);

    const unsigned int size = numberWaveFunctions*numberWaveFunctions;

    MPI_Allreduce(&XtHXValuelocal[0],
		  &ProjHam[0],
		  size,
		  MPI_C_DOUBLE_COMPLEX,
		  MPI_SUM,
		  mpi_communicator);
  }
#else
  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::XtHX(std::vector<vectorType> &src,
				 std::vector<double> & ProjHam)
  {

    //Resize ProjHam
    ProjHam.resize(src.size()*src.size(),0.0);

    std::vector<vectorType> tempPSI3(src.size());

    for(unsigned int i = 0; i < src.size(); ++i)
      {
	tempPSI3[i].reinit(src[0]);
      }

    //HX
    HX(src, tempPSI3);
    for (unsigned int i = 0; i < src.size(); i++)
      {
	tempPSI3[i].update_ghost_values();
      }

    const unsigned int dofs_per_proc=src[0].local_size();


    //
    //required for lapack functions
    //
    const unsigned int k = dofs_per_proc, n = src.size();
    const unsigned int vectorSize = k*n;
    const unsigned int lda=k, ldb=k, ldc=n;


    std::vector<double> hx(dofs_per_proc*src.size()), x(dofs_per_proc*src.size());

    //
    //extract vectors at the processor level
    //
    std::vector<IndexSet::size_type> local_dof_indices(dofs_per_proc);
    src[0].locally_owned_elements().fill_index_vector(local_dof_indices);

    unsigned int index=0;
    for (std::vector<vectorType>::const_iterator it=src.begin(); it!=src.end(); it++)
      {
	(*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), x.begin()+dofs_per_proc*index);
	tempPSI3[index].extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), hx.begin()+dofs_per_proc*index);
	index++;
      }
    char transA  = 'T', transB  = 'N';
    const double alpha = 1.0, beta  = 0.0;
    dgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &ProjHam[0], &ldc);
    Utilities::MPI::sum(ProjHam, mpi_communicator, ProjHam);
  }

  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::XtHX(const std::vector<double> & X,
				 const unsigned int numberWaveFunctions,
				 std::vector<double> & ProjHam)
  {

    //
    //Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size()/numberWaveFunctions;

    //
    //Resize ProjHam
    //
    ProjHam.clear();
    ProjHam.resize(numberWaveFunctions*numberWaveFunctions,0.0);

    //
    //create temporary array XTemp
    //
    dealii::parallel::distributed::Vector<double> XTemp;
    reinit(numberWaveFunctions,
	   XTemp,
	   true);
    for(unsigned int iNode = 0; iNode<numberDofs; ++iNode)
       for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    XTemp.local_element(iNode*numberWaveFunctions
		     +iWave)
		 =X[iNode*numberWaveFunctions+iWave];

    //
    //create temporary array Y
    //
    dealii::parallel::distributed::Vector<double> Y;
    reinit(numberWaveFunctions,
	   Y,
	   true);

    //
    //evaluate H times XTemp and store in Y
    //
    bool scaleFlag = false;
    double scalar = 1.0;
    HX(XTemp,
       numberWaveFunctions,
       scaleFlag,
       scalar,
       false,
       Y);

    char transA = 'N';
    char transB = 'T';
    const double alpha = 1.0, beta = 0.0;

    dgemm_(&transA,
	   &transB,
	   &numberWaveFunctions,
	   &numberWaveFunctions,
	   &numberDofs,
	   &alpha,
	   &X[0],
	   &numberWaveFunctions,
           Y.begin(),
	   &numberWaveFunctions,
	   &beta,
	   &ProjHam[0],
	   &numberWaveFunctions);

    Y.reinit(0);

    Utilities::MPI::sum(ProjHam, mpi_communicator, ProjHam);

  }
#endif

#ifdef DEAL_II_WITH_SCALAPACK
  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::XtHX(const std::vector<dataTypes::number> & X,
				 const unsigned int numberWaveFunctions,
				 const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				 dealii::ScaLAPACKMatrix<dataTypes::number> & projHamPar)
  {
#ifdef USE_COMPLEX
    AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
    //
    //Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size()/numberWaveFunctions;

    //create temporary arrays XBlock,Hx
    dealii::parallel::distributed::Vector<dataTypes::number> XBlock,HXBlock;

    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
						    projHamPar,
						    globalToLocalRowIdMap,
						    globalToLocalColumnIdMap);
   //band group parallelization data structures
   const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
   const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
   std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
   dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
					      numberWaveFunctions,
					      bandGroupLowHighPlusOneIndices);

   /*
    * X^{T}*H*Xc is done in a blocked approach for memory optimization:
    * Sum_{blocks} X^{T}*H*XcBlock. The result of each X^{T}*H*XcBlock
    * has a much smaller memory compared to X^{T}*H*Xc.
    * X^{T} (denoted by X in the code with column major format storage)
    * is a matrix with size (N x MLoc).
    * N is denoted by numberWaveFunctions in the code.
    * MLoc, which is number of local dofs is denoted by numberDofs in the code.
    * Xc denotes complex conjugate of X.
    * XcBlock is a matrix of size (MLoc x B). B is the block size.
    * A further optimization is done to reduce floating point operations:
    * As X^{T}*H*Xc is a Hermitian matrix, it suffices to compute only the lower
    * triangular part. To exploit this, we do
    * X^{T}*H*Xc=Sum_{blocks} XTrunc^{T}*H*XcBlock
    * where XTrunc^{T} is a (D x MLoc) sub matrix of X^{T} with the row indices
    * ranging from the lowest global index of XcBlock (denoted by jvec in the code)
    * to N. D=N-jvec.
    * The parallel ScaLapack matrix projHamPar is directly filled from
    * the XTrunc^{T}*H*XcBlock result
    */

    const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                         bandGroupLowHighPlusOneIndices[1]);

    std::vector<dataTypes::number> projHamBlock(numberWaveFunctions*vectorsBlockSize,0.0);

    if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Inside Blocked XtHX with parallel projected Ham matrix");

    for (unsigned int jvec = 0; jvec < numberWaveFunctions; jvec += vectorsBlockSize)
    {
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, numberWaveFunctions-jvec);
	  if (jvec==0 || B!=vectorsBlockSize)
	  {
	     reinit(B,
		    XBlock,
		    true);
	     HXBlock.reinit(XBlock);
	  }

	  if ((jvec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (jvec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	  {
	      XBlock=0;
	      //fill XBlock^{T} from X:
	      for(unsigned int iNode = 0; iNode<numberDofs; ++iNode)
		  for(unsigned int iWave = 0; iWave < B; ++iWave)
			XBlock.local_element(iNode*B
				 +iWave)
			     =X[iNode*numberWaveFunctions+jvec+iWave];


	      //evaluate H times XBlock^{T} and store in HXBlock^{T}
	      HXBlock=0;
	      const bool scaleFlag = false;
	      const dataTypes::number scalar = 1.0;
	      HX(XBlock,
		 B,
		 scaleFlag,
		 scalar,
		 false,
		 HXBlock);


	      const char transA = 'N';
#ifdef USE_COMPLEX
	      const char transB = 'C';
#else
	      const char transB = 'T';
#endif
	      const dataTypes::number alpha = 1.0,beta = 0.0;
	      std::fill(projHamBlock.begin(),projHamBlock.end(),0.);

	      const unsigned int D=numberWaveFunctions-jvec;

	      // Comptute local XTrunc^{T}*HXcBlock.
	      dgemm_(&transA,
		     &transB,
		     &D,
		     &B,
		     &numberDofs,
		     &alpha,
		     &X[0]+jvec,
		     &numberWaveFunctions,
		     HXBlock.begin(),
		     &B,
		     &beta,
		     &projHamBlock[0],
		     &D);


	      // Sum local XTrunc^{T}*HXcBlock across domain decomposition processors
#ifdef USE_COMPLEX
	      MPI_Allreduce(MPI_IN_PLACE,
			    &projHamBlock[0],
			    D*B,
			    MPI_C_DOUBLE_COMPLEX,
			    MPI_SUM,
			    getMPICommunicator());
#else
	      MPI_Allreduce(MPI_IN_PLACE,
			    &projHamBlock[0],
			    D*B,
			    MPI_DOUBLE,
			    MPI_SUM,
			    getMPICommunicator());
#endif

	      //Copying only the lower triangular part to the ScaLAPACK projected Hamiltonian matrix
	      if (processGrid->is_process_active())
		  for (unsigned int j = 0; j <B; ++j)
		     if(globalToLocalColumnIdMap.find(j+jvec)!=globalToLocalColumnIdMap.end())
		     {
		       const unsigned int localColumnId=globalToLocalColumnIdMap[j+jvec];
		       for (unsigned int i = jvec; i <numberWaveFunctions; ++i)
		       {
			 std::map<unsigned int, unsigned int>::iterator it=
					      globalToLocalRowIdMap.find(i);
			 if (it!=globalToLocalRowIdMap.end())
				 projHamPar.local_el(it->second,
						     localColumnId)
						     =projHamBlock[j*D+i-jvec];
		       }
		     }

	  }//band parallelization

    }//block loop

    linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(processGrid,
						                      projHamPar,
						                      dftPtr->interBandGroupComm);
#endif
  }

  template<unsigned int FEOrder>
  void kohnShamDFTOperatorClass<FEOrder>::XtHXMixedPrec
	             (const std::vector<dataTypes::number> & X,
		      const unsigned int N,
		      const unsigned int Ncore,
		      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		      dealii::ScaLAPACKMatrix<dataTypes::number> & projHamPar)
  {
#ifdef USE_COMPLEX
    AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
    //
    //Get access to number of locally owned nodes on the current processor
    //
    const unsigned int numberDofs = X.size()/N;

    //create temporary arrays XBlock,Hx
    dealii::parallel::distributed::Vector<dataTypes::number> XBlock,HXBlock;

    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
						    projHamPar,
						    globalToLocalRowIdMap,
						    globalToLocalColumnIdMap);
   //band group parallelization data structures
   const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
   const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
   std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
   dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
					      N,
					      bandGroupLowHighPlusOneIndices);

   /*
    * X^{T}*H*Xc is done in a blocked approach for memory optimization:
    * Sum_{blocks} X^{T}*H*XcBlock. The result of each X^{T}*H*XcBlock
    * has a much smaller memory compared to X^{T}*H*Xc.
    * X^{T} (denoted by X in the code with column major format storage)
    * is a matrix with size (N x MLoc).
    * MLoc, which is number of local dofs is denoted by numberDofs in the code.
    * Xc denotes complex conjugate of X.
    * XcBlock is a matrix of size (MLoc x B). B is the block size.
    * A further optimization is done to reduce floating point operations:
    * As X^{T}*H*Xc is a Hermitian matrix, it suffices to compute only the lower
    * triangular part. To exploit this, we do
    * X^{T}*H*Xc=Sum_{blocks} XTrunc^{T}*H*XcBlock
    * where XTrunc^{T} is a (D x MLoc) sub matrix of X^{T} with the row indices
    * ranging from the lowest global index of XcBlock (denoted by jvec in the code)
    * to N. D=N-jvec.
    * The parallel ScaLapack matrix projHamPar is directly filled from
    * the XTrunc^{T}*H*XcBlock result
    */

    const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
	                                         bandGroupLowHighPlusOneIndices[1]);

    std::vector<dataTypes::numberLowPrec> projHamBlockSinglePrec(N*vectorsBlockSize,0.0);
    std::vector<dataTypes::number> projHamBlock(N*vectorsBlockSize,0.0);

    std::vector<dataTypes::numberLowPrec> HXBlockSinglePrec;

    std::vector<dataTypes::numberLowPrec> XSinglePrec(&X[0],&X[0]+X.size());

    if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Inside Blocked XtHX with parallel projected Ham matrix");

    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
    {
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, N-jvec);
	  if (jvec==0 || B!=vectorsBlockSize)
	  {
	     reinit(B,
		    XBlock,
		    true);
	     HXBlock.reinit(XBlock);
	     HXBlockSinglePrec.resize(B*numberDofs);
	  }

	  if ((jvec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (jvec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	  {
	      XBlock=0;
	      //fill XBlock^{T} from X:
	      for(unsigned int iNode = 0; iNode<numberDofs; ++iNode)
		  for(unsigned int iWave = 0; iWave < B; ++iWave)
			XBlock.local_element(iNode*B
				 +iWave)
			     =X[iNode*N+jvec+iWave];


	      //evaluate H times XBlock^{T} and store in HXBlock^{T}
	      HXBlock=0;
	      const bool scaleFlag = false;
	      const dataTypes::number scalar = 1.0;
	      HX(XBlock,
		 B,
		 scaleFlag,
		 scalar,
		 false,
		 HXBlock);


	      const char transA = 'N';
#ifdef USE_COMPLEX
	      const char transB = 'C';
#else
	      const char transB = 'T';
#endif
	      const dataTypes::number alpha = 1.0,beta = 0.0;
	      std::fill(projHamBlock.begin(),projHamBlock.end(),0.);

	      if (jvec+B>Ncore)
	      {

		  const unsigned int D=N-jvec;

		  // Comptute local XTrunc^{T}*HXcBlock.
		  dgemm_(&transA,
			 &transB,
			 &D,
			 &B,
			 &numberDofs,
			 &alpha,
			 &X[0]+jvec,
			 &N,
			 HXBlock.begin(),
			 &B,
			 &beta,
			 &projHamBlock[0],
			 &D);


		  // Sum local XTrunc^{T}*HXcBlock across domain decomposition processors
		  MPI_Allreduce(MPI_IN_PLACE,
				&projHamBlock[0],
				D*B,
				dataTypes::mpi_type_id(&projHamBlock[0]),
				MPI_SUM,
				getMPICommunicator());


		  //Copying only the lower triangular part to the ScaLAPACK projected Hamiltonian matrix
		  if (processGrid->is_process_active())
		      for (unsigned int j = 0; j <B; ++j)
			 if(globalToLocalColumnIdMap.find(j+jvec)!=globalToLocalColumnIdMap.end())
			 {
			   const unsigned int localColumnId=globalToLocalColumnIdMap[j+jvec];
			   for (unsigned int i = jvec; i <N; ++i)
			   {
			     std::map<unsigned int, unsigned int>::iterator it=
						  globalToLocalRowIdMap.find(i);
			     if (it!=globalToLocalRowIdMap.end())
				     projHamPar.local_el(it->second,
							 localColumnId)
							 =projHamBlock[j*D+i-jvec];
			   }
			 }
	      }
	      else
	      {

		  const dataTypes::numberLowPrec alphaSinglePrec = 1.0,betaSinglePrec = 0.0;
	          std::fill(projHamBlockSinglePrec.begin(),projHamBlockSinglePrec.end(),0.);

		  for(unsigned int iNode = 0; iNode<numberDofs; ++iNode)
		      for(unsigned int iWave = 0; iWave < B; ++iWave)
			    HXBlockSinglePrec[iNode*B+iWave]=HXBlock.local_element(iNode*B+iWave);

		  const unsigned int Dcore=Ncore-jvec;

		  sgemm_(&transA,
			 &transB,
			 &Dcore,
			 &B,
			 &numberDofs,
			 &alphaSinglePrec,
			 &XSinglePrec[0]+jvec,
			 &N,
			 &HXBlockSinglePrec[0],
			 &B,
			 &betaSinglePrec,
			 &projHamBlockSinglePrec[0],
			 &Dcore);

		  MPI_Allreduce(MPI_IN_PLACE,
				&projHamBlockSinglePrec[0],
				Dcore*B,
				dataTypes::mpi_type_id(&projHamBlockSinglePrec[0]),
				MPI_SUM,
				getMPICommunicator());


		  if (processGrid->is_process_active())
		      for (unsigned int j = 0; j <B; ++j)
			 if(globalToLocalColumnIdMap.find(j+jvec)!=globalToLocalColumnIdMap.end())
			 {
			   const unsigned int localColumnId=globalToLocalColumnIdMap[j+jvec];
			   for (unsigned int i = jvec; i <Ncore; ++i)
			   {
			     std::map<unsigned int, unsigned int>::iterator it=
						  globalToLocalRowIdMap.find(i);
			     if (it!=globalToLocalRowIdMap.end())
				     projHamPar.local_el(it->second,
							 localColumnId)
							 =(dataTypes::number)
							   projHamBlockSinglePrec[j*Dcore+i-jvec];
			   }
			 }

		  const unsigned int Dvalence=N-Ncore;

		  dgemm_(&transA,
			 &transB,
			 &Dvalence,
			 &B,
			 &numberDofs,
			 &alpha,
			 &X[0]+Ncore,
			 &N,
			 HXBlock.begin(),
			 &B,
			 &beta,
			 &projHamBlock[0],
			 &Dvalence);


		  MPI_Allreduce(MPI_IN_PLACE,
				&projHamBlock[0],
				Dvalence*B,
				dataTypes::mpi_type_id(&projHamBlock[0]),
				MPI_SUM,
				getMPICommunicator());

		  if (processGrid->is_process_active())
		      for (unsigned int j = 0; j <B; ++j)
			 if(globalToLocalColumnIdMap.find(j+jvec)!=globalToLocalColumnIdMap.end())
			 {
			   const unsigned int localColumnId=globalToLocalColumnIdMap[j+jvec];
			   for (unsigned int i = Ncore; i <N; ++i)
			   {
			     std::map<unsigned int, unsigned int>::iterator it=
						  globalToLocalRowIdMap.find(i);
			     if (it!=globalToLocalRowIdMap.end())
				     projHamPar.local_el(it->second,
							 localColumnId)
							 =projHamBlock[j*Dvalence+i-Ncore];
			   }
			 }
	      }


	  }//band parallelization

    }//block loop

    linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(processGrid,
						                      projHamPar,
						                      dftPtr->interBandGroupComm);
#endif
  }
#endif

template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
						   const vectorType & phi,
						   const vectorType & phiExt,
						   const unsigned int spinIndex,
						   const std::map<dealii::CellId,std::vector<double> > & pseudoValues)

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
	  std::vector<double> densityValue(2*n_sub_cells), exchangePotentialVal(2*n_sub_cells), corrPotentialVal(2*n_sub_cells);
	  for (unsigned int v = 0; v < n_sub_cells; ++v)
	    {
	      cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	      densityValue[2*v+1] = (*rhoValues).find(cellPtr->id())->second[2*q+1];
	      densityValue[2*v] = (*rhoValues).find(cellPtr->id())->second[2*q];
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
	  if(dftParameters::isPseudopotential)
	    {
	      VectorizedArray<double>  pseudoPotential;
	      for (unsigned int v = 0; v < n_sub_cells; ++v)
		{
		  cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
		  pseudoPotential[v]=pseudoValues.find(cellPtr->id())->second[q];
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
void kohnShamDFTOperatorClass<FEOrder>::computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
						   const std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
						   const vectorType & phi,
						   const vectorType & phiExt,
						   const unsigned int spinIndex,
						   const std::map<dealii::CellId,std::vector<double> > & pseudoValues)
{
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phi(dftPtr->matrix_free_data, 0 ,0);
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval_phiExt(dftPtr->matrix_free_data, dftPtr->phiExtDofHandlerIndex ,0);
  int numberQuadraturePoints = fe_eval_phi.n_q_points;
  vEff.reinit (n_cells, numberQuadraturePoints);
  derExcWithSigmaTimesGradRho.reinit(TableIndices<2>(n_cells, numberQuadraturePoints));
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
	      densityValue[2*v+1] = (*rhoValues).find(cellPtr->id())->second[2*q+1];
	      densityValue[2*v] = (*rhoValues).find(cellPtr->id())->second[2*q];
	      double gradRhoX1 = (*gradRhoValues).find(cellPtr->id())->second[6*q + 0];
	      double gradRhoY1 = (*gradRhoValues).find(cellPtr->id())->second[6*q + 1];
	      double gradRhoZ1 = (*gradRhoValues).find(cellPtr->id())->second[6*q + 2];
	      double gradRhoX2 = (*gradRhoValues).find(cellPtr->id())->second[6*q + 3];
	      double gradRhoY2 = (*gradRhoValues).find(cellPtr->id())->second[6*q + 4];
	      double gradRhoZ2 = (*gradRhoValues).find(cellPtr->id())->second[6*q + 5];
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
	      double gradRhoX = (*gradRhoValues).find(cellPtr->id())->second[6*q + 0 + 3*spinIndex];
	      double gradRhoY = (*gradRhoValues).find(cellPtr->id())->second[6*q + 1 + 3*spinIndex];
	      double gradRhoZ = (*gradRhoValues).find(cellPtr->id())->second[6*q + 2 + 3*spinIndex];
	      double gradRhoOtherX = (*gradRhoValues).find(cellPtr->id())->second[6*q + 0 + 3*(1-spinIndex)];
	      double gradRhoOtherY = (*gradRhoValues).find(cellPtr->id())->second[6*q + 1 + 3*(1-spinIndex)];
	      double gradRhoOtherZ = (*gradRhoValues).find(cellPtr->id())->second[6*q + 2 + 3*(1-spinIndex)];
	      double term = derExchEnergyWithSigma[3*v+2*spinIndex]+derCorrEnergyWithSigma[3*v+2*spinIndex];
	      double termOff = derExchEnergyWithSigma[3*v+1]+derCorrEnergyWithSigma[3*v+1];
	      derExcWithSigmaTimesGradRhoX[v] = term*gradRhoX + 0.5*termOff*gradRhoOtherX;
	      derExcWithSigmaTimesGradRhoY[v] = term*gradRhoY + 0.5*termOff*gradRhoOtherY;
	      derExcWithSigmaTimesGradRhoZ[v] = term*gradRhoZ + 0.5*termOff*gradRhoOtherZ;
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
		  pseudoPotential[v]=pseudoValues.find(cellPtr->id())->second[q];
		}
	      vEff(cell,q)=fe_eval_phi.get_value(q)+derExchEnergyWithDensity+derCorrEnergyWithDensity+(pseudoPotential-fe_eval_phiExt.get_value(q));
	      derExcWithSigmaTimesGradRho(cell,q)[0] = derExcWithSigmaTimesGradRhoX;
	      derExcWithSigmaTimesGradRho(cell,q)[1] = derExcWithSigmaTimesGradRhoY;
	      derExcWithSigmaTimesGradRho(cell,q)[2] = derExcWithSigmaTimesGradRhoZ;
	    }
	  else
	    {
	      vEff(cell,q)=fe_eval_phi.get_value(q)+derExchEnergyWithDensity+derCorrEnergyWithDensity;
	      derExcWithSigmaTimesGradRho(cell,q)[0] = derExcWithSigmaTimesGradRhoX;
	      derExcWithSigmaTimesGradRho(cell,q)[1] = derExcWithSigmaTimesGradRhoY;
	      derExcWithSigmaTimesGradRho(cell,q)[2] = derExcWithSigmaTimesGradRhoZ;
	    }
	}
    }
}


  template class kohnShamDFTOperatorClass<1>;
  template class kohnShamDFTOperatorClass<2>;
  template class kohnShamDFTOperatorClass<3>;
  template class kohnShamDFTOperatorClass<4>;
  template class kohnShamDFTOperatorClass<5>;
  template class kohnShamDFTOperatorClass<6>;
  template class kohnShamDFTOperatorClass<7>;
  template class kohnShamDFTOperatorClass<8>;
  template class kohnShamDFTOperatorClass<9>;
  template class kohnShamDFTOperatorClass<10>;
  template class kohnShamDFTOperatorClass<11>;
  template class kohnShamDFTOperatorClass<12>;

}
