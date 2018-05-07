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
// @author Shiva Rudraraju, Phani Motamarri
//

#include <eigen.h>
#include <dft.h>
#include <dftParameters.h>
#include <linearAlgebraOperations.h>



namespace dftfe {

#include "computeNonLocalHamiltonianTimesXMemoryOpt.cc"
#include "matrixVectorProductImplementations.cc"
#include "shapeFunctionDataCalculator.cc"
#include "hamiltonianMatrixCalculator.cc"


  //
  //constructor
  //
  template<unsigned int FEOrder>
  eigenClass<FEOrder>::eigenClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
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
  //initialize eigenClass object
  //
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::init()
  {
    computing_timer.enter_section("eigenClass setup");


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

    computing_timer.exit_section("eigenClass setup");
  }

//
//compute mass Vector
//
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeMassVector(const dealii::DoFHandler<3> & dofHandler,
	                                    const dealii::ConstraintMatrix & constraintMatrix,
			                    vectorType & sqrtMassVec,
			                    vectorType & invSqrtMassVec)
{
  computing_timer.enter_section("eigenClass Mass assembly");
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
   computing_timer.exit_section("eigenClass Mass assembly");
}


template<unsigned int FEOrder>
void eigenClass<FEOrder>::reinitkPointIndex(unsigned int & kPointIndex)
{
  d_kPointIndex = kPointIndex;
}


template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
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
void eigenClass<FEOrder>::computeVEff(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
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


#ifdef ENABLE_PERIODIC_BC
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::HX(dealii::parallel::distributed::Vector<std::complex<double> > & src,
			       const unsigned int numberWaveFunctions,
			       const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
			       const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			       const bool scaleFlag,
			       std::complex<double> scalar,
			       dealii::parallel::distributed::Vector<std::complex<double> > & dst)


  {
    const unsigned int numberDofs = src.local_size()/numberWaveFunctions;
    const unsigned int inc = 1;

    //
    //scale src vector with M^{-1/2}
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	std::complex<double> scalingCoeff = d_invSqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i])*scalar;
	zscal_(&numberWaveFunctions,
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
	    std::complex<double> scalingCoeff = d_sqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i]);
	    zscal_(&numberWaveFunctions,
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
    computeLocalHamiltonianTimesX(src,
				  numberWaveFunctions,
				  flattenedArrayMacroCellLocalProcIndexIdMap,
				  dst);


    //
    //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
    //H^{nloc}*M^{-1/2}*X
    if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      computeNonLocalHamiltonianTimesX(src,
				       numberWaveFunctions,
				       flattenedArrayCellLocalProcIndexIdMap,
				       dst);


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
	std::complex<double> scalingCoeff = d_invSqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i]);
	zscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       dst.begin()+i*numberWaveFunctions,
	       &inc);
      }
      

    //
    //unscale src M^{1/2}*X
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	std::complex<double> scalingCoeff = d_sqrtMassVector.local_element(dftPtr->localProc_dof_indicesReal[i])*(1.0/scalar);
	zscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctions,
	       &inc);
      }

  }
#else
 template<unsigned int FEOrder>
  void eigenClass<FEOrder>::HX(dealii::parallel::distributed::Vector<double> & src,
			       const unsigned int numberWaveFunctions,
			       const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
			       const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			       const bool scaleFlag,
			       double scalar,
			       dealii::parallel::distributed::Vector<double> & dst)


  {
    const unsigned int numberDofs = src.local_size()/numberWaveFunctions;
    const unsigned int inc = 1;

    //
    //scale src vector with M^{-1/2}
    //
    for(unsigned int i = 0; i < numberDofs; ++i)
      {
	double scalingCoeff = d_invSqrtMassVector.local_element(i)*scalar;
	dscal_(&numberWaveFunctions,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctions,
	       &inc);
      }


    if(scaleFlag)
      {
	for(int i = 0; i < numberDofs; ++i)
	  {
	    double scalingCoeff = d_sqrtMassVector.local_element(i);
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
    computeLocalHamiltonianTimesX(src,
				  numberWaveFunctions,
				  flattenedArrayMacroCellLocalProcIndexIdMap,
				  dst);

    //
    //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
    //H^{nloc}*M^{-1/2}*X
    if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      computeNonLocalHamiltonianTimesX(src,
				       numberWaveFunctions,
				       flattenedArrayCellLocalProcIndexIdMap,
				       dst);



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
  void eigenClass<FEOrder>::HX(std::vector<vectorType> &src,
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
    dftPtr->matrix_free_data.cell_loop(&eigenClass<FEOrder>::computeLocalHamiltonianTimesXMF, this, dst, src); //HMX

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
#ifdef ENABLE_PERIODIC_BC
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::XtHX(std::vector<vectorType> & src,
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
    int k = dofs_per_proc, n = src.size();
    int vectorSize = k*n;
    int lda=k, ldb=k, ldc=n;


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
    std::complex<double> alpha = 1.0, beta  = 0.0;
    int sizeXtHX = n*n;
    std::vector<std::complex<double> > XtHXValuelocal(sizeXtHX,0.0);
    zgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &XtHXValuelocal[0], &ldc);

    MPI_Allreduce(&XtHXValuelocal[0],
		  &ProjHam[0],
		  sizeXtHX,
		  MPI_C_DOUBLE_COMPLEX,
		  MPI_SUM,
		  mpi_communicator);
 
  }
#else
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::XtHX(std::vector<vectorType> &src,
				 std::vector<double> & ProjHam)
  {

    //Resize ProjHam 
    ProjHam.resize(src.size()*src.size(),0.0);

    std::vector<vectorType> tempPSI3(src.size());

    for(unsigned int i = 0; i < src.size(); ++i)
      {
	tempPSI3[i].reinit(src[0]);
      }

    computing_timer.enter_section("eigenClass XHX");

    //HX
    HX(src, tempPSI3);
    for (unsigned int i = 0; i < src.size(); i++)
      {
	tempPSI3[i].update_ghost_values();
      }

    unsigned int dofs_per_proc=src[0].local_size(); 


    //
    //required for lapack functions
    //
    int k = dofs_per_proc, n = src.size(); 
    int vectorSize = k*n;
    int lda=k, ldb=k, ldc=n;


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
    double alpha = 1.0, beta  = 0.0;
    dgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &ProjHam[0], &ldc);
    Utilities::MPI::sum(ProjHam, mpi_communicator, ProjHam); 
  
    computing_timer.exit_section("eigenClass XHX");

  }
#endif

template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
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
void eigenClass<FEOrder>::computeVEffSpinPolarized(const std::map<dealii::CellId,std::vector<double> >* rhoValues,
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

}
