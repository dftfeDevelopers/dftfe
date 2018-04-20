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
    FE (QGaussLobatto<1>(FEOrder+1)),
    d_kPointIndex(0),
    d_numberNodesPerElement(_dftPtr->matrix_free_data.get_dofs_per_cell()),
    d_numberMacroCells(_dftPtr->matrix_free_data.n_macro_cells()),
    mpi_communicator (mpi_comm_replica),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
    pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times),
    operatorClass(mpi_comm_replica,
		  _dftPtr->getMatrixFreeData(),
		  _dftPtr->getLocalDofIndicesReal(),
		  _dftPtr->getLocalDofIndicesImag(),
		  _dftPtr->getLocalProcDofIndicesReal(),
		  _dftPtr->getLocalProcDofIndicesImag(),
		  _dftPtr->getConstraintMatrixEigen())
  {
    
  }

  //
  //initialize eigenClass object
  //
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::init()
  {
    computing_timer.enter_section("eigenClass setup");

    dftPtr->matrix_free_data.initialize_dof_vector(invSqrtMassVector,dftPtr->eigenDofHandlerIndex);
    sqrtMassVector.reinit(invSqrtMassVector);

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
    computeMassVector();

    computing_timer.exit_section("eigenClass setup");
  }

  //
  //compute mass Vector
  //
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::computeMassVector()
  {
    computing_timer.enter_section("eigenClass Mass assembly");
    invSqrtMassVector = 0.0;
    sqrtMassVector = 0.0;

#ifdef ENABLE_PERIODIC_BC
    Tensor<1,2,VectorizedArray<double> > one;
    one[0] =  make_vectorized_array (1.0);
    one[1] =  make_vectorized_array (1.0);
    FEEvaluation<3,FEOrder,FEOrder+1,2,double>  fe_eval(dftPtr->matrix_free_data, dftPtr->eigenDofHandlerIndex, 1);
#else
    VectorizedArray<double>  one = make_vectorized_array (1.0);
    FEEvaluation<3,FEOrder,FEOrder+1,1,double>  fe_eval(dftPtr->matrix_free_data, dftPtr->eigenDofHandlerIndex, 1); //Selecting GL quadrature points
#endif

    //
    //first evaluate diagonal terms of mass matrix (\integral N_i*N_i) and store in dealii vector named invSqrtMassVector
    //
    const unsigned int n_q_points = fe_eval.n_q_points;
    for(unsigned int cell=0; cell<dftPtr->matrix_free_data.n_macro_cells(); ++cell)
      {
	fe_eval.reinit(cell);
	for (unsigned int q = 0; q < n_q_points; ++q)
	  fe_eval.submit_value(one,q);
	fe_eval.integrate (true,false);
	fe_eval.distribute_local_to_global(invSqrtMassVector);
      }

    invSqrtMassVector.compress(VectorOperation::add);

    //
    //evaluate inverse square root of the diagonal mass matrix and store in the dealii vector "invSqrtMassVector"
    //
    for(types::global_dof_index i = 0; i < invSqrtMassVector.size(); ++i)
      {
	if(invSqrtMassVector.in_local_range(i))
	  {
	    if(!dftPtr->getConstraintMatrixEigen().is_constrained(i))
	      {

		if(std::abs(invSqrtMassVector(i)) > 1.0e-15)
		  {
		    sqrtMassVector(i) = std::sqrt(invSqrtMassVector(i));
		    invSqrtMassVector(i) = 1.0/std::sqrt(invSqrtMassVector(i));
		  }
		AssertThrow(!std::isnan(invSqrtMassVector(i)),ExcMessage("Value of inverse square root of mass matrix on the unconstrained node is undefined"));

	      }
	  }
      }

    invSqrtMassVector.compress(VectorOperation::insert);
    sqrtMassVector.compress(VectorOperation::insert);
    computing_timer.exit_section("eigenClass Mass assembly");
  }

  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::reinitkPointIndex(unsigned int & kPointIndex)
  {
    d_kPointIndex = kPointIndex;
  }



  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
					const vectorType & phi,
					const vectorType & phiExt,
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
  void eigenClass<FEOrder>::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues,
					std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
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
			       const unsigned int numberWaveFunctionsPerBlock,
			       const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			       dealii::parallel::distributed::Vector<std::complex<double> > & dst)


  {
    const unsigned int numberDofs = src.local_size()/numberWaveFunctionsPerBlock;
    const unsigned int inc = 1;

    //
    //scale src vector with M^{-1/2}
    //
    for(int i = 0; i < numberDofs; ++i)
      {
	std::complex<double> scalingCoeff = invSqrtMassVector.local_element(i);
	zscal_(&numberWaveFunctionsPerBlock,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctionsPerBlock,
	       &inc);
      }

    //
    //This may be removed when memory optimization
    //
    const std::complex<double> zeroValue = 0.0;
    dst = zeroValue;

    //
    //update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src,
					      numberWaveFunctionsPerBlock);


    src.update_ghost_values();


    //
    //nonlocal PSP
    //

    //
    //H*M^{-1/2}*X
    //
    computeLocalHamiltonianTimesX(src,
				  numberWaveFunctionsPerBlock,
				  flattenedArrayCellLocalProcIndexIdMap,
				  dst);

    //
    //update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(dst,
							      numberWaveFunctionsPerBlock);



    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    //
    //M^{-1/2}*H*M^{-1/2}*X
    //
    for(int i = 0; i < numberDofs; ++i)
      {
	std::complex<double> scalingCoeff = invSqrtMassVector.local_element(i);
	zscal_(&numberWaveFunctionsPerBlock,
	       &scalingCoeff,
	       dst.begin()+i*numberWaveFunctionsPerBlock,
	       &inc);
      }
      

    //
    //unscale src M^{1/2}*X
    //
    for(int i = 0; i < numberDofs; ++i)
      {
	std::complex<double> scalingCoeff = sqrtMassVector.local_element(i);
	zscal_(&numberWaveFunctionsPerBlock,
	       &scalingCoeff,
	       src.begin()+i*numberWaveFunctionsPerBlock,
	       &inc);
      }

  }
#else
 template<unsigned int FEOrder>
  void eigenClass<FEOrder>::HX(dealii::parallel::distributed::Vector<double> & src,
			       const unsigned int numberWaveFunctionsPerBlock,
			       const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			       dealii::parallel::distributed::Vector<double> & dst)


  {
    const unsigned int numberDofs = src.local_size()/numberWaveFunctionsPerBlock;
    const unsigned int inc = 1;

    //
    //scale src vector with M^{-1/2}
    //
    for(int i = 0; i < numberDofs; ++i)
      {
	dscal_(&numberWaveFunctionsPerBlock,
	       &invSqrtMassVector.local_element(i),
	       src.begin()+i*numberWaveFunctionsPerBlock,
	       &inc);
      }

    //
    //This may be removed when memory optimization
    //
    dst = 0.0;

    //
    //update slave nodes before doing element-level matrix-vec multiplication
    //
    dftPtr->constraintsNoneDataInfo.distribute(src,
					      numberWaveFunctionsPerBlock);

    src.update_ghost_values();

    //
    //nonlocal PSP
    //

    //
    //H*M^{-1/2}*X
    //
    computeLocalHamiltonianTimesX(src,
				  numberWaveFunctionsPerBlock,
				  flattenedArrayCellLocalProcIndexIdMap,
				  dst);


    //
    //update master node contributions from its correponding slave nodes
    //
    dftPtr->constraintsNoneDataInfo.distribute_slave_to_master(dst,
							      numberWaveFunctionsPerBlock);


    src.zero_out_ghosts();
    dst.compress(VectorOperation::add);

    //
    //M^{-1/2}*H*M^{-1/2}*X
    //
    for(int i = 0; i < numberDofs; ++i)
      {
	dscal_(&numberWaveFunctionsPerBlock,
	       &invSqrtMassVector.local_element(i),
	       dst.begin()+i*numberWaveFunctionsPerBlock,
	       &inc);
      }
      

    //
    //unscale src M^{1/2}*X
    //
    for(int i = 0; i < numberDofs; ++i)
      {
	dscal_(&numberWaveFunctionsPerBlock,
	       &sqrtMassVector.local_element(i),
	       src.begin()+i*numberWaveFunctionsPerBlock,
	       &inc);
      }


  }
#endif


  //HX
  template<unsigned int FEOrder>
  void eigenClass<FEOrder>::HX(std::vector<vectorType> &src,
			       std::vector<vectorType> &dst)
  {



    computing_timer.enter_section("eigenClass HX");
    for (unsigned int i = 0; i < src.size(); i++)
      {
	src[i].scale(invSqrtMassVector); //M^{-1/2}*X
	//dftPtr->constraintsNoneEigen.distribute(src[i]);
	dftPtr->getConstraintMatrixEigenDataInfo().distribute(src[i]);
	src[i].update_ghost_values();
	dst[i] = 0.0;
      }


    //
    //required if its a pseudopotential calculation and number of nonlocal atoms are greater than zero
    //H^{nloc}*M^{-1/2}*X
    //if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
    //computeNonLocalHamiltonianTimesXMemoryOpt(src,dst);

    //
    //First evaluate H^{loc}*M^{-1/2}*X and then add to H^{nloc}*M^{-1/2}*X
    //
    dftPtr->matrix_free_data.cell_loop(&eigenClass<FEOrder>::computeLocalHamiltonianTimesXMF, this, dst, src); //HMX

    //
    //Finally evaluate M^{-1/2}*H*M^{-1/2}*X
    //
    for (std::vector<vectorType>::iterator it=dst.begin(); it!=dst.end(); it++)
      {
	(*it).scale(invSqrtMassVector);
      }

    //
    //unscale src back
    //
    for (std::vector<vectorType>::iterator it=src.begin(); it!=src.end(); it++)
      {
	(*it).scale(sqrtMassVector); //MHMX
      }

    computing_timer.exit_section("eigenClass HX");
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


    computing_timer.enter_section("eigenClass XHX");

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
  
    computing_timer.exit_section("eigenClass XHX");
 
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
  void eigenClass<FEOrder>::computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues,
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
  void eigenClass<FEOrder>::computeVEffSpinPolarized(std::map<dealii::CellId,std::vector<double> >* rhoValues,
						     std::map<dealii::CellId,std::vector<double> >* gradRhoValues,
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
