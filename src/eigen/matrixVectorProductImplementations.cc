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
// @author Phani Motamarri (2018)
//


/** @file matrixVectorProductImplementations.cc
 *  @brief Contains linear algebra operations
 *
 */


template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeLocalHamiltonianTimesXMF (const dealii::MatrixFree<3,double>  &data,
							   std::vector<vectorType>  &dst,
							   const std::vector<vectorType>  &src,
							   const std::pair<unsigned int,unsigned int> &cell_range) const
{
  VectorizedArray<double>  half = make_vectorized_array(0.5);
  VectorizedArray<double>  two = make_vectorized_array(2.0);


#ifdef ENABLE_PERIODIC_BC
  int kPointIndex = d_kPointIndex;
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
	      fe_eval.read_dof_values(src[i]);
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
	      fe_eval.distribute_local_to_global (dst[i]);

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
	      fe_eval.read_dof_values(src[i]);
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
	      fe_eval.distribute_local_to_global (dst[i]);

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
	      fe_eval.read_dof_values(src[i]);
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
	      fe_eval.distribute_local_to_global (dst[i]);
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
	      fe_eval.read_dof_values(src[i]);
	      fe_eval.evaluate (true,true,false);
	      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
		{
		  fe_eval.submit_gradient(fe_eval.get_gradient(q)*half, q);
		  fe_eval.submit_value(fe_eval.get_value(q)*vEff(cell,q), q);
		}

	      fe_eval.integrate (true, true);
	      fe_eval.distribute_local_to_global(dst[i]);
	    }
	}

    }
#endif
}


#ifdef ENABLE_PERIODIC_BC
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<std::complex<double> > & src,
							const int numberWaveFunctions,
							const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
							dealii::parallel::distributed::Vector<std::complex<double> > & dst) const
{

  //
  //initialize dst to zeros. Will be changed later
  //
  //const std::complex<double> zeroValue = 0.0;
  //dst = zeroValue;

  //
  //element level matrix-vector multiplications
  //
  char transA = 'N',transB = 'T';
  std::complex<double> scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
  int inc = 1;

  std::vector<std::complex<double> > cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
  std::vector<std::complex<double> > cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  int iElem = 0;
  for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
    {
      for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
	{
	  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      zcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
	    }

	  zgemm_(&transA,
		 &transB,
		 &numberWaveFunctions,
		 &d_numberNodesPerElement,
		 &d_numberNodesPerElement,
		 &scalarCoeffAlpha,
		 &cellWaveFunctionMatrix[0],
		 &numberWaveFunctions,
		 &d_cellHamiltonianMatrix[iElem][0],
		 &d_numberNodesPerElement,
		 &scalarCoeffBeta,
		 &cellHamMatrixTimesWaveMatrix[0],
		 &numberWaveFunctions);

	  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      zaxpy_(&numberWaveFunctions,
		     &scalarCoeffAlpha,
		     &cellHamMatrixTimesWaveMatrix[numberWaveFunctions*iNode],
		     &inc,
		     dst.begin()+localNodeId,
		     &inc);
	    }

	  ++iElem;
	}//subcell loop
    }//macrocell loop

}
#else
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<double> & src,
							const int numberWaveFunctions,
							const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
							dealii::parallel::distributed::Vector<double> & dst) const
{
  //
  //initialize dst to zeros. Will be changed later
  //
  //dst = 0.0;

  //
  //element level matrix-vector multiplications
  //
  char transA = 'N',transB = 'N';
  double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
  int inc = 1;

  std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
  std::vector<double> cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  int iElem = 0;
  for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
    {
      for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
	{
	  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      dcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
	    }

	  dgemm_(&transA,
		 &transB,
		 &numberWaveFunctions,
		 &d_numberNodesPerElement,
		 &d_numberNodesPerElement,
		 &scalarCoeffAlpha,
		 &cellWaveFunctionMatrix[0],
		 &numberWaveFunctions,
		 &d_cellHamiltonianMatrix[iElem][0],
		 &d_numberNodesPerElement,
		 &scalarCoeffBeta,
		 &cellHamMatrixTimesWaveMatrix[0],
		 &numberWaveFunctions);

	  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      daxpy_(&numberWaveFunctions,
		     &scalarCoeffAlpha,
		     &cellHamMatrixTimesWaveMatrix[numberWaveFunctions*iNode],
		     &inc,
		     dst.begin()+localNodeId,
		     &inc);
	    }

	  ++iElem;
	}//subcell loop
    }//macrocell loop

}
#endif

