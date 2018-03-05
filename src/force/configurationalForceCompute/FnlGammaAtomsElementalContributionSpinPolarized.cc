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
// @author Sambit Das (2018)
//

#ifdef ENABLE_PERIODIC_BC 
//(locally used function) compute Fnl contibution due to Gamma(Rj) for given set of cells  
template<unsigned int FEOrder>
void forceClass<FEOrder>::FnlGammaAtomsElementalContributionPeriodicSpinPolarized
                                  (std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
				   FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
				   const unsigned int cell,
				   const std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > > & pspnlGammaAtomsQuads,
                                   const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiSpin0TimesV,
                                   const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiSpin1TimesV,					   
				   const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiSpin0Quads,
				   const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiSpin1Quads)
{
 
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numEigenVectors=dftPtr->eigenVectorsOrig[0].size(); 
  const unsigned int numKPoints=dftPtr->d_kPointWeights.size();
  const unsigned int numSubCells= dftPtr->matrix_free_data.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  unsigned int numNonLocalAtomsCurrentProcess= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

  for(int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
  {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =  dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];
      std::vector<std::vector<std::vector<std::complex<double> > > >   temp2Spin0(numKPoints);
      std::vector<std::vector<std::vector<std::complex<double> > > >   temp2Spin1(numKPoints);      
      for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
      {
	  temp2Spin0[ikPoint].resize(1);
	  temp2Spin0[ikPoint][0]=projectorKetTimesPsiSpin0TimesV[ikPoint][iAtom];
	  temp2Spin1[ikPoint].resize(1);
	  temp2Spin1[ikPoint][0]=projectorKetTimesPsiSpin0TimesV[ikPoint][iAtom];
      }        
      for (unsigned int q=0; q<numQuadPoints; ++q)
      {  	  
	   std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > temp1(1);
	   temp1[0]=pspnlGammaAtomsQuads[q][iAtom];	
   
           Tensor<1,C_DIM,VectorizedArray<double> > 
	       F=-eshelbyTensorSP::getFnlPeriodic(temp1,
						temp2Spin0,
						temp2Spin1,
						psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
                                                psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,		
						dftPtr->d_kPointWeights,
						dftPtr->eigenValues,
						dftPtr->fermiEnergy,
						dftParameters::TVal);
 

           forceEval.submit_value(F,q);
      }
      Tensor<1,C_DIM,VectorizedArray<double> > forceContributionFnlGammaiAtomCells
						     =forceEval.integrate_value();

      if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom)==forceContributionFnlGammaAtoms.end())
	   forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]=std::vector<double>(C_DIM,0.0);
      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
      {
	   for (unsigned int idim=0; idim<C_DIM; idim++)
	   {
	     forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom][idim]+=
		   forceContributionFnlGammaiAtomCells[idim][iSubCell];
	   }
      }
  }//iAtom loop
}

#else

template<unsigned int FEOrder>
void forceClass<FEOrder>::FnlGammaAtomsElementalContributionNonPeriodicSpinPolarized
                                  (std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
				   FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
				   const unsigned int cell,
				   const std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > pspnlGammaAtomQuads,
                                   const std::vector<std::vector<double> >  & projectorKetTimesPsiSpin0TimesV,
                                   const std::vector<std::vector<double> >  & projectorKetTimesPsiSpin1TimesV,
				   const std::vector< VectorizedArray<double> > & psiSpin0Quads,
				   const std::vector< VectorizedArray<double> > & psiSpin1Quads)
{
 
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numEigenVectors=dftPtr->eigenVectorsOrig[0].size(); 
  const unsigned int numSubCells= dftPtr->matrix_free_data.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  unsigned int numNonLocalAtomsCurrentProcess= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

  for(int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
  {
      //
      //get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =  dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];
      std::vector<std::vector<double> >  temp2Spin0(1);
      temp2Spin0[0]=projectorKetTimesPsiSpin0TimesV[iAtom];
      std::vector<std::vector<double> >  temp2Spin1(1);
      temp2Spin1[0]=projectorKetTimesPsiSpin1TimesV[iAtom];      
      for (unsigned int q=0; q<numQuadPoints; ++q)
      {  	  
	   std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > temp1(1);
	   temp1[0]=pspnlGammaAtomQuads[q][iAtom];

           Tensor<1,C_DIM,VectorizedArray<double> > F=
	                  -eshelbyTensorSP::getFnlNonPeriodic(temp1,
					                    temp2Spin0,
							    temp2Spin1,
					                    psiSpin0Quads.begin()+q*numEigenVectors,
							    psiSpin1Quads.begin()+q*numEigenVectors,
					                    (dftPtr->eigenValues)[0],
					                    dftPtr->fermiEnergy,
					                    dftParameters::TVal);  
	   
            							    
           forceEval.submit_value(F,q);
      }
      Tensor<1,C_DIM,VectorizedArray<double> > forceContributionFnlGammaiAtomCells
						     =forceEval.integrate_value();

      if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom)==forceContributionFnlGammaAtoms.end())
	   forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]=std::vector<double>(C_DIM,0.0);
      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
      {
	   for (unsigned int idim=0; idim<C_DIM; idim++)
	   {
	     forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom][idim]+=
		   forceContributionFnlGammaiAtomCells[idim][iSubCell];
	   }
      }
  }//iAtom loop
}

#endif
