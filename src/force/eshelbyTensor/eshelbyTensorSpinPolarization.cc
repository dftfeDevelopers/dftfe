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
// @author Sambit Das (2017)
//
#include "../../../include/eshelbyTensorSpinPolarization.h"

namespace eshelbyTensorSP
{
    
 namespace internal
 {
   double getPartialOccupancy(double eigenValue, double fermiEnergy, double kb, double T)
   {
      const double factor=(eigenValue-fermiEnergy)/(kb*T);       
      return (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
   }
 }
Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicNoKPoints
                                                                      (const VectorizedArray<double> & phiTot,
		                                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
							               const VectorizedArray<double> & rho,
							               const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin0,
                                                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin1,
							               const VectorizedArray<double> & exc,
							               const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoQuadsSpin0,
                                                                       const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoQuadsSpin1,								       
                                                                       const VectorizedArray<double> & pseudoVLoc,
                                                                       const VectorizedArray<double> & phiExt)
{
   Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= make_vectorized_array(1.0/(4.0*M_PI))*outer_product(gradPhiTot,gradPhiTot)-outer_product(derExcGradRhoQuadsSpin0,gradRhoSpin0)-outer_product(derExcGradRhoQuadsSpin1,gradRhoSpin1);
   VectorizedArray<double> identityTensorFactor=make_vectorized_array(-1.0/(8.0*M_PI))*scalar_product(gradPhiTot,gradPhiTot)+rho*phiTot+exc*rho + (pseudoVLoc-phiExt)*rho;
   eshelbyTensor[0][0]+=identityTensorFactor;
   eshelbyTensor[1][1]+=identityTensorFactor;
   eshelbyTensor[2][2]+=identityTensorFactor;
   return eshelbyTensor;   
}

Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicKPoints
								      (std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
                                                                       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin0Begin,
                                                                       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,
                                                                       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin1Begin,
								       
								       const std::vector<double> & kPointCoordinates,
                                                                       const std::vector<double> & kPointWeights,							
								       const std::vector<std::vector<double> > & eigenValues_,
								       const double fermiEnergy_,
								       const double tVal)
{


   Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
   for (unsigned int idim=0; idim<C_DIM; idim++)
   {
     for (unsigned int jdim=0; jdim<C_DIM; jdim++)
     {       
       eshelbyTensor[idim][jdim]=make_vectorized_array(0.0);
     }
   }   
   VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);

   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
   std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator it2Spin0=gradPsiSpin0Begin;
   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;
   std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator it2Spin1=gradPsiSpin1Begin;   const unsigned int numEigenValues=eigenValues_[0].size()/2;

   Tensor<1,C_DIM,VectorizedArray<double> > kPointCoord;
   for (unsigned int ik=0; ik<kPointWeights.size(); ++ik){
     kPointCoord[0]=make_vectorized_array(kPointCoordinates[ik*C_DIM+0]);
     kPointCoord[1]=make_vectorized_array(kPointCoordinates[ik*C_DIM+1]);
     kPointCoord[2]=make_vectorized_array(kPointCoordinates[ik*C_DIM+2]);
     for (unsigned int eigenIndex=0; eigenIndex<numEigenValues; ++it1Spin0, ++it2Spin0,++it1Spin1,++it2Spin1, ++ eigenIndex){
        const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
        const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsiSpin0=*it2Spin0;
        const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;
        const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsiSpin1=*it2Spin1;	

        const double partOccSpin0 =internal::getPartialOccupancy(eigenValues_[ik][eigenIndex],
		                                                 fermiEnergy_,
								 C_kb,
								 tVal); 
        const double partOccSpin1 =internal::getPartialOccupancy(eigenValues_[ik][eigenIndex+numEigenValues],
		                                                 fermiEnergy_,
								 C_kb,
								 tVal); 

	VectorizedArray<double> identityTensorFactorContributionSpin0=make_vectorized_array(0.0);
	VectorizedArray<double> identityTensorFactorContributionSpin1=make_vectorized_array(0.0);	
	const VectorizedArray<double> fnkSpin0=make_vectorized_array(partOccSpin0*kPointWeights[ik]);
        const VectorizedArray<double> fnkSpin1=make_vectorized_array(partOccSpin1*kPointWeights[ik]);

        identityTensorFactorContributionSpin0+=(scalar_product(gradPsiSpin0[0],gradPsiSpin0[0])+scalar_product(gradPsiSpin0[1],gradPsiSpin0[1]));
	identityTensorFactorContributionSpin0+=make_vectorized_array(2.0)*(psiSpin0[0]*scalar_product(kPointCoord,gradPsiSpin0[1])-psiSpin0[1]*scalar_product(kPointCoord,gradPsiSpin0[0]));
	identityTensorFactorContributionSpin0+=(scalar_product(kPointCoord,kPointCoord)-make_vectorized_array(2.0*eigenValues_[ik][eigenIndex]))*(psiSpin0[0]*psiSpin0[0]+psiSpin0[1]*psiSpin0[1]);

        identityTensorFactorContributionSpin1+=(scalar_product(gradPsiSpin1[0],gradPsiSpin1[0])+scalar_product(gradPsiSpin1[1],gradPsiSpin1[1]));
	identityTensorFactorContributionSpin1+=make_vectorized_array(2.0)*(psiSpin1[0]*scalar_product(kPointCoord,gradPsiSpin1[1])-psiSpin1[1]*scalar_product(kPointCoord,gradPsiSpin1[0]));
	identityTensorFactorContributionSpin1+=(scalar_product(kPointCoord,kPointCoord)-make_vectorized_array(2.0*eigenValues_[ik][eigenIndex]))*(psiSpin1[0]*psiSpin1[0]+psiSpin1[1]*psiSpin1[1]);	

	identityTensorFactor+=(identityTensorFactorContributionSpin0*fnkSpin0+identityTensorFactorContributionSpin1*fnkSpin1)*make_vectorized_array(0.5);

        eshelbyTensor-=fnkSpin0*(outer_product(gradPsiSpin0[0],gradPsiSpin0[0])+outer_product(gradPsiSpin0[1],gradPsiSpin0[1])+psiSpin0[0]*outer_product(gradPsiSpin0[1],kPointCoord)-psiSpin0[1]*outer_product(gradPsiSpin0[0],kPointCoord));
        eshelbyTensor-=fnkSpin1*(outer_product(gradPsiSpin1[0],gradPsiSpin1[0])+outer_product(gradPsiSpin1[1],gradPsiSpin1[1])+psiSpin1[0]*outer_product(gradPsiSpin1[1],kPointCoord)-psiSpin1[1]*outer_product(gradPsiSpin1[0],kPointCoord));	
     }
   }

   eshelbyTensor[0][0]+=identityTensorFactor;
   eshelbyTensor[1][1]+=identityTensorFactor;
   eshelbyTensor[2][2]+=identityTensorFactor;
   return eshelbyTensor;
}

Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorNonPeriodic(const VectorizedArray<double> & phiTot,
		                                                          const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
									  const VectorizedArray<double> & rho,
									  const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin0,
                                                                          const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin1,									  
									  const VectorizedArray<double> & exc,
									  const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoQuadsSpin0,
                                                                          const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoQuadsSpin1,									  
                                                                          const VectorizedArray<double> & pseudoVLoc,
                                                                          const VectorizedArray<double> & phiExt,
									  std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
                                                                          std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin0Begin,
									  std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
                                                                          std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin1Begin,									  
									  const std::vector<double> & eigenValues_,
									  const double fermiEnergy_,
									  const double tVal)
{

   Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= make_vectorized_array(1.0/(4.0*M_PI))*outer_product(gradPhiTot,gradPhiTot)-outer_product(derExcGradRhoQuadsSpin0,gradRhoSpin0)-outer_product(derExcGradRhoQuadsSpin1,gradRhoSpin1);
   VectorizedArray<double> identityTensorFactor=make_vectorized_array(-1.0/(8.0*M_PI))*scalar_product(gradPhiTot,gradPhiTot)+rho*phiTot+exc*rho + (pseudoVLoc-phiExt)*rho;

   std::vector<VectorizedArray<double> >::const_iterator it1Spin0=psiSpin0Begin;   
   std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator it2Spin0=gradPsiSpin0Begin;
   std::vector<VectorizedArray<double> >::const_iterator it1Spin1=psiSpin1Begin;   
   std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator it2Spin1=gradPsiSpin1Begin;   
   const unsigned int numEigenValues=eigenValues_.size()/2;   
   for (unsigned int eigenIndex=0; eigenIndex < numEigenValues; ++it1Spin0, ++it2Spin0,  ++it1Spin1, ++it2Spin1, ++ eigenIndex){
      const VectorizedArray<double> & psiSpin0= *it1Spin0;
      const Tensor<1,C_DIM,VectorizedArray<double> > & gradPsiSpin0=*it2Spin0;
      const VectorizedArray<double> & psiSpin1= *it1Spin1;
      const Tensor<1,C_DIM,VectorizedArray<double> > & gradPsiSpin1=*it2Spin1;      
      const double partOccSpin0 =internal::getPartialOccupancy(eigenValues_[eigenIndex],
							       fermiEnergy_,
							       C_kb,
							       tVal); 
      const double partOccSpin1 =internal::getPartialOccupancy(eigenValues_[eigenIndex+numEigenValues],
		                                               fermiEnergy_,
							       C_kb,
							       tVal);    
      identityTensorFactor+=make_vectorized_array(0.5*partOccSpin0)*scalar_product(gradPsiSpin0,gradPsiSpin0)-make_vectorized_array(partOccSpin0*eigenValues_[eigenIndex])*psiSpin0*psiSpin0;
      identityTensorFactor+=make_vectorized_array(0.5*partOccSpin1)*scalar_product(gradPsiSpin1,gradPsiSpin1)-make_vectorized_array(partOccSpin1*eigenValues_[eigenIndex+numEigenValues])*psiSpin1*psiSpin1;      
      eshelbyTensor-=make_vectorized_array(partOccSpin0)*outer_product(gradPsiSpin0,gradPsiSpin0);
      eshelbyTensor-=make_vectorized_array(partOccSpin1)*outer_product(gradPsiSpin1,gradPsiSpin1);
   }
   
   eshelbyTensor[0][0]+=identityTensorFactor;
   eshelbyTensor[1][1]+=identityTensorFactor;
   eshelbyTensor[2][2]+=identityTensorFactor;
   return eshelbyTensor;
}


Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorNonPeriodic(const std::vector<std::vector<VectorizedArray<double> > > & ZetaDeltaV,
								         const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesV,
                                                                         const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesV,
									 
								         std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
                                                                         std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,									 
								         const std::vector<double> & eigenValues_,
								         const double fermiEnergy_,
								         const double tVal)
{

   Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
   VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);
   std::vector<VectorizedArray<double> >::const_iterator it1Spin0=psiSpin0Begin; 
   std::vector<VectorizedArray<double> >::const_iterator it1Spin1=psiSpin1Begin;     
   const unsigned int numEigenValues=eigenValues_.size()/2;    
   for (unsigned int eigenIndex=0; eigenIndex < numEigenValues; ++it1Spin0, ++it1Spin1, ++ eigenIndex)
   {
      const VectorizedArray<double> & psiSpin0= *it1Spin0;
      const VectorizedArray<double> & psiSpin1= *it1Spin1;      
      const double partOccSpin0 =internal::getPartialOccupancy(eigenValues_[eigenIndex],
							       fermiEnergy_,
							       C_kb,
							       tVal); 
      const double partOccSpin1 =internal::getPartialOccupancy(eigenValues_[eigenIndex+numEigenValues],
		                                               fermiEnergy_,
							       C_kb,
							       tVal);   
      for (unsigned int iAtomNonLocal=0; iAtomNonLocal < ZetaDeltaV.size(); ++iAtomNonLocal)
      {
	 const int numberPseudoWaveFunctions = ZetaDeltaV[iAtomNonLocal].size();
         for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
         {	  
             identityTensorFactor+=make_vectorized_array(2.0*partOccSpin0*projectorKetTimesPsiSpin0TimesV[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin0*ZetaDeltaV[iAtomNonLocal][iPseudoWave];
             identityTensorFactor+=make_vectorized_array(2.0*partOccSpin1*projectorKetTimesPsiSpin1TimesV[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin1*ZetaDeltaV[iAtomNonLocal][iPseudoWave];	     
	 }
      }
   }  
   eshelbyTensor[0][0]=identityTensorFactor;
   eshelbyTensor[1][1]=identityTensorFactor;
   eshelbyTensor[2][2]=identityTensorFactor;
   
   return eshelbyTensor;
}

Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
								      const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
                                                                      const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,
								      
								      std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
                                                                      std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,								      
                                                                      const std::vector<double> & kPointWeights,								      
								      const std::vector<std::vector<double> > & eigenValues_,
								      const double fermiEnergy_,
								      const double tVal)
{
   Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
   VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);
   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;   
   VectorizedArray<double> two=make_vectorized_array(2.0);
   const unsigned int numKPoints=kPointWeights.size();
   const unsigned int numEigenValues=eigenValues_[0].size()/2;    
   for (unsigned int ik=0; ik<numKPoints; ++ik){
     for (unsigned int eigenIndex=0; eigenIndex<numEigenValues; ++it1Spin0,++it1Spin1, ++ eigenIndex){
        const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
        const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;	

        const double partOccSpin0 =internal::getPartialOccupancy(eigenValues_[ik][eigenIndex],
							         fermiEnergy_,
							         C_kb,
							         tVal); 
        const double partOccSpin1 =internal::getPartialOccupancy(eigenValues_[ik][eigenIndex+numEigenValues],
		                                                 fermiEnergy_,
							         C_kb,
							         tVal);  

	const VectorizedArray<double> fnkSpin0=make_vectorized_array(partOccSpin0*kPointWeights[ik]);
	const VectorizedArray<double> fnkSpin1=make_vectorized_array(partOccSpin1*kPointWeights[ik]);	
	for (unsigned int iAtomNonLocal=0; iAtomNonLocal < ZetaDeltaV.size(); ++iAtomNonLocal)
	{
	     const unsigned int numberPseudoWaveFunctions = ZetaDeltaV[iAtomNonLocal].size();
	     for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	     {	  
		 const VectorizedArray<double> CRealSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		 const VectorizedArray<double> CImagSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		 const VectorizedArray<double> CRealSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		 const VectorizedArray<double> CImagSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());		 
		 const VectorizedArray<double> zdvR=ZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
		 const VectorizedArray<double> zdvI=ZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];
		 identityTensorFactor+=two*fnkSpin0*((psiSpin0[0]*zdvR+psiSpin0[1]*zdvI)*CRealSpin0-(psiSpin0[0]*zdvI-psiSpin0[1]*zdvR)*CImagSpin0);
		 identityTensorFactor+=two*fnkSpin1*((psiSpin1[0]*zdvR+psiSpin1[1]*zdvI)*CRealSpin1-(psiSpin1[0]*zdvI-psiSpin1[1]*zdvR)*CImagSpin1);		 
	     }
	}	

     }
   }

   eshelbyTensor[0][0]+=identityTensorFactor;
   eshelbyTensor[1][1]+=identityTensorFactor;
   eshelbyTensor[2][2]+=identityTensorFactor;
   return eshelbyTensor;
    
}

Tensor<1,C_DIM,VectorizedArray<double> >  getFnlNonPeriodic(const std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > & gradZetaDeltaV,
						            const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesV,
                                                            const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesV,							    
						            std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
                                                            std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
							    
						            const std::vector<double> & eigenValues_,
						            const double fermiEnergy_,
						            const double tVal)
{

   Tensor<1,C_DIM,VectorizedArray<double> > F; 
   std::vector<VectorizedArray<double> >::const_iterator it1Spin0=psiSpin0Begin;
   std::vector<VectorizedArray<double> >::const_iterator it1Spin1=psiSpin1Begin;    
   const unsigned int numEigenValues=eigenValues_.size()/2;     
   for (unsigned int eigenIndex=0; eigenIndex < numEigenValues; ++it1Spin0,++it1Spin1, ++ eigenIndex)
   {
      const VectorizedArray<double> & psiSpin0= *it1Spin0;
      const VectorizedArray<double> & psiSpin1= *it1Spin1;      
      const double partOccSpin0 =internal::getPartialOccupancy(eigenValues_[eigenIndex],
							       fermiEnergy_,
							       C_kb,
							       tVal); 
      const double partOccSpin1 =internal::getPartialOccupancy(eigenValues_[eigenIndex+numEigenValues],
		                                               fermiEnergy_,
							       C_kb,
							       tVal);  
      for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetaDeltaV.size(); ++iAtomNonLocal)
      {
	 const unsigned int numberPseudoWaveFunctions = gradZetaDeltaV[iAtomNonLocal].size();
         for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
         {	
	     F+=make_vectorized_array(2.0*partOccSpin0*projectorKetTimesPsiSpin0TimesV[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin0*gradZetaDeltaV[iAtomNonLocal][iPseudoWave];
	     F+=make_vectorized_array(2.0*partOccSpin1*projectorKetTimesPsiSpin1TimesV[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin1*gradZetaDeltaV[iAtomNonLocal][iPseudoWave];	     
	 }
      }
   }  
   
   return F;
}

Tensor<1,C_DIM,VectorizedArray<double> >  getFPSPLocal(const VectorizedArray<double> rho,
		                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPseudoVLoc,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt)

{

   return rho*(gradPseudoVLoc-gradPhiExt);
}

Tensor<1,C_DIM,VectorizedArray<double> >  getFnlPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
						         const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
                                                         const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,							 
						         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
                                                         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
							 
                                                         const std::vector<double> & kPointWeights,								      
					                 const std::vector<std::vector<double> > & eigenValues_,
						         const double fermiEnergy_,
						         const double tVal)
{
   Tensor<1,C_DIM,VectorizedArray<double> > F;
   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;   
   VectorizedArray<double> two=make_vectorized_array(2.0);
   const unsigned int numKPoints=kPointWeights.size();
   const unsigned int numEigenValues=eigenValues_[0].size()/2;     
   for (unsigned int ik=0; ik<numKPoints; ++ik){
     for (unsigned int eigenIndex=0; eigenIndex<numEigenValues; ++it1Spin0,++it1Spin1, ++ eigenIndex){
        const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
        const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;	

        const double partOccSpin0 =internal::getPartialOccupancy(eigenValues_[ik][eigenIndex],
							       fermiEnergy_,
							       C_kb,
							       tVal); 
        const double partOccSpin1 =internal::getPartialOccupancy(eigenValues_[ik][eigenIndex+numEigenValues],
		                                               fermiEnergy_,
							       C_kb,
							       tVal);  
        const VectorizedArray<double> fnkSpin0=make_vectorized_array(partOccSpin0*kPointWeights[ik]);
        const VectorizedArray<double> fnkSpin1=make_vectorized_array(partOccSpin1*kPointWeights[ik]);	
	for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetaDeltaV.size(); ++iAtomNonLocal)
	{
	     const unsigned int numberPseudoWaveFunctions = gradZetaDeltaV[iAtomNonLocal].size();
	     for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	     {	  
		 const VectorizedArray<double> CRealSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		 const VectorizedArray<double> CImagSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		 const VectorizedArray<double> CRealSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		 const VectorizedArray<double> CImagSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());		 
		 const Tensor<1,C_DIM,VectorizedArray<double> >  zdvR=gradZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
		 const Tensor<1,C_DIM,VectorizedArray<double> >  zdvI=gradZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];
		 F+=two*fnkSpin0*((psiSpin0[0]*zdvR+psiSpin0[1]*zdvI)*CRealSpin0-(psiSpin0[0]*zdvI-psiSpin0[1]*zdvR)*CImagSpin0);
		 F+=two*fnkSpin1*((psiSpin1[0]*zdvR+psiSpin1[1]*zdvI)*CRealSpin1-(psiSpin1[0]*zdvI-psiSpin1[1]*zdvR)*CImagSpin1);		 
	     }
	}	

     }
   }

   return F;    
}

Tensor<1,C_DIM,VectorizedArray<double> >  getNonSelfConsistentForce(const VectorizedArray<double> & vEffRhoInSpin0,
								    const VectorizedArray<double> & vEffRhoOutSpin0,
                                                                    const VectorizedArray<double> & vEffRhoInSpin1,
								    const VectorizedArray<double> & vEffRhoOutSpin1,
								    
							            const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoOutSpin0,
                                                                    const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoOutSpin1,								    
							            const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoInSpin0,
								    const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoInSpin1,
                                                                    const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoOutSpin0,
								    const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoOutSpin1,
                                                                    const Tensor<2,C_DIM,VectorizedArray<double> > & hessianRhoOutSpin0,
                                                                    const Tensor<2,C_DIM,VectorizedArray<double> > & hessianRhoOutSpin1)
{
   Tensor<1,C_DIM,VectorizedArray<double> > F; 

   F+= (vEffRhoOutSpin0-vEffRhoInSpin0)*gradRhoOutSpin0+(vEffRhoOutSpin1-vEffRhoInSpin1)*gradRhoOutSpin1+(derExchCorrEnergyWithGradRhoOutSpin0-derExchCorrEnergyWithGradRhoInSpin0)*hessianRhoOutSpin0+(derExchCorrEnergyWithGradRhoOutSpin1-derExchCorrEnergyWithGradRhoInSpin1)*hessianRhoOutSpin1; 
   return F;
}

}
