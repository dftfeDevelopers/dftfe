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
// @author Sambit Das (2017)
//
#include <eshelbyTensorSpinPolarized.h>
#include <dftUtils.h>
#include <dftParameters.h>

namespace dftfe {

namespace eshelbyTensorSP
{

    Tensor<2,C_DIM,VectorizedArray<double> >  getELocXcEshelbyTensor
						       (const VectorizedArray<double> & rho,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin0,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin1,
						       const VectorizedArray<double> & exc,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin0,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin1)
    {
       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= -outer_product(derExcGradRhoSpin0,gradRhoSpin0)
	                                                       -outer_product(derExcGradRhoSpin1,gradRhoSpin1);
       VectorizedArray<double> identityTensorFactor=exc*rho;
       eshelbyTensor[0][0]+=identityTensorFactor;
       eshelbyTensor[1][1]+=identityTensorFactor;
       eshelbyTensor[2][2]+=identityTensorFactor;
       return eshelbyTensor;
    }

    Tensor<2,C_DIM,VectorizedArray<double> >  getELocWfcEshelbyTensorPeriodicKPoints
		    (std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
		     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,
		     std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin0Begin,
		     std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin1Begin,
		     const std::vector<double> & kPointCoordinates,
		     const std::vector<double> & kPointWeights,
		     const std::vector<std::vector<double> > & eigenValues_,
		     const double fermiEnergy_,
		     const double fermiEnergyUp_,
		     const double fermiEnergyDown_,
		     const double tVal)
    {


       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
       for (unsigned int idim=0; idim<C_DIM; idim++)
	 for (unsigned int jdim=0; jdim<C_DIM; jdim++)
	   eshelbyTensor[idim][jdim]=make_vectorized_array(0.0);

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

	    double partOccSpin0 =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
								     fermiEnergy_,
								     C_kb,
								     tVal);
	    double partOccSpin1 =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex+numEigenValues],
								     fermiEnergy_,
								     C_kb,
								     tVal);

	    if(dftParameters::constraintMagnetization)
	    {
		 partOccSpin0 = 1.0 , partOccSpin1 = 1.0 ;
		 if (eigenValues_[ik][eigenIndex+numEigenValues]> fermiEnergyDown_)
			partOccSpin1 = 0.0 ;
		 if (eigenValues_[ik][eigenIndex] > fermiEnergyUp_)
			partOccSpin0 = 0.0 ;
	    }

	    VectorizedArray<double> identityTensorFactorContributionSpin0=make_vectorized_array(0.0);
	    VectorizedArray<double> identityTensorFactorContributionSpin1=make_vectorized_array(0.0);
	    const VectorizedArray<double> fnkSpin0=make_vectorized_array(partOccSpin0*kPointWeights[ik]);
	    const VectorizedArray<double> fnkSpin1=make_vectorized_array(partOccSpin1*kPointWeights[ik]);

	    identityTensorFactorContributionSpin0+=(scalar_product(gradPsiSpin0[0],gradPsiSpin0[0])+scalar_product(gradPsiSpin0[1],gradPsiSpin0[1]));
	    identityTensorFactorContributionSpin0+=make_vectorized_array(2.0)*(psiSpin0[0]*scalar_product(kPointCoord,gradPsiSpin0[1])-psiSpin0[1]*scalar_product(kPointCoord,gradPsiSpin0[0]));
	    identityTensorFactorContributionSpin0+=(scalar_product(kPointCoord,kPointCoord)-make_vectorized_array(2.0*eigenValues_[ik][eigenIndex]))*(psiSpin0[0]*psiSpin0[0]+psiSpin0[1]*psiSpin0[1]);

	    identityTensorFactorContributionSpin1+=(scalar_product(gradPsiSpin1[0],gradPsiSpin1[0])+scalar_product(gradPsiSpin1[1],gradPsiSpin1[1]));
	    identityTensorFactorContributionSpin1+=make_vectorized_array(2.0)*(psiSpin1[0]*scalar_product(kPointCoord,gradPsiSpin1[1])-psiSpin1[1]*scalar_product(kPointCoord,gradPsiSpin1[0]));
	    identityTensorFactorContributionSpin1+=(scalar_product(kPointCoord,kPointCoord)-make_vectorized_array(2.0*eigenValues_[ik][eigenIndex+numEigenValues]))*(psiSpin1[0]*psiSpin1[0]+psiSpin1[1]*psiSpin1[1]);

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

    Tensor<2,C_DIM,VectorizedArray<double> >  getELocWfcEshelbyTensorNonPeriodic
			(std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
			std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
			std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin0Begin,
			std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin1Begin,
			const std::vector<double> & eigenValues_,
		        const double fermiEnergy_,
		        const double fermiEnergyUp_,
		        const double fermiEnergyDown_,
			const double tVal)
    {

       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
       for (unsigned int idim=0; idim<C_DIM; idim++)
	 for (unsigned int jdim=0; jdim<C_DIM; jdim++)
	   eshelbyTensor[idim][jdim]=make_vectorized_array(0.0);

       VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);

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

	  double partOccSpin0 =dftUtils::getPartialOccupancy(eigenValues_[eigenIndex],
								     fermiEnergy_,
								     C_kb,
								     tVal);

	  double partOccSpin1 =dftUtils::getPartialOccupancy(eigenValues_[eigenIndex+numEigenValues],
								     fermiEnergy_,
								     C_kb,
								     tVal);

	  if(dftParameters::constraintMagnetization)
	  {
		 partOccSpin0 = 1.0 , partOccSpin1 = 1.0 ;
		 if (eigenValues_[eigenIndex+numEigenValues]> fermiEnergyDown_)
			partOccSpin1 = 0.0 ;
		 if (eigenValues_[eigenIndex] > fermiEnergyUp_)
			partOccSpin0 = 0.0 ;
	  }

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
									     const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesVTimesPartOcc,
									     const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesVTimesPartOcc,
									     std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
									     std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,

									     const unsigned int numBlockedEigenvectors)
    {

       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
       VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);
       std::vector<VectorizedArray<double> >::const_iterator it1Spin0=psiSpin0Begin;
       std::vector<VectorizedArray<double> >::const_iterator it1Spin1=psiSpin1Begin;
       for (unsigned int eigenIndex=0; eigenIndex < numBlockedEigenvectors; ++it1Spin0, ++it1Spin1, ++ eigenIndex)
       {
	  const VectorizedArray<double> & psiSpin0= *it1Spin0;
	  const VectorizedArray<double> & psiSpin1= *it1Spin1;

	  for (unsigned int iAtomNonLocal=0; iAtomNonLocal < ZetaDeltaV.size(); ++iAtomNonLocal)
	  {
	     const int numberPseudoWaveFunctions = ZetaDeltaV[iAtomNonLocal].size();
	     for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	     {
		 identityTensorFactor+=make_vectorized_array(2.0*projectorKetTimesPsiSpin0TimesVTimesPartOcc[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin0*ZetaDeltaV[iAtomNonLocal][iPseudoWave];
		 identityTensorFactor+=make_vectorized_array(2.0*projectorKetTimesPsiSpin1TimesVTimesPartOcc[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin1*ZetaDeltaV[iAtomNonLocal][iPseudoWave];
	     }
	  }
       }
       eshelbyTensor[0][0]=identityTensorFactor;
       eshelbyTensor[1][1]=identityTensorFactor;
       eshelbyTensor[2][2]=identityTensorFactor;

       return eshelbyTensor;
    }

    Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
									  const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesVTimesPartOcc,
									  const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesVTimesPartOcc,
									  std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
									  std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,
									  const std::vector<double> & kPointWeights,
									  const unsigned int numBlockedEigenvectors)
    {
       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
       VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;
       VectorizedArray<double> two=make_vectorized_array(2.0);
       const unsigned int numKPoints=kPointWeights.size();
       for (unsigned int ik=0; ik<numKPoints; ++ik){
	 for (unsigned int eigenIndex=0; eigenIndex<numBlockedEigenvectors; ++it1Spin0,++it1Spin1, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;
	    const VectorizedArray<double> fnkSpin0=make_vectorized_array(kPointWeights[ik]);
	    const VectorizedArray<double> fnkSpin1=make_vectorized_array(kPointWeights[ik]);
	    for (unsigned int iAtomNonLocal=0; iAtomNonLocal < ZetaDeltaV.size(); ++iAtomNonLocal)
	    {
		 const unsigned int numberPseudoWaveFunctions = ZetaDeltaV[iAtomNonLocal].size();
		 for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		 {
		     const VectorizedArray<double> CRealSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     const VectorizedArray<double> CImagSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     const VectorizedArray<double> CRealSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     const VectorizedArray<double> CImagSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
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
								const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesVTimesPartOcc,
								const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesVTimesPartOcc,
								std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
								std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
								const unsigned int numBlockedEigenvectors)
    {

       Tensor<1,C_DIM,VectorizedArray<double> > F;
       std::vector<VectorizedArray<double> >::const_iterator it1Spin0=psiSpin0Begin;
       std::vector<VectorizedArray<double> >::const_iterator it1Spin1=psiSpin1Begin;
       for (unsigned int eigenIndex=0; eigenIndex < numBlockedEigenvectors; ++it1Spin0,++it1Spin1, ++ eigenIndex)
       {
	  const VectorizedArray<double> & psiSpin0= *it1Spin0;
	  const VectorizedArray<double> & psiSpin1= *it1Spin1;

	  for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetaDeltaV.size(); ++iAtomNonLocal)
	  {
	     const unsigned int numberPseudoWaveFunctions = gradZetaDeltaV[iAtomNonLocal].size();
	     for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	     {
		 F+=make_vectorized_array(2.0*projectorKetTimesPsiSpin0TimesVTimesPartOcc[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin0*gradZetaDeltaV[iAtomNonLocal][iPseudoWave];
		 F+=make_vectorized_array(2.0*projectorKetTimesPsiSpin1TimesVTimesPartOcc[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psiSpin1*gradZetaDeltaV[iAtomNonLocal][iPseudoWave];
	     }
	  }
       }

       return F;
    }


    Tensor<1,C_DIM,VectorizedArray<double> >  getFnlPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesVTimesPartOcc,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesVTimesPartOcc,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
							     const std::vector<double> & kPointWeights,
							     const unsigned int numBlockedEigenvectors)
    {
       Tensor<1,C_DIM,VectorizedArray<double> > F;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;
       VectorizedArray<double> two=make_vectorized_array(2.0);
       const unsigned int numKPoints=kPointWeights.size();
       for (unsigned int ik=0; ik<numKPoints; ++ik){
	 for (unsigned int eigenIndex=0; eigenIndex<numBlockedEigenvectors; ++it1Spin0,++it1Spin1, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;
	    const VectorizedArray<double> fnkSpin0=make_vectorized_array(kPointWeights[ik]);
	    const VectorizedArray<double> fnkSpin1=make_vectorized_array(kPointWeights[ik]);
	    for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetaDeltaV.size(); ++iAtomNonLocal)
	    {
		 const unsigned int numberPseudoWaveFunctions = gradZetaDeltaV[iAtomNonLocal].size();
		 for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		 {
		     const VectorizedArray<double> CRealSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     const VectorizedArray<double> CImagSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     const VectorizedArray<double> CRealSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     const VectorizedArray<double> CImagSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
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
									const VectorizedArray<double> & vEffRhoInSpin1,
									const VectorizedArray<double> & vEffRhoOutSpin0,
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

       return (vEffRhoOutSpin0-vEffRhoInSpin0)*gradRhoOutSpin0+(vEffRhoOutSpin1-vEffRhoInSpin1)*gradRhoOutSpin1+(derExchCorrEnergyWithGradRhoOutSpin0-derExchCorrEnergyWithGradRhoInSpin0)*hessianRhoOutSpin0+(derExchCorrEnergyWithGradRhoOutSpin1-derExchCorrEnergyWithGradRhoInSpin1)*hessianRhoOutSpin1;
    }

    Tensor<2,C_DIM,VectorizedArray<double> > getEKStress(std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
	                                         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
						 std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin0Begin,
						 std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin1Begin,
						 const std::vector<double> & kPointCoordinates,
						 const std::vector<double> & kPointWeights,
						 const std::vector<std::vector<double> > & eigenValues_,
		                                 const double fermiEnergy_,
		                                 const double fermiEnergyUp_,
		                                 const double fermiEnergyDown_,
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

       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;
       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator it2Spin0=gradPsiSpin0Begin;
       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator it2Spin1=gradPsiSpin1Begin;
       const unsigned int numEigenValues=eigenValues_[0].size()/2;

       Tensor<1,C_DIM,VectorizedArray<double> > kPointCoord;
       for (unsigned int ik=0; ik<kPointWeights.size(); ++ik){
	 kPointCoord[0]=make_vectorized_array(kPointCoordinates[ik*C_DIM+0]);
	 kPointCoord[1]=make_vectorized_array(kPointCoordinates[ik*C_DIM+1]);
	 kPointCoord[2]=make_vectorized_array(kPointCoordinates[ik*C_DIM+2]);
	 for (unsigned int eigenIndex=0; eigenIndex<numEigenValues; ++it1Spin0,++it1Spin1, ++it2Spin0, ++it2Spin1, ++eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;
	    const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsiSpin0=*it2Spin0;
	    const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsiSpin1=*it2Spin1;

	    double partOccSpin0 =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
								     fermiEnergy_,
								     C_kb,
								     tVal);
	    double partOccSpin1 =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex+numEigenValues],
								     fermiEnergy_,
								     C_kb,
								     tVal);

	    if(dftParameters::constraintMagnetization)
	    {
		 partOccSpin0 = 1.0 , partOccSpin1 = 1.0 ;
		 if (eigenValues_[ik][eigenIndex+numEigenValues]> fermiEnergyDown_)
			partOccSpin1 = 0.0 ;
		 if (eigenValues_[ik][eigenIndex] > fermiEnergyUp_)
			partOccSpin0 = 0.0 ;
	    }

	    VectorizedArray<double> fnkSpin0=make_vectorized_array(partOccSpin0*kPointWeights[ik]);
	    VectorizedArray<double> fnkSpin1=make_vectorized_array(partOccSpin1*kPointWeights[ik]);

	    eshelbyTensor+=fnkSpin0*(psiSpin0[1]*outer_product(kPointCoord,gradPsiSpin0[0])-psiSpin0[0]*outer_product(kPointCoord,gradPsiSpin0[1])
		    -outer_product(kPointCoord,kPointCoord)*(psiSpin0[0]*psiSpin0[0]+psiSpin0[1]*psiSpin0[1]));
	    eshelbyTensor+=fnkSpin1*(psiSpin1[1]*outer_product(kPointCoord,gradPsiSpin1[0])-psiSpin1[0]*outer_product(kPointCoord,gradPsiSpin1[1])
		    -outer_product(kPointCoord,kPointCoord)*(psiSpin1[0]*psiSpin1[0]+psiSpin1[1]*psiSpin1[1]));

	 }
       }

       return eshelbyTensor;
    }

  Tensor<2,C_DIM,VectorizedArray<double> >  getEnlStress(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > & gradZetalmDeltaVlDyadicDistImageAtoms,
							 const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesVTimesPartOcc,
							 const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesVTimesPartOcc,
							 std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
							 std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
							 const std::vector<double> & kPointWeights,
                                                         const unsigned int numBlockedEigenvectors)
 {
       Tensor<2,C_DIM,VectorizedArray<double> > E;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin0=psiSpin0Begin;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1Spin1=psiSpin1Begin;
       VectorizedArray<double> two=make_vectorized_array(2.0);
       const unsigned int numKPoints=kPointWeights.size();
       for (unsigned int ik=0; ik<numKPoints; ++ik){
	 for (unsigned int eigenIndex=0; eigenIndex<numBlockedEigenvectors; ++it1Spin0,++it1Spin1, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin0= *it1Spin0;
	    const Tensor<1,2,VectorizedArray<double> > & psiSpin1= *it1Spin1;
	    const VectorizedArray<double> fnkSpin0=make_vectorized_array(kPointWeights[ik]);
	    const VectorizedArray<double> fnkSpin1=make_vectorized_array(kPointWeights[ik]);
	    for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetalmDeltaVlDyadicDistImageAtoms.size(); ++iAtomNonLocal)
	    {
		 const unsigned int numberPseudoWaveFunctions =gradZetalmDeltaVlDyadicDistImageAtoms[iAtomNonLocal].size();
		 for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		 {
		     const VectorizedArray<double> CRealSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     const VectorizedArray<double> CImagSpin0=make_vectorized_array(projectorKetTimesPsiSpin0TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     const VectorizedArray<double> CRealSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     const VectorizedArray<double> CImagSpin1=make_vectorized_array(projectorKetTimesPsiSpin1TimesVTimesPartOcc[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     const Tensor<2,C_DIM,VectorizedArray<double> >  zdvR=gradZetalmDeltaVlDyadicDistImageAtoms[iAtomNonLocal][iPseudoWave][ik][0];
		     const Tensor<2,C_DIM,VectorizedArray<double> >  zdvI=gradZetalmDeltaVlDyadicDistImageAtoms[iAtomNonLocal][iPseudoWave][ik][1];
		     E+=two*fnkSpin0*((psiSpin0[0]*zdvR+psiSpin0[1]*zdvI)*CRealSpin0-(psiSpin0[0]*zdvI-psiSpin0[1]*zdvR)*CImagSpin0);
		     E+=two*fnkSpin1*((psiSpin1[0]*zdvR+psiSpin1[1]*zdvI)*CRealSpin1-(psiSpin1[0]*zdvI-psiSpin1[1]*zdvR)*CImagSpin1);
		 }
	    }

	 }
       }

       return E;
    }

}

}
