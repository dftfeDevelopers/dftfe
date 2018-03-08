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
#include "../../../include/eshelbyTensor.h"
#include "../../../include/dftUtils.h"

namespace eshelbyTensor
{
    Tensor<2,C_DIM,VectorizedArray<double> >  getPhiExtEshelbyTensor(const VectorizedArray<double> & phiExt,const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt)
    {

       Tensor<2,C_DIM,VectorizedArray<double> > identityTensor;
       identityTensor[0][0]=make_vectorized_array (1.0);
       identityTensor[1][1]=make_vectorized_array (1.0);
       identityTensor[2][2]=make_vectorized_array (1.0);



       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= make_vectorized_array(1.0/(4.0*M_PI))*outer_product(gradPhiExt,gradPhiExt)-make_vectorized_array(1.0/(8.0*M_PI))*scalar_product(gradPhiExt,gradPhiExt)*identityTensor;

      return eshelbyTensor;
    }

    Tensor<2,C_DIM,VectorizedArray<double> >  getVselfBallEshelbyTensor(const Tensor<1,C_DIM,VectorizedArray<double> > & gradVself)
    {

       Tensor<2,C_DIM,VectorizedArray<double> > identityTensor;
       identityTensor[0][0]=make_vectorized_array (1.0);
       identityTensor[1][1]=make_vectorized_array (1.0);
       identityTensor[2][2]=make_vectorized_array (1.0);



       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= make_vectorized_array(1.0/(8.0*M_PI))*scalar_product(gradVself,gradVself)*identityTensor-make_vectorized_array(1.0/(4.0*M_PI))*outer_product(gradVself,gradVself);

      return eshelbyTensor;
    }


    Tensor<2,C_DIM,double >  getVselfBallEshelbyTensor(const Tensor<1,C_DIM,double> & gradVself)
    {

       double identityTensorFactor=1.0/(8.0*M_PI)*scalar_product(gradVself,gradVself);
       Tensor<2,C_DIM,double > eshelbyTensor= -1.0/(4.0*M_PI)*outer_product(gradVself,gradVself);

       eshelbyTensor[0][0]+=identityTensorFactor;
       eshelbyTensor[1][1]+=identityTensorFactor;
       eshelbyTensor[2][2]+=identityTensorFactor;
      
       return eshelbyTensor;
    }

    Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicNoKPoints
									  (const VectorizedArray<double> & phiTot,
									   const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
									   const VectorizedArray<double> & rho,
									   const Tensor<1,C_DIM,VectorizedArray<double> > & gradRho,
									   const VectorizedArray<double> & exc,
									   const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRho,
									   const VectorizedArray<double> & pseudoVLoc,
									   const VectorizedArray<double> & phiExt)
    {
       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= make_vectorized_array(1.0/(4.0*M_PI))*outer_product(gradPhiTot,gradPhiTot)-outer_product(derExcGradRho,gradRho);
       VectorizedArray<double> identityTensorFactor=make_vectorized_array(-1.0/(8.0*M_PI))*scalar_product(gradPhiTot,gradPhiTot)+rho*phiTot+exc*rho + (pseudoVLoc-phiExt)*rho;
       eshelbyTensor[0][0]+=identityTensorFactor;
       eshelbyTensor[1][1]+=identityTensorFactor;
       eshelbyTensor[2][2]+=identityTensorFactor;
       return eshelbyTensor;   
    }

    Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicKPoints
									   (std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiBegin,
									   std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiBegin,
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

       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1=psiBegin;
       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator it2=gradPsiBegin;

       Tensor<1,C_DIM,VectorizedArray<double> > kPointCoord;
       for (unsigned int ik=0; ik<eigenValues_.size(); ++ik){
	 kPointCoord[0]=make_vectorized_array(kPointCoordinates[ik*C_DIM+0]);
	 kPointCoord[1]=make_vectorized_array(kPointCoordinates[ik*C_DIM+1]);
	 kPointCoord[2]=make_vectorized_array(kPointCoordinates[ik*C_DIM+2]);
	 for (unsigned int eigenIndex=0; eigenIndex<eigenValues_[0].size(); ++it1, ++it2, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psi= *it1;
	    const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsi=*it2;
	    const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
								fermiEnergy_,
								C_kb,
								tVal);  
	    VectorizedArray<double> identityTensorFactorContribution=make_vectorized_array(0.0);
	    VectorizedArray<double> fnk=make_vectorized_array(partOcc*kPointWeights[ik]);
	    identityTensorFactorContribution+=(scalar_product(gradPsi[0],gradPsi[0])+scalar_product(gradPsi[1],gradPsi[1]));
	    identityTensorFactorContribution+=make_vectorized_array(2.0)*(psi[0]*scalar_product(kPointCoord,gradPsi[1])-psi[1]*scalar_product(kPointCoord,gradPsi[0]));
	    identityTensorFactorContribution+=(scalar_product(kPointCoord,kPointCoord)-make_vectorized_array(2.0*eigenValues_[ik][eigenIndex]))*(psi[0]*psi[0]+psi[1]*psi[1]);
	    identityTensorFactorContribution*=fnk;
	    identityTensorFactor+=identityTensorFactorContribution;

	    eshelbyTensor-=make_vectorized_array(2.0)*fnk*(outer_product(gradPsi[0],gradPsi[0])+outer_product(gradPsi[1],gradPsi[1])+psi[0]*outer_product(gradPsi[1],kPointCoord)-psi[1]*outer_product(gradPsi[0],kPointCoord));
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
									      const Tensor<1,C_DIM,VectorizedArray<double> > & gradRho,
									      const VectorizedArray<double> & exc,
									      const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRho,
									      const VectorizedArray<double> & pseudoVLoc,
									      const VectorizedArray<double> & phiExt,
									      std::vector<VectorizedArray<double> >::const_iterator psiBegin,
									      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiBegin,
									      const std::vector<double> & eigenValues_,
									      const double fermiEnergy_,
									      const double tVal)
    {

       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor= make_vectorized_array(1.0/(4.0*M_PI))*outer_product(gradPhiTot,gradPhiTot)-outer_product(derExcGradRho,gradRho);
       VectorizedArray<double> identityTensorFactor=make_vectorized_array(-1.0/(8.0*M_PI))*scalar_product(gradPhiTot,gradPhiTot)+rho*phiTot+exc*rho + (pseudoVLoc-phiExt)*rho;

       std::vector<VectorizedArray<double> >::const_iterator it1=psiBegin;   
       std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator it2=gradPsiBegin;
       for (unsigned int eigenIndex=0; eigenIndex < eigenValues_.size(); ++it1, ++it2, ++ eigenIndex){
	  const VectorizedArray<double> & psi= *it1;
	  const Tensor<1,C_DIM,VectorizedArray<double> > & gradPsi=*it2;
	  const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[eigenIndex],
							      fermiEnergy_,
							      C_kb,
							      tVal);       
	  identityTensorFactor+=make_vectorized_array(partOcc)*scalar_product(gradPsi,gradPsi)-make_vectorized_array(2*partOcc*eigenValues_[eigenIndex])*psi*psi;
	  eshelbyTensor-=make_vectorized_array(2.0*partOcc)*outer_product(gradPsi,gradPsi);
       }
       
       eshelbyTensor[0][0]+=identityTensorFactor;
       eshelbyTensor[1][1]+=identityTensorFactor;
       eshelbyTensor[2][2]+=identityTensorFactor;
       return eshelbyTensor;
    }


    Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorNonPeriodic(const std::vector<std::vector<VectorizedArray<double> > > & ZetaDeltaV,
									     const std::vector<std::vector<double> > & projectorKetTimesPsiTimesV,
									     std::vector<VectorizedArray<double> >::const_iterator psiBegin,
									     const std::vector<double> & eigenValues_,
									     const double fermiEnergy_,
									     const double tVal)
    {

       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
       VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);
       std::vector<VectorizedArray<double> >::const_iterator it1=psiBegin;   
       for (unsigned int eigenIndex=0; eigenIndex < eigenValues_.size(); ++it1, ++ eigenIndex)
       {
	  const VectorizedArray<double> & psi= *it1;
	  const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[eigenIndex],
							      fermiEnergy_,
							      C_kb,
							      tVal);       
	  for (unsigned int iAtomNonLocal=0; iAtomNonLocal < ZetaDeltaV.size(); ++iAtomNonLocal)
	  {
	     const int numberPseudoWaveFunctions = ZetaDeltaV[iAtomNonLocal].size();
	     for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	     {	  
		 identityTensorFactor+=make_vectorized_array(4.0*partOcc*projectorKetTimesPsiTimesV[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psi*ZetaDeltaV[iAtomNonLocal][iPseudoWave];
	     }
	  }
       }  
       eshelbyTensor[0][0]=identityTensorFactor;
       eshelbyTensor[1][1]=identityTensorFactor;
       eshelbyTensor[2][2]=identityTensorFactor;
       
       return eshelbyTensor;
    }

    Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
									  const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesV,
									  std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiBegin,
									  const std::vector<double> & kPointWeights,								      
									  const std::vector<std::vector<double> > & eigenValues_,
									  const double fermiEnergy_,
									  const double tVal)
    {
       Tensor<2,C_DIM,VectorizedArray<double> > eshelbyTensor;
       VectorizedArray<double> identityTensorFactor=make_vectorized_array(0.0);
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1=psiBegin;
       VectorizedArray<double> four=make_vectorized_array(4.0);
       const int numKPoints=eigenValues_.size();
       for (unsigned int ik=0; ik<numKPoints; ++ik){
	 for (unsigned int eigenIndex=0; eigenIndex<eigenValues_[0].size(); ++it1, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psi= *it1;
	    const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
							      fermiEnergy_,
							      C_kb,
							      tVal); 
	    VectorizedArray<double> fnk=make_vectorized_array(partOcc*kPointWeights[ik]);
	    for (unsigned int iAtomNonLocal=0; iAtomNonLocal < ZetaDeltaV.size(); ++iAtomNonLocal)
	    {
		 const int numberPseudoWaveFunctions = ZetaDeltaV[iAtomNonLocal].size();
		 for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		 {	  
		     VectorizedArray<double> CReal=make_vectorized_array(projectorKetTimesPsiTimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     VectorizedArray<double> CImag=make_vectorized_array(projectorKetTimesPsiTimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     VectorizedArray<double> zdvR=ZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
		     VectorizedArray<double> zdvI=ZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];
		     identityTensorFactor+=four*fnk*((psi[0]*zdvR+psi[1]*zdvI)*CReal-(psi[0]*zdvI-psi[1]*zdvR)*CImag);
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
								const std::vector<std::vector<double> > & projectorKetTimesPsiTimesV,
								std::vector<VectorizedArray<double> >::const_iterator psiBegin,
								const std::vector<double> & eigenValues_,
								const double fermiEnergy_,
								const double tVal)
    {

       Tensor<1,C_DIM,VectorizedArray<double> > F; 
       std::vector<VectorizedArray<double> >::const_iterator it1=psiBegin;   
       for (unsigned int eigenIndex=0; eigenIndex < eigenValues_.size(); ++it1, ++ eigenIndex)
       {
	  const VectorizedArray<double> & psi= *it1;
	  const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[eigenIndex],
							      fermiEnergy_,
							      C_kb,
							      tVal);        
	  for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetaDeltaV.size(); ++iAtomNonLocal)
	  {
	     const int numberPseudoWaveFunctions = gradZetaDeltaV[iAtomNonLocal].size();
	     for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	     {	
		 F+=make_vectorized_array(4.0*partOcc*projectorKetTimesPsiTimesV[iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave])*psi*gradZetaDeltaV[iAtomNonLocal][iPseudoWave];
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
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesV,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
							     const std::vector<double> & kPointWeights,								      
							     const std::vector<std::vector<double> > & eigenValues_,
							     const double fermiEnergy_,
							     const double tVal)
    {
       Tensor<1,C_DIM,VectorizedArray<double> > F;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1=psiBegin;
       VectorizedArray<double> four=make_vectorized_array(4.0);
       const int numKPoints=eigenValues_.size();
       for (unsigned int ik=0; ik<numKPoints; ++ik){
	 for (unsigned int eigenIndex=0; eigenIndex<eigenValues_[0].size(); ++it1, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psi= *it1;
	    const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
							      fermiEnergy_,
							      C_kb,
							      tVal); 
	    VectorizedArray<double> fnk=make_vectorized_array(partOcc*kPointWeights[ik]);
	    for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetaDeltaV.size(); ++iAtomNonLocal)
	    {
		 const int numberPseudoWaveFunctions = gradZetaDeltaV[iAtomNonLocal].size();
		 for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		 {	  
		     VectorizedArray<double> CReal=make_vectorized_array(projectorKetTimesPsiTimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     VectorizedArray<double> CImag=make_vectorized_array(projectorKetTimesPsiTimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     Tensor<1,C_DIM,VectorizedArray<double> >  zdvR=gradZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][0];
		     Tensor<1,C_DIM,VectorizedArray<double> >  zdvI=gradZetaDeltaV[iAtomNonLocal][iPseudoWave][ik][1];
		     F+=four*fnk*((psi[0]*zdvR+psi[1]*zdvI)*CReal-(psi[0]*zdvI-psi[1]*zdvR)*CImag);
		 }
	    }	

	 }
       }

       return F;    
    }

    Tensor<1,C_DIM,VectorizedArray<double> >  getNonSelfConsistentForce(const VectorizedArray<double> & vEffRhoIn,
									const VectorizedArray<double> & vEffRhoOut,
									const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoOut,
									const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoIn,
									const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoOut,
									const Tensor<2,C_DIM,VectorizedArray<double> > & hessianRhoOut
									)
    {
       return (vEffRhoOut-vEffRhoIn)*gradRhoOut+(derExchCorrEnergyWithGradRhoOut-derExchCorrEnergyWithGradRhoIn)*hessianRhoOut; 
    }


    Tensor<2,C_DIM,VectorizedArray<double> > getEKStress(std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
						 std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiBegin,
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

       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1=psiBegin;
       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator it2=gradPsiBegin;

       Tensor<1,C_DIM,VectorizedArray<double> > kPointCoord;
       for (unsigned int ik=0; ik<eigenValues_.size(); ++ik){
	 kPointCoord[0]=make_vectorized_array(kPointCoordinates[ik*C_DIM+0]);
	 kPointCoord[1]=make_vectorized_array(kPointCoordinates[ik*C_DIM+1]);
	 kPointCoord[2]=make_vectorized_array(kPointCoordinates[ik*C_DIM+2]);
	 for (unsigned int eigenIndex=0; eigenIndex<eigenValues_[0].size(); ++it1, ++it2, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psi= *it1;
	    const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsi=*it2;
	    const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
								fermiEnergy_,
								C_kb,
								tVal);  
	    VectorizedArray<double> fnk=make_vectorized_array(2.0*partOcc*kPointWeights[ik]);
	    eshelbyTensor+=fnk*(psi[1]*outer_product(kPointCoord,gradPsi[0])-psi[0]*outer_product(kPointCoord,gradPsi[1])-outer_product(kPointCoord,kPointCoord)*(psi[0]*psi[0]+psi[1]*psi[1]));
	 }
       }

       return eshelbyTensor;	
    }

  Tensor<2,C_DIM,VectorizedArray<double> >  getEnlStress(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > & gradZetalmDeltaVlDyadicDistImageAtoms,
						         const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesV,
						         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
                                                         const std::vector<double> & kPointWeights,
					                 const std::vector<std::vector<double> > & eigenValues_,
						         const double fermiEnergy_,
						         const double tVal)
  {
       Tensor<2,C_DIM,VectorizedArray<double> > E;
       std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator it1=psiBegin;
       VectorizedArray<double> four=make_vectorized_array(4.0);
       const int numKPoints=eigenValues_.size();
       for (unsigned int ik=0; ik<numKPoints; ++ik){
	 for (unsigned int eigenIndex=0; eigenIndex<eigenValues_[0].size(); ++it1, ++ eigenIndex){
	    const Tensor<1,2,VectorizedArray<double> > & psi= *it1;
	    const double partOcc =dftUtils::getPartialOccupancy(eigenValues_[ik][eigenIndex],
							      fermiEnergy_,
							      C_kb,
							      tVal); 
	    VectorizedArray<double> fnk=make_vectorized_array(partOcc*kPointWeights[ik]);
	    for (unsigned int iAtomNonLocal=0; iAtomNonLocal < gradZetalmDeltaVlDyadicDistImageAtoms.size(); ++iAtomNonLocal)
	    {
		 const int numberPseudoWaveFunctions = gradZetalmDeltaVlDyadicDistImageAtoms[iAtomNonLocal].size();
		 for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		 {	  
		     VectorizedArray<double> CReal=make_vectorized_array(projectorKetTimesPsiTimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].real());
		     VectorizedArray<double> CImag=make_vectorized_array(projectorKetTimesPsiTimesV[ik][iAtomNonLocal][numberPseudoWaveFunctions*eigenIndex + iPseudoWave].imag());
		     Tensor<2,C_DIM,VectorizedArray<double> >  zdvR=gradZetalmDeltaVlDyadicDistImageAtoms[iAtomNonLocal][iPseudoWave][ik][0];
		     Tensor<2,C_DIM,VectorizedArray<double> >  zdvI=gradZetalmDeltaVlDyadicDistImageAtoms[iAtomNonLocal][iPseudoWave][ik][1];
		     E+=four*fnk*((psi[0]*zdvR+psi[1]*zdvI)*CReal-(psi[0]*zdvI-psi[1]*zdvR)*CImag);
		 }
	    }	

	 }
       }

       return E;          
  }
}
