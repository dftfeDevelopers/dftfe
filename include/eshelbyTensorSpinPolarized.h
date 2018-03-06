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

#ifndef eshelbySP_H_
#define eshelbySP_H_
#include "headers.h"
#include "constants.h"

using namespace dealii;
/**
 * The functions in this namespace contain the expressions for the various terms of the configurational force (https://arxiv.org/abs/1712.05535)
 * for both periodic and non-periodic case. The nature of the terms are similar to the ones in eshelbyTensor.h except the ones here are specialized
 * for spin polarized case. Refer to eshelbyTensor.h for information about the individual functions. Spin0 and Spin1 refer to up and down spins respectively.
 *  
 */
namespace eshelbyTensorSP
{
  Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicNoKPoints
					      (const VectorizedArray<double> & phiTot,
					       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
					       const VectorizedArray<double> & rho,
					       const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin0,
					       const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin1,
					       const VectorizedArray<double> & exc,
					       const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin0,
					       const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin1,
					       const VectorizedArray<double> & pseudoVLoc,
					       const VectorizedArray<double> & phiExt);									       

  Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicKPoints
		(std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
                 std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,
                 std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin0Begin,		 
                 std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin1Begin,
                 const std::vector<double> & kPointCoordinates,
                 const std::vector<double> & kPointWeights,							
	         const std::vector<std::vector<double> > & eigenValues_,
	         const double fermiEnergy_,
	         const double tVal);

  Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorNonPeriodic
                   (const VectorizedArray<double> & phiTot,
		    const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
		    const VectorizedArray<double> & rho,
		    const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin0,
                    const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin1,	
		    const VectorizedArray<double> & exc,
		    const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin0,
                    const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin1,			  
                    const VectorizedArray<double> & pseudoVLoc,
                    const VectorizedArray<double> & phiExt,
		    std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
                    std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin, 
		    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin0Begin,
		    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin1Begin,
		    const std::vector<double> & eigenValues_,
		    const double fermiEnergy_,
		    const double tVal);

  Tensor<1,C_DIM,VectorizedArray<double> >  getFPSPLocal(const VectorizedArray<double> rho,
		                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPseudoVLoc,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt);


  Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorNonPeriodic(const std::vector<std::vector<VectorizedArray<double> > > & ZetaDeltaV,
								         const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesV,
                                                                         const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesV,									 
								         std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
                                                                         std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,									 
								         const std::vector<double> & eigenValues_,
								         const double fermiEnergy_,
								         const double tVal);     

  Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
								      const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
                                                                      const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,
								      std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
                                                                      std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,								      
                                                                      const std::vector<double> & kPointWeights,								      
								      const std::vector<std::vector<double> > & eigenValues_,
								      const double fermiEnergy_,
								      const double tVal);


  Tensor<1,C_DIM,VectorizedArray<double> >  getFnlNonPeriodic(const std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > & gradZetaDeltaV,
						            const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesV,
                                                            const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesV,    
						            std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
                                                            std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
						            const std::vector<double> & eigenValues_,
						            const double fermiEnergy_,
						            const double tVal);



  Tensor<1,C_DIM,VectorizedArray<double> >  getFnlPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
						         const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
                                                         const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,							 
						         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
                                                         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
                                                         const std::vector<double> & kPointWeights,								      
					                 const std::vector<std::vector<double> > & eigenValues_,
						         const double fermiEnergy_,
						         const double tVal);

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
                                                                    const Tensor<2,C_DIM,VectorizedArray<double> > & hessianRhoOutSpin1);
};
#endif
