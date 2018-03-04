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

#ifndef eshelby_H_
#define eshelby_H_
#include "headers.h"
#include "constants.h"
//#include "dft.h"

using namespace dealii;
//
//Declare eshelby namespace functions
//
namespace eshelbyTensorSP
{
  //Eshelby tensor functions
  Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicNoKPoints
                                                                      (const VectorizedArray<double> & phiTot,
		                                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
							               const VectorizedArray<double> & rho,
							               const Tensor<1,C_DIM,VectorizedArray<double> > & gradRho,
							               const VectorizedArray<double> & exc,
							               const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoQuads,
                                                                       const VectorizedArray<double> & pseudoVLoc,
                                                                       const VectorizedArray<double> & phiExt);										       

  Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorPeriodicKPoints
							               (std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiBegin,
                                                                       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiBegin,
								       const std::vector<double> & kPointCoordinates,
                                                                       const std::vector<double> & kPointWeights,
								       const std::vector<std::vector<double> > & eigenValues_,
								       const double fermiEnergy_,
								       const double tVal); 

  Tensor<2,C_DIM,VectorizedArray<double> >  getELocEshelbyTensorNonPeriodic(const VectorizedArray<double> & phiTot,
		                                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
							               const VectorizedArray<double> & rho,
							               const Tensor<1,C_DIM,VectorizedArray<double> > & gradRho,
							               const VectorizedArray<double> & exc,
							               const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoQuads,
                                                                       const VectorizedArray<double> & pseudoVLoc,
                                                                       const VectorizedArray<double> & phiExt,								       
								       std::vector<VectorizedArray<double> >::const_iterator psiBegin,
                                                                       std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator  gradPsiBegin,
								       const std::vector<double> & eigenValues_,
								       const double fermiEnergy_,
								       const double tVal);

Tensor<1,C_DIM,VectorizedArray<double> >  getFPSPLocal(const VectorizedArray<double> rho,
		                                       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPseudoVLoc,
						       const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt);


Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorNonPeriodic(const std::vector<std::vector<VectorizedArray<double> > > & ZetaDeltaV,
								         const std::vector<std::vector<double> >& projectorKetTimesPsiTimesV,
								         std::vector<VectorizedArray<double> >::const_iterator psiBegin,
								         const std::vector<double> & eigenValues_,
								         const double fermiEnergy_,
								         const double tVal);

Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
								      const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesV,
								      std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
                                                                      const std::vector<double> & kPointWeights,								      
								      const std::vector<std::vector<double> > & eigenValues_,
								      const double fermiEnergy_,
								      const double tVal);




Tensor<1,C_DIM,VectorizedArray<double> >  getFnlNonPeriodic(const std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > & gradZetaDeltaV,
						            const std::vector<std::vector<double> > & projectorKetTimesPsiTimesV,
						            std::vector<VectorizedArray<double> >::const_iterator psiBegin,
						            const std::vector<double> & eigenValues_,
						            const double fermiEnergy_,
						            const double tVal);



Tensor<1,C_DIM,VectorizedArray<double> >  getFnlPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
						         const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesV,
						         std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
                                                         const std::vector<double> & kPointWeights,								      
					                 const std::vector<std::vector<double> > & eigenValues_,
						         const double fermiEnergy_,
						         const double tVal);

Tensor<1,C_DIM,VectorizedArray<double> >  getNonSelfConsistentForce(const VectorizedArray<double> & vEffRhoIn,
								    const VectorizedArray<double> & vEffRhoOut,
							            const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoOut,
							            const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoIn,
								    const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoOut,
                                                                    const Tensor<2,C_DIM,VectorizedArray<double> > & hessianRhoOut
								    );


};
#endif
