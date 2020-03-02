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



#ifndef eshelby_H_
#define eshelby_H_
#include "headers.h"
#include "constants.h"

namespace dftfe {

    using namespace dealii;
/**
  * @brief The functions in this namespace contain the expressions for the various terms of the configurational force (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
  * for both periodic (see Eq. 38) and non-periodic (see Eqs. 28-29) case.
  *
  * Basically, the configurational force is the Gateaux derivative
  * of the Kohn-Sham saddle point problem with respect to perturbations of the underlying space due to generic generator (which can be affine perturbation in
  * case of stress computation or an atom centered generator with a compact support for computing the forces). The terms in the configurational force can be
  * grouped into two types: one type can be written as contraction of Eshelby tensors (second order tensor) with the gradient of the Generator. Another type involves
  * contraction of first order tensors with the Generator. The functions in this class provide expressions for the left side of the contraction operation- the second
  * order Eshelby tensors (denoted by E*) and the first order force tensors (denoted by F*).
  * General nomenclature of the input arguments:
  * a) phiTot- total electrostatic potential
  * b) phiExt- sum of electrostatic potential from all nuclear charges
  * c) rho- electron density
  * d) gradRho- gradient of electron density
  * e) exc- exchange correlation energy
  * f) derExcGradRho- derivative of exc with gradient of rho
  * g) psiBegin- begin iterator to vector eigenvectors stored as a flattened array over k points and number of eigenvectors for each k point
  * (periodic case has complex valued eigenvectors which is why Tensor<1,2,VectorizedArray<double> is used in functions for periodic case)
  * h) gradPsiBegin- gradient of eigenvectors
  * i) eigenValues- Kohn sham grounstate eigenvalues stored in a vector. For periodic problems with multiple k points the outer vector should be over k points
  * j) tVal- smearing temperature in K
  * k) pseudoVLoc- local part of the pseudopotential
  * l) gradPseudoVLoc- gradient of local part of pseudopotential
  * m) ZetaDeltaV- nonlocal pseudowavefunctions times deltaV (see Eq. 11 in https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
  * n) gradZetaDeltaV- gradient of ZetaDeltaV
  * o) projectorKetTimesPsiTimesV- nonlocal pseudopotential projector ket times eigenvectors which are precomputed.
  * The nonlocal pseudopotential constants are also multiplied to this quantity. (see Eq. 11 in https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
  *
  * @author Sambit Das
  */
    namespace eshelbyTensor
    {
      /// Eshelby tensor from sum of electrostatic potential from all nuclear charges (only used for testing purpose)
      Tensor<2,C_DIM,VectorizedArray<double> >  getPhiExtEshelbyTensor(const VectorizedArray<double> & phiExt, const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt);

      /// Eshelby tensor corresponding to nuclear self energy (only used for testing purpose)
      Tensor<2,C_DIM,VectorizedArray<double> >  getVselfBallEshelbyTensor(const Tensor<1,C_DIM,VectorizedArray<double> > & gradVself);

      /// Eshelby tensor corresponding to nuclear self energy
      Tensor<2,C_DIM,double >  getVselfBallEshelbyTensor(const Tensor<1,C_DIM,double > & gradVself);


      ///Local part of the Eshelby tensor for periodic case (only considers terms which are summed over k points)
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocWfcEshelbyTensorPeriodicKPoints
	       (std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiBegin,
	       std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiBegin,
	       const std::vector<double> & kPointCoordinates,
	       const std::vector<double> & kPointWeights,
	       const std::vector<std::vector<double> > & eigenValues_,
	       const double fermiEnergy_,
	       const double tVal);

      /// Local part of the Eshelby tensor for non-periodic case
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocWfcEshelbyTensorNonPeriodic
			     (std::vector<VectorizedArray<double> >::const_iterator psiBegin,
			      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiBegin,
			      const std::vector<double> & eigenValues_,
                              const std::vector<double> & partialOccupancies_);

      /// All-electron electrostatic part of the Eshelby tensor
      Tensor<2,C_DIM,VectorizedArray<double> >  getEElectroEshelbyTensor
	                     (const VectorizedArray<double> & phiTot,
			      const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiTot,
			      const VectorizedArray<double> & rho);

      /// exchange-correlation part of the ELoc Eshelby tensor
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocXcEshelbyTensor
			     (const VectorizedArray<double> & rho,
			     const Tensor<1,C_DIM,VectorizedArray<double> > & gradRho,
			     const VectorizedArray<double> & exc,
			     const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRho);


      /// exchange-correlation part of the shadow potential (XL-BOMD) Eshelby tensor
      Tensor<2,C_DIM,VectorizedArray<double> >  getShadowPotentialForceRhoDiffXcEshelbyTensor
			     (const VectorizedArray<double> & shadowKSRhoMinMinusRho,
                              const Tensor<1,C_DIM,VectorizedArray<double> > & shadowKSGradRhoMinMinusGradRho,
			      const Tensor<1,C_DIM,VectorizedArray<double> > & gradRho,
			      const VectorizedArray<double> & vxc,
                              const Tensor<1,C_DIM,VectorizedArray<double> > & derVxcGradRho,
                              const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRho,
                              const Tensor<2,C_DIM,VectorizedArray<double> > & der2ExcGradRho);

      /// psp part of the ELoc Eshelby tensor
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocPspEshelbyTensor
			     (const VectorizedArray<double> & rho,
			      const VectorizedArray<double> & pseudoVLoc,
			      const VectorizedArray<double> & phiExt);

      /// Local pseudopotential force contribution
      Tensor<1,C_DIM,VectorizedArray<double> >  getFPSPLocal(const VectorizedArray<double> rho,
							     const Tensor<1,C_DIM,VectorizedArray<double> > & gradPseudoVLoc,
							     const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt);



      /// Nonlocal pseudopotential Eshelby tensor (for periodic case)
      Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
									  const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesVTimesPartOcc,
									  std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
									  const std::vector<double> & kPointWeights,
									  const std::vector<unsigned int> & nonlocalAtomsCompactSupportList,
									  const unsigned int numBlockedEigenvectors);

      /// Nonlocal pseudopotential force contribution (for periodic case)
      void  getFnlEnlMergedPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
	                            const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
				    const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesVTimesPartOcc,
				    std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
				    const std::vector<double> & kPointWeights,
				    const unsigned int numBlockedEigenvectors,
				    const std::vector<unsigned int> & nonlocalAtomsCompactSupportList,
				    Tensor<1,C_DIM,VectorizedArray<double> > & Fnl,
				    Tensor<2,C_DIM,VectorizedArray<double> > & Enl);

      /// Nonlocal pseudopotential force contribution (for non periodic case)
      void  getFnlEnlMergedNonPeriodic(const std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > & gradZetaDeltaV,
	                               const std::vector<std::vector<VectorizedArray<double> > > & ZetaDeltaV,
				       const std::vector<VectorizedArray<double> > & projectorKetTimesPsiTimesVTimesPartOcc,
				       const std::vector<unsigned int> & nonlocalAtomsCompactSupportList,
                                       const std::vector<unsigned int> & nonlocalPseudoWfcsAccum,
				       Tensor<1,C_DIM,VectorizedArray<double> > & Fnl,
				       Tensor<2,C_DIM,VectorizedArray<double> > & Enl);

      Tensor<1,C_DIM,VectorizedArray<double> >  getFnlNonPeriodic(const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >  & gradZetaDeltaV,
								  const std::vector<VectorizedArray<double> > & projectorKetTimesPsiTimesVTimesPartOcc,
                                                                  const unsigned int startingId);

      /// Nonlocal pseudopotential force contribution (for periodic case)
      Tensor<1,C_DIM,VectorizedArray<double> >  getFnlPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesVTimesPartOcc,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
							     const std::vector<double> & kPointWeights,
							     const unsigned int numBlockedEigenvectors);

      /** Force contribution due to the numerical difference between the input and output electron density (rhoIn and rhoOut)
	* of the final scf iteration. vEff denotes the Kohn-Sham effective potential.
	*/
      Tensor<1,C_DIM,VectorizedArray<double> >  getNonSelfConsistentForce(const VectorizedArray<double> & vEffRhoIn,
									 const VectorizedArray<double> & vEffRhoOut,
									 const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoOut,
									 const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoIn,
									 const Tensor<1,C_DIM,VectorizedArray<double> > & derExchCorrEnergyWithGradRhoOut,
									 const Tensor<2,C_DIM,VectorizedArray<double> > & hessianRhoOut
									);

      /// EK Eshelby tensor (used only for stress computation)
      Tensor<2,C_DIM,VectorizedArray<double> > getEKStress(std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
						   std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiBegin,
						   const std::vector<double> & kPointCoordinates,
						   const std::vector<double> & kPointWeights,
						   const std::vector<std::vector<double> > & eigenValues_,
						   const double fermiEnergy_,
						   const double tVal);

      /// Nonlocal pseudopotential Eshelby tensor (used only for stress computation)
      Tensor<2,C_DIM,VectorizedArray<double> >  getEnlStress(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > & gradZetalmDeltaVlDyadicDistImageAtoms,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiTimesVTimesPartOcc,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiBegin,
							     const std::vector<double> & kPointWeights,
							     const std::vector<unsigned int> & nonlocalAtomsCompactSupportList,
							     const unsigned int numBlockedEigenvectors);

    };

}
#endif
