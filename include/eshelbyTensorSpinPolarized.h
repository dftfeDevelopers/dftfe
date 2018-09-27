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
#ifndef eshelbySP_H_
#define eshelbySP_H_
#include "headers.h"
#include "constants.h"

namespace dftfe {

    using namespace dealii;
    /**
     * @brief The functions in this namespace contain the expressions for the various terms of the configurational force (https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
     * for both periodic and non-periodic case.
     *
     * The functions in this namespace are similar to the ones in eshelbyTensor.h
     * except the ones here are specialized
     * for spin polarized case. Spin0 and Spin1 refer to up and down spins respectively.
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
     * k) pseudoVLoc- local part of the pseuodopotential
     * l) gradPseudoVLoc- gradient of local part of pseudopotential
     * m) ZetaDeltaV- nonlocal pseudowavefunctions times deltaV (see Eq. 11 in https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
     * n) gradZetaDeltaV- gradient of ZetaDeltaV
     * o) projectorKetTimesPsiTimesV- nonlocal pseudopotential projector ket times eigenvectors
     * which are precomputed. The nonlocal pseudopotential constants are also multiplied to this quantity.
     * (see Eq. 11 in https://link.aps.org/doi/10.1103/PhysRevB.97.165132)
     *
     * @author Sambit Das
     */
    namespace eshelbyTensorSP
    {
      ///Local part of the Eshelby tensor for periodic case (only considers terms which are summed over k points)
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocWfcEshelbyTensorPeriodicKPoints
		    (std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
		     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,
		     std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin0Begin,
		     std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin1Begin,
		     const std::vector<double> & kPointCoordinates,
		     const std::vector<double> & kPointWeights,
		     const std::vector<std::vector<double> > & eigenValues_,
		     const double fermiEnergy_,
		     const double tVal);

      ///Local part of the Eshelby tensor for non-periodic case
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocWfcEshelbyTensorNonPeriodic
			(std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
			std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
			std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin0Begin,
			std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >::const_iterator gradPsiSpin1Begin,
			const std::vector<double> & eigenValues_,
			const double fermiEnergy_,
			const double tVal);

      /// exchange-correlation and psp part of the ELoc Eshelby tensor
      Tensor<2,C_DIM,VectorizedArray<double> >  getELocXcPspEshelbyTensor
			     (const VectorizedArray<double> & rho,
			      const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin0,
			      const Tensor<1,C_DIM,VectorizedArray<double> > & gradRhoSpin1,
			      const VectorizedArray<double> & exc,
			      const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin0,
			      const Tensor<1,C_DIM,VectorizedArray<double> > & derExcGradRhoSpin1,
			     const VectorizedArray<double> & pseudoVLoc);

      ///Local pseudotential force contribution
      Tensor<1,C_DIM,VectorizedArray<double> >  getFPSPLocal(const VectorizedArray<double> rho,
							   const Tensor<1,C_DIM,VectorizedArray<double> > & gradPseudoVLoc,
							   const Tensor<1,C_DIM,VectorizedArray<double> > & gradPhiExt);

      ///Nonlocal pseudotential Eshelby tensor (for non-periodic case)
      Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorNonPeriodic(const std::vector<std::vector<VectorizedArray<double> > > & ZetaDeltaV,
									     const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesV,
									     const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesV,
									     std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
									     std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
									     const std::vector<double> & eigenValues_,
									     const double fermiEnergy_,
									     const double tVal);

      ///Nonlocal pseudotential Eshelby tensor (for periodic case)
      Tensor<2,C_DIM,VectorizedArray<double> >  getEnlEshelbyTensorPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > & ZetaDeltaV,
									  const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
									  const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,
									  std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin0Begin,
									  std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator psiSpin1Begin,
									  const std::vector<double> & kPointWeights,
									  const std::vector<std::vector<double> > & eigenValues_,
									  const double fermiEnergy_,
									  const double tVal);

      ///Nonlocal pseudotential force contribution (for non periodic case)
      Tensor<1,C_DIM,VectorizedArray<double> >  getFnlNonPeriodic(const std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > & gradZetaDeltaV,
								const std::vector<std::vector<double> > & projectorKetTimesPsiSpin0TimesV,
								const std::vector<std::vector<double> > & projectorKetTimesPsiSpin1TimesV,
								std::vector<VectorizedArray<double> >::const_iterator psiSpin0Begin,
								std::vector<VectorizedArray<double> >::const_iterator psiSpin1Begin,
								const std::vector<double> & eigenValues_,
								const double fermiEnergy_,
								const double tVal);
      ///Nonlocal pseudotential force contribution (for periodic case)
      Tensor<1,C_DIM,VectorizedArray<double> >  getFnlPeriodic(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > & gradZetaDeltaV,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
							     const std::vector<double> & kPointWeights,
							     const std::vector<std::vector<double> > & eigenValues_,
							     const double fermiEnergy_,
							     const double tVal);

      ///Force contribution due to the numerical difference between the input and output electron density (rhoIn and rhoOut)
      ///of the final scf iteration. vEff denotes the Kohn-Sham effective potential.
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

      /// EK Eshelby tensor (used only for stress computation)
      Tensor<2,C_DIM,VectorizedArray<double> > getEKStress(std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
						   std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
						   std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin0Begin,
						   std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > >::const_iterator gradPsiSpin1Begin,
						   const std::vector<double> & kPointCoordinates,
						   const std::vector<double> & kPointWeights,
						   const std::vector<std::vector<double> > & eigenValues_,
						   const double fermiEnergy_,
						   const double tVal);

      /// Nonlocal pseudotential Eshelby tensor (used only for stress computation)
      Tensor<2,C_DIM,VectorizedArray<double> >  getEnlStress(const std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > & gradZetalmDeltaVlDyadicDistImageAtoms,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin0TimesV,
							     const std::vector<std::vector<std::vector<std::complex<double> > > >& projectorKetTimesPsiSpin1TimesV,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin0Begin,
							     std::vector<Tensor<1,2,VectorizedArray<double> > >::const_iterator  psiSpin1Begin,
							     const std::vector<double> & kPointWeights,
							     const std::vector<std::vector<double> > & eigenValues_,
							     const double fermiEnergy_,
							     const double tVal);
    };

}
#endif
