// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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
// @author Sambit Das(2018)
//
#include <force.h>
#include <dft.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::computeStress(
    const dealii::MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorDevice,
#endif
    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
    const dispersionCorrection &                       dispersionCorr,
    const unsigned int                                 eigenDofHandlerIndex,
    const unsigned int                   smearedChargeQuadratureId,
    const unsigned int                   lpspQuadratureIdElectro,
    const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
    const unsigned int                   phiTotDofHandlerIndexElectro,
    const distributedCPUVec<double> &    phiTotRhoOutElectro,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradRhoOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoTotalOutValuesLpsp,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &gradRhoTotalOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                                  pseudoVLocAtomsElectro,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                      hessianRhoCoreAtoms,
    const dealii::AffineConstraints<double> &hangingPlusPBCConstraintsElectro,
    const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro)
  {
    createBinObjectsForce(matrixFreeDataElectro.get_dof_handler(
                            phiTotDofHandlerIndexElectro),
                          d_dofHandlerForceElectro,
                          hangingPlusPBCConstraintsElectro,
                          vselfBinsManagerElectro,
                          d_cellsVselfBallsDofHandlerElectro,
                          d_cellsVselfBallsDofHandlerForceElectro,
                          d_cellsVselfBallsClosestAtomIdDofHandlerElectro,
                          d_AtomIdBinIdLocalDofHandlerElectro,
                          d_cellFacesVselfBallSurfacesDofHandlerElectro,
                          d_cellFacesVselfBallSurfacesDofHandlerForceElectro);

    // reset to zero
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          {
            d_stress[idim][jdim]        = 0.0;
            d_stressKPoints[idim][jdim] = 0.0;
          }
      }

    // configurational stress contribution from all terms except those from
    // nuclear self energy
    computeStressEEshelbyEPSPEnlEk(matrixFreeData,
#ifdef DFTFE_WITH_DEVICE
                                   kohnShamDFTEigenOperatorDevice,
#endif
                                   kohnShamDFTEigenOperator,
                                   eigenDofHandlerIndex,
                                   smearedChargeQuadratureId,
                                   lpspQuadratureIdElectro,
                                   matrixFreeDataElectro,
                                   phiTotDofHandlerIndexElectro,
                                   phiTotRhoOutElectro,
                                   rhoOutValues,
                                   gradRhoOutValues,
                                   rhoTotalOutValuesLpsp,
                                   gradRhoTotalOutValuesLpsp,
                                   pseudoVLocElectro,
                                   pseudoVLocAtomsElectro,
                                   rhoCoreValues,
                                   gradRhoCoreValues,
                                   hessianRhoCoreValues,
                                   gradRhoCoreAtoms,
                                   hessianRhoCoreAtoms,
                                   vselfBinsManagerElectro);

    // configurational stress contribution from nuclear self energy. This is
    // handled separately as it involves
    // a surface integral over the vself ball surface
    if (dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm) ==
        0)
      computeStressEself(matrixFreeDataElectro.get_dof_handler(
                           phiTotDofHandlerIndexElectro),
                         vselfBinsManagerElectro,
                         matrixFreeDataElectro,
                         smearedChargeQuadratureId);

    // Sum all processor contributions and distribute to all processors
    d_stress = dealii::Utilities::MPI::sum(d_stress, mpi_communicator);
    d_stress =
      dealii::Utilities::MPI::sum(d_stress, dftPtr->interBandGroupComm);
    d_stress = dealii::Utilities::MPI::sum(d_stress, dftPtr->interpoolcomm);

    // Sum k point stress contribution over all processors
    // and k point pools and add to total stress
    d_stressKPoints =
      dealii::Utilities::MPI::sum(d_stressKPoints, mpi_communicator);
    d_stressKPoints =
      dealii::Utilities::MPI::sum(d_stressKPoints, dftPtr->interBandGroupComm);
    d_stressKPoints =
      dealii::Utilities::MPI::sum(d_stressKPoints, dftPtr->interpoolcomm);
    d_stress += d_stressKPoints;

    if (d_dftParams.dc_dispersioncorrectiontype != 0)
      {
        for (unsigned int irow = 0; irow < 3; irow++)
          {
            for (unsigned int icol = 0; icol < 3; icol++)
              {
                d_stress[irow][icol] +=
                  dispersionCorr.getStressCorrection(irow, icol);
              }
          }
      }
    // Scale by inverse of domain volume
    d_stress = d_stress * (1.0 / dftPtr->d_domainVolume);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::printStress()
  {
    if (!d_dftParams.reproducible_output)
      {
        pcout << std::endl;
        pcout << "Cell stress (Hartree/Bohr^3)" << std::endl;
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
        for (unsigned int idim = 0; idim < 3; idim++)
          pcout << d_stress[idim][0] << "  " << d_stress[idim][1] << "  "
                << d_stress[idim][2] << std::endl;
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
      }
    else
      {
        pcout << std::endl;
        pcout << "Absolute value of cell stress (Hartree/Bohr^3)" << std::endl;
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            std::vector<double> truncatedStress(3);
            for (unsigned int jdim = 0; jdim < 3; jdim++)
              truncatedStress[jdim] = std::fabs(
                std::floor(10000000 * d_stress[idim][jdim]) / 10000000.0);
            pcout << std::fixed << std::setprecision(6) << truncatedStress[0]
                  << "  " << truncatedStress[1] << "  " << truncatedStress[2]
                  << std::endl;
          }
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
