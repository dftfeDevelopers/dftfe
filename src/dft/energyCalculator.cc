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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das, Krishnendu Ghosh
//


// source file for energy computations
#include <constants.h>
#include <dftUtils.h>
#include <energyCalculator.h>

namespace dftfe
{
  namespace internalEnergy
  {
    template <typename T>
    double
    computeFieldTimesDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
        &                                                  basisOperationsPtr,
      const unsigned int                                   quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValues)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (unsigned int iCell = 0; iCell < basisOperationsPtr->nCells();
           ++iCell)
        {
          const std::vector<double> &cellFieldValues =
            fieldValues.find(basisOperationsPtr->cellID(iCell))->second;
          for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            result +=
              cellFieldValues[iQuad] *
              densityQuadValues[iCell * nQuadsPerCell + iQuad] *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
        }
      return result;
    }
    template <typename T>
    double
    computeFieldTimesDensityResidual(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
        &                                                  basisOperationsPtr,
      const unsigned int                                   quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesIn,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesOut)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (unsigned int iCell = 0; iCell < basisOperationsPtr->nCells();
           ++iCell)
        {
          const std::vector<double> &cellFieldValues =
            fieldValues.find(basisOperationsPtr->cellID(iCell))->second;
          for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            result +=
              cellFieldValues[iQuad] *
              (densityQuadValuesOut[iCell * nQuadsPerCell + iQuad] -
               densityQuadValuesIn[iCell * nQuadsPerCell + iQuad]) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
        }
      return result;
    }
    template <typename T>
    double
    computeFieldTimesDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValues)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (unsigned int iCell = 0; iCell < basisOperationsPtr->nCells();
           ++iCell)
        {
          for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            result +=
              fieldValues[iCell * nQuadsPerCell + iQuad] *
              densityQuadValues[iCell * nQuadsPerCell + iQuad] *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
        }
      return result;
    }
    template <typename T>
    double
    computeFieldTimesDensityResidual(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesIn,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesOut)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (unsigned int iCell = 0; iCell < basisOperationsPtr->nCells();
           ++iCell)
        {
          for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            result +=
              fieldValues[iCell * nQuadsPerCell + iQuad] *
              (densityQuadValuesOut[iCell * nQuadsPerCell + iQuad] -
               densityQuadValuesIn[iCell * nQuadsPerCell + iQuad]) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
        }
      return result;
    }
    void
    printEnergy(const double                      bandEnergy,
                const double                      totalkineticEnergy,
                const double                      totalexchangeEnergy,
                const double                      totalcorrelationEnergy,
                const double                      totalElectrostaticEnergy,
                const double                      dispersionEnergy,
                const double                      totalEnergy,
                const unsigned int                numberAtoms,
                const dealii::ConditionalOStream &pcout,
                const bool                        reproducibleOutput,
                const bool                        isPseudo,
                const unsigned int                verbosity,
                const dftParameters &             dftParams)
    {
      if (reproducibleOutput)
        {
          const double bandEnergyTrunc =
            std::floor(1000000000 * (bandEnergy)) / 1000000000.0;
          const double totalkineticEnergyTrunc =
            std::floor(1000000000 * (totalkineticEnergy)) / 1000000000.0;
          const double totalexchangeEnergyTrunc =
            std::floor(1000000000 * (totalexchangeEnergy)) / 1000000000.0;
          const double totalcorrelationEnergyTrunc =
            std::floor(1000000000 * (totalcorrelationEnergy)) / 1000000000.0;
          const double totalElectrostaticEnergyTrunc =
            std::floor(1000000000 * (totalElectrostaticEnergy)) / 1000000000.0;
          const double totalEnergyTrunc =
            std::floor(1000000000 * (totalEnergy)) / 1000000000.0;
          const double totalEnergyPerAtomTrunc =
            std::floor(1000000000 * (totalEnergy / numberAtoms)) / 1000000000.0;

          pcout << std::endl << "Energy computations (Hartree) " << std::endl;
          pcout << "-------------------" << std::endl;
          if (dftParams.useMixedPrecCGS_O || dftParams.useMixedPrecCGS_SR ||
              dftParams.useSinglePrecCommunCheby)
            pcout << std::setw(25) << "Total energy"
                  << ": " << std::fixed << std::setprecision(6) << std::setw(20)
                  << totalEnergyTrunc << std::endl;
          else
            pcout << std::setw(25) << "Total energy"
                  << ": " << std::fixed << std::setprecision(8) << std::setw(20)
                  << totalEnergyTrunc << std::endl;
        }
      else
        {
          pcout << std::endl;
          char bufferEnergy[200];
          pcout << "Energy computations (Hartree)\n";
          pcout
            << "-------------------------------------------------------------------------------\n";
          sprintf(bufferEnergy, "%-52s:%25.16e\n", "Band energy", bandEnergy);
          pcout << bufferEnergy;
          if (verbosity >= 2)
            {
              if (isPseudo)
                sprintf(bufferEnergy,
                        "%-52s:%25.16e\n",
                        "Kinetic energy plus nonlocal PSP energy",
                        totalkineticEnergy);
              else
                sprintf(bufferEnergy,
                        "%-52s:%25.16e\n",
                        "Kinetic energy",
                        totalkineticEnergy);
              pcout << bufferEnergy;
            }

          sprintf(bufferEnergy,
                  "%-52s:%25.16e\n",
                  "Exchange energy",
                  totalexchangeEnergy);
          pcout << bufferEnergy;
          sprintf(bufferEnergy,
                  "%-52s:%25.16e\n",
                  "Correlation energy",
                  totalcorrelationEnergy);
          pcout << bufferEnergy;
          if (verbosity >= 2)
            {
              if (isPseudo)
                sprintf(bufferEnergy,
                        "%-52s:%25.16e\n",
                        "Local PSP Electrostatic energy",
                        totalElectrostaticEnergy);
              else
                sprintf(bufferEnergy,
                        "%-52s:%25.16e\n",
                        "Electrostatic energy",
                        totalElectrostaticEnergy);
              pcout << bufferEnergy;
            }

          if (dftParams.dc_dispersioncorrectiontype != 0)
            {
              sprintf(bufferEnergy,
                      "%-52s:%25.16e\n",
                      "Dispersion energy",
                      dispersionEnergy);
              pcout << bufferEnergy;
            }
          sprintf(bufferEnergy,
                  "%-52s:%25.16e\n",
                  "Total internal energy",
                  totalEnergy);
          pcout << bufferEnergy;
          sprintf(bufferEnergy,
                  "%-52s:%25.16e\n",
                  "Total internal energy per atom",
                  totalEnergy / numberAtoms);
          pcout << bufferEnergy;
          pcout
            << "-------------------------------------------------------------------------------\n";
        }
    }

    double
    localBandEnergy(const std::vector<std::vector<double>> &eigenValues,
                    const std::vector<double> &             kPointWeights,
                    const double                            fermiEnergy,
                    const double                            fermiEnergyUp,
                    const double                            fermiEnergyDown,
                    const double                            TVal,
                    const unsigned int                      spinPolarized,
                    const dealii::ConditionalOStream &      scout,
                    const MPI_Comm &                        interpoolcomm,
                    const unsigned int                      lowerBoundKindex,
                    const unsigned int                      verbosity,
                    const dftParameters &                   dftParams)
    {
      double       bandEnergyLocal = 0.0;
      unsigned int numEigenValues = eigenValues[0].size() / (1 + spinPolarized);
      //
      for (unsigned int ipool = 0;
           ipool < dealii::Utilities::MPI::n_mpi_processes(interpoolcomm);
           ++ipool)
        {
          MPI_Barrier(interpoolcomm);
          if (ipool == dealii::Utilities::MPI::this_mpi_process(interpoolcomm))
            {
              for (unsigned int kPoint = 0; kPoint < kPointWeights.size();
                   ++kPoint)
                {
                  if (verbosity > 1)
                    {
                      scout
                        << " Printing KS eigen values (spin split if this is a spin polarized calculation ) and fractional occupancies for kPoint "
                        << (lowerBoundKindex + kPoint) << std::endl;
                      scout << "  " << std::endl;
                    }
                  for (unsigned int i = 0; i < numEigenValues; i++)
                    {
                      if (spinPolarized == 0)
                        {
                          const double partialOccupancy =
                            dftUtils::getPartialOccupancy(
                              eigenValues[kPoint][i], fermiEnergy, C_kb, TVal);
                          bandEnergyLocal += 2.0 * partialOccupancy *
                                             kPointWeights[kPoint] *
                                             eigenValues[kPoint][i];
                          //

                          if (verbosity > 1)
                            scout << i << " : " << eigenValues[kPoint][i]
                                  << "       " << partialOccupancy << std::endl;
                          //
                        }
                      if (spinPolarized == 1)
                        {
                          double partialOccupancy =
                            dftUtils::getPartialOccupancy(
                              eigenValues[kPoint][i], fermiEnergy, C_kb, TVal);
                          double partialOccupancy2 =
                            dftUtils::getPartialOccupancy(
                              eigenValues[kPoint][i + numEigenValues],
                              fermiEnergy,
                              C_kb,
                              TVal);

                          if (dftParams.constraintMagnetization)
                            {
                              partialOccupancy = 1.0, partialOccupancy2 = 1.0;
                              if (eigenValues[kPoint][i + numEigenValues] >
                                  fermiEnergyDown)
                                partialOccupancy2 = 0.0;
                              if (eigenValues[kPoint][i] > fermiEnergyUp)
                                partialOccupancy = 0.0;
                            }
                          bandEnergyLocal += partialOccupancy *
                                             kPointWeights[kPoint] *
                                             eigenValues[kPoint][i];
                          bandEnergyLocal +=
                            partialOccupancy2 * kPointWeights[kPoint] *
                            eigenValues[kPoint][i + numEigenValues];
                          //
                          if (verbosity > 1)
                            scout << i << " : " << eigenValues[kPoint][i]
                                  << "       "
                                  << eigenValues[kPoint][i + numEigenValues]
                                  << "       " << partialOccupancy << "       "
                                  << partialOccupancy2 << std::endl;
                        }
                    } // eigen state
                  //
                  if (verbosity > 1)
                    scout
                      << "============================================================================================================"
                      << std::endl;
                } // kpoint
            }     // is it current pool
          //
          MPI_Barrier(interpoolcomm);
          //
        } // loop over pool

      return bandEnergyLocal;
    }

    // get nuclear electrostatic energy 0.5*sum_I*(Z_I*phi_tot(RI) -
    // Z_I*VselfI(RI))
    double
    nuclearElectrostaticEnergyLocal(
      const distributedCPUVec<double> &                    phiTotRhoOut,
      const std::vector<std::vector<double>> &             localVselfs,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<unsigned int>>
        &                          smearedbNonTrivialAtomIds,
      const dealii::DoFHandler<3> &dofHandlerElectrostatic,
      const dealii::Quadrature<3> &quadratureElectrostatic,
      const dealii::Quadrature<3> &quadratureSmearedCharge,
      const std::map<dealii::types::global_dof_index, double>
        &        atomElectrostaticNodeIdToChargeMap,
      const bool smearedNuclearCharges)
    {
      double phiContribution = 0.0, vSelfContribution = 0.0;

      if (!smearedNuclearCharges)
        {
          for (std::map<dealii::types::global_dof_index, double>::const_iterator
                 it = atomElectrostaticNodeIdToChargeMap.begin();
               it != atomElectrostaticNodeIdToChargeMap.end();
               ++it)
            phiContribution +=
              (-it->second) * phiTotRhoOut(it->first); //-charge*potential

          //
          // Then evaluate sum_I*(Z_I*Vself_I(R_I)) on atoms belonging to
          // current processor
          //
          for (unsigned int i = 0; i < localVselfs.size(); ++i)
            vSelfContribution +=
              (-localVselfs[i][0]) * (localVselfs[i][1]); //-charge*potential
        }
      else
        {
          dealii::FEValues<3> fe_values(dofHandlerElectrostatic.get_fe(),
                                        quadratureSmearedCharge,
                                        dealii::update_values |
                                          dealii::update_JxW_values);
          const unsigned int  n_q_points = quadratureSmearedCharge.size();
          dealii::DoFHandler<3>::active_cell_iterator
            cell = dofHandlerElectrostatic.begin_active(),
            endc = dofHandlerElectrostatic.end();

          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                if ((smearedbNonTrivialAtomIds.find(cell->id())->second)
                      .size() > 0)
                  {
                    const std::vector<double> &bQuadValuesCell =
                      smearedbValues.find(cell->id())->second;
                    fe_values.reinit(cell);

                    std::vector<double> tempPhiTot(n_q_points);
                    fe_values.get_function_values(phiTotRhoOut, tempPhiTot);

                    double temp = 0;
                    for (unsigned int q = 0; q < n_q_points; ++q)
                      temp +=
                        tempPhiTot[q] * bQuadValuesCell[q] * fe_values.JxW(q);

                    phiContribution += temp;
                  }
              }

          vSelfContribution = localVselfs[0][0];
        }

      return 0.5 * (phiContribution - vSelfContribution);
    }

    double
    nuclearElectrostaticEnergyResidualLocal(
      const distributedCPUVec<double> &                    phiTotRhoIn,
      const distributedCPUVec<double> &                    phiTotRhoOut,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<unsigned int>>
        &                          smearedbNonTrivialAtomIds,
      const dealii::DoFHandler<3> &dofHandlerElectrostatic,
      const dealii::Quadrature<3> &quadratureSmearedCharge,
      const std::map<dealii::types::global_dof_index, double>
        &        atomElectrostaticNodeIdToChargeMap,
      const bool smearedNuclearCharges)
    {
      double phiContribution = 0.0, vSelfContribution = 0.0;

      if (!smearedNuclearCharges)
        {
          for (std::map<dealii::types::global_dof_index, double>::const_iterator
                 it = atomElectrostaticNodeIdToChargeMap.begin();
               it != atomElectrostaticNodeIdToChargeMap.end();
               ++it)
            phiContribution +=
              (-it->second) * (phiTotRhoOut(it->first) -
                               phiTotRhoIn(it->first)); //-charge*potential
        }
      else
        {
          distributedCPUVec<double> phiRes;
          phiRes = phiTotRhoOut;
          phiRes -= phiTotRhoIn;
          dealii::FEValues<3> fe_values(dofHandlerElectrostatic.get_fe(),
                                        quadratureSmearedCharge,
                                        dealii::update_values |
                                          dealii::update_JxW_values);
          const unsigned int  n_q_points = quadratureSmearedCharge.size();
          dealii::DoFHandler<3>::active_cell_iterator
            cell = dofHandlerElectrostatic.begin_active(),
            endc = dofHandlerElectrostatic.end();

          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                if ((smearedbNonTrivialAtomIds.find(cell->id())->second)
                      .size() > 0)
                  {
                    const std::vector<double> &bQuadValuesCell =
                      smearedbValues.find(cell->id())->second;
                    fe_values.reinit(cell);

                    std::vector<double> tempPhiTot(n_q_points);
                    fe_values.get_function_values(phiRes, tempPhiTot);

                    double temp = 0;
                    for (unsigned int q = 0; q < n_q_points; ++q)
                      temp +=
                        tempPhiTot[q] * bQuadValuesCell[q] * fe_values.JxW(q);

                    phiContribution += temp;
                  }
              }
        }

      return 0.5 * (phiContribution);
    }


    double
    computeRepulsiveEnergy(
      const std::vector<std::vector<double>> &atomLocationsAndCharge,
      const bool                              isPseudopotential)
    {
      double energy = 0.0;
      for (unsigned int n1 = 0; n1 < atomLocationsAndCharge.size(); n1++)
        {
          for (unsigned int n2 = n1 + 1; n2 < atomLocationsAndCharge.size();
               n2++)
            {
              double Z1, Z2;
              if (isPseudopotential)
                {
                  Z1 = atomLocationsAndCharge[n1][1];
                  Z2 = atomLocationsAndCharge[n2][1];
                }
              else
                {
                  Z1 = atomLocationsAndCharge[n1][0];
                  Z2 = atomLocationsAndCharge[n2][0];
                }
              const dealii::Point<3> atom1(atomLocationsAndCharge[n1][2],
                                           atomLocationsAndCharge[n1][3],
                                           atomLocationsAndCharge[n1][4]);
              const dealii::Point<3> atom2(atomLocationsAndCharge[n2][2],
                                           atomLocationsAndCharge[n2][3],
                                           atomLocationsAndCharge[n2][4]);
              energy += (Z1 * Z2) / atom1.distance(atom2);
            }
        }
      return energy;
    }

  } // namespace internalEnergy

  template <dftfe::utils::MemorySpace memorySpace>
  energyCalculator<memorySpace>::energyCalculator(
    const MPI_Comm &     mpi_comm_parent,
    const MPI_Comm &     mpi_comm_domain,
    const MPI_Comm &     interpool_comm,
    const MPI_Comm &     interbandgroup_comm,
    const dftParameters &dftParams)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , interpoolcomm(interpool_comm)
    , interBandGroupComm(interbandgroup_comm)
    , d_dftParams(dftParams)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  // compute energies
  double
  energyCalculator<memorySpace>::computeEnergy(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                                     basisOperationsPtrElectro,
    const unsigned int                      densityQuadratureID,
    const unsigned int                      densityQuadratureIDElectro,
    const unsigned int                      smearedChargeQuadratureIDElectro,
    const unsigned int                      lpspQuadratureIDElectro,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<double> &             kPointWeights,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const dispersionCorrection &                   dispersionCorr,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &phiTotRhoInValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                              phiTotRhoOutValues,
    const distributedCPUVec<double> &phiTotRhoOut,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityInValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoOutValuesLpsp,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCInRepresentationPtr,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
                                                         auxDensityXCOutRepresentationPtr,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<unsigned int>>
      &                                     smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::CellId, std::vector<double>> &pseudoLocValues,
    const std::map<dealii::types::global_dof_index, double>
      &                atomElectrostaticNodeIdToChargeMap,
    const unsigned int numberGlobalAtoms,
    const unsigned int lowerBoundKindex,
    const unsigned int scfConverged,
    const bool         print,
    const bool         smearedNuclearCharges)
  {
    const dealii::ConditionalOStream scout(
      std::cout,
      (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
    const double bandEnergy = dealii::Utilities::MPI::sum(
      internalEnergy::localBandEnergy(eigenValues,
                                      kPointWeights,
                                      fermiEnergy,
                                      fermiEnergyUp,
                                      fermiEnergyDown,
                                      d_dftParams.TVal,
                                      d_dftParams.spinPolarized,
                                      scout,
                                      interpoolcomm,
                                      lowerBoundKindex,
                                      (d_dftParams.verbosity + scfConverged),
                                      d_dftParams),
      interpoolcomm);
    double excCorrPotentialTimesRho = 0.0, electrostaticPotentialTimesRho = 0.0,
           exchangeEnergy = 0.0, correlationEnergy = 0.0,
           electrostaticEnergyTotPot = 0.0;


    electrostaticPotentialTimesRho =
      internalEnergy::computeFieldTimesDensity(basisOperationsPtr,
                                               densityQuadratureID,
                                               phiTotRhoInValues,
                                               densityOutValues[0]);
    if (d_dftParams.isPseudopotential || smearedNuclearCharges)
      electrostaticPotentialTimesRho +=
        internalEnergy::computeFieldTimesDensity(basisOperationsPtrElectro,
                                                 lpspQuadratureIDElectro,
                                                 pseudoLocValues,
                                                 rhoOutValuesLpsp);
    electrostaticEnergyTotPot =
      0.5 * internalEnergy::computeFieldTimesDensity(basisOperationsPtrElectro,
                                                     densityQuadratureIDElectro,
                                                     phiTotRhoOutValues,
                                                     densityOutValues[0]);
    if (d_dftParams.isPseudopotential || smearedNuclearCharges)
      electrostaticEnergyTotPot +=
        internalEnergy::computeFieldTimesDensity(basisOperationsPtrElectro,
                                                 lpspQuadratureIDElectro,
                                                 pseudoLocValues,
                                                 rhoOutValuesLpsp);

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityOutQuadValuesSpinPolarized = densityOutValues;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradDensityOutQuadValuesSpinPolarized;

    if (d_dftParams.spinPolarized == 0)
      densityOutQuadValuesSpinPolarized.push_back(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          densityOutValues[0].size(), 0.0));

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        gradDensityOutQuadValuesSpinPolarized = gradDensityOutValues;

        if (d_dftParams.spinPolarized == 0)
          gradDensityOutQuadValuesSpinPolarized.push_back(
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(
              gradDensityOutValues[0].size(), 0.0));
      }

    computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                      densityQuadratureID,
                                      excManagerPtr,
                                      densityOutQuadValuesSpinPolarized,
                                      gradDensityOutQuadValuesSpinPolarized,
                                      auxDensityXCInRepresentationPtr,
                                      auxDensityXCOutRepresentationPtr,
                                      exchangeEnergy,
                                      correlationEnergy,
                                      excCorrPotentialTimesRho);

    const double potentialTimesRho =
      excCorrPotentialTimesRho + electrostaticPotentialTimesRho;

    double energy = -potentialTimesRho + exchangeEnergy + correlationEnergy +
                    electrostaticEnergyTotPot;

    const double nuclearElectrostaticEnergy =
      internalEnergy::nuclearElectrostaticEnergyLocal(
        phiTotRhoOut,
        localVselfs,
        smearedbValues,
        smearedbNonTrivialAtomIds,
        basisOperationsPtrElectro->getDofHandler(),
        basisOperationsPtrElectro->matrixFreeData().get_quadrature(
          densityQuadratureIDElectro),
        basisOperationsPtrElectro->matrixFreeData().get_quadrature(
          smearedChargeQuadratureIDElectro),
        atomElectrostaticNodeIdToChargeMap,
        smearedNuclearCharges);

    // sum over all processors
    double totalEnergy = dealii::Utilities::MPI::sum(energy, mpi_communicator);
    double totalpotentialTimesRho =
      dealii::Utilities::MPI::sum(potentialTimesRho, mpi_communicator);
    double totalexchangeEnergy =
      dealii::Utilities::MPI::sum(exchangeEnergy, mpi_communicator);
    double totalcorrelationEnergy =
      dealii::Utilities::MPI::sum(correlationEnergy, mpi_communicator);
    double totalelectrostaticEnergyPot =
      dealii::Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
    double totalNuclearElectrostaticEnergy =
      dealii::Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);

    double d_energyDispersion = 0;
    if (d_dftParams.dc_dispersioncorrectiontype != 0)
      {
        d_energyDispersion = dispersionCorr.getEnergyCorrection();
        totalEnergy += d_energyDispersion;
      }

    //
    // total energy
    //
    totalEnergy += bandEnergy;


    totalEnergy += totalNuclearElectrostaticEnergy;

    // subtracting the expectation of the wavefunction dependent potential from
    // the total energy and
    // adding the part of Exc energy dependent on wavefunction
    totalEnergy -= excManagerPtr->getExcSSDFunctionalObj()
                     ->getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi();

    totalEnergy += excManagerPtr->getExcSSDFunctionalObj()
                     ->getWaveFunctionDependentExcEnergy();

    const double allElectronElectrostaticEnergy =
      (totalelectrostaticEnergyPot + totalNuclearElectrostaticEnergy);


    double totalkineticEnergy = -totalpotentialTimesRho + bandEnergy;

    // output
    if (print)
      {
        internalEnergy::printEnergy(bandEnergy,
                                    totalkineticEnergy,
                                    totalexchangeEnergy,
                                    totalcorrelationEnergy,
                                    allElectronElectrostaticEnergy,
                                    d_energyDispersion,
                                    totalEnergy,
                                    numberGlobalAtoms,
                                    pcout,
                                    d_dftParams.reproducible_output,
                                    d_dftParams.isPseudopotential,
                                    d_dftParams.verbosity,
                                    d_dftParams);
      }
    return totalEnergy;
  }

  // compute energie residual,
  // E_KS-E_HWF=\int(V_{in}(\rho_{out}-\rho_{in}))+E_{pot}[\rho_{out}]-E_{pot}[\rho_{in}]
  template <dftfe::utils::MemorySpace memorySpace>
  double
  energyCalculator<memorySpace>::computeEnergyResidual(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtrElectro,
    const unsigned int densityQuadratureID,
    const unsigned int densityQuadratureIDElectro,
    const unsigned int smearedChargeQuadratureIDElectro,
    const unsigned int lpspQuadratureIDElectro,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &phiTotRhoInValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                              phiTotRhoOutValues,
    const distributedCPUVec<double> &phiTotRhoIn,
    const distributedCPUVec<double> &phiTotRhoOut,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityInValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityInValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCInRepresentationPtr,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
                                                         auxDensityXCOutRepresentationPtr,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<unsigned int>>
      &                                     smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::types::global_dof_index, double>
      &        atomElectrostaticNodeIdToChargeMap,
    const bool smearedNuclearCharges)
  {
    const dealii::ConditionalOStream scout(
      std::cout,
      (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
    double excCorrPotentialTimesRho = 0.0, electrostaticPotentialTimesRho = 0.0,
           exchangeEnergy = 0.0, correlationEnergy = 0.0,
           electrostaticEnergyTotPot = 0.0;


    electrostaticPotentialTimesRho =
      internalEnergy::computeFieldTimesDensityResidual(basisOperationsPtr,
                                                       densityQuadratureID,
                                                       phiTotRhoInValues,
                                                       densityInValues[0],
                                                       densityOutValues[0]);
    electrostaticEnergyTotPot =
      0.5 *
      (internalEnergy::computeFieldTimesDensity(basisOperationsPtrElectro,
                                                densityQuadratureIDElectro,
                                                phiTotRhoOutValues,
                                                densityOutValues[0]) -
       internalEnergy::computeFieldTimesDensity(basisOperationsPtrElectro,
                                                densityQuadratureIDElectro,
                                                phiTotRhoInValues,
                                                densityInValues[0]));

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityInQuadValuesSpinPolarized = densityInValues;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityOutQuadValuesSpinPolarized = densityOutValues;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradDensityInQuadValuesSpinPolarized;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradDensityOutQuadValuesSpinPolarized;

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        gradDensityInQuadValuesSpinPolarized  = gradDensityInValues;
        gradDensityOutQuadValuesSpinPolarized = gradDensityOutValues;
      }

    if (d_dftParams.spinPolarized == 0)
      {
        densityInQuadValuesSpinPolarized.push_back(
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            densityInValues[0].size(), 0.0));
        densityOutQuadValuesSpinPolarized.push_back(
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            densityOutValues[0].size(), 0.0));

        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            gradDensityInQuadValuesSpinPolarized.push_back(
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                gradDensityInValues[0].size(), 0.0));
            gradDensityOutQuadValuesSpinPolarized.push_back(
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                gradDensityOutValues[0].size(), 0.0));
          }
      }

    computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                      densityQuadratureID,
                                      excManagerPtr,
                                      densityInQuadValuesSpinPolarized,
                                      gradDensityInQuadValuesSpinPolarized,
                                      auxDensityXCInRepresentationPtr,
                                      auxDensityXCInRepresentationPtr,
                                      exchangeEnergy,
                                      correlationEnergy,
                                      excCorrPotentialTimesRho);


    excCorrPotentialTimesRho *= -1.0;
    exchangeEnergy *= -1.0;
    correlationEnergy *= -1.0;
    computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                      densityQuadratureID,
                                      excManagerPtr,
                                      densityOutQuadValuesSpinPolarized,
                                      gradDensityOutQuadValuesSpinPolarized,
                                      auxDensityXCInRepresentationPtr,
                                      auxDensityXCOutRepresentationPtr,
                                      exchangeEnergy,
                                      correlationEnergy,
                                      excCorrPotentialTimesRho);
    const double potentialTimesRho =
      excCorrPotentialTimesRho + electrostaticPotentialTimesRho;
    const double nuclearElectrostaticEnergy =
      internalEnergy::nuclearElectrostaticEnergyResidualLocal(
        phiTotRhoIn,
        phiTotRhoOut,
        smearedbValues,
        smearedbNonTrivialAtomIds,
        basisOperationsPtrElectro->getDofHandler(),
        basisOperationsPtrElectro->matrixFreeData().get_quadrature(
          smearedChargeQuadratureIDElectro),
        atomElectrostaticNodeIdToChargeMap,
        smearedNuclearCharges);

    double energy = -potentialTimesRho + exchangeEnergy + correlationEnergy +
                    electrostaticEnergyTotPot + nuclearElectrostaticEnergy;


    // sum over all processors
    double totalEnergy = dealii::Utilities::MPI::sum(energy, mpi_communicator);

    return std::abs(totalEnergy);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  energyCalculator<memorySpace>::computeXCEnergyTermsSpinPolarized(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &                                            basisOperationsPtr,
    const unsigned int                             quadratureId,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCInRepresentationPtr,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
            auxDensityXCOutRepresentationPtr,
    double &exchangeEnergy,
    double &correlationEnergy,
    double &excCorrPotentialTimesRho)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nCells        = basisOperationsPtr->nCells();
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();


    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      xDensityInDataOut;
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      cDensityInDataOut;

    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      xDensityOutDataOut;
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      cDensityOutDataOut;

    std::vector<double> &xEnergyDensityOut =
      xDensityOutDataOut[xcRemainderOutputDataAttributes::e];
    std::vector<double> &cEnergyDensityOut =
      cDensityOutDataOut[xcRemainderOutputDataAttributes::e];

    std::vector<double> &pdexDensityInSpinUp =
      xDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdexDensityInSpinDown =
      xDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
    std::vector<double> &pdecDensityInSpinUp =
      cDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdecDensityInSpinDown =
      cDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        xDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
        cDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
      }

    auto quadPointsAll = basisOperationsPtr->quadPoints();

    auto quadWeightsAll = basisOperationsPtr->JxW();


    auto dot3 = [](const std::array<double, 3> &a,
                   const std::array<double, 3> &b) {
      double sum = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        {
          sum += a[i] * b[i];
        }
      return sum;
    };


    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        std::vector<double> quadPointsInCell(nQuadsPerCell * 3);
        std::vector<double> quadWeightsInCell(nQuadsPerCell);
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            for (unsigned int idim = 0; idim < 3; ++idim)
              quadPointsInCell[3 * iQuad + idim] =
                quadPointsAll[iCell * nQuadsPerCell * 3 + 3 * iQuad + idim];
            quadWeightsInCell[iQuad] =
              std::real(quadWeightsAll[iCell * nQuadsPerCell + iQuad]);
          }

        excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCInRepresentationPtr,
          quadPointsInCell,
          xDensityInDataOut,
          cDensityInDataOut);

        excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCOutRepresentationPtr,
          quadPointsInCell,
          xDensityOutDataOut,
          cDensityOutDataOut);

        std::vector<double> pdexDensityInSigma;
        std::vector<double> pdecDensityInSigma;
        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            pdexDensityInSigma =
              xDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma];
            pdecDensityInSigma =
              cDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma];
          }

        std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
                             densityXCInData;
        std::vector<double> &gradDensityXCInSpinUp =
          densityXCInData[DensityDescriptorDataAttributes::gradValuesSpinUp];
        std::vector<double> &gradDensityXCInSpinDown =
          densityXCInData[DensityDescriptorDataAttributes::gradValuesSpinDown];

        if (isIntegrationByPartsGradDensityDependenceVxc)
          auxDensityXCInRepresentationPtr->applyLocalOperations(
            quadPointsInCell, densityXCInData);

        std::vector<double> gradXCRhoInDotgradRhoOut;
        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            gradXCRhoInDotgradRhoOut.resize(nQuadsPerCell * 3);

            std::array<double, 3> gradXCRhoIn1, gradXCRhoIn2, gradRhoOut1,
              gradRhoOut2;
            for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              {
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  {
                    gradXCRhoIn1[iDim] =
                      gradDensityXCInSpinUp[3 * iQuad + iDim];
                    gradXCRhoIn2[iDim] =
                      gradDensityXCInSpinDown[3 * iQuad + iDim];
                    gradRhoOut1[iDim] =
                      (gradDensityOutValues[0][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim] +
                       gradDensityOutValues[1][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim]) /
                      2.0;
                    gradRhoOut2[iDim] =
                      (gradDensityOutValues[0][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim] -
                       gradDensityOutValues[1][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim]) /
                      2.0;
                  }

                gradXCRhoInDotgradRhoOut[iQuad * 3 + 0] =
                  dot3(gradXCRhoIn1, gradRhoOut1);
                gradXCRhoInDotgradRhoOut[iQuad * 3 + 1] =
                  (dot3(gradXCRhoIn1, gradRhoOut2) +
                   dot3(gradXCRhoIn2, gradRhoOut1)) /
                  2.0;
                gradXCRhoInDotgradRhoOut[iQuad * 3 + 2] =
                  dot3(gradXCRhoIn2, gradRhoOut2);
              }
          } // GGA

        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            double Vxc =
              pdexDensityInSpinUp[iQuad] + pdecDensityInSpinUp[iQuad];
            excCorrPotentialTimesRho +=
              Vxc *
              ((densityOutValues[0][iCell * nQuadsPerCell + iQuad] +
                densityOutValues[1][iCell * nQuadsPerCell + iQuad]) /
               2.0) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];

            Vxc = pdexDensityInSpinDown[iQuad] + pdecDensityInSpinDown[iQuad];
            excCorrPotentialTimesRho +=
              Vxc *
              ((densityOutValues[0][iCell * nQuadsPerCell + iQuad] -
                densityOutValues[1][iCell * nQuadsPerCell + iQuad]) /
               2.0) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];

            exchangeEnergy +=
              (xEnergyDensityOut[iQuad]) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];

            correlationEnergy +=
              (cEnergyDensityOut[iQuad]) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
            if (isIntegrationByPartsGradDensityDependenceVxc)
              {
                double VxcGrad = 0.0;
                for (unsigned int isigma = 0; isigma < 3; ++isigma)
                  VxcGrad += 2.0 *
                             (pdexDensityInSigma[iQuad * 3 + isigma] +
                              pdecDensityInSigma[iQuad * 3 + isigma]) *
                             gradXCRhoInDotgradRhoOut[iQuad * 3 + isigma];
                excCorrPotentialTimesRho +=
                  VxcGrad * basisOperationsPtr
                              ->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
              }
          }
      } // cell loop
  }


  template <dftfe::utils::MemorySpace memorySpace>
  double
  energyCalculator<memorySpace>::computeEntropicEnergy(
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<double> &             kPointWeights,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    const bool                              isSpinPolarized,
    const bool                              isConstraintMagnetization,
    const double                            temperature) const
  {
    // computation of entropic term only for one k-pt
    double             entropy = 0.0;
    const unsigned int numEigenValues =
      isSpinPolarized ? eigenValues[0].size() / 2 : eigenValues[0].size();

    for (unsigned int kPoint = 0; kPoint < eigenValues.size(); ++kPoint)
      for (int i = 0; i < numEigenValues; ++i)
        {
          if (isSpinPolarized)
            {
              double partOccSpin0 = dftUtils::getPartialOccupancy(
                eigenValues[kPoint][i], fermiEnergy, C_kb, temperature);
              double partOccSpin1 = dftUtils::getPartialOccupancy(
                eigenValues[kPoint][i + numEigenValues],
                fermiEnergy,
                C_kb,
                temperature);

              if (d_dftParams.constraintMagnetization)
                {
                  partOccSpin0 = 1.0, partOccSpin1 = 1.0;
                  if (eigenValues[kPoint][i + numEigenValues] > fermiEnergyDown)
                    partOccSpin1 = 0.0;
                  if (eigenValues[kPoint][i] > fermiEnergyUp)
                    partOccSpin0 = 0.0;
                }


              double fTimeslogfSpin0, oneminusfTimeslogoneminusfSpin0;

              if (std::abs(partOccSpin0 - 1.0) <= 1e-07 ||
                  std::abs(partOccSpin0) <= 1e-07)
                {
                  fTimeslogfSpin0                 = 0.0;
                  oneminusfTimeslogoneminusfSpin0 = 0.0;
                }
              else
                {
                  fTimeslogfSpin0 = partOccSpin0 * log(partOccSpin0);
                  oneminusfTimeslogoneminusfSpin0 =
                    (1.0 - partOccSpin0) * log(1.0 - partOccSpin0);
                }
              entropy += -C_kb * kPointWeights[kPoint] *
                         (fTimeslogfSpin0 + oneminusfTimeslogoneminusfSpin0);

              double fTimeslogfSpin1, oneminusfTimeslogoneminusfSpin1;

              if (std::abs(partOccSpin1 - 1.0) <= 1e-07 ||
                  std::abs(partOccSpin1) <= 1e-07)
                {
                  fTimeslogfSpin1                 = 0.0;
                  oneminusfTimeslogoneminusfSpin1 = 0.0;
                }
              else
                {
                  fTimeslogfSpin1 = partOccSpin1 * log(partOccSpin1);
                  oneminusfTimeslogoneminusfSpin1 =
                    (1.0 - partOccSpin1) * log(1.0 - partOccSpin1);
                }
              entropy += -C_kb * kPointWeights[kPoint] *
                         (fTimeslogfSpin1 + oneminusfTimeslogoneminusfSpin1);
            }
          else
            {
              const double partialOccupancy = dftUtils::getPartialOccupancy(
                eigenValues[kPoint][i], fermiEnergy, C_kb, temperature);
              double fTimeslogf, oneminusfTimeslogoneminusf;

              if (std::abs(partialOccupancy - 1.0) <= 1e-07 ||
                  std::abs(partialOccupancy) <= 1e-07)
                {
                  fTimeslogf                 = 0.0;
                  oneminusfTimeslogoneminusf = 0.0;
                }
              else
                {
                  fTimeslogf = partialOccupancy * log(partialOccupancy);
                  oneminusfTimeslogoneminusf =
                    (1.0 - partialOccupancy) * log(1.0 - partialOccupancy);
                }
              entropy += -2.0 * C_kb * kPointWeights[kPoint] *
                         (fTimeslogf + oneminusfTimeslogoneminusf);
            }
        }

    // Sum across k point parallelization pools
    entropy = dealii::Utilities::MPI::sum(entropy, interpoolcomm);

    return temperature * entropy;
  }

  template class energyCalculator<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class energyCalculator<dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
