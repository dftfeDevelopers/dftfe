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
  namespace internal
  {
    double
    computeFieldTimesDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
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
            result += cellFieldValues[iQuad] *
                      densityQuadValues[iCell * nQuadsPerCell + iQuad] *
                      basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
        }
      return result;
    }
    double
    computeFieldTimesDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
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
            result += fieldValues[iCell * nQuadsPerCell + iQuad] *
                      densityQuadValues[iCell * nQuadsPerCell + iQuad] *
                      basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
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
              dftParams.useMixedPrecCheby)
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
      const bool smearedNuclearCharges = false)
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

  } // namespace internal

  energyCalculator::energyCalculator(const MPI_Comm &     mpi_comm_parent,
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

  // compute energies
  double
  energyCalculator::computeEnergy(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
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
    const excManager *                      excManagerPtr,
    const dispersionCorrection &            dispersionCorr,
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
      &gradDensityInValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  rhoOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
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
      internal::localBandEnergy(eigenValues,
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
      internal::computeFieldTimesDensity(basisOperationsPtr,
                                         densityQuadratureID,
                                         phiTotRhoInValues,
                                         densityOutValues[0]);
    if (d_dftParams.isPseudopotential || smearedNuclearCharges)
      electrostaticPotentialTimesRho +=
        internal::computeFieldTimesDensity(basisOperationsPtrElectro,
                                           lpspQuadratureIDElectro,
                                           pseudoLocValues,
                                           rhoOutValuesLpsp);
    electrostaticEnergyTotPot =
      0.5 * internal::computeFieldTimesDensity(basisOperationsPtrElectro,
                                               densityQuadratureIDElectro,
                                               phiTotRhoOutValues,
                                               densityOutValues[0]);
    if (d_dftParams.isPseudopotential || smearedNuclearCharges)
      electrostaticEnergyTotPot +=
        internal::computeFieldTimesDensity(basisOperationsPtrElectro,
                                           lpspQuadratureIDElectro,
                                           pseudoLocValues,
                                           rhoOutValuesLpsp);
    if (d_dftParams.spinPolarized == 1)
      computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                        densityQuadratureID,
                                        excManagerPtr,
                                        densityInValues,
                                        densityOutValues,
                                        gradDensityInValues,
                                        gradDensityOutValues,
                                        rhoCoreValues,
                                        gradRhoCoreValues,
                                        exchangeEnergy,
                                        correlationEnergy,
                                        excCorrPotentialTimesRho);
    else
      computeXCEnergyTerms(basisOperationsPtr,
                           densityQuadratureID,
                           excManagerPtr,
                           densityInValues,
                           densityOutValues,
                           gradDensityInValues,
                           gradDensityOutValues,
                           rhoCoreValues,
                           gradRhoCoreValues,
                           exchangeEnergy,
                           correlationEnergy,
                           excCorrPotentialTimesRho);
    const double potentialTimesRho =
      excCorrPotentialTimesRho + electrostaticPotentialTimesRho;

    double energy = -potentialTimesRho + exchangeEnergy + correlationEnergy +
                    electrostaticEnergyTotPot;

    const double nuclearElectrostaticEnergy =
      internal::nuclearElectrostaticEnergyLocal(
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

    const double allElectronElectrostaticEnergy =
      (totalelectrostaticEnergyPot + totalNuclearElectrostaticEnergy);


    double totalkineticEnergy = -totalpotentialTimesRho + bandEnergy;

    // output
    if (print)
      {
        internal::printEnergy(bandEnergy,
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

  double
  energyCalculator::computeXCEnergyTermsSpinPolarized(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtr,
    const unsigned int quadratureId,
    const excManager * excManagerPtr,
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
      &                                                  gradDensityOutValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    double &                                             exchangeEnergy,
    double &                                             correlationEnergy,
    double &excCorrPotentialTimesRho)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int  nCells        = basisOperationsPtr->nCells();
    const unsigned int  nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    std::vector<double> densityValueInXC(2 * nQuadsPerCell, 0.0);
    std::vector<double> densityValueOutXC(2 * nQuadsPerCell, 0.0);
    std::vector<double> exchangeEnergyDensity(nQuadsPerCell, 0.0);
    std::vector<double> corrEnergyDensity(nQuadsPerCell, 0.0);
    std::vector<double> derExchEnergyWithInputDensity(2 * nQuadsPerCell, 0.0);
    std::vector<double> derCorrEnergyWithInputDensity(2 * nQuadsPerCell, 0.0);
    std::vector<double> derExchEnergyWithSigmaGradDenInput,
      derCorrEnergyWithSigmaGradDenInput;
    std::vector<double> sigmaWithOutputGradDensity, sigmaWithInputGradDensity;
    std::vector<double> gradXCRhoInDotgradRhoOut;
    std::vector<std::vector<double>> gradRhoIn, gradRhoOut;
    if (excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        derExchEnergyWithSigmaGradDenInput.resize(3 * nQuadsPerCell);
        derCorrEnergyWithSigmaGradDenInput.resize(3 * nQuadsPerCell);
        sigmaWithOutputGradDensity.resize(3 * nQuadsPerCell);
        sigmaWithInputGradDensity.resize(3 * nQuadsPerCell);
        gradXCRhoInDotgradRhoOut.resize(3 * nQuadsPerCell);
      }
    auto dot3 = [](const std::array<double, 3> &a,
                   const std::array<double, 3> &b) {
      double sum = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        {
          sum += a[i] * b[i];
        }
      return sum;
    };
    const std::vector<double> dummy;
    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        auto cellId = basisOperationsPtr->cellID(iCell);
        std::map<rhoDataAttributes, const std::vector<double> *> rhoOutData;
        std::map<rhoDataAttributes, const std::vector<double> *> rhoInData;

        std::map<VeffOutputDataAttributes, std::vector<double> *>
          outputDerExchangeEnergy;
        std::map<VeffOutputDataAttributes, std::vector<double> *>
                                   outputDerCorrEnergy;
        const std::vector<double> &tempRhoCore =
          d_dftParams.nonLinearCoreCorrection ?
            rhoCoreValues.find(cellId)->second :
            dummy;
        const std::vector<double> &tempGradRhoCore =
          (d_dftParams.nonLinearCoreCorrection &&
           excManagerPtr->getDensityBasedFamilyType() ==
             densityFamilyType::GGA) ?
            gradRhoCoreValues.find(cellId)->second :
            dummy;
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            densityValueInXC[2 * iQuad + 0] =
              (densityInValues[0][iCell * nQuadsPerCell + iQuad] +
               densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
              2.0;
            densityValueInXC[2 * iQuad + 1] =
              (densityInValues[0][iCell * nQuadsPerCell + iQuad] -
               densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
              2.0;
            densityValueOutXC[2 * iQuad + 0] =
              (densityOutValues[0][iCell * nQuadsPerCell + iQuad] +
               densityOutValues[1][iCell * nQuadsPerCell + iQuad]) /
              2.0;
            densityValueOutXC[2 * iQuad + 1] =
              (densityOutValues[0][iCell * nQuadsPerCell + iQuad] -
               densityOutValues[1][iCell * nQuadsPerCell + iQuad]) /
              2.0;
            if (d_dftParams.nonLinearCoreCorrection == true)
              {
                densityValueInXC[2 * iQuad + 0] += tempRhoCore[iQuad] / 2.0;
                densityValueInXC[2 * iQuad + 1] += tempRhoCore[iQuad] / 2.0;
                densityValueOutXC[2 * iQuad + 0] += tempRhoCore[iQuad] / 2.0;
                densityValueOutXC[2 * iQuad + 1] += tempRhoCore[iQuad] / 2.0;
              }
          }
        rhoOutData[rhoDataAttributes::values] = &densityValueOutXC;

        rhoInData[rhoDataAttributes::values] = &densityValueInXC;

        outputDerExchangeEnergy
          [VeffOutputDataAttributes::derEnergyWithDensity] =
            &derExchEnergyWithInputDensity;

        outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
          &derCorrEnergyWithInputDensity;

        if (excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          {
            std::array<double, 3> gradXCRhoIn1, gradXCRhoIn2, gradXCRhoOut1,
              gradXCRhoOut2, gradRhoOut1, gradRhoOut2;
            for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              {
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  {
                    gradXCRhoIn1[iDim] =
                      (gradDensityInValues[0][iCell * 3 * nQuadsPerCell +
                                              3 * iQuad + iDim] +
                       gradDensityInValues[1][iCell * 3 * nQuadsPerCell +
                                              3 * iQuad + iDim]) /
                      2.0;
                    gradXCRhoIn2[iDim] =
                      (gradDensityInValues[0][iCell * 3 * nQuadsPerCell +
                                              3 * iQuad + iDim] -
                       gradDensityInValues[1][iCell * 3 * nQuadsPerCell +
                                              3 * iQuad + iDim]) /
                      2.0;
                    gradXCRhoOut1[iDim] =
                      (gradDensityOutValues[0][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim] +
                       gradDensityOutValues[1][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim]) /
                      2.0;
                    gradXCRhoOut2[iDim] =
                      (gradDensityOutValues[0][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim] -
                       gradDensityOutValues[1][iCell * 3 * nQuadsPerCell +
                                               3 * iQuad + iDim]) /
                      2.0;
                  }
                gradRhoOut1 = gradXCRhoOut1;
                gradRhoOut2 = gradXCRhoOut2;
                if (d_dftParams.nonLinearCoreCorrection == true)
                  {
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      {
                        gradXCRhoIn1[iDim] +=
                          tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                        gradXCRhoIn2[iDim] +=
                          tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                        gradXCRhoOut1[iDim] +=
                          tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                        gradXCRhoOut2[iDim] +=
                          tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                      }
                  }
                sigmaWithInputGradDensity[3 * iQuad + 0] =
                  dot3(gradXCRhoIn1, gradXCRhoIn1);
                sigmaWithInputGradDensity[3 * iQuad + 1] =
                  dot3(gradXCRhoIn1, gradXCRhoIn2);
                sigmaWithInputGradDensity[3 * iQuad + 2] =
                  dot3(gradXCRhoIn2, gradXCRhoIn2);
                sigmaWithOutputGradDensity[3 * iQuad + 0] =
                  dot3(gradXCRhoOut1, gradXCRhoOut1);
                sigmaWithOutputGradDensity[3 * iQuad + 1] =
                  dot3(gradXCRhoOut1, gradXCRhoOut2);
                sigmaWithOutputGradDensity[3 * iQuad + 2] =
                  dot3(gradXCRhoOut2, gradXCRhoOut2);
                gradXCRhoInDotgradRhoOut[3 * iQuad + 0] =
                  dot3(gradXCRhoIn1, gradRhoOut1);
                gradXCRhoInDotgradRhoOut[3 * iQuad + 1] =
                  dot3(gradXCRhoIn1, gradRhoOut2);
                gradXCRhoInDotgradRhoOut[3 * iQuad + 2] =
                  dot3(gradXCRhoIn2, gradRhoOut2);
              }
            rhoOutData[rhoDataAttributes::sigmaGradValue] =
              &sigmaWithOutputGradDensity;
            rhoInData[rhoDataAttributes::sigmaGradValue] =
              &sigmaWithInputGradDensity;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigmaGradDenInput;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigmaGradDenInput;
          }
        excManagerPtr->getExcDensityObj()->computeDensityBasedEnergyDensity(
          nQuadsPerCell, rhoOutData, exchangeEnergyDensity, corrEnergyDensity);

        excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
          nQuadsPerCell,
          rhoInData,
          outputDerExchangeEnergy,
          outputDerCorrEnergy);
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            double Vxc = derExchEnergyWithInputDensity[2 * iQuad + 0] +
                         derCorrEnergyWithInputDensity[2 * iQuad + 0];
            excCorrPotentialTimesRho +=
              Vxc *
              ((densityInValues[0][iCell * nQuadsPerCell + iQuad] +
                densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
               2.0) *
              basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
            Vxc = derExchEnergyWithInputDensity[2 * iQuad + 1] +
                  derCorrEnergyWithInputDensity[2 * iQuad + 1];
            excCorrPotentialTimesRho +=
              Vxc *
              ((densityInValues[0][iCell * nQuadsPerCell + iQuad] -
                densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
               2.0) *
              basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
            exchangeEnergy +=
              (exchangeEnergyDensity[iQuad]) *
              (densityValueOutXC[2 * iQuad] +
               densityValueOutXC[2 * iQuad + 1]) *
              basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];

            correlationEnergy +=
              (corrEnergyDensity[iQuad]) *
              (densityValueOutXC[2 * iQuad] +
               densityValueOutXC[2 * iQuad + 1]) *
              basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
            if (excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              {
                double VxcGrad = 0.0;
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  VxcGrad +=
                    2.0 *
                    (derExchEnergyWithSigmaGradDenInput[3 * iQuad + iDim] +
                     derCorrEnergyWithSigmaGradDenInput[3 * iQuad + iDim]) *
                    gradXCRhoInDotgradRhoOut[3 * iQuad + iDim];
                excCorrPotentialTimesRho +=
                  VxcGrad *
                  basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
              }
          }
      }
  }


  double
  energyCalculator::computeEntropicEnergy(
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
} // namespace dftfe
