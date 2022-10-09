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
    const dealii::DoFHandler<3> &           dofHandlerElectrostatic,
    const dealii::DoFHandler<3> &           dofHandlerElectronic,
    const dealii::Quadrature<3> &           quadratureElectrostatic,
    const dealii::Quadrature<3> &           quadratureElectronic,
    const dealii::Quadrature<3> &           quadratureSmearedCharge,
    const dealii::Quadrature<3> &           quadratureLpsp,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<double> &             kPointWeights,
    const double                            fermiEnergy,
    const excWavefunctionBaseClass * excFunctionalPtr,
    const dispersionCorrection &            dispersionCorr,
    const std::map<dealii::CellId, std::vector<double>> &phiTotRhoInValues,
    const distributedCPUVec<double> &                    phiTotRhoOut,
    const std::map<dealii::CellId, std::vector<double>> &rhoInValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectrostatic,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectrostaticLpsp,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoInValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<unsigned int>>
      &                                     smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::CellId, std::vector<double>> &pseudoValuesElectronic,
    const std::map<dealii::CellId, std::vector<double>>
      &pseudoValuesElectrostatic,
    const std::map<dealii::types::global_dof_index, double>
      &                atomElectrostaticNodeIdToChargeMap,
    const unsigned int numberGlobalAtoms,
    const unsigned int lowerBoundKindex,
    const unsigned int scfConverged,
    const bool         print,
    const bool         smearedNuclearCharges) const
  {
    dealii::FEValues<3> feValuesElectrostatic(dofHandlerElectrostatic.get_fe(),
                                              quadratureElectrostatic,
                                              dealii::update_values |
                                                dealii::update_JxW_values);
    dealii::FEValues<3> feValuesElectronic(dofHandlerElectronic.get_fe(),
                                           quadratureElectronic,
                                           dealii::update_values |
                                             dealii::update_JxW_values);

    dealii::FEValues<3> feValuesElectronicLpsp(dofHandlerElectronic.get_fe(),
                                               quadratureLpsp,
                                               dealii::update_JxW_values);

    dealii::FEValues<3> feValuesElectrostaticLpsp(
      dofHandlerElectrostatic.get_fe(),
      quadratureLpsp,
      dealii::update_JxW_values);

    const unsigned int num_quad_points_electrostatic =
      quadratureElectrostatic.size();
    const unsigned int num_quad_points_electronic = quadratureElectronic.size();
    const unsigned int num_quad_points_lpsp       = quadratureLpsp.size();

    if (rhoOutValues.size() != 0)
      {
        AssertThrow(
          num_quad_points_electronic == rhoOutValues.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
          AssertThrow(
            num_quad_points_electronic * 3 ==
              gradRhoOutValues.begin()->second.size(),
            dealii::ExcMessage(
              "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        AssertThrow(
          num_quad_points_lpsp == rhoOutValuesLpsp.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
      }

    if (rhoOutValuesElectrostaticLpsp.size() != 0)
      {
        AssertThrow(
          num_quad_points_electrostatic ==
            rhoOutValuesElectrostatic.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        AssertThrow(
          num_quad_points_lpsp ==
            rhoOutValuesElectrostaticLpsp.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
      }

    const double TVal = d_dftParams.TVal;
    // std::vector<double> cellPhiTotRhoIn(num_quad_points_electronic);
    std::vector<double> cellPhiTotRhoOut(num_quad_points_electrostatic);

    const dealii::ConditionalOStream scout(
      std::cout,
      (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
       dealii::Utilities::MPI::this_mpi_process(interBandGroupComm) == 0));
    const double bandEnergy = dealii::Utilities::MPI::sum(
      internal::localBandEnergy(eigenValues,
                                kPointWeights,
                                fermiEnergy,
                                fermiEnergy,
                                fermiEnergy,
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

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellElectrostatic = dofHandlerElectrostatic.begin_active(),
      endcElectrostatic = dofHandlerElectrostatic.end();

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellElectronic = dofHandlerElectronic.begin_active(),
      endcElectronic = dofHandlerElectronic.end();

    for (; cellElectronic != endcElectronic; ++cellElectronic)
      if (cellElectronic->is_locally_owned())
        {
          feValuesElectronic.reinit(cellElectronic);
          // feValuesElectronic.get_function_values(phiTotRhoIn,cellPhiTotRhoIn);

          feValuesElectronicLpsp.reinit(cellElectronic);
          std::vector<double> densityValueInXC,
            densityValueOutXC;
          std::vector<double> exchangeEnergyDensity,
            corrEnergyDensity;
          std::vector<double> derExchEnergyWithInputDensity,
            derCorrEnergyWithInputDensity;
          std::vector<double> derExchEnergyWithSigmaGradDenInput,
            derCorrEnergyWithSigmaGradDenInput;
          std::vector<double> sigmaWithOutputGradDensity,
            sigmaWithInputGradDensity;
          std::vector<double> gradXCRhoInDotgradRhoOut;

          std::map<rhoDataAttributes,const std::vector<double>*>  rhoOutData;
          std::map<rhoDataAttributes,const std::vector<double>*>  rhoInData;

          std::map<VeffOutputDataAttributes, std::vector<double>*> outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes, std::vector<double>*> outputDerCorrEnergy;


          if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
            {
              // Get exc
              densityValueInXC.resize(num_quad_points_electronic);
              densityValueOutXC.resize(num_quad_points_electronic);
              exchangeEnergyDensity.resize(num_quad_points_electronic);
              corrEnergyDensity.resize(num_quad_points_electronic);
              derExchEnergyWithInputDensity.resize(num_quad_points_electronic);
              derCorrEnergyWithInputDensity.resize(num_quad_points_electronic);
              derExchEnergyWithSigmaGradDenInput.resize(num_quad_points_electronic);
              derCorrEnergyWithSigmaGradDenInput.resize(num_quad_points_electronic);
              sigmaWithOutputGradDensity.resize(num_quad_points_electronic);
              sigmaWithInputGradDensity.resize(num_quad_points_electronic);
              gradXCRhoInDotgradRhoOut.resize(num_quad_points_electronic);

              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[q_point] =
                        rhoInValues.find(cellElectronic->id())
                          ->second[q_point] +
                        rhoCoreValues.find(cellElectronic->id())
                          ->second[q_point];
                      densityValueOutXC[q_point] =
                        rhoOutValues.find(cellElectronic->id())
                          ->second[q_point] +
                        rhoCoreValues.find(cellElectronic->id())
                          ->second[q_point];
                      const double gradRhoInX =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradRhoInY =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradRhoInZ =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);
                      const double gradRhoOutX =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradRhoOutY =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradRhoOutZ =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);
                      const double gradValRhoOutX =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradValRhoOutY =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradValRhoOutZ =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);

                      sigmaWithInputGradDensity[q_point] =
                        gradRhoInX * gradRhoInX + gradRhoInY * gradRhoInY +
                        gradRhoInZ * gradRhoInZ;
                      sigmaWithOutputGradDensity[q_point] =
                        gradRhoOutX * gradRhoOutX + gradRhoOutY * gradRhoOutY +
                        gradRhoOutZ * gradRhoOutZ;
                      gradXCRhoInDotgradRhoOut[q_point] =
                        gradRhoInX * gradValRhoOutX +
                        gradRhoInY * gradValRhoOutY +
                        gradRhoInZ * gradValRhoOutZ;
                    }
                }
              else
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[q_point] =
                        rhoInValues.find(cellElectronic->id())->second[q_point];
                      densityValueOutXC[q_point] =
                        rhoOutValues.find(cellElectronic->id())
                          ->second[q_point];
                      const double gradRhoInX =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradRhoInY =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradRhoInZ =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);
                      const double gradRhoOutX =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradRhoOutY =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradRhoOutZ =
                        (gradRhoOutValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);
                      sigmaWithInputGradDensity[q_point] =
                        gradRhoInX * gradRhoInX + gradRhoInY * gradRhoInY +
                        gradRhoInZ * gradRhoInZ;
                      sigmaWithOutputGradDensity[q_point] =
                        gradRhoOutX * gradRhoOutX + gradRhoOutY * gradRhoOutY +
                        gradRhoOutZ * gradRhoOutZ;
                      gradXCRhoInDotgradRhoOut[q_point] =
                        gradRhoInX * gradRhoOutX + gradRhoInY * gradRhoOutY +
                        gradRhoInZ * gradRhoOutZ;
                    }
                }

              rhoOutData [rhoDataAttributes::values] = &densityValueOutXC;
              rhoOutData [rhoDataAttributes::sigmaGradValue] = &sigmaWithOutputGradDensity;

              rhoInData [rhoDataAttributes::values] = &densityValueInXC;
              rhoInData [rhoDataAttributes::sigmaGradValue] = &sigmaWithInputGradDensity;

              outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity]  = &derExchEnergyWithInputDensity;
              outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] = &derExchEnergyWithSigmaGradDenInput;

              outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] = &derCorrEnergyWithInputDensity;
              outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] = &derCorrEnergyWithSigmaGradDenInput;

            }
          else if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::LDA)
            {
              densityValueInXC.resize(num_quad_points_electronic);
              densityValueOutXC.resize(num_quad_points_electronic);
              exchangeEnergyDensity.resize(num_quad_points_electronic);
              corrEnergyDensity.resize(num_quad_points_electronic);
              derExchEnergyWithInputDensity.resize(num_quad_points_electronic);
              derCorrEnergyWithInputDensity.resize(num_quad_points_electronic);

              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[q_point] =
                        rhoInValues.find(cellElectronic->id())
                          ->second[q_point] +
                        rhoCoreValues.find(cellElectronic->id())
                          ->second[q_point];
                      densityValueOutXC[q_point] =
                        rhoOutValues.find(cellElectronic->id())
                          ->second[q_point] +
                        rhoCoreValues.find(cellElectronic->id())
                          ->second[q_point];
                    }
                }
              else
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[q_point] =
                        rhoInValues.find(cellElectronic->id())->second[q_point];
                      densityValueOutXC[q_point] =
                        rhoOutValues.find(cellElectronic->id())
                          ->second[q_point];
                    }
                }

              rhoOutData [rhoDataAttributes::values] = &densityValueOutXC;

              rhoInData [rhoDataAttributes::values] = &densityValueInXC;

              outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity]  = &derExchEnergyWithInputDensity;

              outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] = &derCorrEnergyWithInputDensity;


            }

          excFunctionalPtr->computeDensityBasedEnergyDensity(
            num_quad_points_electronic,
            rhoOutData,
            exchangeEnergyDensity,
            corrEnergyDensity);

          excFunctionalPtr->computeDensityBasedVxc(
            num_quad_points_electronic,
            rhoInData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);


          if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
            {
              for (unsigned int q_point = 0;
                   q_point < num_quad_points_electronic;
                   ++q_point)
                {
                  // Vxc computed with rhoIn
                  const double Vxc = derExchEnergyWithInputDensity[q_point] +
                                     derCorrEnergyWithInputDensity[q_point];
                  const double VxcGrad =
                    2.0 *
                    (derExchEnergyWithSigmaGradDenInput[q_point] +
                     derCorrEnergyWithSigmaGradDenInput[q_point]) *
                    gradXCRhoInDotgradRhoOut[q_point];

                  excCorrPotentialTimesRho +=
                    (Vxc * (rhoOutValues.find(cellElectronic->id())
                              ->second[q_point]) +
                     VxcGrad) *
                    feValuesElectronic.JxW(q_point);

                  exchangeEnergy += (exchangeEnergyDensity[q_point]) *
                                    densityValueOutXC[q_point] *
                                    feValuesElectronic.JxW(q_point);
                  correlationEnergy += (corrEnergyDensity[q_point]) *
                                       densityValueOutXC[q_point] *
                                       feValuesElectronic.JxW(q_point);

                  electrostaticPotentialTimesRho +=
                    (phiTotRhoInValues.find(cellElectronic->id())
                       ->second[q_point]) *
                    (rhoOutValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);
                }

            }
          else if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::LDA)
            {
              for (unsigned int q_point = 0;
                   q_point < num_quad_points_electronic;
                   ++q_point)
                {
                  excCorrPotentialTimesRho +=
                    (derExchEnergyWithInputDensity[q_point] +
                     derCorrEnergyWithInputDensity[q_point]) *
                    (rhoOutValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);

                  exchangeEnergy += (exchangeEnergyDensity[q_point]) *
                                    densityValueOutXC[q_point] *
                                    feValuesElectronic.JxW(q_point);
                  correlationEnergy += (corrEnergyDensity[q_point]) *
                                       densityValueOutXC[q_point] *
                                       feValuesElectronic.JxW(q_point);

                  electrostaticPotentialTimesRho +=
                    (phiTotRhoInValues.find(cellElectronic->id())
                       ->second[q_point]) *
                    (rhoOutValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);
                }

            }


          if (d_dftParams.isPseudopotential || smearedNuclearCharges)
            {
              const std::vector<double> &tempRho =
                rhoOutValuesLpsp.find(cellElectronic->id())->second;
              const std::vector<double> &tempPspCorr =
                pseudoValuesElectronic.find(cellElectronic->id())->second;
              for (unsigned int q_point = 0; q_point < num_quad_points_lpsp;
                   ++q_point)
                electrostaticPotentialTimesRho +=
                  tempPspCorr[q_point] * tempRho[q_point] *
                  feValuesElectronicLpsp.JxW(q_point);
            }

        }  // cell loop

    for (; cellElectrostatic != endcElectrostatic; ++cellElectrostatic)
      if (cellElectrostatic->is_locally_owned())
        {
          // Compute values for current cell.
          feValuesElectrostatic.reinit(cellElectrostatic);
          feValuesElectrostatic.get_function_values(phiTotRhoOut,
                                                    cellPhiTotRhoOut);

          feValuesElectrostaticLpsp.reinit(cellElectrostatic);

          for (unsigned int q_point = 0;
               q_point < num_quad_points_electrostatic;
               ++q_point)
            {
              electrostaticEnergyTotPot +=
                0.5 * (cellPhiTotRhoOut[q_point]) *
                (rhoOutValuesElectrostatic.find(cellElectrostatic->id())
                   ->second[q_point]) *
                feValuesElectrostatic.JxW(q_point);
            }

          if (d_dftParams.isPseudopotential || smearedNuclearCharges)
            {
              const std::vector<double> &tempRho =
                rhoOutValuesElectrostaticLpsp.find(cellElectrostatic->id())
                  ->second;
              const std::vector<double> &tempPspCorr =
                pseudoValuesElectrostatic.find(cellElectrostatic->id())->second;
              for (unsigned int q_point = 0; q_point < num_quad_points_lpsp;
                   ++q_point)
                electrostaticEnergyTotPot +=
                  tempPspCorr[q_point] * tempRho[q_point] *
                  feValuesElectrostaticLpsp.JxW(q_point);
            }
        }

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
        dofHandlerElectrostatic,
        quadratureElectrostatic,
        quadratureSmearedCharge,
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


  // compute energies
  // energy=Eband(rho)-rho*pot(n)-del{Exc}/del{gradRho}|_{gradRho=gradN}} \dot
  // gradRho+Exc(n)+Etotelec(n+b)-Eself-int{n*sumvself}
  //        + int{(rho-n)*(phiTot(n+b)-sumvself+vxc(n))} +
  //        int{rho*vpsp}+int{(gradRho-gradN) \dot
  //        del{Exc}/del{gradRho}|_{gradRho=gradN}}
  //       =Eband(rho)-rho*pot(n)-del{Exc}/del{gradRho}|_{gradRho=gradN}} \dot
  //       gradRho+Exc(n)+Etotelec(n)-Eself+init{n*vpsp}-int{n*sumvself}
  //        +
  //        int{(rho-n)*(phiTot(n+b)+vpsp-sumvself+vxc(n))}+int{(gradRho-gradN)
  //        \dot del{Exc}/del{gradRho}|_{gradRho=gradN}}
  // rho*pot(n)=int{rho*(phiTot(n+b)+vxc(n)+vpsp-sumvself)}
  // = Eband(rho)+Exc(n)+Etotelec(n+b)-Eself- int{n*(phiTot(n+b)+vxc(n))+gradN
  // \dot del{Exc}/del{gradRho}|_{gradRho=gradN}} (Note that many of the
  // cancellations above assume that the electrostatic terms over
  // dofHandlerElectrostatic and dofHandlerElectronic cancel each other as the
  // integrations use the same quadrature rule even though the polynomial order
  // of dofHandlerElectrostatic can be different from dofHandlerElectronic. h
  // refined electrostatics is not allowed.
  double
  energyCalculator::computeShadowPotentialEnergyExtendedLagrangian(
    const dealii::DoFHandler<3> &           dofHandlerElectrostatic,
    const dealii::DoFHandler<3> &           dofHandlerElectronic,
    const dealii::Quadrature<3> &           quadratureDensity,
    const dealii::Quadrature<3> &           quadratureSmearedCharge,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<double> &             kPointWeights,
    const double                            fermiEnergy,
    const xc_func_type &                    funcX,
    const xc_func_type &                    funcC,
    const excWavefunctionBaseClass * excFunctionalPtr,
    const std::map<dealii::CellId, std::vector<double>> &phiTotRhoInValues,
    const distributedCPUVec<double> &                    phiTotRhoIn,
    const std::map<dealii::CellId, std::vector<double>> &rhoInValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoInValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<unsigned int>>
      &                                     smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::types::global_dof_index, double>
      &                atomElectrostaticNodeIdToChargeMap,
    const unsigned int numberGlobalAtoms,
    const unsigned int lowerBoundKindex,
    const bool         smearedNuclearCharges) const
  {
    dealii::FEValues<3> feValuesElectrostatic(dofHandlerElectrostatic.get_fe(),
                                              quadratureDensity,
                                              dealii::update_values |
                                                dealii::update_JxW_values);
    dealii::FEValues<3> feValuesElectronic(dofHandlerElectronic.get_fe(),
                                           quadratureDensity,
                                           dealii::update_values |
                                             dealii::update_JxW_values);

    const unsigned int num_quad_points_density = quadratureDensity.size();

    if (rhoInValues.size() != 0)
      {
        AssertThrow(
          num_quad_points_density == rhoInValues.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
          AssertThrow(
            num_quad_points_density * 3 ==
              gradRhoInValues.begin()->second.size(),
            dealii::ExcMessage(
              "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
      }


    const double TVal = d_dftParams.TVal;

    std::vector<double> cellPhiTotRhoIn(num_quad_points_density);

    const dealii::ConditionalOStream scout(
      std::cout,
      (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
       dealii::Utilities::MPI::this_mpi_process(interBandGroupComm) == 0));
    const double bandEnergy = dealii::Utilities::MPI::sum(
      internal::localBandEnergy(eigenValues,
                                kPointWeights,
                                fermiEnergy,
                                fermiEnergy,
                                fermiEnergy,
                                d_dftParams.TVal,
                                d_dftParams.spinPolarized,
                                scout,
                                interpoolcomm,
                                lowerBoundKindex,
                                (d_dftParams.verbosity + 1),
                                d_dftParams),
      interpoolcomm);

    double excCorrPotentialTimesRhoIn       = 0.0,
           electrostaticPotentialTimesRhoIn = 0.0, exchangeEnergy = 0.0,
           correlationEnergy = 0.0, electrostaticEnergyTotPot = 0.0;

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellElectrostatic = dofHandlerElectrostatic.begin_active(),
      endcElectrostatic = dofHandlerElectrostatic.end();

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellElectronic = dofHandlerElectronic.begin_active(),
      endcElectronic = dofHandlerElectronic.end();

    for (; cellElectronic != endcElectronic; ++cellElectronic)
      if (cellElectronic->is_locally_owned())
        {
          feValuesElectronic.reinit(cellElectronic);

          if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
            {
              // Get exc
              std::vector<double> densityValueIn(num_quad_points_density);
              std::vector<double> exchangeEnergyDensity(
                num_quad_points_density),
                corrEnergyDensity(num_quad_points_density);
              std::vector<double> derExchEnergyWithInputDensity(
                num_quad_points_density),
                derCorrEnergyWithInputDensity(num_quad_points_density);
              std::vector<double> derExchEnergyWithSigmaGradDenInput(
                num_quad_points_density),
                derCorrEnergyWithSigmaGradDenInput(num_quad_points_density);
              std::vector<double> sigmaWithInputGradDensity(
                num_quad_points_density);

              std::vector<double> gradXCDensityDotGradDensity(
                num_quad_points_density);

              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_density;
                       ++q_point)
                    {
                      densityValueIn[q_point] =
                        rhoInValues.find(cellElectronic->id())
                          ->second[q_point] +
                        rhoCoreValues.find(cellElectronic->id())
                          ->second[q_point];
                      const double gradXCRhoInX =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradXCRhoInY =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradXCRhoInZ =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]) +
                        (gradRhoCoreValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);

                      const double gradRhoInX =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradRhoInY =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradRhoInZ =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);

                      sigmaWithInputGradDensity[q_point] =
                        gradXCRhoInX * gradXCRhoInX +
                        gradXCRhoInY * gradXCRhoInY +
                        gradXCRhoInZ * gradXCRhoInZ;
                      gradXCDensityDotGradDensity[q_point] =
                        gradXCRhoInX * gradRhoInX + gradXCRhoInY * gradRhoInY +
                        gradXCRhoInZ * gradRhoInZ;
                    }
                }
              else
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_density;
                       ++q_point)
                    {
                      densityValueIn[q_point] =
                        rhoInValues.find(cellElectronic->id())->second[q_point];
                      const double gradRhoInX =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 0]);
                      const double gradRhoInY =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 1]);
                      const double gradRhoInZ =
                        (gradRhoInValues.find(cellElectronic->id())
                           ->second[3 * q_point + 2]);
                      sigmaWithInputGradDensity[q_point] =
                        gradRhoInX * gradRhoInX + gradRhoInY * gradRhoInY +
                        gradRhoInZ * gradRhoInZ;
                      gradXCDensityDotGradDensity[q_point] =
                        sigmaWithInputGradDensity[q_point];
                    }
                }

              xc_gga_exc(&funcX,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &sigmaWithInputGradDensity[0],
                         &exchangeEnergyDensity[0]);
              xc_gga_exc(&funcC,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &sigmaWithInputGradDensity[0],
                         &corrEnergyDensity[0]);

              xc_gga_vxc(&funcX,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &sigmaWithInputGradDensity[0],
                         &derExchEnergyWithInputDensity[0],
                         &derExchEnergyWithSigmaGradDenInput[0]);
              xc_gga_vxc(&funcC,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &sigmaWithInputGradDensity[0],
                         &derCorrEnergyWithInputDensity[0],
                         &derCorrEnergyWithSigmaGradDenInput[0]);

              for (unsigned int q_point = 0; q_point < num_quad_points_density;
                   ++q_point)
                {
                  // Vxc computed with rhoIn
                  const double Vxc = derExchEnergyWithInputDensity[q_point] +
                                     derCorrEnergyWithInputDensity[q_point];
                  const double VxcGrad =
                    2.0 *
                    (derExchEnergyWithSigmaGradDenInput[q_point] +
                     derCorrEnergyWithSigmaGradDenInput[q_point]) *
                    gradXCDensityDotGradDensity[q_point];

                  excCorrPotentialTimesRhoIn +=
                    (Vxc * (rhoInValues.find(cellElectronic->id())
                              ->second[q_point]) +
                     VxcGrad) *
                    feValuesElectronic.JxW(q_point);

                  if (d_dftParams.nonLinearCoreCorrection)
                    {
                      exchangeEnergy +=
                        (exchangeEnergyDensity[q_point]) *
                        (rhoInValues.find(cellElectronic->id())
                           ->second[q_point] +
                         rhoCoreValues.find(cellElectronic->id())
                           ->second[q_point]) *
                        feValuesElectronic.JxW(q_point);
                      correlationEnergy +=
                        (corrEnergyDensity[q_point]) *
                        (rhoInValues.find(cellElectronic->id())
                           ->second[q_point] +
                         rhoCoreValues.find(cellElectronic->id())
                           ->second[q_point]) *
                        feValuesElectronic.JxW(q_point);
                    }
                  else
                    {
                      exchangeEnergy += (exchangeEnergyDensity[q_point]) *
                                        (rhoInValues.find(cellElectronic->id())
                                           ->second[q_point]) *
                                        feValuesElectronic.JxW(q_point);
                      correlationEnergy +=
                        (corrEnergyDensity[q_point]) *
                        (rhoInValues.find(cellElectronic->id())
                           ->second[q_point]) *
                        feValuesElectronic.JxW(q_point);
                    }

                  electrostaticPotentialTimesRhoIn +=
                    (phiTotRhoInValues.find(cellElectronic->id())
                       ->second[q_point]) *
                    (rhoInValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);
                }
            }
          else if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::LDA)
            {
              // Get Exc
              std::vector<double> densityValueIn(num_quad_points_density);
              std::vector<double> exchangeEnergyDensity(num_quad_points_density),
                corrEnergyDensity(num_quad_points_density);
              std::vector<double> derExchEnergyWithInputDensity(num_quad_points_density),
                derCorrEnergyWithInputDensity(num_quad_points_density);

              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_density;
                       ++q_point)
                    {
                      densityValueIn[q_point] =
                        rhoInValues.find(cellElectronic->id())
                          ->second[q_point] +
                        rhoCoreValues.find(cellElectronic->id())
                          ->second[q_point];
                    }
                }
              else
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_density;
                       ++q_point)
                    {
                      densityValueIn[q_point] =
                        rhoInValues.find(cellElectronic->id())->second[q_point];
                    }
                }

              xc_lda_exc(&funcX,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &exchangeEnergyDensity[0]);
              xc_lda_exc(&funcC,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &corrEnergyDensity[0]);
              xc_lda_vxc(&funcX,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &derExchEnergyWithInputDensity[0]);
              xc_lda_vxc(&funcC,
                         num_quad_points_density,
                         &densityValueIn[0],
                         &derCorrEnergyWithInputDensity[0]);

              for (unsigned int q_point = 0; q_point < num_quad_points_density;
                   ++q_point)
                {
                  excCorrPotentialTimesRhoIn +=
                    (derExchEnergyWithInputDensity[q_point] +
                     derCorrEnergyWithInputDensity[q_point]) *
                    (rhoInValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);

                  if (d_dftParams.nonLinearCoreCorrection)
                    {
                      exchangeEnergy +=
                        (exchangeEnergyDensity[q_point]) *
                        (rhoInValues.find(cellElectronic->id())
                           ->second[q_point] +
                         rhoCoreValues.find(cellElectronic->id())
                           ->second[q_point]) *
                        feValuesElectronic.JxW(q_point);
                      correlationEnergy +=
                        (corrEnergyDensity[q_point]) *
                        (rhoInValues.find(cellElectronic->id())
                           ->second[q_point] +
                         rhoCoreValues.find(cellElectronic->id())
                           ->second[q_point]) *
                        feValuesElectronic.JxW(q_point);
                    }
                  else
                    {
                      exchangeEnergy += (exchangeEnergyDensity[q_point]) *
                                        (rhoInValues.find(cellElectronic->id())
                                           ->second[q_point]) *
                                        feValuesElectronic.JxW(q_point);
                      correlationEnergy +=
                        (corrEnergyDensity[q_point]) *
                        (rhoInValues.find(cellElectronic->id())
                           ->second[q_point]) *
                        feValuesElectronic.JxW(q_point);
                    }

                  electrostaticPotentialTimesRhoIn +=
                    (phiTotRhoInValues.find(cellElectronic->id())
                       ->second[q_point]) *
                    (rhoInValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);
                }
            }
        } // cell loop



    for (; cellElectrostatic != endcElectrostatic; ++cellElectrostatic)
      if (cellElectrostatic->is_locally_owned())
        {
          // Compute values for current cell.
          feValuesElectrostatic.reinit(cellElectrostatic);
          feValuesElectrostatic.get_function_values(phiTotRhoIn,
                                                    cellPhiTotRhoIn);

          for (unsigned int q_point = 0; q_point < num_quad_points_density;
               ++q_point)
            {
              electrostaticEnergyTotPot +=
                0.5 * (cellPhiTotRhoIn[q_point]) *
                (rhoInValues.find(cellElectrostatic->id())->second[q_point]) *
                feValuesElectrostatic.JxW(q_point);
            }
        }

    const double potentialTimesRhoIn =
      excCorrPotentialTimesRhoIn + electrostaticPotentialTimesRhoIn;

    double energy = -potentialTimesRhoIn + exchangeEnergy + correlationEnergy +
                    electrostaticEnergyTotPot;


    const double nuclearElectrostaticEnergy =
      internal::nuclearElectrostaticEnergyLocal(
        phiTotRhoIn,
        localVselfs,
        smearedbValues,
        smearedbNonTrivialAtomIds,
        dofHandlerElectrostatic,
        quadratureDensity,
        quadratureSmearedCharge,
        atomElectrostaticNodeIdToChargeMap,
        smearedNuclearCharges);

    // sum over all processors
    double totalEnergy = dealii::Utilities::MPI::sum(energy, mpi_communicator);
    double totalNuclearElectrostaticEnergy =
      dealii::Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);



    //
    // total energy
    //
    totalEnergy += bandEnergy;


    totalEnergy += totalNuclearElectrostaticEnergy;


    pcout << std::endl;
    char bufferEnergy[200];
    pcout << "Energy computations (Hartree)\n";
    pcout
      << "-------------------------------------------------------------------------------\n";

    sprintf(bufferEnergy,
            "%-52s:%25.16e\n",
            "Total shadow potential energy",
            totalEnergy);
    pcout << bufferEnergy;
    sprintf(bufferEnergy,
            "%-52s:%25.16e\n",
            "Total shadow potential energy per atom",
            totalEnergy / numberGlobalAtoms);
    pcout << bufferEnergy;
    pcout
      << "-------------------------------------------------------------------------------\n";

    return totalEnergy;
  }

  // compute energies
  double
  energyCalculator::computeEnergySpinPolarized(
    const dealii::DoFHandler<3> &           dofHandlerElectrostatic,
    const dealii::DoFHandler<3> &           dofHandlerElectronic,
    const dealii::Quadrature<3> &           quadratureElectrostatic,
    const dealii::Quadrature<3> &           quadratureElectronic,
    const dealii::Quadrature<3> &           quadratureSmearedCharge,
    const dealii::Quadrature<3> &           quadratureLpsp,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<double> &             kPointWeights,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    const excWavefunctionBaseClass * excFunctionalPtr,
    const dispersionCorrection &            dispersionCorr,
    const std::map<dealii::CellId, std::vector<double>> &phiTotRhoInValues,
    const distributedCPUVec<double> &                    phiTotRhoOut,
    const std::map<dealii::CellId, std::vector<double>> &rhoInValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectrostatic,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectrostaticLpsp,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoInValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValues,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoInValuesSpinPolarized,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesSpinPolarized,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoInValuesSpinPolarized,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesSpinPolarized,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<unsigned int>>
      &                                     smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::CellId, std::vector<double>> &pseudoValuesElectronic,
    const std::map<dealii::CellId, std::vector<double>>
      &pseudoValuesElectrostatic,
    const std::map<dealii::types::global_dof_index, double>
      &                atomElectrostaticNodeIdToChargeMap,
    const unsigned int numberGlobalAtoms,
    const unsigned int lowerBoundKindex,
    const unsigned int scfConverged,
    const bool         print,
    const bool         smearedNuclearCharges) const
  {
    dealii::FEValues<3> feValuesElectrostatic(dofHandlerElectrostatic.get_fe(),
                                              quadratureElectrostatic,
                                              dealii::update_values |
                                                dealii::update_JxW_values);
    dealii::FEValues<3> feValuesElectronic(dofHandlerElectronic.get_fe(),
                                           quadratureElectronic,
                                           dealii::update_values |
                                             dealii::update_JxW_values);

    dealii::FEValues<3> feValuesElectronicLpsp(dofHandlerElectronic.get_fe(),
                                               quadratureLpsp,
                                               dealii::update_JxW_values);

    dealii::FEValues<3> feValuesElectrostaticLpsp(
      dofHandlerElectrostatic.get_fe(),
      quadratureLpsp,
      dealii::update_JxW_values);

    const unsigned int num_quad_points_electrostatic =
      quadratureElectrostatic.size();
    const unsigned int num_quad_points_electronic = quadratureElectronic.size();
    const unsigned int num_quad_points_lpsp       = quadratureLpsp.size();


    if (rhoOutValues.size() != 0)
      {
        AssertThrow(
          num_quad_points_electronic == rhoOutValues.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
          AssertThrow(
            num_quad_points_electronic * 3 ==
              gradRhoOutValues.begin()->second.size(),
            dealii::ExcMessage(
              "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        AssertThrow(
          num_quad_points_lpsp == rhoOutValuesLpsp.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
      }

    if (rhoOutValuesElectrostaticLpsp.size() != 0)
      {
        AssertThrow(
          num_quad_points_electrostatic ==
            rhoOutValuesElectrostatic.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
        AssertThrow(
          num_quad_points_lpsp ==
            rhoOutValuesElectrostaticLpsp.begin()->second.size(),
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature data in energyCalculator::computeEnergy."));
      }

    // std::vector<double> cellPhiTotRhoIn(num_quad_points_electronic);
    std::vector<double> cellPhiTotRhoOut(num_quad_points_electrostatic);
    //
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

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellElectrostatic = dofHandlerElectrostatic.begin_active(),
      endcElectrostatic = dofHandlerElectrostatic.end();

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellElectronic = dofHandlerElectronic.begin_active(),
      endcElectronic = dofHandlerElectronic.end();

    for (; cellElectronic != endcElectronic; ++cellElectronic)
      if (cellElectronic->is_locally_owned())
        {
          feValuesElectronic.reinit(cellElectronic);
          feValuesElectronicLpsp.reinit(cellElectronic);

          std::vector<double> densityValueInXC,
            densityValueOutXC;
          std::vector<double> exchangeEnergyDensity,
            corrEnergyDensity;
          std::vector<double> derExchEnergyWithInputDensity,
            derCorrEnergyWithInputDensity;
          std::vector<double> derExchEnergyWithSigmaGradDenInput,
            derCorrEnergyWithSigmaGradDenInput;
          std::vector<double> sigmaWithOutputGradDensity,
            sigmaWithInputGradDensity;
          std::vector<double> gradXCRhoInDotgradRhoOut;

          std::map<rhoDataAttributes,const std::vector<double>*>  rhoOutData;
          std::map<rhoDataAttributes,const std::vector<double>*>  rhoInData;

          std::map<VeffOutputDataAttributes,std::vector<double>*> outputDerExchangeEnergy;
          std::map<VeffOutputDataAttributes,std::vector<double>*> outputDerCorrEnergy;


          if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
            {
              densityValueInXC.resize(2 *num_quad_points_electronic);
              densityValueOutXC.resize(2 * num_quad_points_electronic);
              exchangeEnergyDensity.resize(num_quad_points_electronic);
              corrEnergyDensity.resize(num_quad_points_electronic);
              derExchEnergyWithInputDensity.resize(2 * num_quad_points_electronic);
              derCorrEnergyWithInputDensity.resize(2 * num_quad_points_electronic);
              derExchEnergyWithSigmaGradDenInput.resize(3 * num_quad_points_electronic);
              derCorrEnergyWithSigmaGradDenInput.resize(3 *num_quad_points_electronic);
              sigmaWithOutputGradDensity.resize(3 * num_quad_points_electronic);
              sigmaWithInputGradDensity.resize(3 * num_quad_points_electronic);
              gradXCRhoInDotgradRhoOut.resize(3 * num_quad_points_electronic);

              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  const std::vector<double> &tempRhoCore =
                    rhoCoreValues.find(cellElectronic->id())->second;
                  const std::vector<double> &tempGradRhoCore =
                    gradRhoCoreValues.find(cellElectronic->id())->second;
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[2 * q_point + 0] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0] +
                        tempRhoCore[q_point] / 2.0;
                      densityValueInXC[2 * q_point + 1] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1] +
                        tempRhoCore[q_point] / 2.0;
                      densityValueOutXC[2 * q_point + 0] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0] +
                        tempRhoCore[q_point] / 2.0;
                      densityValueOutXC[2 * q_point + 1] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1] +
                        tempRhoCore[q_point] / 2.0;
                      //
                      const double gradXCRhoInX1 =
                        (gradRhoInValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 0]) +
                        tempGradRhoCore[3 * q_point + 0] / 2.0;
                      const double gradXCRhoInY1 =
                        (gradRhoInValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 1]) +
                        tempGradRhoCore[3 * q_point + 1] / 2.0;
                      const double gradXCRhoInZ1 =
                        (gradRhoInValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 2]) +
                        tempGradRhoCore[3 * q_point + 2] / 2.0;
                      const double gradXCRhoOutX1 =
                        (gradRhoOutValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 0]) +
                        tempGradRhoCore[3 * q_point + 0] / 2.0;
                      const double gradXCRhoOutY1 =
                        (gradRhoOutValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 1]) +
                        tempGradRhoCore[3 * q_point + 1] / 2.0;
                      const double gradXCRhoOutZ1 =
                        (gradRhoOutValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 2]) +
                        tempGradRhoCore[3 * q_point + 2] / 2.0;
                      //
                      const double gradXCRhoInX2 =
                        (gradRhoInValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 3]) +
                        tempGradRhoCore[3 * q_point + 0] / 2.0;
                      const double gradXCRhoInY2 =
                        (gradRhoInValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 4]) +
                        tempGradRhoCore[3 * q_point + 1] / 2.0;
                      const double gradXCRhoInZ2 =
                        (gradRhoInValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 5]) +
                        tempGradRhoCore[3 * q_point + 2] / 2.0;
                      const double gradXCRhoOutX2 =
                        (gradRhoOutValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 3]) +
                        tempGradRhoCore[3 * q_point + 0] / 2.0;
                      const double gradXCRhoOutY2 =
                        (gradRhoOutValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 4]) +
                        tempGradRhoCore[3 * q_point + 1] / 2.0;
                      const double gradXCRhoOutZ2 =
                        (gradRhoOutValuesSpinPolarized
                           .find(cellElectronic->id())
                           ->second[6 * q_point + 5]) +
                        tempGradRhoCore[3 * q_point + 2] / 2.0;

                      const double gradRhoOutX1 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 0]);
                      const double gradRhoOutY1 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 1]);
                      const double gradRhoOutZ1 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 2]);
                      //
                      const double gradRhoOutX2 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 3]);
                      const double gradRhoOutY2 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 4]);
                      const double gradRhoOutZ2 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 5]);
                      //
                      sigmaWithInputGradDensity[3 * q_point + 0] =
                        gradXCRhoInX1 * gradXCRhoInX1 +
                        gradXCRhoInY1 * gradXCRhoInY1 +
                        gradXCRhoInZ1 * gradXCRhoInZ1;
                      sigmaWithInputGradDensity[3 * q_point + 1] =
                        gradXCRhoInX1 * gradXCRhoInX2 +
                        gradXCRhoInY1 * gradXCRhoInY2 +
                        gradXCRhoInZ1 * gradXCRhoInZ2;
                      sigmaWithInputGradDensity[3 * q_point + 2] =
                        gradXCRhoInX2 * gradXCRhoInX2 +
                        gradXCRhoInY2 * gradXCRhoInY2 +
                        gradXCRhoInZ2 * gradXCRhoInZ2;
                      sigmaWithOutputGradDensity[3 * q_point + 0] =
                        gradXCRhoOutX1 * gradXCRhoOutX1 +
                        gradXCRhoOutY1 * gradXCRhoOutY1 +
                        gradXCRhoOutZ1 * gradXCRhoOutZ1;
                      sigmaWithOutputGradDensity[3 * q_point + 1] =
                        gradXCRhoOutX1 * gradXCRhoOutX2 +
                        gradXCRhoOutY1 * gradXCRhoOutY2 +
                        gradXCRhoOutZ1 * gradXCRhoOutZ2;
                      sigmaWithOutputGradDensity[3 * q_point + 2] =
                        gradXCRhoOutX2 * gradXCRhoOutX2 +
                        gradXCRhoOutY2 * gradXCRhoOutY2 +
                        gradXCRhoOutZ2 * gradXCRhoOutZ2;
                      gradXCRhoInDotgradRhoOut[3 * q_point + 0] =
                        gradXCRhoInX1 * gradRhoOutX1 +
                        gradXCRhoInY1 * gradRhoOutY1 +
                        gradXCRhoInZ1 * gradRhoOutZ1;
                      gradXCRhoInDotgradRhoOut[3 * q_point + 1] =
                        gradXCRhoInX1 * gradRhoOutX2 +
                        gradXCRhoInY1 * gradRhoOutY2 +
                        gradXCRhoInZ1 * gradRhoOutZ2;
                      gradXCRhoInDotgradRhoOut[3 * q_point + 2] =
                        gradXCRhoInX2 * gradRhoOutX2 +
                        gradXCRhoInY2 * gradRhoOutY2 +
                        gradXCRhoInZ2 * gradRhoOutZ2;
                    }
                }
              else
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[2 * q_point + 0] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0];
                      densityValueInXC[2 * q_point + 1] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1];
                      densityValueOutXC[2 * q_point + 0] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0];
                      densityValueOutXC[2 * q_point + 1] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1];
                      //
                      const double gradRhoInX1  = (gradRhoInValuesSpinPolarized
                                                    .find(cellElectronic->id())
                                                    ->second[6 * q_point + 0]);
                      const double gradRhoInY1  = (gradRhoInValuesSpinPolarized
                                                    .find(cellElectronic->id())
                                                    ->second[6 * q_point + 1]);
                      const double gradRhoInZ1  = (gradRhoInValuesSpinPolarized
                                                    .find(cellElectronic->id())
                                                    ->second[6 * q_point + 2]);
                      const double gradRhoOutX1 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 0]);
                      const double gradRhoOutY1 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 1]);
                      const double gradRhoOutZ1 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 2]);
                      //
                      const double gradRhoInX2  = (gradRhoInValuesSpinPolarized
                                                    .find(cellElectronic->id())
                                                    ->second[6 * q_point + 3]);
                      const double gradRhoInY2  = (gradRhoInValuesSpinPolarized
                                                    .find(cellElectronic->id())
                                                    ->second[6 * q_point + 4]);
                      const double gradRhoInZ2  = (gradRhoInValuesSpinPolarized
                                                    .find(cellElectronic->id())
                                                    ->second[6 * q_point + 5]);
                      const double gradRhoOutX2 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 3]);
                      const double gradRhoOutY2 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 4]);
                      const double gradRhoOutZ2 = (gradRhoOutValuesSpinPolarized
                                                     .find(cellElectronic->id())
                                                     ->second[6 * q_point + 5]);
                      //
                      sigmaWithInputGradDensity[3 * q_point + 0] =
                        gradRhoInX1 * gradRhoInX1 + gradRhoInY1 * gradRhoInY1 +
                        gradRhoInZ1 * gradRhoInZ1;
                      sigmaWithInputGradDensity[3 * q_point + 1] =
                        gradRhoInX1 * gradRhoInX2 + gradRhoInY1 * gradRhoInY2 +
                        gradRhoInZ1 * gradRhoInZ2;
                      sigmaWithInputGradDensity[3 * q_point + 2] =
                        gradRhoInX2 * gradRhoInX2 + gradRhoInY2 * gradRhoInY2 +
                        gradRhoInZ2 * gradRhoInZ2;
                      sigmaWithOutputGradDensity[3 * q_point + 0] =
                        gradRhoOutX1 * gradRhoOutX1 +
                        gradRhoOutY1 * gradRhoOutY1 +
                        gradRhoOutZ1 * gradRhoOutZ1;
                      sigmaWithOutputGradDensity[3 * q_point + 1] =
                        gradRhoOutX1 * gradRhoOutX2 +
                        gradRhoOutY1 * gradRhoOutY2 +
                        gradRhoOutZ1 * gradRhoOutZ2;
                      sigmaWithOutputGradDensity[3 * q_point + 2] =
                        gradRhoOutX2 * gradRhoOutX2 +
                        gradRhoOutY2 * gradRhoOutY2 +
                        gradRhoOutZ2 * gradRhoOutZ2;
                      gradXCRhoInDotgradRhoOut[3 * q_point + 0] =
                        gradRhoInX1 * gradRhoOutX1 +
                        gradRhoInY1 * gradRhoOutY1 + gradRhoInZ1 * gradRhoOutZ1;
                      gradXCRhoInDotgradRhoOut[3 * q_point + 1] =
                        gradRhoInX1 * gradRhoOutX2 +
                        gradRhoInY1 * gradRhoOutY2 + gradRhoInZ1 * gradRhoOutZ2;
                      gradXCRhoInDotgradRhoOut[3 * q_point + 2] =
                        gradRhoInX2 * gradRhoOutX2 +
                        gradRhoInY2 * gradRhoOutY2 + gradRhoInZ2 * gradRhoOutZ2;
                    }
                }

              rhoOutData [rhoDataAttributes::values] = &densityValueOutXC;
              rhoOutData [rhoDataAttributes::sigmaGradValue] = &sigmaWithOutputGradDensity;

              rhoInData [rhoDataAttributes::values] = &densityValueInXC;
              rhoInData [rhoDataAttributes::sigmaGradValue] = &sigmaWithInputGradDensity;

              outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity]  = &derExchEnergyWithInputDensity;
              outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] = &derExchEnergyWithSigmaGradDenInput;

              outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] = &derCorrEnergyWithInputDensity;
              outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] = &derCorrEnergyWithSigmaGradDenInput;
            }
          else if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::LDA)
            {
              densityValueInXC.resize(2 *num_quad_points_electronic);
              densityValueOutXC.resize(2 * num_quad_points_electronic);
              exchangeEnergyDensity.resize(num_quad_points_electronic);
              corrEnergyDensity.resize(num_quad_points_electronic);
              derExchEnergyWithInputDensity.resize(2 * num_quad_points_electronic);
              derCorrEnergyWithInputDensity.resize(2 * num_quad_points_electronic);

              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  const std::vector<double> &tempRhoCore =
                    rhoCoreValues.find(cellElectronic->id())->second;
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[2 * q_point + 0] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0] +
                        tempRhoCore[q_point] / 2.0;
                      densityValueInXC[2 * q_point + 1] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1] +
                        tempRhoCore[q_point] / 2.0;
                      densityValueOutXC[2 * q_point + 0] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0] +
                        tempRhoCore[q_point] / 2.0;
                      densityValueOutXC[2 * q_point + 1] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1] +
                        tempRhoCore[q_point] / 2.0;
                    }
                }
              else
                {
                  for (unsigned int q_point = 0;
                       q_point < num_quad_points_electronic;
                       ++q_point)
                    {
                      densityValueInXC[2 * q_point + 0] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0];
                      densityValueInXC[2 * q_point + 1] =
                        rhoInValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1];
                      densityValueOutXC[2 * q_point + 0] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0];
                      densityValueOutXC[2 * q_point + 1] =
                        rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 1];
                    }
                }


              rhoOutData [rhoDataAttributes::values] = &densityValueOutXC;

              rhoInData [rhoDataAttributes::values] = &densityValueInXC;

              outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity]  = &derExchEnergyWithInputDensity;

              outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] = &derCorrEnergyWithInputDensity;


            }



          excFunctionalPtr->computeDensityBasedEnergyDensity(
            num_quad_points_electronic,
            rhoOutData,
            exchangeEnergyDensity,
            corrEnergyDensity);

          excFunctionalPtr->computeDensityBasedVxc(
            num_quad_points_electronic,
            rhoInData,
            outputDerExchangeEnergy,
            outputDerCorrEnergy);

          if (excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
            {
              for (unsigned int q_point = 0;
                   q_point < num_quad_points_electronic;
                   ++q_point)
                {
                  // Vxc computed with rhoIn
                  double Vxc = derExchEnergyWithInputDensity[2 * q_point + 0] +
                               derCorrEnergyWithInputDensity[2 * q_point + 0];
                  double VxcGrad =
                    2.0 *
                    (derExchEnergyWithSigmaGradDenInput[3 * q_point + 0] +
                     derCorrEnergyWithSigmaGradDenInput[3 * q_point + 0]) *
                    gradXCRhoInDotgradRhoOut[3 * q_point + 0];

                  VxcGrad +=
                    2.0 *
                    (derExchEnergyWithSigmaGradDenInput[3 * q_point + 1] +
                     derCorrEnergyWithSigmaGradDenInput[3 * q_point + 1]) *
                    gradXCRhoInDotgradRhoOut[3 * q_point + 1];

                  VxcGrad +=
                    2.0 *
                    (derExchEnergyWithSigmaGradDenInput[3 * q_point + 2] +
                     derCorrEnergyWithSigmaGradDenInput[3 * q_point + 2]) *
                    gradXCRhoInDotgradRhoOut[3 * q_point + 2];

                  excCorrPotentialTimesRho +=
                    (Vxc *
                       (rhoOutValuesSpinPolarized.find(cellElectronic->id())
                          ->second[2 * q_point + 0]) +
                     VxcGrad) *
                    feValuesElectronic.JxW(q_point);

                  Vxc = derExchEnergyWithInputDensity[2 * q_point + 1] +
                        derCorrEnergyWithInputDensity[2 * q_point + 1];

                  excCorrPotentialTimesRho +=
                    (Vxc *
                     (rhoOutValuesSpinPolarized.find(cellElectronic->id())
                        ->second[2 * q_point + 1])) *
                    feValuesElectronic.JxW(q_point);

                  exchangeEnergy += (exchangeEnergyDensity[q_point]) *
                                    (densityValueOutXC[2 * q_point] +
                                     densityValueOutXC[2 * q_point + 1]) *
                                    feValuesElectronic.JxW(q_point);

                  correlationEnergy += (corrEnergyDensity[q_point]) *
                                       (densityValueOutXC[2 * q_point] +
                                        densityValueOutXC[2 * q_point + 1]) *
                                       feValuesElectronic.JxW(q_point);

                  electrostaticPotentialTimesRho +=
                    (phiTotRhoInValues.find(cellElectronic->id())
                       ->second[q_point]) *
                    (rhoOutValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);
                }
            }
          else if(excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::LDA)
            {
              for (unsigned int q_point = 0;
                   q_point < num_quad_points_electronic;
                   ++q_point)
                {
                  // Vxc computed with rhoIn
                  double Vxc = derExchEnergyWithInputDensity[2 * q_point] +
                               derCorrEnergyWithInputDensity[2 * q_point];
                  excCorrPotentialTimesRho +=
                    Vxc *
                    (rhoOutValuesSpinPolarized.find(cellElectronic->id())
                       ->second[2 * q_point]) *
                    feValuesElectronic.JxW(q_point);
                  //
                  Vxc = derExchEnergyWithInputDensity[2 * q_point + 1] +
                        derCorrEnergyWithInputDensity[2 * q_point + 1];
                  excCorrPotentialTimesRho +=
                    Vxc *
                    (rhoOutValuesSpinPolarized.find(cellElectronic->id())
                       ->second[2 * q_point + 1]) *
                    feValuesElectronic.JxW(q_point);
                  //
                  exchangeEnergy += (exchangeEnergyDensity[q_point]) *
                                    (densityValueOutXC[2 * q_point] +
                                     densityValueOutXC[2 * q_point + 1]) *
                                    feValuesElectronic.JxW(q_point);
                  correlationEnergy += (corrEnergyDensity[q_point]) *
                                       (densityValueOutXC[2 * q_point] +
                                        densityValueOutXC[2 * q_point + 1]) *
                                       feValuesElectronic.JxW(q_point);

                  electrostaticPotentialTimesRho +=
                    (phiTotRhoInValues.find(cellElectronic->id())
                       ->second[q_point]) *
                    (rhoOutValues.find(cellElectronic->id())->second[q_point]) *
                    feValuesElectronic.JxW(q_point);
                }

            }

          if (d_dftParams.isPseudopotential || smearedNuclearCharges)
            {
              const std::vector<double> &tempRho =
                rhoOutValuesLpsp.find(cellElectronic->id())->second;
              const std::vector<double> &tempPspCorr =
                pseudoValuesElectronic.find(cellElectronic->id())->second;
              for (unsigned int q_point = 0; q_point < num_quad_points_lpsp;
                   ++q_point)
                electrostaticPotentialTimesRho +=
                  tempPspCorr[q_point] * tempRho[q_point] *
                  feValuesElectronicLpsp.JxW(q_point);
            }

        } //cell loop


    for (; cellElectrostatic != endcElectrostatic; ++cellElectrostatic)
      if (cellElectrostatic->is_locally_owned())
        {
          // Compute values for current cell.
          feValuesElectrostatic.reinit(cellElectrostatic);
          feValuesElectrostatic.get_function_values(phiTotRhoOut,
                                                    cellPhiTotRhoOut);

          feValuesElectrostaticLpsp.reinit(cellElectrostatic);

          for (unsigned int q_point = 0;
               q_point < num_quad_points_electrostatic;
               ++q_point)
            {
              electrostaticEnergyTotPot +=
                0.5 * (cellPhiTotRhoOut[q_point]) *
                (rhoOutValuesElectrostatic.find(cellElectrostatic->id())
                   ->second[q_point]) *
                feValuesElectrostatic.JxW(q_point);
            }

          if (d_dftParams.isPseudopotential || smearedNuclearCharges)
            {
              const std::vector<double> &tempRho =
                rhoOutValuesElectrostaticLpsp.find(cellElectrostatic->id())
                  ->second;
              const std::vector<double> &tempPspCorr =
                pseudoValuesElectrostatic.find(cellElectrostatic->id())->second;
              for (unsigned int q_point = 0; q_point < num_quad_points_lpsp;
                   ++q_point)
                electrostaticEnergyTotPot +=
                  tempPspCorr[q_point] * tempRho[q_point] *
                  feValuesElectrostaticLpsp.JxW(q_point);
            }
        }



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
        dofHandlerElectrostatic,
        quadratureElectrostatic,
        quadratureSmearedCharge,
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
