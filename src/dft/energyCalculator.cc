// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
                                                          &basisOperationsPtr,
      const dftfe::uInt                                    quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValues)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (dftfe::uInt iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
        {
          const std::vector<double> &cellFieldValues =
            fieldValues.find(basisOperationsPtr->cellID(iCell))->second;
          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
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
                                                          &basisOperationsPtr,
      const dftfe::uInt                                    quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesIn,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesOut)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (dftfe::uInt iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
        {
          const std::vector<double> &cellFieldValues =
            fieldValues.find(basisOperationsPtr->cellID(iCell))->second;
          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
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
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValues)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (dftfe::uInt iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
        {
          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
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
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesIn,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesOut)
    {
      double result = 0.0;
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      for (dftfe::uInt iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
        {
          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            result +=
              fieldValues[iCell * nQuadsPerCell + iQuad] *
              (densityQuadValuesOut[iCell * nQuadsPerCell + iQuad] -
               densityQuadValuesIn[iCell * nQuadsPerCell + iQuad]) *
              basisOperationsPtr->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
        }
      return result;
    }
    template <typename T>
    void
    transformNonColinDensityToSpinPolarizedDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      bool              isGGA,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &nonColinDensityValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &nonColinGradDensityValues,
      std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &spinPolarizedDensityValues,
      std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &spinPolarizedGradDensityValues)
    {
      spinPolarizedDensityValues.clear();
      spinPolarizedGradDensityValues.clear();
      spinPolarizedDensityValues.resize(2);
      spinPolarizedGradDensityValues.resize(isGGA ? 2 : 0);
      spinPolarizedDensityValues[0] = nonColinDensityValues[0];
      spinPolarizedDensityValues[1].resize(spinPolarizedDensityValues[0].size(),
                                           0.0);
      if (isGGA)
        {
          spinPolarizedGradDensityValues[0] = nonColinGradDensityValues[0];
          spinPolarizedGradDensityValues[1].resize(
            spinPolarizedGradDensityValues[0].size(), 0.0);
        }
      basisOperationsPtr->reinit(0, 0, quadratureId, false);
      const dftfe::uInt   nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
      std::vector<double> cellMagAxisVals(isGGA ? nQuadsPerCell * 3 : 0, 0.0);
      for (dftfe::uInt iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
        {
          const double *cellRhoValues =
            nonColinDensityValues[0].data() + iCell * nQuadsPerCell;
          const double *cellMagZValues =
            nonColinDensityValues[1].data() + iCell * nQuadsPerCell;
          const double *cellMagYValues =
            nonColinDensityValues[2].data() + iCell * nQuadsPerCell;
          const double *cellMagXValues =
            nonColinDensityValues[3].data() + iCell * nQuadsPerCell;
          double *cellMagNormValues =
            spinPolarizedDensityValues[1].data() + iCell * nQuadsPerCell;

          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            cellMagNormValues[iQuad] =
              std::sqrt(cellMagZValues[iQuad] * cellMagZValues[iQuad] +
                        cellMagYValues[iQuad] * cellMagYValues[iQuad] +
                        cellMagXValues[iQuad] * cellMagXValues[iQuad]);
          if (isGGA)
            {
              const double *cellGradRhoValues =
                nonColinGradDensityValues[0].data() + 3 * iCell * nQuadsPerCell;
              const double *cellGradMagZValues =
                nonColinGradDensityValues[1].data() + 3 * iCell * nQuadsPerCell;
              const double *cellGradMagYValues =
                nonColinGradDensityValues[2].data() + 3 * iCell * nQuadsPerCell;
              const double *cellGradMagXValues =
                nonColinGradDensityValues[3].data() + 3 * iCell * nQuadsPerCell;
              double *cellGradMagNormValues =
                spinPolarizedGradDensityValues[1].data() +
                3 * iCell * nQuadsPerCell;
              double *cellMagAxisValues = cellMagAxisVals.data();
              for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                {
                  if (cellMagNormValues[iQuad] > 1e-12)
                    {
                      cellMagAxisValues[3 * iQuad + 0] =
                        cellMagXValues[iQuad] / cellMagNormValues[iQuad];
                      cellMagAxisValues[3 * iQuad + 1] =
                        cellMagYValues[iQuad] / cellMagNormValues[iQuad];
                      cellMagAxisValues[3 * iQuad + 2] =
                        cellMagZValues[iQuad] / cellMagNormValues[iQuad];
                    }
                  else
                    {
                      cellMagAxisValues[3 * iQuad + 0] = 0.0;
                      cellMagAxisValues[3 * iQuad + 1] = 0.0;
                      cellMagAxisValues[3 * iQuad + 2] = 0.0;
                    }
                  for (dftfe::uInt idim = 0; idim < 3; ++idim)
                    {
                      cellGradMagNormValues[iQuad * 3 + idim] =
                        cellMagAxisValues[3 * iQuad + 2] *
                          cellGradMagZValues[3 * iQuad + idim] +
                        cellMagAxisValues[3 * iQuad + 1] *
                          cellGradMagYValues[3 * iQuad + idim] +
                        cellMagAxisValues[3 * iQuad + 0] *
                          cellGradMagXValues[3 * iQuad + idim];
                    }
                }
            }
        }
    }

    void
    printEnergy(const double                      bandEnergy,
                const double                      totalkineticEnergy,
                const double                      totalexchangeEnergy,
                const double                      totalcorrelationEnergy,
                const double                      totalElectrostaticEnergy,
                const double                      dispersionEnergy,
                const double                      totalEnergy,
                const dftfe::uInt                 numberAtoms,
                const dealii::ConditionalOStream &pcout,
                const bool                        reproducibleOutput,
                const bool                        isPseudo,
                const dftfe::uInt                 verbosity,
                const dftParameters              &dftParams)
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
          if (dftParams.useMixedPrecXtOX || dftParams.useMixedPrecCGS_SR ||
              dftParams.useMixedPrecXtHX ||
              (dftParams.communPrecCheby != "STANDARD"))
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
                    const std::vector<std::vector<double>> &partialOccupancies,
                    const std::vector<double>              &kPointWeights,
                    const double                            fermiEnergy,
                    const double                            fermiEnergyUp,
                    const double                            fermiEnergyDown,
                    const double                            TVal,
                    const dftfe::uInt                       spinPolarized,
                    const dealii::ConditionalOStream       &scout,
                    const MPI_Comm                         &interpoolcomm,
                    const dftfe::uInt                       lowerBoundKindex,
                    const dftfe::uInt                       verbosity,
                    const dftParameters                    &dftParams)
    {
      double      bandEnergyLocal = 0.0;
      dftfe::uInt numEigenValues =
        eigenValues[0].size() / (spinPolarized == 1 ? 2 : 1);
      //
      for (dftfe::uInt ipool = 0;
           ipool < dealii::Utilities::MPI::n_mpi_processes(interpoolcomm);
           ++ipool)
        {
          MPI_Barrier(interpoolcomm);
          if (ipool == dealii::Utilities::MPI::this_mpi_process(interpoolcomm))
            {
              for (dftfe::uInt kPoint = 0; kPoint < kPointWeights.size();
                   ++kPoint)
                {
                  if (verbosity > 1)
                    {
                      scout
                        << " Printing KS eigen values (spin split if this is a spin polarized calculation ) and fractional occupancies for kPoint "
                        << (lowerBoundKindex + kPoint) << std::endl;
                      scout << "  " << std::endl;
                    }
                  for (dftfe::uInt i = 0; i < numEigenValues; i++)
                    {
                      if (spinPolarized == 0)
                        {
                          bandEnergyLocal +=
                            2.0 * partialOccupancies[kPoint][i] *
                            kPointWeights[kPoint] * eigenValues[kPoint][i];
                          //

                          if (verbosity > 1)
                            scout << i << " : " << eigenValues[kPoint][i]
                                  << "       " << partialOccupancies[kPoint][i]
                                  << std::endl;
                          //
                        }
                      if (spinPolarized == 1)
                        {
                          bandEnergyLocal += partialOccupancies[kPoint][i] *
                                             kPointWeights[kPoint] *
                                             eigenValues[kPoint][i];
                          bandEnergyLocal +=
                            partialOccupancies[kPoint][i + numEigenValues] *
                            kPointWeights[kPoint] *
                            eigenValues[kPoint][i + numEigenValues];
                          //
                          if (verbosity > 1)
                            scout
                              << i << " : " << eigenValues[kPoint][i]
                              << "       "
                              << eigenValues[kPoint][i + numEigenValues]
                              << "       " << partialOccupancies[kPoint][i]
                              << "       "
                              << partialOccupancies[kPoint][i + numEigenValues]
                              << std::endl;
                        }
                      if (spinPolarized == 2)
                        {
                          bandEnergyLocal += partialOccupancies[kPoint][i] *
                                             kPointWeights[kPoint] *
                                             eigenValues[kPoint][i];
                          //

                          if (verbosity > 1)
                            scout << i << " : " << eigenValues[kPoint][i]
                                  << "       " << partialOccupancies[kPoint][i]
                                  << std::endl;
                          //
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
      const distributedCPUVec<double>                     &phiTotRhoOut,
      const std::vector<std::vector<double>>              &localVselfs,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                  &smearedbNonTrivialAtomIds,
      const dealii::DoFHandler<3> &dofHandlerElectrostatic,
      const dealii::Quadrature<3> &quadratureElectrostatic,
      const dealii::Quadrature<3> &quadratureSmearedCharge,
      const std::map<dealii::types::global_dof_index, double>
                &atomElectrostaticNodeIdToChargeMap,
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
          for (dftfe::uInt i = 0; i < localVselfs.size(); ++i)
            vSelfContribution +=
              (-localVselfs[i][0]) * (localVselfs[i][1]); //-charge*potential
        }
      else
        {
          dealii::FEValues<3> fe_values(dofHandlerElectrostatic.get_fe(),
                                        quadratureSmearedCharge,
                                        dealii::update_values |
                                          dealii::update_JxW_values);
          const dftfe::uInt   n_q_points = quadratureSmearedCharge.size();
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
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
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
      const distributedCPUVec<double>                     &phiTotRhoIn,
      const distributedCPUVec<double>                     &phiTotRhoOut,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                  &smearedbNonTrivialAtomIds,
      const dealii::DoFHandler<3> &dofHandlerElectrostatic,
      const dealii::Quadrature<3> &quadratureSmearedCharge,
      const std::map<dealii::types::global_dof_index, double>
                &atomElectrostaticNodeIdToChargeMap,
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
          const dftfe::uInt   n_q_points = quadratureSmearedCharge.size();
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
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
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
      for (dftfe::uInt n1 = 0; n1 < atomLocationsAndCharge.size(); n1++)
        {
          for (dftfe::uInt n2 = n1 + 1; n2 < atomLocationsAndCharge.size();
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
    const MPI_Comm      &mpi_comm_parent,
    const MPI_Comm      &mpi_comm_domain,
    const MPI_Comm      &interpool_comm,
    const MPI_Comm      &interbandgroup_comm,
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
                                           &basisOperationsPtrElectro,
    const dftfe::uInt                       densityQuadratureID,
    const dftfe::uInt                       densityQuadratureIDElectro,
    const dftfe::uInt                       smearedChargeQuadratureIDElectro,
    const dftfe::uInt                       lpspQuadratureIDElectro,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<std::vector<double>> &partialOccupancies,
    const std::vector<double>              &kPointWeights,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const dispersionCorrection                    &dispersionCorr,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &phiTotRhoInValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                    &phiTotRhoOutValues,
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
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauInValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoOutValuesLpsp,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCInRepresentationPtr,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCOutRepresentationPtr,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                           &smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::CellId, std::vector<double>> &pseudoLocValues,
    const std::map<dealii::types::global_dof_index, double>
                     &atomElectrostaticNodeIdToChargeMap,
    const dftfe::uInt numberGlobalAtoms,
    const dftfe::uInt lowerBoundKindex,
    const dftfe::uInt scfConverged,
    const bool        print,
    const bool        smearedNuclearCharges)
  {
    const dealii::ConditionalOStream scout(
      std::cout,
      (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
    const double bandEnergy = dealii::Utilities::MPI::sum(
      internalEnergy::localBandEnergy(eigenValues,
                                      partialOccupancies,
                                      kPointWeights,
                                      fermiEnergy,
                                      fermiEnergyUp,
                                      fermiEnergyDown,
                                      d_dftParams.TVal,
                                      (d_dftParams.noncolin ||
                                       d_dftParams.hasSOC) ?
                                        2 :
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
      gradDensityOutQuadValuesSpinPolarized = gradDensityOutValues;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      tauOutQuadValuesSpinPolarized = tauOutValues;

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    if (d_dftParams.noncolin)
      {
        internalEnergy::transformNonColinDensityToSpinPolarizedDensity(
          basisOperationsPtr,
          densityQuadratureID,
          isIntegrationByPartsGradDensityDependenceVxc,
          densityOutValues,
          gradDensityOutValues,
          densityOutQuadValuesSpinPolarized,
          gradDensityOutQuadValuesSpinPolarized);
      }
    else if (d_dftParams.spinPolarized == 0)
      {
        densityOutQuadValuesSpinPolarized.push_back(
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            densityOutValues[0].size(), 0.0));

        const bool isTauMGGA =
          (excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
           ExcFamilyType::TauMGGA);

        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            gradDensityOutQuadValuesSpinPolarized = gradDensityOutValues;

            gradDensityOutQuadValuesSpinPolarized.push_back(
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                gradDensityOutValues[0].size(), 0.0));
          }

        if (isTauMGGA)
          {
            if (d_dftParams.spinPolarized == 0)
              {
                tauOutQuadValuesSpinPolarized.push_back(
                  dftfe::utils::MemoryStorage<double,
                                              dftfe::utils::MemorySpace::HOST>(
                    tauOutValues[0].size(), 0.0));
              }
          }
      }
    computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                      densityQuadratureID,
                                      excManagerPtr,
                                      densityOutQuadValuesSpinPolarized,
                                      gradDensityOutQuadValuesSpinPolarized,
                                      tauOutQuadValuesSpinPolarized,
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
                     &basisOperationsPtrElectro,
    const dftfe::uInt densityQuadratureID,
    const dftfe::uInt densityQuadratureIDElectro,
    const dftfe::uInt smearedChargeQuadratureIDElectro,
    const dftfe::uInt lpspQuadratureIDElectro,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &phiTotRhoInValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                    &phiTotRhoOutValues,
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
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauInValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauOutValues,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCInRepresentationPtr,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCOutRepresentationPtr,
    const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
    const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                           &smearedbNonTrivialAtomIds,
    const std::vector<std::vector<double>> &localVselfs,
    const std::map<dealii::types::global_dof_index, double>
              &atomElectrostaticNodeIdToChargeMap,
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

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      tauInQuadValuesSpinPolarized;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      tauOutQuadValuesSpinPolarized;

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        gradDensityInQuadValuesSpinPolarized  = gradDensityInValues;
        gradDensityOutQuadValuesSpinPolarized = gradDensityOutValues;
      }
    if (isTauMGGA)
      {
        tauInQuadValuesSpinPolarized  = tauInValues;
        tauOutQuadValuesSpinPolarized = tauOutValues;
      }

    if (d_dftParams.noncolin)
      {
        internalEnergy::transformNonColinDensityToSpinPolarizedDensity(
          basisOperationsPtr,
          densityQuadratureID,
          isIntegrationByPartsGradDensityDependenceVxc,
          densityInValues,
          gradDensityInValues,
          densityInQuadValuesSpinPolarized,
          gradDensityInQuadValuesSpinPolarized);
        internalEnergy::transformNonColinDensityToSpinPolarizedDensity(
          basisOperationsPtr,
          densityQuadratureID,
          isIntegrationByPartsGradDensityDependenceVxc,
          densityOutValues,
          gradDensityOutValues,
          densityOutQuadValuesSpinPolarized,
          gradDensityOutQuadValuesSpinPolarized);
      }
    else if (d_dftParams.spinPolarized == 0)
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
        if (isTauMGGA)
          {
            tauInQuadValuesSpinPolarized.push_back(
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                tauInValues[0].size(), 0.0));
            tauOutQuadValuesSpinPolarized.push_back(
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                tauOutValues[0].size(), 0.0));
          }
      }

    computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                      densityQuadratureID,
                                      excManagerPtr,
                                      densityInQuadValuesSpinPolarized,
                                      gradDensityInQuadValuesSpinPolarized,
                                      tauInQuadValuesSpinPolarized,
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
                                      tauOutQuadValuesSpinPolarized,
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
                                                  &basisOperationsPtr,
    const dftfe::uInt                              quadratureId,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauOutValues,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCInRepresentationPtr,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
            auxDensityXCOutRepresentationPtr,
    double &exchangeEnergy,
    double &correlationEnergy,
    double &excCorrPotentialTimesRho)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const dftfe::uInt nCells        = basisOperationsPtr->nCells();
    const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();


    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      xDensityInDataOut;
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      cDensityInDataOut;

    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      xDensityOutDataOut;
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      cDensityOutDataOut;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &xEnergyDensityOut =
        xDensityOutDataOut[xcRemainderOutputDataAttributes::e];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &cEnergyDensityOut =
        cDensityOutDataOut[xcRemainderOutputDataAttributes::e];

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdexDensityInSpinUp =
        xDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdexDensityInSpinDown =
        xDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdecDensityInSpinUp =
        cDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &pdecDensityInSpinDown =
        cDensityInDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        xDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          dftfe::utils::MemoryStorage<double,
                                      dftfe::utils::MemorySpace::HOST>();
        cDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          dftfe::utils::MemoryStorage<double,
                                      dftfe::utils::MemorySpace::HOST>();
      }

    if (isTauMGGA)
      {
        xDensityInDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] =
          dftfe::utils::MemoryStorage<double,
                                      dftfe::utils::MemorySpace::HOST>();
        xDensityInDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown] =
          dftfe::utils::MemoryStorage<double,
                                      dftfe::utils::MemorySpace::HOST>();
        cDensityInDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp] =
          dftfe::utils::MemoryStorage<double,
                                      dftfe::utils::MemorySpace::HOST>();
        cDensityInDataOut[xcRemainderOutputDataAttributes::pdeTauSpinDown] =
          dftfe::utils::MemoryStorage<double,
                                      dftfe::utils::MemorySpace::HOST>();
      }

    auto quadPointsAll = basisOperationsPtr->quadPoints();

    auto quadWeightsAll = basisOperationsPtr->JxW();


    auto dot3 = [](const std::array<double, 3> &a,
                   const std::array<double, 3> &b) {
      double sum = 0.0;
      for (dftfe::uInt i = 0; i < 3; i++)
        {
          sum += a[i] * b[i];
        }
      return sum;
    };


    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      {
        excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCInRepresentationPtr,
          std::make_pair(iCell * nQuadsPerCell, (iCell + 1) * nQuadsPerCell),
          xDensityInDataOut,
          cDensityInDataOut);

        excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCOutRepresentationPtr,
          std::make_pair(iCell * nQuadsPerCell, (iCell + 1) * nQuadsPerCell),
          xDensityOutDataOut,
          cDensityOutDataOut);

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexDensityInSigma;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdecDensityInSigma;
        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            pdexDensityInSigma =
              xDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma];
            pdecDensityInSigma =
              cDensityInDataOut[xcRemainderOutputDataAttributes::pdeSigma];
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexTauInSpinUp;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexTauInSpinDown;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdecTauInSpinUp;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdecTauInSpinDown;
        if (isTauMGGA)
          {
            pdexTauInSpinUp =
              xDensityInDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp];
            pdexTauInSpinDown = xDensityInDataOut
              [xcRemainderOutputDataAttributes::pdeTauSpinDown];
            pdecTauInSpinUp =
              cDensityInDataOut[xcRemainderOutputDataAttributes::pdeTauSpinUp];
            pdecTauInSpinDown = cDensityInDataOut
              [xcRemainderOutputDataAttributes::pdeTauSpinDown];
          }

        std::unordered_map<
          DensityDescriptorDataAttributes,
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          densityXCInData;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &gradDensityXCInSpinUp =
            densityXCInData[DensityDescriptorDataAttributes::gradValuesSpinUp];
        dftfe::utils::MemoryStorage<
          double,
          dftfe::utils::MemorySpace::HOST> &gradDensityXCInSpinDown =
          densityXCInData[DensityDescriptorDataAttributes::gradValuesSpinDown];

        if (isIntegrationByPartsGradDensityDependenceVxc)
          auxDensityXCInRepresentationPtr->applyLocalOperations(
            std::make_pair(iCell * nQuadsPerCell, (iCell + 1) * nQuadsPerCell),
            densityXCInData);

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          gradXCRhoInDotgradRhoOut;
        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            gradXCRhoInDotgradRhoOut.resize(nQuadsPerCell * 3);

            std::array<double, 3> gradXCRhoIn1, gradXCRhoIn2, gradRhoOut1,
              gradRhoOut2;
            for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              {
                for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
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

        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
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
                for (dftfe::uInt isigma = 0; isigma < 3; ++isigma)
                  VxcGrad += 2.0 *
                             (pdexDensityInSigma[iQuad * 3 + isigma] +
                              pdecDensityInSigma[iQuad * 3 + isigma]) *
                             gradXCRhoInDotgradRhoOut[iQuad * 3 + isigma];
                excCorrPotentialTimesRho +=
                  VxcGrad * basisOperationsPtr
                              ->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
              }
            if (isTauMGGA)
              {
                double VxcTauContribution =
                  pdexTauInSpinUp[iQuad] + pdecTauInSpinUp[iQuad];
                excCorrPotentialTimesRho +=
                  VxcTauContribution *
                  ((tauOutValues[0][iCell * nQuadsPerCell + iQuad] +
                    tauOutValues[1][iCell * nQuadsPerCell + iQuad]) /
                   2.0) *
                  basisOperationsPtr
                    ->JxWBasisData()[iCell * nQuadsPerCell + iQuad];

                VxcTauContribution =
                  pdexTauInSpinDown[iQuad] + pdecTauInSpinDown[iQuad];
                excCorrPotentialTimesRho +=
                  VxcTauContribution *
                  ((tauOutValues[0][iCell * nQuadsPerCell + iQuad] -
                    tauOutValues[1][iCell * nQuadsPerCell + iQuad]) /
                   2.0) *
                  basisOperationsPtr
                    ->JxWBasisData()[iCell * nQuadsPerCell + iQuad];
              } // TauMGGA loop
          }     // iQuad loop
      }         // cell loop
  }


  template <dftfe::utils::MemorySpace memorySpace>
  double
  energyCalculator<memorySpace>::computeEntropicEnergy(
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<std::vector<double>> &partialOccupancies,
    const std::vector<double>              &kPointWeights,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    const bool                              isSpinPolarized,
    const bool                              isConstraintMagnetization,
    const double                            temperature) const
  {
    // computation of entropic term only for one k-pt
    double            entropy = 0.0;
    const dftfe::uInt numEigenValues =
      isSpinPolarized ? eigenValues[0].size() / 2 : eigenValues[0].size();

    for (dftfe::uInt kPoint = 0; kPoint < eigenValues.size(); ++kPoint)
      for (dftfe::Int i = 0; i < numEigenValues; ++i)
        {
          if (isSpinPolarized)
            {
              double partOccSpin0 = partialOccupancies[kPoint][i];
              double partOccSpin1 =
                partialOccupancies[kPoint][i + numEigenValues];

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
              double partialOccupancy = partialOccupancies[kPoint][i];


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
