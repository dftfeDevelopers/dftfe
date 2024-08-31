// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das
//

#include <dft.h>
#include <densityFirstOrderResponseCalculator.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeOutputDensityDirectionalDerivative(
      const distributedCPUVec<double> &v,
      const distributedCPUVec<double> &vSpin0,
      const distributedCPUVec<double> &vSpin1,
      distributedCPUVec<double> &      fv,
      distributedCPUVec<double> &      fvSpin0,
      distributedCPUVec<double> &      fvSpin1)
  {
    computing_timer.enter_subsection("Output density direction derivative");

    KohnShamHamiltonianOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                   dftfe::utils::MemorySpace::DEVICE>::
        copy(d_eigenVectorsFlattenedDevice.size(),
             d_eigenVectorsDensityMatrixPrimeFlattenedDevice.begin(),
             d_eigenVectorsFlattenedDevice.begin());
#endif
    if (!d_dftParamsPtr->useDevice)
      d_eigenVectorsDensityMatrixPrimeHost = d_eigenVectorsFlattenedHost;


    // set up linear solver
    dealiiLinearSolver CGSolver(d_mpiCommParent,
                                mpi_communicator,
                                dealiiLinearSolver::CG);

#ifdef DFTFE_WITH_DEVICE
    // set up linear solver Device
    linearSolverCGDevice CGSolverDevice(d_mpiCommParent,
                                        mpi_communicator,
                                        linearSolverCGDevice::CG,
                                        d_BLASWrapperPtr);
#endif


    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> charge;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;
    std::map<dealii::CellId, std::vector<double>> dummyMap;
    interpolateDensityNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      v,
      charge,
      dummy,
      dummy,
      false,
      false);

    d_phiPrime = 0;

    // Reuses diagonalA and mean value constraints
    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->floatingNuclearCharges and
        d_dftParamsPtr->poissonGPU and not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        d_phiPrimeSolverProblemDevice.reinit(
          d_basisOperationsPtrElectroHost,
          d_phiPrime,
          *d_constraintsVectorElectro[d_phiPrimeDofHandlerIndexElectro],
          d_phiPrimeDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          std::map<dealii::types::global_dof_index, double>(),
          dummyMap,
          d_smearedChargeQuadratureIdElectro,
          charge,
          d_BLASWrapperPtr,
          true,
          d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
            d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC,
          false,
          true,
          false,
          0,
          true,
          false,
          d_dftParamsPtr->multipoleBoundaryConditions);
#endif
      }
    else
      {
        d_phiPrimeSolverProblem.reinit(
          d_basisOperationsPtrElectroHost,
          d_phiPrime,
          *d_constraintsVectorElectro[d_phiPrimeDofHandlerIndexElectro],
          d_phiPrimeDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          std::map<dealii::types::global_dof_index, double>(),
          dummyMap,
          d_smearedChargeQuadratureIdElectro,
          charge,
          true,
          d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
            d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC,
          false,
          true,
          false,
          0,
          true,
          false,
          d_dftParamsPtr->multipoleBoundaryConditions);
      }

    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges and
        not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        CGSolverDevice.solve(d_phiPrimeSolverProblemDevice,
                             d_dftParamsPtr->absPoissonSolverToleranceLRD,
                             d_dftParamsPtr->maxLinearSolverIterations,
                             d_dftParamsPtr->verbosity);
#endif
      }
    else
      {
        CGSolver.solve(d_phiPrimeSolverProblem,
                       d_dftParamsPtr->absPoissonSolverToleranceLRD,
                       d_dftParamsPtr->maxLinearSolverIterations,
                       d_dftParamsPtr->verbosity);
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      electrostaticPotPrimeValues;
    interpolateElectroNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_phiPrimeDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_phiPrime,
      electrostaticPotPrimeValues,
      dummy,
      false);

    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    // interpolate nodal data to quadrature data
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoPrimeValues(2);
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradRhoPrimeValues(2);
    interpolateDensityNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      v,
      rhoPrimeValues[0],
      gradRhoPrimeValues[0],
      dummy,
      isGradDensityDataDependent);


    if (d_dftParamsPtr->spinPolarized == 1)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          vSpin0Values;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          gradvSpin0Values;

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          vSpin1Values;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          gradvSpin1Values;

        interpolateDensityNodalDataToQuadratureDataGeneral(
          d_basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          vSpin0,
          vSpin0Values,
          gradvSpin0Values,
          dummy,
          isGradDensityDataDependent,
          false);

        interpolateDensityNodalDataToQuadratureDataGeneral(
          d_basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          vSpin1,
          vSpin1Values,
          gradvSpin1Values,
          dummy,
          isGradDensityDataDependent,
          false);

        rhoPrimeValues[0].resize(vSpin0Values.size());
        rhoPrimeValues[1].resize(vSpin0Values.size());
        gradRhoPrimeValues[0].resize(gradvSpin0Values.size());
        gradRhoPrimeValues[1].resize(gradvSpin0Values.size());

        auto &rhoTotalPrimeQuadVals = rhoPrimeValues[0];
        auto &rhoMagPrimeQuadVals   = rhoPrimeValues[1];
        for (unsigned int i = 0; i < vSpin0Values.size(); ++i)
          {
            rhoTotalPrimeQuadVals[i] = vSpin0Values[i] + vSpin1Values[i];
            rhoMagPrimeQuadVals[i]   = vSpin0Values[i] - vSpin1Values[i];
          }

        auto &gradRhoTotalPrimeQuadVals = gradRhoPrimeValues[0];
        auto &gradRhoMagPrimeQuadVals   = gradRhoPrimeValues[1];
        for (unsigned int i = 0; i < gradvSpin0Values.size(); ++i)
          {
            gradRhoTotalPrimeQuadVals[i] =
              gradvSpin0Values[i] + gradvSpin1Values[i];
            gradRhoMagPrimeQuadVals[i] =
              gradvSpin0Values[i] - gradvSpin1Values[i];
          }
      }
    else
      {
        rhoPrimeValues[1].clear();
        rhoPrimeValues[1].resize(rhoPrimeValues[0].size(), 0);
        gradRhoPrimeValues[1].clear();
        gradRhoPrimeValues[1].resize(gradRhoPrimeValues[0].size(), 0);
      }

    for (unsigned int s = 0; s < (1 + d_dftParamsPtr->spinPolarized); ++s)
      {
        computing_timer.enter_subsection("VEffPrime Computation");

        updateAuxDensityXCMatrix(d_densityInQuadValues,
                                 d_gradDensityInQuadValues,
                                 d_rhoCore,
                                 d_gradRhoCore,
                                 getEigenVectors(),
                                 eigenValues,
                                 fermiEnergy,
                                 fermiEnergyUp,
                                 fermiEnergyDown,
                                 d_auxDensityMatrixXCInPtr);

        kohnShamDFTEigenOperator.computeVEffPrime(d_auxDensityMatrixXCInPtr,
                                                  rhoPrimeValues,
                                                  gradRhoPrimeValues,
                                                  electrostaticPotPrimeValues,
                                                  s);
        computing_timer.leave_subsection("VEffPrime Computation");

        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          {
            if (kPoint == 0)
              {
                kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);

                computing_timer.enter_subsection(
                  "Hamiltonian matrix prime computation");
                kohnShamDFTEigenOperator.computeCellHamiltonianMatrix(true);

                computing_timer.leave_subsection(
                  "Hamiltonian matrix prime computation");
              }

#ifdef DFTFE_WITH_DEVICE
            if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
              kohnShamEigenSpaceFirstOrderDensityMatResponse(
                s,
                kPoint,
                kohnShamDFTEigenOperator,
                *d_elpaScala,
                d_subspaceIterationSolverDevice);
#endif
            if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
              kohnShamEigenSpaceFirstOrderDensityMatResponse(
                s, kPoint, kohnShamDFTEigenOperator, *d_elpaScala);
          }
      }

    computing_timer.enter_subsection(
      "Density first order response computation");

    computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(fv, fvSpin0, fvSpin1);

    computing_timer.leave_subsection(
      "Density first order response computation");



    computing_timer.leave_subsection("Output density direction derivative");
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
      distributedCPUVec<double> &fv,
      distributedCPUVec<double> &fvSpin0,
      distributedCPUVec<double> &fvSpin1)
  {
    distributedCPUVec<double> fvHam, fvFermiEnergy;
    fvHam.reinit(fv);
    fvFermiEnergy.reinit(fv);
    fvHam         = 0;
    fvFermiEnergy = 0;

    distributedCPUVec<double> fvHamSpin0, fvHamSpin1, fvFermiEnergySpin0,
      fvFermiEnergySpin1;

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        fvHamSpin0.reinit(fv);
        fvHamSpin1.reinit(fv);
        fvFermiEnergySpin0.reinit(fv);
        fvFermiEnergySpin1.reinit(fv);
        fvHamSpin0         = 0;
        fvHamSpin1         = 0;
        fvFermiEnergySpin0 = 0;
        fvFermiEnergySpin1 = 0;
      }

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoResponseHamPRefinedNodalData;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoResponseFermiEnergyPRefinedNodalData;


    // initialize variables to be used later
    d_basisOperationsPtrHost->reinit(0, 0, d_gllQuadratureId, false);
    const unsigned int numLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const unsigned int dofs_per_cell =
      d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    const dealii::IndexSet &locallyOwnedDofs =
      d_dofHandlerRhoNodal.locally_owned_dofs();
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_gllQuadratureId);
    const unsigned int numQuadPoints = quadrature_formula.size();

    // get access to quadrature point coordinates and density DoFHandler nodal
    // points
    const std::vector<dealii::Point<3>> &quadraturePointCoor =
      quadrature_formula.get_points();
    const std::vector<dealii::Point<3>> &supportPointNaturalCoor =
      d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
    std::vector<unsigned int> renumberingMap(numQuadPoints);

    // create renumbering map between the numbering order of quadrature points
    // and lobatto support points
    for (unsigned int i = 0; i < numQuadPoints; ++i)
      {
        const dealii::Point<3> &nodalCoor = supportPointNaturalCoor[i];
        for (unsigned int j = 0; j < numQuadPoints; ++j)
          {
            const dealii::Point<3> &quadCoor = quadraturePointCoor[j];
            double                  dist     = quadCoor.distance(nodalCoor);
            if (dist <= 1e-08)
              {
                renumberingMap[i] = j;
                break;
              }
          }
      }

    // allocate the storage to compute 2p nodal values from wavefunctions

    rhoResponseHamPRefinedNodalData.resize(
      d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    rhoResponseFermiEnergyPRefinedNodalData.resize(
      d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);

    for (unsigned int iComp = 0; iComp < rhoResponseHamPRefinedNodalData.size();
         ++iComp)
      {
        rhoResponseHamPRefinedNodalData[iComp].resize(numLocallyOwnedCells *
                                                        numQuadPoints,
                                                      0);
        rhoResponseFermiEnergyPRefinedNodalData[iComp].resize(
          numLocallyOwnedCells * numQuadPoints, 0);
      }


      // compute first order density response at nodal locations of 2p
      // DoFHandler nodes in each cell
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        computeRhoFirstOrderResponse(
          d_eigenVectorsFlattenedDevice,
          d_eigenVectorsDensityMatrixPrimeFlattenedDevice,
          d_numEigenValues,
          d_densityMatDerFermiEnergy,
          d_basisOperationsPtrDevice,
          d_BLASWrapperPtr,
          d_densityDofHandlerIndex,
          d_gllQuadratureId,
          d_kPointWeights,
          rhoResponseHamPRefinedNodalData,
          rhoResponseFermiEnergyPRefinedNodalData,
          d_mpiCommParent,
          interpoolcomm,
          interBandGroupComm,
          *d_dftParamsPtr);
      }
#endif
    if (!d_dftParamsPtr->useDevice)
      {
        computeRhoFirstOrderResponse(d_eigenVectorsFlattenedHost,
                                     d_eigenVectorsDensityMatrixPrimeHost,
                                     d_numEigenValues,
                                     d_densityMatDerFermiEnergy,
                                     d_basisOperationsPtrHost,
                                     d_BLASWrapperPtrHost,
                                     d_densityDofHandlerIndex,
                                     d_gllQuadratureId,
                                     d_kPointWeights,
                                     rhoResponseHamPRefinedNodalData,
                                     rhoResponseFermiEnergyPRefinedNodalData,
                                     d_mpiCommParent,
                                     interpoolcomm,
                                     interBandGroupComm,
                                     *d_dftParamsPtr);
      }

    // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
    dealii::DoFHandler<3>::active_cell_iterator cellP = d_dofHandlerRhoNodal
                                                          .begin_active(),
                                                endcP =
                                                  d_dofHandlerRhoNodal.end();

    unsigned int iCell = 0;
    for (; cellP != endcP; ++cellP)
      if (cellP->is_locally_owned())
        {
          std::vector<dealii::types::global_dof_index> cell_dof_indices(
            dofs_per_cell);
          cellP->get_dof_indices(cell_dof_indices);
          const double *nodalValuesResponseHam =
            rhoResponseHamPRefinedNodalData[0].data() + iCell * dofs_per_cell;


          const double *nodalValuesResponseFermiEnergy =
            rhoResponseFermiEnergyPRefinedNodalData[0].data() +
            iCell * dofs_per_cell;

          for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
            {
              const dealii::types::global_dof_index nodeID =
                cell_dof_indices[iNode];
              if (!d_constraintsRhoNodal.is_constrained(nodeID))
                {
                  if (locallyOwnedDofs.is_element(nodeID))
                    {
                      fvHam(nodeID) =
                        nodalValuesResponseHam[renumberingMap[iNode]];
                      fvFermiEnergy(nodeID) =
                        nodalValuesResponseFermiEnergy[renumberingMap[iNode]];
                    }
                }
            }
          iCell++;
        }

    const double firstOrderResponseFermiEnergy =
      -totalCharge(d_matrixFreeDataPRefined, fvHam) /
      totalCharge(d_matrixFreeDataPRefined, fvFermiEnergy);

    for (unsigned int i = 0; i < fv.locally_owned_size(); i++)
      fv.local_element(i) =
        fvHam.local_element(i) +
        firstOrderResponseFermiEnergy * fvFermiEnergy.local_element(i);

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
        cellP = d_dofHandlerRhoNodal.begin_active();
        endcP = d_dofHandlerRhoNodal.end();

        iCell = 0;
        for (; cellP != endcP; ++cellP)
          if (cellP->is_locally_owned())
            {
              std::vector<dealii::types::global_dof_index> cell_dof_indices(
                dofs_per_cell);
              cellP->get_dof_indices(cell_dof_indices);
              const double *nodalValuesRhoTotResponseHam =
                rhoResponseHamPRefinedNodalData[0].data() +
                iCell * dofs_per_cell;

              const double *nodalValuesRhoTotResponseFermiEnergy =
                rhoResponseFermiEnergyPRefinedNodalData[0].data() +
                iCell * dofs_per_cell;

              const double *nodalValuesRhoMagResponseHam =
                rhoResponseHamPRefinedNodalData[1].data() +
                iCell * dofs_per_cell;

              const double *nodalValuesRhoMagResponseFermiEnergy =
                rhoResponseFermiEnergyPRefinedNodalData[1].data() +
                iCell * dofs_per_cell;


              for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
                {
                  const dealii::types::global_dof_index nodeID =
                    cell_dof_indices[iNode];
                  if (!d_constraintsRhoNodal.is_constrained(nodeID))
                    {
                      if (locallyOwnedDofs.is_element(nodeID))
                        {
                          fvHamSpin0(nodeID) =
                            0.5 * (nodalValuesRhoTotResponseHam
                                     [renumberingMap[iNode]] +
                                   nodalValuesRhoMagResponseHam
                                     [renumberingMap[iNode]]);
                          fvHamSpin1(nodeID) =
                            0.5 * (nodalValuesRhoTotResponseHam
                                     [renumberingMap[iNode]] -
                                   nodalValuesRhoMagResponseHam
                                     [renumberingMap[iNode]]);
                          fvFermiEnergySpin0(nodeID) =
                            0.5 * (nodalValuesRhoTotResponseFermiEnergy
                                     [renumberingMap[iNode]] +
                                   nodalValuesRhoMagResponseFermiEnergy
                                     [renumberingMap[iNode]]);
                          fvFermiEnergySpin1(nodeID) =
                            0.5 * (nodalValuesRhoTotResponseFermiEnergy
                                     [renumberingMap[iNode]] -
                                   nodalValuesRhoMagResponseFermiEnergy
                                     [renumberingMap[iNode]]);
                        }
                    }
                }
              iCell++;
            }

        for (unsigned int i = 0; i < fvHamSpin0.locally_owned_size(); i++)
          {
            fvSpin0.local_element(i) = fvHamSpin0.local_element(i) +
                                       firstOrderResponseFermiEnergy *
                                         fvFermiEnergySpin0.local_element(i);
            fvSpin1.local_element(i) = fvHamSpin1.local_element(i) +
                                       firstOrderResponseFermiEnergy *
                                         fvFermiEnergySpin1.local_element(i);
          }
      }
  }
#include "dft.inst.cc"
} // namespace dftfe
