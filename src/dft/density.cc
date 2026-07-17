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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh, Sambit Das
//

// source file for electron density related computations
#include <dftfe/config.h>
#include <dftfe/dft.h>
#include <dftfe/densityCalculator.h>

namespace dftfe
{
  // calculate electron density
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::compute_rhoOut(const bool isGroundState)
  {
    const bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);
    const dftfe::uInt numDensityComponents =
      d_dftParamsPtr->noncolin ? 4 :
                                 (d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA" ||
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_LDOS" ||
        d_dftParamsPtr->useSymm)
      {
        computeRhoNodalFromPSI();

        // normalize rho
        const double charge =
          totalCharge(d_matrixFreeDataPRefined, d_densityOutNodalValues[0]);


        const double scalingFactor = ((double)numElectrons) / charge;

        // scale nodal vector with scalingFactor
        for (dftfe::uInt iComp = 0; iComp < d_densityOutNodalValues.size();
             ++iComp)
          d_densityOutNodalValues[iComp] *= scalingFactor;

        // interpolate nodal rhoOut data to quadrature data
        for (dftfe::uInt iComp = 0; iComp < d_densityOutNodalValues.size();
             ++iComp)
          d_basisOperationsPtrElectroHost->interpolate(
            d_densityOutNodalValues[iComp],
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_densityOutQuadValues[iComp],
            d_gradDensityOutQuadValues[iComp],
            d_gradDensityOutQuadValues[iComp],
            isGradDensityDataDependent);

        if (isTauMGGA || (d_dftParamsPtr->printKE && isGroundState))
          {
            d_basisOperationsPtrHost->reinit(0,
                                             0,
                                             d_densityQuadratureId,
                                             false);
            const dftfe::uInt nQuadsPerCell =
              d_basisOperationsPtrHost->nQuadsPerCell();
            const dftfe::uInt nCells = d_basisOperationsPtrHost->nCells();
            std::vector<
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>>
              dummy;
            dummy.resize(numDensityComponents);
            d_tauOutQuadValues.resize(numDensityComponents);
            for (dftfe::uInt iComp = 0; iComp < d_tauOutQuadValues.size();
                 ++iComp)
              {
                d_tauOutQuadValues[iComp].resize(nQuadsPerCell * nCells);
              }

#ifdef DFTFE_WITH_DEVICE
            if (d_dftParamsPtr->useDevice)
              computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
                                d_numEigenValues,
                                d_partialOccupancies,
                                d_basisOperationsPtrDevice,
                                d_BLASWrapperPtr,
                                d_densityDofHandlerIndex,
                                d_intermediateDensityQuadratureId,
                                d_densityQuadratureId,
                                d_kPointCoordinates,
                                d_kPointWeights,
                                dummy,
                                dummy,
                                d_tauOutQuadValues,
                                isGradDensityDataDependent ||
                                  (d_dftParamsPtr->printKE && isGroundState),
                                isTauMGGA ||
                                  (d_dftParamsPtr->printKE && isGroundState),
                                d_mpiCommParent,
                                interpoolcomm,
                                interBandGroupComm,
                                *d_dftParamsPtr);
#endif
            if (!d_dftParamsPtr->useDevice)
              computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
                                d_numEigenValues,
                                d_partialOccupancies,
                                d_basisOperationsPtrHost,
                                d_BLASWrapperPtrHost,
                                d_densityDofHandlerIndex,
                                d_intermediateDensityQuadratureId,
                                d_densityQuadratureId,
                                d_kPointCoordinates,
                                d_kPointWeights,
                                dummy,
                                dummy,
                                d_tauOutQuadValues,
                                isGradDensityDataDependent ||
                                  (d_dftParamsPtr->printKE && isGroundState),
                                isTauMGGA ||
                                  (d_dftParamsPtr->printKE && isGroundState),
                                d_mpiCommParent,
                                interpoolcomm,
                                interBandGroupComm,
                                *d_dftParamsPtr);
          }

        if (d_dftParamsPtr->useSymm)
          {
            for (dftfe::uInt iComp = 0; iComp < d_tauOutQuadValues.size();
                 ++iComp)
              {
                l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                        d_constraintsRhoNodal,
                                        d_densityDofHandlerIndexElectro,
                                        d_densityQuadratureIdElectro,
                                        d_tauOutQuadValues[iComp],
                                        d_tauOutNodalValues[iComp]);
                groupSymmetryPtr->symmetrizeScalarFieldFromLocalValues(
                  d_tauOutNodalValues[iComp], d_dofHandlerRhoNodal);
                d_basisOperationsPtrElectroHost->interpolate(
                  d_tauOutNodalValues[iComp],
                  d_densityDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_tauOutQuadValues[iComp],
                  d_tauOutQuadValues[iComp],
                  d_tauOutQuadValues[iComp],
                  false);
              }
          }
        if (d_dftParamsPtr->verbosity >= 3)
          {
            pcout << "Total Charge before scaling: " << charge << std::endl;
            pcout << "Total Charge using nodal Rho out: "
                  << totalCharge(d_matrixFreeDataPRefined,
                                 d_densityOutNodalValues[0])
                  << std::endl;
          }
      }
    else
      {
        d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId, false);
        const dftfe::uInt nQuadsPerCell =
          d_basisOperationsPtrHost->nQuadsPerCell();
        const dftfe::uInt nCells = d_basisOperationsPtrHost->nCells();
        d_densityOutQuadValues.resize(numDensityComponents);
        if (isGradDensityDataDependent ||
            (d_dftParamsPtr->printKE && isGroundState))
          {
            d_gradDensityOutQuadValues.resize(numDensityComponents);
          }
        for (dftfe::uInt iComp = 0; iComp < d_densityOutQuadValues.size();
             ++iComp)
          d_densityOutQuadValues[iComp].resize(nQuadsPerCell * nCells);

        for (dftfe::uInt iComp = 0; iComp < d_gradDensityOutQuadValues.size();
             ++iComp)
          d_gradDensityOutQuadValues[iComp].resize(3 * nQuadsPerCell * nCells);

        if (isTauMGGA || (d_dftParamsPtr->printKE && isGroundState))
          {
            d_tauOutQuadValues.resize(numDensityComponents);
            for (dftfe::uInt iComp = 0; iComp < d_tauOutQuadValues.size();
                 ++iComp)
              {
                d_tauOutQuadValues[iComp].resize(nQuadsPerCell * nCells);
              }
          }

#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice)
          computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
                            d_numEigenValues,
                            d_partialOccupancies,
                            d_basisOperationsPtrDevice,
                            d_BLASWrapperPtr,
                            d_densityDofHandlerIndex,
                            d_intermediateDensityQuadratureId,
                            d_densityQuadratureId,
                            d_kPointCoordinates,
                            d_kPointWeights,
                            d_densityOutQuadValues,
                            d_gradDensityOutQuadValues,
                            d_tauOutQuadValues,
                            isGradDensityDataDependent ||
                              (d_dftParamsPtr->printKE && isGroundState),
                            isTauMGGA ||
                              (d_dftParamsPtr->printKE && isGroundState),
                            d_mpiCommParent,
                            interpoolcomm,
                            interBandGroupComm,
                            *d_dftParamsPtr);
#endif
        if (!d_dftParamsPtr->useDevice)
          computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
                            d_numEigenValues,
                            d_partialOccupancies,
                            d_basisOperationsPtrHost,
                            d_BLASWrapperPtrHost,
                            d_densityDofHandlerIndex,
                            d_intermediateDensityQuadratureId,
                            d_densityQuadratureId,
                            d_kPointCoordinates,
                            d_kPointWeights,
                            d_densityOutQuadValues,
                            d_gradDensityOutQuadValues,
                            d_tauOutQuadValues,
                            isGradDensityDataDependent ||
                              (d_dftParamsPtr->printKE && isGroundState),
                            isTauMGGA ||
                              (d_dftParamsPtr->printKE && isGroundState),
                            d_mpiCommParent,
                            interpoolcomm,
                            interBandGroupComm,
                            *d_dftParamsPtr);
        // normalizeRhoOutQuadValues();

        if (d_dftParamsPtr->computeEnergyEverySCF || isGroundState)
          {
            computeRhoNodalFromPSI();

            // normalize rho
            const double charge =
              totalCharge(d_matrixFreeDataPRefined, d_densityOutNodalValues[0]);


            const double scalingFactor = ((double)numElectrons) / charge;

            // scale nodal vector with scalingFactor
            d_densityOutNodalValues[0] *= scalingFactor;
          }
      }

    if (d_dftParamsPtr->computeEnergyEverySCF || isGroundState)
      {
        d_rhoOutNodalValuesDistributed = d_densityOutNodalValues[0];
        d_rhoOutNodalValuesDistributed.update_ghost_values();
        d_constraintsRhoNodalInfo.distribute(d_rhoOutNodalValuesDistributed);
        d_basisOperationsPtrElectroHost->interpolate(
          d_densityOutNodalValues[0],
          d_densityDofHandlerIndexElectro,
          d_lpspQuadratureIdElectro,
          d_densityTotalOutValuesLpspQuad,
          d_gradDensityTotalOutValuesLpspQuad,
          d_gradDensityTotalOutValuesLpspQuad,
          true);
      }

    if (isGroundState &&
        ((d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
          d_dftParamsPtr->solverMode == "GEOOPT") ||
         (d_dftParamsPtr->extrapolateDensity == 2 &&
          d_dftParamsPtr->solverMode == "MD")) &&
        d_dftParamsPtr->spinPolarized != 1 && !d_dftParamsPtr->noncolin)
      {
        d_rhoOutNodalValuesSplit = d_densityOutNodalValues[0];
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          rhoOutValuesCopy = d_densityOutQuadValues[0];
        const dealii::Quadrature<3> &quadrature_formula =
          matrix_free_data.get_quadrature(d_densityQuadratureId);
        const dftfe::uInt n_q_points = quadrature_formula.size();

        const double charge =
          totalCharge(d_dofHandlerRhoNodal, d_densityOutQuadValues[0]);
        const double scaling = ((double)numElectrons) / charge;

        // scaling rho
        for (dftfe::uInt i = 0; i < rhoOutValuesCopy.size(); ++i)
          rhoOutValuesCopy[i] *= scaling;
        l2ProjectionQuadDensityMinusAtomicDensity(
          d_basisOperationsPtrElectroHost,
          d_constraintsRhoNodal,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          rhoOutValuesCopy,
          d_rhoOutNodalValuesSplit);
      }
  }

  // calculate LDOS (Local Density of States)
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::compute_ldosOut()
  {
    // Map 1p GLL points to 2p nodal mesh
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_ldosNodalValues, d_densityDofHandlerIndexElectro);
    d_ldosNodalValues = 0.0;

    const dealii::IndexSet &locallyOwnedDofs =
      d_dofHandlerRhoNodal.locally_owned_dofs();

    const dftfe::uInt dofs_per_cell =
      d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_gllQuadratureId);
    const dftfe::uInt numQuadPoints = quadrature_formula.size();

    // get access to quadrature point coordinates and density DoFHandler nodal
    // points
    const std::vector<dealii::Point<3>> &quadraturePointCoor =
      quadrature_formula.get_points();
    const std::vector<dealii::Point<3>> &supportPointNaturalCoor =
      d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
    std::vector<dftfe::uInt> renumberingMap(numQuadPoints);

    // create renumbering map between the numbering order of quadrature points
    // and lobatto support points
    for (dftfe::uInt i = 0; i < numQuadPoints; ++i)
      {
        const dealii::Point<3> &nodalCoor = supportPointNaturalCoor[i];
        for (dftfe::uInt j = 0; j < numQuadPoints; ++j)
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

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellP           = d_dofHandlerRhoNodal.begin_active(),
      endcP           = d_dofHandlerRhoNodal.end();
    dftfe::uInt iCell = 0;

    for (; cellP != endcP; ++cellP)
      {
        if (cellP->is_locally_owned())
          {
            std::vector<dealii::types::global_dof_index> cell_dof_indices(
              dofs_per_cell);
            cellP->get_dof_indices(cell_dof_indices);

            const double *nodalValues =
              d_ldosQuadValues.data() + iCell * dofs_per_cell;

            for (dftfe::uInt iNode = 0; iNode < dofs_per_cell; ++iNode)
              {
                const dealii::types::global_dof_index nodeID =
                  cell_dof_indices[iNode];
                if (!d_constraintsRhoNodal.is_constrained(nodeID))
                  {
                    if (locallyOwnedDofs.is_element(nodeID))
                      d_ldosNodalValues(nodeID) =
                        nodalValues[renumberingMap[iNode]];
                  }
              }
            ++iCell;
          }
      }

    d_constraintsRhoNodal.distribute(d_ldosNodalValues);
    d_ldosNodalValues.update_ghost_values();

    if (d_dftParamsPtr->useSymm)
      {
        groupSymmetryPtr->symmetrizeScalarFieldFromLocalValues(
          d_ldosNodalValues, d_dofHandlerRhoNodal);
        d_constraintsRhoNodal.set_zero(d_ldosNodalValues);
        d_ldosNodalValues.zero_out_ghost_values();
      }

    // Interpolate LDOS nodal field to the quadrature used by the Helmholtz
    // operator and its low-rank correction.
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;
    d_basisOperationsPtrElectroHost->interpolate(
      d_ldosNodalValues,
      d_densityDofHandlerIndexElectro,
      d_kerkerAXQuadratureIdElectro,
      d_ldosAxQuadValuesElectro,
      dummy,
      dummy,
      false);
  }



  // rho data reinitilization without remeshing. The rho out of last ground
  // state solve is made the rho in of the new solve
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::noRemeshRhoDataInit()
  {
    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);

    // cleanup of existing rho Out and rho In data
    clearRhoData();
    d_densityInQuadValues = d_densityOutQuadValues;
    if (isGradDensityDataDependent)
      {
        d_gradDensityInQuadValues = d_gradDensityOutQuadValues;
      }
    if (isTauMGGA)
      {
        d_tauInQuadValues = d_tauOutQuadValues;
      }

    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA" ||
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_LDOS" ||
        d_dftParamsPtr->useSymm)
      {
        d_densityInNodalValues = d_densityOutNodalValues;

        // normalize rho
        const double charge =
          totalCharge(d_matrixFreeDataPRefined, d_densityInNodalValues[0]);

        const double scalingFactor = ((double)numElectrons) / charge;

        // scale nodal vector with scalingFactor
        // why this scaling is neeed?
        for (dftfe::uInt iComp = 0; iComp < d_densityInNodalValues.size();
             ++iComp)
          d_densityInNodalValues[iComp] *= scalingFactor;

        for (dftfe::uInt iComp = 0; iComp < d_densityInNodalValues.size();
             ++iComp)
          d_basisOperationsPtrElectroHost->interpolate(
            d_densityInNodalValues[iComp],
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_densityInQuadValues[iComp],
            d_gradDensityInQuadValues[iComp],
            d_gradDensityInQuadValues[iComp],
            isGradDensityDataDependent);

        d_densityOutQuadValues.resize(d_densityInNodalValues.size());
        for (dftfe::uInt iComp = 0; iComp < d_densityOutQuadValues.size();
             ++iComp)
          d_densityOutQuadValues[iComp].resize(
            d_densityInQuadValues[iComp].size());
        if (isGradDensityDataDependent)
          {
            d_gradDensityOutQuadValues.resize(d_gradDensityInQuadValues.size());
            for (dftfe::uInt iComp = 0; iComp < d_densityOutQuadValues.size();
                 ++iComp)
              d_gradDensityOutQuadValues[iComp].resize(
                d_gradDensityInQuadValues[iComp].size());
          }
      }

    // scale quadrature values
    normalizeRhoInQuadValues();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeRhoNodalFromPSI()
  {
    const bool computeLDOS =
      d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_LDOS";
    if (computeLDOS)
      compute_ldosOccupanciesAndTotalDOS();

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityPRefinedNodalData;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradDensityPRefinedNodalData;

    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      dummy;
    // initialize variables to be used later
    const dftfe::uInt dofs_per_cell =
      d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    const dealii::IndexSet &locallyOwnedDofs =
      d_dofHandlerRhoNodal.locally_owned_dofs();
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_gllQuadratureId);
    const dftfe::uInt numQuadPoints = quadrature_formula.size();

    // get access to quadrature point coordinates and density DoFHandler nodal
    // points
    const std::vector<dealii::Point<3>> &quadraturePointCoor =
      quadrature_formula.get_points();
    const std::vector<dealii::Point<3>> &supportPointNaturalCoor =
      d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
    std::vector<dftfe::uInt> renumberingMap(numQuadPoints);

    // create renumbering map between the numbering order of quadrature points
    // and lobatto support points
    for (dftfe::uInt i = 0; i < numQuadPoints; ++i)
      {
        const dealii::Point<3> &nodalCoor = supportPointNaturalCoor[i];
        for (dftfe::uInt j = 0; j < numQuadPoints; ++j)
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
    const dftfe::uInt numDensityComponents =
      d_dftParamsPtr->noncolin ? 4 :
                                 (d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    densityPRefinedNodalData.resize(numDensityComponents);
    dummy.resize(numDensityComponents);

    // compute rho from wavefunctions at nodal locations of 2p DoFHandler
    // nodes in each cell
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
                        d_numEigenValues,
                        d_partialOccupancies,
                        d_basisOperationsPtrDevice,
                        d_BLASWrapperPtr,
                        d_densityDofHandlerIndex,
                        d_intermediateDensityQuadratureId,
                        d_gllQuadratureId,
                        d_kPointCoordinates,
                        d_kPointWeights,
                        densityPRefinedNodalData,
                        gradDensityPRefinedNodalData,
                        dummy,
                        false,
                        false,
                        d_mpiCommParent,
                        interpoolcomm,
                        interBandGroupComm,
                        *d_dftParamsPtr,
                        computeLDOS ? &d_ldosOccupancies : nullptr,
                        computeLDOS ? &d_ldosQuadValues : nullptr);
#endif
    if (!d_dftParamsPtr->useDevice)
      computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
                        d_numEigenValues,
                        d_partialOccupancies,
                        d_basisOperationsPtrHost,
                        d_BLASWrapperPtrHost,
                        d_densityDofHandlerIndex,
                        d_intermediateDensityQuadratureId,
                        d_gllQuadratureId,
                        d_kPointCoordinates,
                        d_kPointWeights,
                        densityPRefinedNodalData,
                        gradDensityPRefinedNodalData,
                        dummy,
                        false,
                        false,
                        d_mpiCommParent,
                        interpoolcomm,
                        interBandGroupComm,
                        *d_dftParamsPtr,
                        computeLDOS ? &d_ldosOccupancies : nullptr,
                        computeLDOS ? &d_ldosQuadValues : nullptr);

    // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
    for (dftfe::uInt iComp = 0; iComp < densityPRefinedNodalData.size();
         ++iComp)
      {
        dealii::DoFHandler<3>::active_cell_iterator
          cellP           = d_dofHandlerRhoNodal.begin_active(),
          endcP           = d_dofHandlerRhoNodal.end();
        dftfe::uInt iCell = 0;
        for (; cellP != endcP; ++cellP)
          {
            if (cellP->is_locally_owned())
              {
                std::vector<dealii::types::global_dof_index> cell_dof_indices(
                  dofs_per_cell);
                cellP->get_dof_indices(cell_dof_indices);
                const double *nodalValues =
                  densityPRefinedNodalData[iComp].data() +
                  iCell * dofs_per_cell;

                for (dftfe::uInt iNode = 0; iNode < dofs_per_cell; ++iNode)
                  {
                    const dealii::types::global_dof_index nodeID =
                      cell_dof_indices[iNode];
                    if (!d_constraintsRhoNodal.is_constrained(nodeID))
                      {
                        if (locallyOwnedDofs.is_element(nodeID))
                          d_densityOutNodalValues[iComp](nodeID) =
                            nodalValues[renumberingMap[iNode]];
                      }
                  }
                ++iCell;
              }
          }
      }
    if (d_dftParamsPtr->useSymm)
      if (d_dftParamsPtr->noncolin)
        {
          for (dftfe::uInt iComp = 0; iComp < d_densityOutNodalValues.size();
               ++iComp)
            {
              d_constraintsRhoNodal.distribute(d_densityOutNodalValues[iComp]);
              d_densityOutNodalValues[iComp].update_ghost_values();
            }
          groupSymmetryPtr->symmetrizeScalarFieldFromLocalValues(
            d_densityOutNodalValues[0], d_dofHandlerRhoNodal);
          groupSymmetryPtr->symmetrizeVectorFieldFromLocalValues(
            d_densityOutNodalValues[3],
            d_densityOutNodalValues[2],
            d_densityOutNodalValues[1],
            d_dofHandlerRhoNodal);
          for (dftfe::uInt iComp = 0; iComp < d_densityOutNodalValues.size();
               ++iComp)
            {
              d_constraintsRhoNodal.set_zero(d_densityOutNodalValues[iComp]);
              d_densityOutNodalValues[iComp].zero_out_ghost_values();
            }
        }
      else
        for (dftfe::uInt iComp = 0; iComp < d_densityOutNodalValues.size();
             ++iComp)
          {
            d_constraintsRhoNodal.distribute(d_densityOutNodalValues[iComp]);
            d_densityOutNodalValues[iComp].update_ghost_values();
            groupSymmetryPtr->symmetrizeScalarFieldFromLocalValues(
              d_densityOutNodalValues[iComp], d_dofHandlerRhoNodal);
            d_constraintsRhoNodal.set_zero(d_densityOutNodalValues[iComp]);
            d_densityOutNodalValues[iComp].zero_out_ghost_values();
          }
  }
#include "dft.inst.cc"

} // namespace dftfe
