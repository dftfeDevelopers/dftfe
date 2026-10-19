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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//
#include <dftfe/dft.h>
#include <cstdint>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
#include <dftfe/atomTypeSplineTables.h>
#include <dftfe/dftUtils.h>
#include <dftfe/fileReaders.h>
#include <dftfe/vectorUtilities.h>
#include <dftfe/atomicInitKernels.h>
#include <dftfe/feevaluationWrapper.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initAnalyticSmearedChargeData()
  {
    d_smearedChargeMoments.assign(13, 0.0);
    d_smearedChargeMomentsComputed = false;
    d_analyticSmearedChargeManager.initialize(
      atomLocations,
      d_imageIdsTrunc,
      d_imageChargesTrunc,
      d_imagePositionsTrunc,
      d_meshSizes,
      d_domainBoundingVectors,
      d_minDist,
      d_pspCutOffTrunc,
      d_oncvClassPtr,
      d_dofHandlerPRefined,
      d_basisOperationsPtrElectroHost,
      d_smearedChargeQuadratureIdElectro,
      mpi_communicator,
      *d_dftParamsPtr,
      pcout);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeNuclearSelfPotential()
  {
    if (d_dftParamsPtr->smearedNuclearChargePathway ==
        "ANALYTIC_SMEARED_CHARGE")
      {
        initAnalyticSmearedChargeData();
        return;
      }

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice && d_dftParamsPtr->vselfGPU)
      d_vselfBinsManager.solveVselfInBinsDevice(
        d_basisOperationsPtrElectroHost,
        d_baseDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_binsStartDofHandlerIndexElectro,
        d_dftParamsPtr->finiteElementPolynomialOrder ==
            d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics ?
          d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
          d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
        d_BLASWrapperPtr,
        d_constraintsPRefined,
        d_imagePositionsTrunc,
        d_imageIdsTrunc,
        d_imageChargesTrunc,
        d_localVselfs,
        d_bQuadValuesAllAtoms,
        d_bQuadAtomIdsAllAtoms,
        d_bQuadAtomIdsAllAtomsImages,
        d_bCellNonTrivialAtomIds,
        d_bCellNonTrivialAtomIdsBins,
        d_bCellNonTrivialAtomImageIds,
        d_bCellNonTrivialAtomImageIdsBins,
        d_smearedChargeWidths,
        d_smearedChargeScaling,
        d_smearedChargeQuadratureIdElectro,
        d_dftParamsPtr->smearedNuclearCharges);
    else
#endif
      d_vselfBinsManager.solveVselfInBins(
        d_basisOperationsPtrElectroHost,
        d_binsStartDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_constraintsPRefined,
        d_imagePositionsTrunc,
        d_imageIdsTrunc,
        d_imageChargesTrunc,
        d_localVselfs,
        d_bQuadValuesAllAtoms,
        d_bQuadAtomIdsAllAtoms,
        d_bQuadAtomIdsAllAtomsImages,
        d_bCellNonTrivialAtomIds,
        d_bCellNonTrivialAtomIdsBins,
        d_bCellNonTrivialAtomImageIds,
        d_bCellNonTrivialAtomImageIdsBins,
        d_smearedChargeWidths,
        d_smearedChargeScaling,
        d_smearedChargeQuadratureIdElectro,
        d_dftParamsPtr->smearedNuclearCharges);
  }

  //
  // Initialize rho by reading in single-atom electron-density and fit a spline
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initLocalPseudoPotential(
    const dealii::DoFHandler<3>             &_dofHandler,
    const dftfe::uInt                        lpspQuadratureId,
    const dealii::MatrixFree<3, double>     &_matrix_free_data,
    const dftfe::uInt                        _phiExtDofHandlerIndex,
    const dealii::AffineConstraints<double> &_phiExtConstraintMatrix,
    const std::map<dealii::types::global_dof_index, dealii::Point<3>>
                                                  &_supportPoints,
    const vselfBinsManager                        &vselfBinManager,
    distributedCPUVec<double>                     &phiExt,
    std::map<dealii::CellId, std::vector<double>> &_pseudoValues,
    std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      &_pseudoValuesAtoms)
  {
    _pseudoValues.clear();
    _pseudoValuesAtoms.clear();

    //
    // Reading single atom rho initial guess
    //
    std::map<dftfe::uInt, double> outerMostDataPoint;
    // Larger max allowed Tail is important for pseudo-dojo database ONCV
    // pseudopotential local potentials which have a larger data range
    // with slow convergence to -Z/r
    // Same value of 10.0 used as rcut in QUANTUM ESPRESSO
    // (cf. Modules/read_pseudo.f90)
    const double maxAllowedTail =
      d_dftParamsPtr->reproducible_output ? 8.0001 : 10.0001;
    double maxTail = 0.0;
    if (d_dftParamsPtr->isPseudopotential)
      {
        //
        // loop over atom types
        //
        for (std::set<dftfe::uInt>::iterator it = atomTypes.begin();
             it != atomTypes.end();
             it++)
          {
            outerMostDataPoint[*it] = d_oncvClassPtr->getRmaxLocalPot(*it);
            if (outerMostDataPoint[*it] > maxTail)
              maxTail = outerMostDataPoint[*it];
          }
      }
    else
      {
        maxTail = maxAllowedTail;
        for (std::set<dftfe::uInt>::iterator it = atomTypes.begin();
             it != atomTypes.end();
             it++)
          outerMostDataPoint[*it] = maxAllowedTail;
      }
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "initLocalPSP, max psp tail considered: " << maxTail
            << std::endl;
    const bool analyticSmearedChargeRoute =
      d_dftParamsPtr->smearedNuclearChargePathway == "ANALYTIC_SMEARED_CHARGE";
    const std::vector<double> &smearedChargeWidths =
      analyticSmearedChargeRoute ?
        d_analyticSmearedChargeManager.smearedChargeWidths() :
        d_smearedChargeWidths;
    double maxSmearedChargeWidth = 0.0;
    if (analyticSmearedChargeRoute && !smearedChargeWidths.empty())
      maxSmearedChargeWidth = *std::max_element(smearedChargeWidths.begin(),
                                                smearedChargeWidths.end());
    const double cutOffForPsp =
      analyticSmearedChargeRoute ?
        std::max(maxSmearedChargeWidth + 6.0, maxTail + 2.0) :
        std::max(vselfBinManager.getStoredAdaptiveBallRadius() + 6.0,
                 maxTail + 2.0);

    //
    // Initialize pseudopotential
    //
    const dftfe::uInt n_q_points =
      _matrix_free_data.get_quadrature(lpspQuadratureId).size();

    const dftfe::Int numberGlobalCharges = atomLocations.size();
    //
    // get number of image charges used only for periodic
    //
    const dftfe::Int numberImageCharges = d_imageIds.size();

    // distributedCPUVec<double> phiExt;
    //_matrix_free_data.initialize_dof_vector(phiExt,_phiExtDofHandlerIndex);
    phiExt = 0;

    double init_1;
    MPI_Barrier(d_mpiCommParent);
    init_1 = MPI_Wtime();

    dealii::BoundingBox<3> boundingBoxTria(
      vectorTools::createBoundingBoxTriaLocallyOwned(_dofHandler));
    dealii::Tensor<1, 3, double> tempDisp;
    tempDisp[0] = cutOffForPsp;
    tempDisp[1] = cutOffForPsp;
    tempDisp[2] = cutOffForPsp;

    std::vector<double> atomsImagesPositions(
      (numberGlobalCharges + numberImageCharges) * 3);
    std::vector<double> atomsImagesCharges(
      (numberGlobalCharges + numberImageCharges));
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt iAtom = 0;
         iAtom < numberGlobalCharges + numberImageCharges;
         iAtom++)
      {
        if (iAtom < numberGlobalCharges)
          {
            atomsImagesPositions[iAtom * 3 + 0] = atomLocations[iAtom][2];
            atomsImagesPositions[iAtom * 3 + 1] = atomLocations[iAtom][3];
            atomsImagesPositions[iAtom * 3 + 2] = atomLocations[iAtom][4];
            if (d_dftParamsPtr->isPseudopotential)
              atomsImagesCharges[iAtom] = atomLocations[iAtom][1];
            else
              atomsImagesCharges[iAtom] = atomLocations[iAtom][0];
          }
        else
          {
            const dftfe::uInt iImageCharge = iAtom - numberGlobalCharges;
            atomsImagesPositions[iAtom * 3 + 0] =
              d_imagePositions[iImageCharge][0];
            atomsImagesPositions[iAtom * 3 + 1] =
              d_imagePositions[iImageCharge][1];
            atomsImagesPositions[iAtom * 3 + 2] =
              d_imagePositions[iImageCharge][2];
            if (d_dftParamsPtr->isPseudopotential)
              atomsImagesCharges[iAtom] =
                atomLocations[d_imageIds[iImageCharge]][1];
            else
              atomsImagesCharges[iAtom] =
                atomLocations[d_imageIds[iImageCharge]][0];
          }
      }

    for (dftfe::uInt iCell = 0;
         iCell < d_basisOperationsPtrElectroHost->nCells();
         ++iCell)
      {
        std::vector<double> &pseudoVLoc =
          _pseudoValues[d_basisOperationsPtrElectroHost->cellID(iCell)];
        pseudoVLoc.resize(n_q_points, 0.0);
      }

    const dftfe::Int numberDofs = phiExt.locally_owned_size();
    // kpoint group parallelization data structures
    const dftfe::uInt numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(interpoolcomm);

    const dftfe::uInt kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    if (!analyticSmearedChargeRoute)
      {
        const std::vector<std::map<dealii::types::global_dof_index, dftfe::Int>>
          &boundaryNodeMapBinsOnlyChargeId =
            vselfBinManager.getBoundaryFlagsBinsOnlyChargeId();
        const std::vector<
          std::map<dealii::types::global_dof_index, dealii::Point<3>>>
          &dofClosestChargeLocationMapBins =
            vselfBinManager.getClosestAtomLocationsBins();
        const std::map<dftfe::uInt, dftfe::uInt> &atomIdBinIdMap =
          vselfBinManager.getAtomIdBinIdMapLocalAllImages();
        const auto             &partitioner = phiExt.get_partitioner();
        std::vector<dftfe::Int> kptGroupLowHighPlusOneIndicesStep1;

        if (numberDofs > 0)
          dftUtils::createKpointParallelizationIndices(
            interpoolcomm, numberDofs, kptGroupLowHighPlusOneIndicesStep1);

#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt localDofId = 0;
             localDofId < phiExt.locally_owned_size();
             ++localDofId)
          {
            if (localDofId <
                  kptGroupLowHighPlusOneIndicesStep1[2 * kptGroupTaskId + 1] &&
                localDofId >=
                  kptGroupLowHighPlusOneIndicesStep1[2 * kptGroupTaskId])
              {
                const dealii::types::global_dof_index dofId =
                  partitioner->local_to_global(localDofId);
                const dealii::Point<3> &nodalCoor =
                  _supportPoints.find(dofId)->second;
                if (!_phiExtConstraintMatrix.is_constrained(dofId))
                  {
                    dealii::Point<3> atom;
                    double           atomCharge;
                    dftfe::Int       chargeId;
                    double           distanceToAtom;
                    double           sumVal = 0.0;
                    double           val;
                    double           diffx;
                    double           diffy;
                    double           diffz;
                    for (dftfe::uInt iAtom = 0;
                         iAtom < (atomLocations.size() + numberImageCharges);
                         ++iAtom)
                      {
                        diffx =
                          nodalCoor[0] - atomsImagesPositions[iAtom * 3 + 0];
                        diffy =
                          nodalCoor[1] - atomsImagesPositions[iAtom * 3 + 1];
                        diffz =
                          nodalCoor[2] - atomsImagesPositions[iAtom * 3 + 2];
                        atomCharge = atomsImagesCharges[iAtom];

                        distanceToAtom = std::sqrt(
                          diffx * diffx + diffy * diffy + diffz * diffz);

                        if (distanceToAtom < cutOffForPsp)
                          {
                            if (iAtom < numberGlobalCharges)
                              {
                                chargeId = iAtom;
                              }
                            else
                              {
                                const dftfe::uInt iImageCharge =
                                  iAtom - numberGlobalCharges;
                                chargeId = d_imageIds[iImageCharge];
                              }

                            if (atomIdBinIdMap.find(chargeId) !=
                                atomIdBinIdMap.end())
                              {
                                const dftfe::uInt binId =
                                  atomIdBinIdMap.find(chargeId)->second;
                                const dftfe::Int boundaryFlagChargeId =
                                  boundaryNodeMapBinsOnlyChargeId[binId]
                                    .find(dofId)
                                    ->second;

                                if (boundaryFlagChargeId == chargeId)
                                  {
                                    atom[0] =
                                      atomsImagesPositions[iAtom * 3 + 0];
                                    atom[1] =
                                      atomsImagesPositions[iAtom * 3 + 1];
                                    atom[2] =
                                      atomsImagesPositions[iAtom * 3 + 2];

                                    if (dofClosestChargeLocationMapBins[binId]
                                          .find(dofId)
                                          ->second.distance(atom) < 1e-5)
                                      {
                                        const distributedCPUVec<double>
                                          &vselfBin =
                                            vselfBinManager
                                              .getVselfFieldBins()[binId];
                                        val =
                                          vselfBin.local_element(localDofId);
                                      }
                                    else
                                      val = -atomCharge / distanceToAtom;
                                  }
                                else
                                  val = -atomCharge / distanceToAtom;
                              }
                          }
                        else
                          {
                            val = -atomCharge / distanceToAtom;
                          }

                        sumVal += val;
                      }
                    phiExt.local_element(localDofId) = sumVal;
                  }
              } // interpool comm parallelization
          }     // dof loop

        if (numberDofs > 0 && numberKptGroups > 1)
          MPI_Allreduce(MPI_IN_PLACE,
                        phiExt.begin(),
                        numberDofs,
                        MPI_DOUBLE,
                        MPI_SUM,
                        interpoolcomm);
        MPI_Barrier(interpoolcomm);
        phiExt.update_ghost_values();
        d_basisOperationsPtrElectroHost
          ->d_constraintInfo[d_phiExtDofHandlerIndexElectro]
          .distribute(phiExt);
      }

    MPI_Barrier(d_mpiCommParent);
    init_1 = MPI_Wtime() - init_1;
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "initLocalPSP: Time taken for init1: " << init_1 << std::endl;

    double init_2;
    MPI_Barrier(d_mpiCommParent);
    init_2 = MPI_Wtime();

    const dftfe::Int numMacroCells = _matrix_free_data.n_cell_batches();

    std::vector<dftfe::Int> kptGroupLowHighPlusOneIndicesStep2;

    if (numMacroCells > 0)
      dftUtils::createKpointParallelizationIndices(
        interpoolcomm, numMacroCells, kptGroupLowHighPlusOneIndicesStep2);
    d_basisOperationsPtrElectroHost->reinit(0, 0, lpspQuadratureId);

    // per-atom quantities, independent of the quadrature point
    std::vector<dftfe::Int>  atomsImagesAtomicNumbers(numberGlobalCharges +
                                                     numberImageCharges);
    std::vector<dftfe::uInt> atomsImagesChargeIds(numberGlobalCharges +
                                                  numberImageCharges);
    std::vector<double>      atomsImagesTailCutoffs(numberGlobalCharges +
                                               numberImageCharges);
    std::vector<double>      atomsImagesSmearedWidths(numberGlobalCharges +
                                                   numberImageCharges,
                                                 0.0);
    for (dftfe::uInt iAtom = 0;
         iAtom < numberGlobalCharges + numberImageCharges;
         ++iAtom)
      {
        const dftfe::uInt chargeId  = iAtom < numberGlobalCharges ?
                                        iAtom :
                                        d_imageIds[iAtom - numberGlobalCharges];
        atomsImagesChargeIds[iAtom] = chargeId;
        atomsImagesAtomicNumbers[iAtom] =
          std::round(atomLocations[chargeId][0]);
        atomsImagesTailCutoffs[iAtom] =
          outerMostDataPoint[atomsImagesAtomicNumbers[iAtom]];
        if (analyticSmearedChargeRoute)
          atomsImagesSmearedWidths[iAtom] = smearedChargeWidths[chargeId];
      }
    // A pseudopotential run evaluates steps 2 and 3 through
    // atomicInitKernels from these spline tables; an all-electron run
    // evaluates the bare Coulomb potential on the host.
    const bool isPseudopotentialRun = d_dftParamsPtr->isPseudopotential;
    std::map<dftfe::uInt, dftfe::uInt> lpspZnumToSplineId;
    AtomTypeSplineTables<memorySpace>  lpspTablesMem;
    if (isPseudopotentialRun)
      {
        splineHostTableUtils::SplineTableGroup lpspTables;
        buildAtomTypeSplineTables(
          atomTypes,
          [this](const dftfe::uInt Znum) {
            return d_oncvClassPtr->getRadialLocalPseudoFunction(Znum, 0);
          },
          lpspTables,
          lpspZnumToSplineId);
        lpspTablesMem.stage(lpspTables);
        computeLocalPseudoPotentialTotal(lpspTablesMem,
                                         lpspZnumToSplineId,
                                         atomsImagesPositions,
                                         atomsImagesCharges,
                                         atomsImagesAtomicNumbers,
                                         atomsImagesTailCutoffs,
                                         atomsImagesSmearedWidths,
                                         analyticSmearedChargeRoute,
                                         maxTail,
                                         _pseudoValues);
      }
    else
      {
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt macrocell = 0;
             macrocell < _matrix_free_data.n_cell_batches();
             ++macrocell)
          {
            if (macrocell <
                  kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId + 1] &&
                macrocell >=
                  kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId])
              {
                double atomCharge;

                for (dftfe::uInt iSubCell = 0;
                     iSubCell <
                     _matrix_free_data.n_active_entries_per_cell_batch(
                       macrocell);
                     ++iSubCell)
                  {
                    dealii::DoFHandler<3>::active_cell_iterator subCellPtr =
                      _matrix_free_data.get_cell_iterator(
                        macrocell, iSubCell, _phiExtDofHandlerIndex);
                    dealii::CellId subCellId = subCellPtr->id();

                    std::vector<double> &pseudoVLoc = _pseudoValues[subCellId];
                    dftfe::uInt          cellIndex =
                      d_basisOperationsPtrElectroHost->cellIndex(subCellId);
                    double        value, distanceToAtom, distanceToAtomInv;
                    const double *quadPointPtr =
                      d_basisOperationsPtrElectroHost->quadPoints().data() +
                      cellIndex * n_q_points * 3;

                    // loop over quad points
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        const dealii::Point<3> quadPoint(
                          quadPointPtr[q * 3],
                          quadPointPtr[q * 3 + 1],
                          quadPointPtr[q * 3 + 2]);

                        double tempVal = 0.0;
                        double diffx;
                        double diffy;
                        double diffz;
                        // loop over atoms
                        for (dftfe::uInt iAtom = 0;
                             iAtom < numberGlobalCharges + numberImageCharges;
                             iAtom++)
                          {
                            diffx = quadPoint[0] -
                                    atomsImagesPositions[iAtom * 3 + 0];
                            diffy = quadPoint[1] -
                                    atomsImagesPositions[iAtom * 3 + 1];
                            diffz = quadPoint[2] -
                                    atomsImagesPositions[iAtom * 3 + 2];

                            atomCharge = atomsImagesCharges[iAtom];

                            distanceToAtom = std::sqrt(
                              diffx * diffx + diffy * diffy + diffz * diffz);
                            distanceToAtomInv = 1.0 / distanceToAtom;

                            value = -atomCharge * distanceToAtomInv;

                            if (analyticSmearedChargeRoute)
                              value -= distanceToAtom >
                                           atomsImagesSmearedWidths[iAtom] ?
                                         -atomCharge / distanceToAtom :
                                         -atomCharge *
                                           atomicInitKernels::smearedPot(
                                             distanceToAtom,
                                             atomsImagesSmearedWidths[iAtom]);

                            tempVal += value;
                          } // atom loop
                        pseudoVLoc[q] = tempVal;
                      } // quad loop
                  }     // subcell loop
              }         // intercomm paral
          }             // cell loop
      }

    if (!analyticSmearedChargeRoute)
      {
        FEEvaluationWrapperClass<1> feEvalObj(_matrix_free_data,
                                              _phiExtDofHandlerIndex,
                                              lpspQuadratureId);
        AssertThrow(
          _matrix_free_data.get_quadrature(lpspQuadratureId).size() ==
            feEvalObj.n_q_points,
          dealii::ExcMessage(
            "DFT-FE Error: mismatch in quadrature rule usage in initLocalPseudoPotential."));

        for (dftfe::uInt macrocell = 0;
             macrocell < _matrix_free_data.n_cell_batches();
             ++macrocell)
          {
            if (macrocell <
                  kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId + 1] &&
                macrocell >=
                  kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId])
              {
                feEvalObj.reinit(macrocell);
                feEvalObj.read_dof_values(phiExt);
                feEvalObj.evaluate(dealii::EvaluationFlags::values);
                for (dftfe::uInt iSubCell = 0;
                     iSubCell <
                     _matrix_free_data.n_active_entries_per_cell_batch(
                       macrocell);
                     ++iSubCell)
                  {
                    dealii::DoFHandler<3>::active_cell_iterator subCellPtr =
                      _matrix_free_data.get_cell_iterator(
                        macrocell, iSubCell, _phiExtDofHandlerIndex);
                    dealii::CellId       subCellId  = subCellPtr->id();
                    std::vector<double> &pseudoVLoc = _pseudoValues[subCellId];
                    // loop over quad points
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        pseudoVLoc[q] -= feEvalObj.get_value(q)[iSubCell];
                      } // loop over quad points
                  }     // subcell loop
              }
          }
      }
    if (numMacroCells > 0 && numberKptGroups > 1)
      {
        std::vector<double> tempPseudoValuesFlattened(
          d_basisOperationsPtrElectroHost->nCells() * n_q_points, 0.0);

#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0;
             iCell < d_basisOperationsPtrElectroHost->nCells();
             ++iCell)
          {
            std::vector<double> &pseudoVLoc =
              _pseudoValues[d_basisOperationsPtrElectroHost->cellID(iCell)];
            for (dftfe::uInt q = 0; q < n_q_points; ++q)
              tempPseudoValuesFlattened[iCell * n_q_points + q] = pseudoVLoc[q];
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &tempPseudoValuesFlattened[0],
                      d_basisOperationsPtrElectroHost->nCells() * n_q_points,
                      MPI_DOUBLE,
                      MPI_SUM,
                      interpoolcomm);
        MPI_Barrier(interpoolcomm);

#pragma omp parallel for num_threads(d_nOMPThreads)
        for (dftfe::uInt iCell = 0;
             iCell < d_basisOperationsPtrElectroHost->nCells();
             ++iCell)
          {
            std::vector<double> &pseudoVLoc =
              _pseudoValues[d_basisOperationsPtrElectroHost->cellID(iCell)];
            for (dftfe::uInt q = 0; q < n_q_points; ++q)
              pseudoVLoc[q] = tempPseudoValuesFlattened[iCell * n_q_points + q];
          }
      }


    MPI_Barrier(d_mpiCommParent);
    init_2 = MPI_Wtime() - init_2;
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "initLocalPSP: Time taken for init2: " << init_2 << std::endl;

    double init_3;
    MPI_Barrier(d_mpiCommParent);
    init_3 = MPI_Wtime();

    std::vector<dftfe::Int> kptGroupLowHighPlusOneIndicesStep3;

    if (d_basisOperationsPtrElectroHost->nCells() > 0)
      dftUtils::createKpointParallelizationIndices(
        interpoolcomm,
        d_basisOperationsPtrElectroHost->nCells(),
        kptGroupLowHighPlusOneIndicesStep3);

    // Per-cell bounding spheres over the quadrature points, computed once and
    // reused for every atom, for the prefilter of step 3. The exact
    // storage criterion is inclusive, so the prefilter keeps the
    // cellBoundingSphereTolerance margin.
    std::vector<dealii::Point<3>> cellCentroids;
    std::vector<double>           cellRadii;
    dftUtils::computeCellBoundingSpheres(
      d_basisOperationsPtrElectroHost->quadPoints().data(),
      d_basisOperationsPtrElectroHost->nCells(),
      n_q_points,
      cellCentroids,
      cellRadii);

    // _pseudoValuesAtoms is only read by the force and stress computation
    const bool storePerAtomPseudoData =
      d_dftParamsPtr->isIonForce || d_dftParamsPtr->isCellStress;

    std::vector<double> pseudoVLocAtom(n_q_points);
    if (storePerAtomPseudoData && isPseudopotentialRun)
      computeLocalPseudoPotentialPerAtom(lpspTablesMem,
                                         lpspZnumToSplineId,
                                         cutOffForPsp,
                                         boundingBoxTria,
                                         cellCentroids,
                                         cellRadii,
                                         kptGroupLowHighPlusOneIndicesStep3,
                                         _pseudoValuesAtoms);
    else if (storePerAtomPseudoData)
      {
#pragma omp parallel for num_threads(d_nOMPThreads) firstprivate(pseudoVLocAtom)
        for (dftfe::uInt iCell = 0;
             iCell < d_basisOperationsPtrElectroHost->nCells();
             ++iCell)
          {
            if ((iCell <
                   kptGroupLowHighPlusOneIndicesStep3[2 * kptGroupTaskId + 1] &&
                 iCell >=
                   kptGroupLowHighPlusOneIndicesStep3[2 * kptGroupTaskId]))
              {
                // compute values for the current elements

                dealii::Point<3> atom;
                double           atomCharge;
                const double    *quadPointPtr =
                  d_basisOperationsPtrElectroHost->quadPoints().data() +
                  iCell * n_q_points * 3;

                // loop over atoms
                for (dftfe::uInt iAtom = 0;
                     iAtom < numberGlobalCharges + d_imagePositionsTrunc.size();
                     iAtom++)
                  {
                    if (iAtom < numberGlobalCharges)
                      {
                        atom[0]    = atomLocations[iAtom][2];
                        atom[1]    = atomLocations[iAtom][3];
                        atom[2]    = atomLocations[iAtom][4];
                        atomCharge = atomLocations[iAtom][0];
                      }
                    else
                      {
                        const dftfe::uInt iImageCharge =
                          iAtom - numberGlobalCharges;
                        atom[0] = d_imagePositionsTrunc[iImageCharge][0];
                        atom[1] = d_imagePositionsTrunc[iImageCharge][1];
                        atom[2] = d_imagePositionsTrunc[iImageCharge][2];
                        atomCharge =
                          atomLocations[d_imageIdsTrunc[iImageCharge]][0];
                      }

                    std::pair<dealii::Point<3, double>,
                              dealii::Point<3, double>>
                      boundaryPoints(atom - tempDisp, atom + tempDisp);

                    dealii::BoundingBox<3> boundingBoxAroundAtom(
                      boundaryPoints);

                    if (boundingBoxTria.get_neighbor_type(
                          boundingBoxAroundAtom) ==
                        dealii::NeighborType::not_neighbors)
                      continue;

                    // nothing is stored unless a quadrature point of this cell
                    // lies within cutOffForPsp of the atom
                    if (cellCentroids[iCell].distance(atom) - cellRadii[iCell] >
                        cutOffForPsp + dftUtils::cellBoundingSphereTolerance)
                      continue;

                    bool   isPseudoDataInCell = false;
                    double distanceToAtom;
                    // loop over quad points
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        const dealii::Point<3> quadPoint(
                          quadPointPtr[q * 3],
                          quadPointPtr[q * 3 + 1],
                          quadPointPtr[q * 3 + 2]);
                        distanceToAtom = quadPoint.distance(atom);

                        if (distanceToAtom <= cutOffForPsp)
                          isPseudoDataInCell = true;

                        pseudoVLocAtom[q] = -atomCharge / distanceToAtom;
                      } // loop over quad points
                    if (isPseudoDataInCell)
                      {
#pragma omp critical(pseudovalsatoms)
                        _pseudoValuesAtoms[iAtom]
                                          [d_basisOperationsPtrElectroHost
                                             ->cellID(iCell)] = pseudoVLocAtom;
                      }
                  } // loop over atoms
              }     // kpt paral loop
          }         // cell loop
      }

    if (d_basisOperationsPtrElectroHost->nCells() > 0 && numberKptGroups > 1 &&
        storePerAtomPseudoData)
      {
        // arranged as iAtom, elemid, and quad data
        std::vector<double> sendData;
        int                 sendCount = 0;
        // loop over atoms
        for (dftfe::uInt iAtom = 0;
             iAtom < numberGlobalCharges + d_imagePositionsTrunc.size();
             iAtom++)
          {
            if (_pseudoValuesAtoms.find(iAtom) != _pseudoValuesAtoms.end())
              {
                for (dftfe::uInt iCell = 0;
                     iCell < d_basisOperationsPtrElectroHost->nCells();
                     ++iCell)
                  {
                    auto cellid =
                      d_basisOperationsPtrElectroHost->cellID(iCell);
                    if (_pseudoValuesAtoms[iAtom].find(cellid) !=
                        _pseudoValuesAtoms[iAtom].end())
                      {
                        sendCount++;
                        pseudoVLocAtom = _pseudoValuesAtoms[iAtom][cellid];
                        sendData.push_back(iAtom);
                        sendData.push_back(iCell);
                        sendData.insert(sendData.end(),
                                        pseudoVLocAtom.begin(),
                                        pseudoVLocAtom.end());
                      }
                  } // cell locally owned loop
              }
          } // iatom loop

        sendCount = sendCount * (2 + n_q_points);

        if (sendCount == 0)
          {
            sendCount = (2 + n_q_points);
            sendData.resize(sendCount, 0);
            sendData[0] = -1;
          }

        std::vector<int> recvCounts(numberKptGroups, 0);
        int              ierr =
          MPI_Allgather(&sendCount,
                        1,
                        dftfe::dataTypes::mpi_type_id(&sendCount),
                        &recvCounts[0],
                        1,
                        dftfe::dataTypes::mpi_type_id(recvCounts.data()),
                        interpoolcomm);

        if (ierr)
          AssertThrow(false,
                      dealii::ExcMessage(
                        "DFT-FE Error: MPI Error in init local psp"));


        const dftfe::Int recvDataSize =
          std::accumulate(recvCounts.begin(), recvCounts.end(), 0);


        std::vector<int> displacements(numberKptGroups, 0);
        int              disp = 0;
        for (dftfe::Int i = 0; i < numberKptGroups; ++i)
          {
            displacements[i] = disp;
            disp += recvCounts[i];
          }

        std::vector<double> recvData(recvDataSize, 0.0);

        ierr = MPI_Allgatherv(&sendData[0],
                              sendCount,
                              MPI_DOUBLE,
                              &recvData[0],
                              &recvCounts[0],
                              &displacements[0],
                              MPI_DOUBLE,
                              interpoolcomm);

        if (ierr)
          AssertThrow(false,
                      dealii::ExcMessage(
                        "DFT-FE Error: MPI Error in init local psp"));


        for (dftfe::uInt i = 0; i < recvDataSize / (2 + n_q_points); i++)
          {
            const dftfe::Int iatom =
              std::round(recvData[i * (2 + n_q_points) + 0]);
            const dftfe::uInt elementId =
              std::round(recvData[i * (2 + n_q_points) + 1]);


            if (iatom != -1)
              {
                const dealii::CellId writeCellId =
                  d_basisOperationsPtrElectroHost->cellID(elementId);
                if (_pseudoValuesAtoms[iatom].find(writeCellId) ==
                    _pseudoValuesAtoms[iatom].end())
                  {
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      pseudoVLocAtom[q] =
                        recvData[i * (2 + n_q_points) + 2 + q];

                    _pseudoValuesAtoms[iatom][writeCellId] = pseudoVLocAtom;
                  }
              }
          }

        MPI_Barrier(interpoolcomm);
      }

    MPI_Barrier(d_mpiCommParent);
    init_3 = MPI_Wtime() - init_3;
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "initLocalPSP: Time taken for init3: " << init_3 << std::endl;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeLocalPseudoPotentialTotal(
    const AtomTypeSplineTables<memorySpace>       &lpspTables,
    const std::map<dftfe::uInt, dftfe::uInt>      &znumToSplineId,
    const std::vector<double>                     &centerCoords,
    const std::vector<double>                     &centerCharges,
    const std::vector<dftfe::Int>                 &centerAtomicNumbers,
    const std::vector<double>                     &centerTailCutoffs,
    const std::vector<double>                     &centerSmearedWidths,
    const bool                                     analyticSmearedChargeRoute,
    const double                                   maxTail,
    std::map<dealii::CellId, std::vector<double>> &pseudoValues)
  {
    // One thread or iteration per (cell, quadrature point) of this k-point
    // group, centers looped inside the kernel; a device run dispatches to
    // the device specialization and a CPU run to the host one.
    const dftfe::uInt n_q_points =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();
    const dftfe::uInt kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const dftfe::uInt        nCenters = centerCharges.size();
    std::vector<dftfe::uInt> centerSplineIdHost(nCenters);
    for (dftfe::uInt iAtom = 0; iAtom < nCenters; ++iAtom)
      centerSplineIdHost[iAtom] = znumToSplineId.at(centerAtomicNumbers[iAtom]);

    const dftfe::utils::MemoryStorage<double, memorySpace>
      dCenterCoords =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(centerCoords),
      dCenterCharges =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(centerCharges),
      dCenterCutoff = dftfe::utils::memoryStorageFromSTL<double, memorySpace>(
        centerTailCutoffs),
      dCenterWidth = dftfe::utils::memoryStorageFromSTL<double, memorySpace>(
        centerSmearedWidths);
    const dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace>
      dCenterSplineId =
        dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
          centerSplineIdHost);
    dftfe::utils::MemoryStorage<double, memorySpace> dQuad, dOut;

    // block over cells so the device transient stays bounded. Each
    // cell stages three coordinates per quadrature point and returns
    // one value, so four doubles per point.
    const dftfe::uInt nCellsLocal = d_basisOperationsPtrElectroHost->nCells();
    const dftfe::uInt cellsBlockSize =
      std::max<dftfe::uInt>(1,
                            atomicInitKernels::maxBlockBytes /
                              (std::size_t)(4 * n_q_points * sizeof(double)));
    std::vector<double> hOut;

    // The caller sums over interpoolcomm, which only needs the k-point
    // group slices to be disjoint and to cover every cell, so the cells
    // are split directly rather than by macrocell.
    std::vector<dftfe::Int> kptGroupCellIndices;
    dftUtils::createKpointParallelizationIndices(interpoolcomm,
                                                 nCellsLocal,
                                                 kptGroupCellIndices);
    const dftfe::Int cellLow = kptGroupCellIndices[2 * kptGroupTaskId];
    const dftfe::Int cellHighPlusOne =
      kptGroupCellIndices[2 * kptGroupTaskId + 1];
    const dftfe::uInt cellBegin =
      cellLow < 0 ? nCellsLocal : (dftfe::uInt)cellLow;
    const dftfe::uInt cellEnd =
      cellHighPlusOne < 0 ? nCellsLocal :
                            std::min(nCellsLocal, (dftfe::uInt)cellHighPlusOne);

    for (dftfe::uInt cellBlockStart = cellBegin; cellBlockStart < cellEnd;
         cellBlockStart += cellsBlockSize)
      {
        const dftfe::uInt nCellsInBlock =
          std::min(cellsBlockSize, cellEnd - cellBlockStart);
        const double *quadPointsBlock =
          d_basisOperationsPtrElectroHost->quadPoints().data() +
          (std::size_t)cellBlockStart * n_q_points * 3;
        const std::size_t nQuadInBlock =
          (std::size_t)nCellsInBlock * n_q_points * 3;
        dQuad.resize(nQuadInBlock);
        dQuad.template copyFrom<dftfe::utils::MemorySpace::HOST>(
          quadPointsBlock);
        dOut.resize((std::size_t)nCellsInBlock * n_q_points);

        atomicInitKernels::localPseudoPotentialTotal<memorySpace>(
          nCellsInBlock,
          n_q_points,
          nCenters,
          analyticSmearedChargeRoute,
          maxTail,
          dQuad.data(),
          dCenterCoords.data(),
          dCenterCharges.data(),
          dCenterCutoff.data(),
          dCenterWidth.data(),
          dCenterSplineId.data(),
          lpspTables.view(),
          dOut.data(),
          d_nOMPThreads);

        hOut.resize((std::size_t)nCellsInBlock * n_q_points);
        dOut.copyTo(hOut);
        for (dftfe::uInt iCell = 0; iCell < nCellsInBlock; ++iCell)
          {
            std::vector<double> &pv =
              pseudoValues[d_basisOperationsPtrElectroHost->cellID(
                cellBlockStart + iCell)];
            std::copy(hOut.begin() + (std::size_t)iCell * n_q_points,
                      hOut.begin() + (std::size_t)(iCell + 1) * n_q_points,
                      pv.begin());
          }
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeLocalPseudoPotentialPerAtom(
    const AtomTypeSplineTables<memorySpace>  &lpspTables,
    const std::map<dftfe::uInt, dftfe::uInt> &znumToSplineId,
    const double                              cutOffForPsp,
    const dealii::BoundingBox<3>             &boundingBoxTria,
    const std::vector<dealii::Point<3>>      &cellCentroids,
    const std::vector<double>                &cellRadii,
    const std::vector<dftfe::Int>            &kptGroupCellIndices,
    std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      &pseudoValuesAtoms)
  {
    // One thread or iteration per (atom-cell pair, quadrature point); a
    // device run dispatches to the device specialization and a CPU run to
    // the host one.
    const dftfe::uInt n_q_points =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();
    const dftfe::uInt kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const dftfe::uInt            numberGlobalCharges = atomLocations.size();
    dealii::Tensor<1, 3, double> tempDisp;
    tempDisp[0] = cutOffForPsp;
    tempDisp[1] = cutOffForPsp;
    tempDisp[2] = cutOffForPsp;

    // blocked over pairs: each pair outputs n_q_points doubles
    const dftfe::uInt pairsBlockSize =
      std::max<dftfe::uInt>(1,
                            atomicInitKernels::maxBlockBytes /
                              (std::size_t)(n_q_points * sizeof(double)));

    std::vector<double>      blockCoords, blockCenters, blockCharges;
    std::vector<dftfe::uInt> blockCellSlot, blockSplineId, blockAtomId,
      blockCellId;
    std::map<dftfe::uInt, dftfe::uInt> cellToSlot;

    dftfe::utils::MemoryStorage<double, memorySpace> dCoords, dCenters,
      dCharges, dValues;
    dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> dCellSlot, dSplineId,
      dHasSupport;
    std::vector<double>      hValues;
    std::vector<dftfe::uInt> hHasSupport;

    auto flushBlock = [&]() {
      const dftfe::uInt nPairs = blockAtomId.size();
      if (nPairs == 0)
        return;
      dCoords =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(blockCoords);
      dCenters =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(blockCenters);
      dCharges =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(blockCharges);
      dCellSlot = dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
        blockCellSlot);
      dSplineId = dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
        blockSplineId);
      dValues.resize((std::size_t)nPairs * n_q_points);
      dHasSupport.resize(nPairs);
      dHasSupport.setValue(0);

      atomicInitKernels::localPseudoPotentialValues<memorySpace>(
        nPairs,
        n_q_points,
        cutOffForPsp,
        dCoords.data(),
        dCenters.data(),
        dCharges.data(),
        dCellSlot.data(),
        dSplineId.data(),
        lpspTables.view(),
        dValues.data(),
        dHasSupport.data(),
        d_nOMPThreads);

      hValues.resize((std::size_t)nPairs * n_q_points);
      dValues.copyTo(hValues);
      hHasSupport.resize(nPairs);
      dHasSupport.copyTo(hHasSupport);

      for (dftfe::uInt iPair = 0; iPair < nPairs; ++iPair)
        if (hHasSupport[iPair] == 1)
          pseudoValuesAtoms[blockAtomId[iPair]][d_basisOperationsPtrElectroHost
                                                  ->cellID(blockCellId[iPair])]
            .assign(hValues.begin() + (std::size_t)iPair * n_q_points,
                    hValues.begin() + (std::size_t)(iPair + 1) * n_q_points);

      blockCoords.clear();
      blockCenters.clear();
      blockCharges.clear();
      blockCellSlot.clear();
      blockSplineId.clear();
      blockAtomId.clear();
      blockCellId.clear();
      cellToSlot.clear();
    };

    for (dftfe::uInt iCell = 0;
         iCell < d_basisOperationsPtrElectroHost->nCells();
         ++iCell)
      {
        if (!(iCell < kptGroupCellIndices[2 * kptGroupTaskId + 1] &&
              iCell >= kptGroupCellIndices[2 * kptGroupTaskId]))
          continue;
        for (dftfe::uInt iAtom = 0;
             iAtom < numberGlobalCharges + d_imagePositionsTrunc.size();
             ++iAtom)
          {
            dealii::Point<3> atom;
            dftfe::Int       atomicNumberLocal;
            double           atomChargeLocal;
            if (iAtom < numberGlobalCharges)
              {
                atom[0]           = atomLocations[iAtom][2];
                atom[1]           = atomLocations[iAtom][3];
                atom[2]           = atomLocations[iAtom][4];
                atomChargeLocal   = atomLocations[iAtom][1];
                atomicNumberLocal = std::round(atomLocations[iAtom][0]);
              }
            else
              {
                const dftfe::uInt iImageCharge = iAtom - numberGlobalCharges;
                atom[0] = d_imagePositionsTrunc[iImageCharge][0];
                atom[1] = d_imagePositionsTrunc[iImageCharge][1];
                atom[2] = d_imagePositionsTrunc[iImageCharge][2];
                atomChargeLocal =
                  atomLocations[d_imageIdsTrunc[iImageCharge]][1];
                atomicNumberLocal =
                  std::round(atomLocations[d_imageIdsTrunc[iImageCharge]][0]);
              }

            // nothing is stored unless the cutOffForPsp sphere of the
            // atom reaches the locally owned domain and this cell
            std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
              boundaryPoints(atom - tempDisp, atom + tempDisp);
            if (boundingBoxTria.get_neighbor_type(dealii::BoundingBox<3>(
                  boundaryPoints)) == dealii::NeighborType::not_neighbors)
              continue;
            if (cellCentroids[iCell].distance(atom) - cellRadii[iCell] >
                cutOffForPsp + dftUtils::cellBoundingSphereTolerance)
              continue;

            if (blockAtomId.size() >= pairsBlockSize)
              flushBlock();

            auto slotIt = cellToSlot.find(iCell);
            if (slotIt == cellToSlot.end())
              {
                const dftfe::uInt slot = cellToSlot.size();
                cellToSlot[iCell]      = slot;
                const double *qp =
                  d_basisOperationsPtrElectroHost->quadPoints().data() +
                  (std::size_t)iCell * n_q_points * 3;
                blockCoords.insert(blockCoords.end(), qp, qp + n_q_points * 3);
                slotIt = cellToSlot.find(iCell);
              }

            blockCenters.push_back(atom[0]);
            blockCenters.push_back(atom[1]);
            blockCenters.push_back(atom[2]);
            blockCharges.push_back(atomChargeLocal);
            blockCellSlot.push_back(slotIt->second);
            blockSplineId.push_back(znumToSplineId.at(atomicNumberLocal));
            blockAtomId.push_back(iAtom);
            blockCellId.push_back(iCell);
          }
      }
    flushBlock();
  }
#include "dft.inst.cc"
} // namespace dftfe
