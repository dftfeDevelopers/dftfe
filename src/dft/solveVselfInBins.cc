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

#include <constants.h>
#include <constraintMatrixInfo.h>
#include <dealiiLinearSolver.h>
#include <dftUtils.h>
#include <poissonSolverProblemWrapper.h>
#include <feevaluationWrapper.h>
#ifdef DFTFE_WITH_DEVICE
#  include <solveVselfInBinsDevice.h>
#endif

namespace dftfe
{
  namespace
  {
    //
    // compute smeared nuclear charges at quad point values
    //
    void
    smearedNuclearCharges(
      const dealii::DoFHandler<3>         &dofHandlerOfField,
      const dealii::Quadrature<3>         &quadrature_formula,
      const std::vector<dealii::Point<3>> &atomLocations,
      const std::vector<double>           &atomCharges,
      const dftfe::uInt                    numberDomainAtomsInBin,
      const std::vector<dftfe::Int>       &imageIdToDomainAtomIdMapCurrentBin,
      const std::vector<dftfe::Int>       &binAtomIdToGlobalAtomIdMapCurrentBin,
      const MPI_Comm                      &mpi_communicator,
      const std::vector<double>           &rc,
      const std::map<dealii::CellId, std::pair<dealii::Point<3>, double>>
                                                         enclosingBallCells,
      std::map<dealii::CellId, std::vector<double>>     &bQuadValues,
      std::map<dealii::CellId, std::vector<dftfe::Int>> &bQuadAtomIdsAllAtoms,
      std::map<dealii::CellId, std::vector<dftfe::Int>>
        &bQuadAtomIdsAllAtomsImages,
      std::map<dealii::CellId, std::vector<dftfe::uInt>>
        &bCellNonTrivialAtomIdsAllAtoms,
      std::map<dealii::CellId, std::vector<dftfe::uInt>>
        &bCellNonTrivialAtomIdsBin,
      std::map<dealii::CellId, std::vector<dftfe::uInt>>
        &bCellNonTrivialAtomImageIdsAllAtoms,
      std::map<dealii::CellId, std::vector<dftfe::uInt>>
                          &bCellNonTrivialAtomImageIdsBin,
      std::vector<double> &smearedChargeScaling)
    {
      // dealii::FESystem<3>
      // FETemp(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(2)),
      //                           1);
      dealii::FEValues<3> fe_values(dofHandlerOfField.get_fe(),
                                    quadrature_formula,
                                    dealii::update_quadrature_points |
                                      dealii::update_JxW_values);
      const dftfe::uInt   vertices_per_cell =
        dealii::GeometryInfo<3>::vertices_per_cell;
      const dftfe::uInt n_q_points = quadrature_formula.size();

      dealii::DoFHandler<3>::active_cell_iterator cell = dofHandlerOfField
                                                           .begin_active(),
                                                  endc =
                                                    dofHandlerOfField.end();

      const dftfe::uInt   numberTotalAtomsInBin = atomLocations.size();
      std::vector<double> smearedNuclearChargeIntegral(numberTotalAtomsInBin,
                                                       0.0);

      cell = dofHandlerOfField.begin_active();
      std::map<dealii::CellId, std::vector<dftfe::uInt>>
        cellNonTrivialIdsBinMap;
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            std::vector<dftfe::uInt> &nonTrivialIdsBin =
              cellNonTrivialIdsBinMap[cell->id()];
            const std::pair<dealii::Point<3>, double> &enclosingBallCell =
              enclosingBallCells.find(cell->id())->second;
            const dealii::Point<3> &enclosingBallCellCenter =
              enclosingBallCell.first;
            const double enclosingBallCellRadius = enclosingBallCell.second;
            for (dftfe::uInt iatom = 0; iatom < numberTotalAtomsInBin; ++iatom)
              {
                const dealii::Point<3> &atomLocation = atomLocations[iatom];
                const double            distFromCellCenter =
                  (enclosingBallCellCenter - atomLocation).norm();

                if (distFromCellCenter > 10.0)
                  continue;

                const dftfe::uInt atomId =
                  iatom < numberDomainAtomsInBin ?
                    iatom :
                    imageIdToDomainAtomIdMapCurrentBin[iatom -
                                                       numberDomainAtomsInBin];
                const double cutoff =
                  rc[binAtomIdToGlobalAtomIdMapCurrentBin[atomId]];

                if (distFromCellCenter < (enclosingBallCellRadius + cutoff))
                  nonTrivialIdsBin.push_back(iatom);
              }
          }

      std::map<dealii::CellId, std::vector<double>> cellNonTrivialQuadPoints;
      cell = dofHandlerOfField.begin_active();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            bool                      isCellTrivial = true;
            std::vector<dftfe::uInt> &nonTrivialIdsBin =
              cellNonTrivialIdsBinMap[cell->id()];
            std::vector<dftfe::uInt> &nonTrivialAtomIdsBin =
              bCellNonTrivialAtomIdsBin[cell->id()];
            std::vector<dftfe::uInt> &nonTrivialAtomIdsAllAtoms =
              bCellNonTrivialAtomIdsAllAtoms[cell->id()];
            std::vector<dftfe::uInt> &nonTrivialAtomImageIdsBin =
              bCellNonTrivialAtomImageIdsBin[cell->id()];
            std::vector<dftfe::uInt> &nonTrivialAtomImageIdsAllAtoms =
              bCellNonTrivialAtomImageIdsAllAtoms[cell->id()];

            if (nonTrivialIdsBin.size() != 0)
              {
                fe_values.reinit(cell);
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    const dealii::Point<3> &quadPoint =
                      fe_values.quadrature_point(q);
                    const double jxw = fe_values.JxW(q);
                    for (dftfe::uInt iatomNonTrivial = 0;
                         iatomNonTrivial < nonTrivialIdsBin.size();
                         ++iatomNonTrivial)
                      {
                        const dftfe::uInt iatom =
                          nonTrivialIdsBin[iatomNonTrivial];
                        const double r =
                          (quadPoint - atomLocations[iatom]).norm();
                        const dftfe::uInt atomId =
                          iatom < numberDomainAtomsInBin ?
                            iatom :
                            imageIdToDomainAtomIdMapCurrentBin
                              [iatom - numberDomainAtomsInBin];
                        if (r >
                            rc[binAtomIdToGlobalAtomIdMapCurrentBin[atomId]])
                          continue;
                        const double chargeVal = dftUtils::smearedCharge(
                          r, rc[binAtomIdToGlobalAtomIdMapCurrentBin[atomId]]);
                        smearedNuclearChargeIntegral[atomId] += chargeVal * jxw;
                        isCellTrivial = false;
                        break;
                      }
                  }

                if (!isCellTrivial)
                  {
                    bQuadValues[cell->id()].resize(n_q_points, 0.0);
                    std::fill(bQuadValues[cell->id()].begin(),
                              bQuadValues[cell->id()].end(),
                              0.0);

                    for (dftfe::uInt iatomNonTrivial = 0;
                         iatomNonTrivial < nonTrivialIdsBin.size();
                         ++iatomNonTrivial)
                      {
                        const dftfe::uInt iatom =
                          nonTrivialIdsBin[iatomNonTrivial];
                        const dftfe::uInt atomId =
                          iatom < numberDomainAtomsInBin ?
                            iatom :
                            imageIdToDomainAtomIdMapCurrentBin
                              [iatom - numberDomainAtomsInBin];
                        const dftfe::uInt chargeId =
                          binAtomIdToGlobalAtomIdMapCurrentBin[atomId];
                        nonTrivialAtomIdsAllAtoms.push_back(chargeId);
                        nonTrivialAtomIdsBin.push_back(chargeId);
                        nonTrivialAtomImageIdsAllAtoms.push_back(
                          binAtomIdToGlobalAtomIdMapCurrentBin[iatom]);
                        nonTrivialAtomImageIdsBin.push_back(
                          binAtomIdToGlobalAtomIdMapCurrentBin[iatom]);
                      }


                    std::vector<double> &quadPointsVec =
                      cellNonTrivialQuadPoints[cell->id()];
                    quadPointsVec.resize(n_q_points * 3, 0.0);
                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        const dealii::Point<3> &quadPoint =
                          fe_values.quadrature_point(q);
                        quadPointsVec[3 * q + 0] = quadPoint[0];
                        quadPointsVec[3 * q + 1] = quadPoint[1];
                        quadPointsVec[3 * q + 2] = quadPoint[2];
                      }
                  }
                else
                  {
                    nonTrivialIdsBin.resize(0);
                  }
              }
          }

      MPI_Allreduce(MPI_IN_PLACE,
                    &smearedNuclearChargeIntegral[0],
                    numberTotalAtomsInBin,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      for (dftfe::uInt iatom = 0; iatom < numberDomainAtomsInBin; ++iatom)
        {
          smearedChargeScaling[binAtomIdToGlobalAtomIdMapCurrentBin[iatom]] =
            1.0 / smearedNuclearChargeIntegral[iatom];
        }

      /*
      if (d_dftParams.verbosity >= 5)
        {
          if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            for (dftfe::uInt iatom = 0; iatom < numberDomainAtomsInBin;
                 ++iatom)
              std::cout
                << "Smeared charge integral before scaling (charge val=1):"
                << smearedNuclearChargeIntegral[iatom] << std::endl;
        }
      */


      std::vector<double> smearedNuclearChargeIntegralCheck(
        numberTotalAtomsInBin, 0.0);
      cell = dofHandlerOfField.begin_active();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            std::vector<double>     &bQuadValuesCell = bQuadValues[cell->id()];
            std::vector<dftfe::Int> &bQuadAtomIdsCell =
              bQuadAtomIdsAllAtoms[cell->id()];
            std::vector<dftfe::Int> &bQuadAtomImageIdsCell =
              bQuadAtomIdsAllAtomsImages[cell->id()];
            const std::vector<dftfe::uInt> &nonTrivialIdsBin =
              cellNonTrivialIdsBinMap[cell->id()];

            if (nonTrivialIdsBin.size() != 0)
              {
                // fe_values.reinit(cell);
                const std::vector<double> &quadPointsVec =
                  cellNonTrivialQuadPoints[cell->id()];
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint;
                    quadPoint[0] = quadPointsVec[3 * q + 0];
                    quadPoint[1] = quadPointsVec[3 * q + 1];
                    quadPoint[2] = quadPointsVec[3 * q + 2];
                    // const double jxw = fe_values.JxW(q);
                    for (dftfe::uInt iatomNonTrivial = 0;
                         iatomNonTrivial < nonTrivialIdsBin.size();
                         ++iatomNonTrivial)
                      {
                        const dftfe::uInt iatom =
                          nonTrivialIdsBin[iatomNonTrivial];
                        const double r =
                          (quadPoint - atomLocations[iatom]).norm();
                        const dftfe::uInt atomId =
                          iatom < numberDomainAtomsInBin ?
                            iatom :
                            imageIdToDomainAtomIdMapCurrentBin
                              [iatom - numberDomainAtomsInBin];
                        if (r >
                            rc[binAtomIdToGlobalAtomIdMapCurrentBin[atomId]])
                          continue;
                        const dftfe::uInt atomChargeId =
                          binAtomIdToGlobalAtomIdMapCurrentBin[atomId];
                        const double chargeVal =
                          dftUtils::smearedCharge(r, rc[atomChargeId]);

                        const double scalingFac =
                          (-atomCharges[atomId]) /
                          smearedNuclearChargeIntegral[atomId];

                        bQuadValuesCell[q] = chargeVal * scalingFac;
                        // smearedNuclearChargeIntegralCheck[atomId]+=bQuadValuesCell[q]*jxw;
                        bQuadAtomIdsCell[q] = atomChargeId;
                        bQuadAtomImageIdsCell[q] =
                          binAtomIdToGlobalAtomIdMapCurrentBin[iatom];
                        break;
                      }
                  }
              }
          }
      /*
      MPI_Allreduce(MPI_IN_PLACE,
          &smearedNuclearChargeIntegralCheck[0],
          numberTotalAtomsInBin,
          MPI_DOUBLE,
          MPI_SUM,
          mpi_communicator);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==0)
        for (dftfe::uInt iatom=0; iatom< numberDomainAtomsInBin; ++iatom)
          std::cout<<"Smeared charge integral after scaling:
      "<<smearedNuclearChargeIntegralCheck[iatom]<<std::endl;
      */
    }
  } // namespace


  void
  vselfBinsManager::solveVselfInBins(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                            &basisOperationsPtr,
    const dftfe::uInt                        offset,
    const dftfe::uInt                        matrixFreeQuadratureIdAX,
    const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
    const std::vector<std::vector<double>>  &imagePositions,
    const std::vector<dftfe::Int>           &imageIds,
    const std::vector<double>               &imageCharges,
    std::vector<std::vector<double>>        &localVselfs,
    std::map<dealii::CellId, std::vector<double>>     &bQuadValuesAllAtoms,
    std::map<dealii::CellId, std::vector<dftfe::Int>> &bQuadAtomIdsAllAtoms,
    std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtomsImages,
    std::map<dealii::CellId, std::vector<dftfe::uInt>> &bCellNonTrivialAtomIds,
    std::vector<std::map<dealii::CellId, std::vector<dftfe::uInt>>>
      &bCellNonTrivialAtomIdsBins,
    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      &bCellNonTrivialAtomImageIds,
    std::vector<std::map<dealii::CellId, std::vector<dftfe::uInt>>>
                              &bCellNonTrivialAtomImageIdsBins,
    const std::vector<double> &smearingWidths,
    std::vector<double>       &smearedChargeScaling,
    const dftfe::uInt          smearedChargeQuadratureId,
    const bool                 useSmearedCharges,
    const bool                 isVselfPerturbationSolve)
  {
    auto matrix_free_data = basisOperationsPtr->matrixFreeData();
    if (!isVselfPerturbationSolve)
      d_binsImages = d_bins;
    smearedChargeScaling.clear();
    localVselfs.clear();
    if (!isVselfPerturbationSolve)
      {
        d_vselfFieldBins.clear();
        d_vselfFieldDerRBins.clear();
      }
    else
      {
        d_vselfFieldPerturbedBins.clear();
      }

    bQuadValuesAllAtoms.clear();
    bQuadAtomIdsAllAtoms.clear();
    bQuadAtomIdsAllAtomsImages.clear();
    bCellNonTrivialAtomIds.clear();
    bCellNonTrivialAtomIdsBins.clear();
    bCellNonTrivialAtomImageIds.clear();
    bCellNonTrivialAtomImageIdsBins.clear();

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    const dftfe::uInt numberBins          = d_boundaryFlagOnlyChargeId.size();
    const dftfe::uInt numberGlobalCharges = d_atomLocations.size();

    smearedChargeScaling.resize(numberGlobalCharges, 0.0);
    bCellNonTrivialAtomIdsBins.resize(numberBins);
    bCellNonTrivialAtomImageIdsBins.resize(numberBins);

    const dealii::DoFHandler<3> &dofHandler =
      matrix_free_data.get_dof_handler(offset);
    const dealii::Quadrature<3> &quadratureFormula =
      matrix_free_data.get_quadrature();

    dealii::FEValues<3> fe_values_sc(
      dofHandler.get_fe(),
      matrix_free_data.get_quadrature(smearedChargeQuadratureId),
      dealii::update_values | dealii::update_JxW_values);
    const dftfe::uInt n_q_points_sc =
      matrix_free_data.get_quadrature(smearedChargeQuadratureId).size();

    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  dofHandler.begin_active(),
                                                endc = dofHandler.end();

    std::map<dealii::CellId, std::pair<dealii::Point<3>, double>>
      enclosingBallCells;

    if (useSmearedCharges)
      {
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              bQuadValuesAllAtoms[cell->id()].resize(n_q_points_sc, 0.0);
              bQuadAtomIdsAllAtoms[cell->id()].resize(n_q_points_sc, -1);
              bQuadAtomIdsAllAtomsImages[cell->id()].resize(n_q_points_sc, -1);
              enclosingBallCells[cell->id()] = cell->enclosing_ball();
            }

        localVselfs.resize(1, std::vector<double>(1));
        localVselfs[0][0] = 0.0;
      }

    // set up poisson solver
    dealiiLinearSolver CGSolver(d_mpiCommParent,
                                mpi_communicator,
                                dealiiLinearSolver::CG);

    poissonSolverProblemWrapperClass vselfSolverProblem(
      d_dftParams.finiteElementPolynomialOrderElectrostatics, mpi_communicator);

    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(
      dealii::MappingQ1<3, 3>(),
      matrix_free_data.get_dof_handler(offset),
      supportPoints);

    std::map<dealii::types::global_dof_index, dftfe::Int>::iterator iterMap;
    std::map<dealii::types::global_dof_index, double>::iterator     iterMapVal;
    if (!isVselfPerturbationSolve)
      {
        d_vselfFieldBins.resize(numberBins);
        d_vselfFieldDerRBins.resize(numberBins * 3);
      }
    else
      d_vselfFieldPerturbedBins.resize(numberBins);

    std::map<dealii::CellId, std::vector<double>> bQuadValuesBin;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;

    distributedCPUVec<double>              vselfBinScratch;
    std::vector<distributedCPUVec<double>> vselfDerRBinScratch(3);
    std::vector<dftfe::uInt>               constraintMatrixIdVselfDerR(3);

    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        double init_time;
        MPI_Barrier(d_mpiCommParent);
        init_time = MPI_Wtime();


        std::set<dftfe::Int>   &atomsInBinSet       = d_bins[iBin];
        std::set<dftfe::Int>   &atomsImagesInBinSet = d_binsImages[iBin];
        std::vector<dftfe::Int> atomsInCurrentBin(atomsInBinSet.begin(),
                                                  atomsInBinSet.end());
        const dftfe::uInt numberGlobalAtomsInBin = atomsInCurrentBin.size();

        std::vector<dftfe::Int> imageIdsOfAtomsInCurrentBin;
        std::vector<dftfe::Int> imageChargeIdsOfAtomsInCurrentBin;
        std::vector<dftfe::Int> imageIdToDomainAtomIdMapCurrentBin;
        for (dftfe::Int index = 0; index < numberGlobalAtomsInBin; ++index)
          {
            dftfe::Int globalChargeIdInCurrentBin = atomsInCurrentBin[index];
            for (dftfe::Int iImageAtom = 0; iImageAtom < imageIds.size();
                 ++iImageAtom)
              if (imageIds[iImageAtom] == globalChargeIdInCurrentBin)
                {
                  imageIdsOfAtomsInCurrentBin.push_back(iImageAtom);
                  imageChargeIdsOfAtomsInCurrentBin.push_back(
                    imageIds[iImageAtom]);
                  imageIdToDomainAtomIdMapCurrentBin.push_back(index);
                  if (!isVselfPerturbationSolve)
                    atomsImagesInBinSet.insert(iImageAtom +
                                               numberGlobalCharges);
                }
          }

        std::vector<dealii::Point<3>> atomPointsBin(numberGlobalAtomsInBin);
        std::vector<double>           atomChargesBin(numberGlobalAtomsInBin);
        for (dftfe::uInt i = 0; i < numberGlobalAtomsInBin; ++i)
          {
            atomPointsBin[i][0] = d_atomLocations[atomsInCurrentBin[i]][2];
            atomPointsBin[i][1] = d_atomLocations[atomsInCurrentBin[i]][3];
            atomPointsBin[i][2] = d_atomLocations[atomsInCurrentBin[i]][4];
            if (d_dftParams.isPseudopotential)
              atomChargesBin[i] = d_atomLocations[atomsInCurrentBin[i]][1];
            else
              atomChargesBin[i] = d_atomLocations[atomsInCurrentBin[i]][0];
          }

        for (dftfe::uInt i = 0; i < imageIdsOfAtomsInCurrentBin.size(); ++i)
          {
            dealii::Point<3> imagePoint;
            imagePoint[0] = imagePositions[imageIdsOfAtomsInCurrentBin[i]][0];
            imagePoint[1] = imagePositions[imageIdsOfAtomsInCurrentBin[i]][1];
            imagePoint[2] = imagePositions[imageIdsOfAtomsInCurrentBin[i]][2];
            atomPointsBin.push_back(imagePoint);
            if (d_dftParams.isPseudopotential)
              atomChargesBin.push_back(
                d_atomLocations[imageChargeIdsOfAtomsInCurrentBin[i]][1]);
            else
              atomChargesBin.push_back(
                d_atomLocations[imageChargeIdsOfAtomsInCurrentBin[i]][0]);
            atomsInCurrentBin.push_back(imageIdsOfAtomsInCurrentBin[i] +
                                        numberGlobalCharges);
          }

        bQuadValuesBin.clear();
        if (useSmearedCharges)
          smearedNuclearCharges(dofHandler,
                                matrix_free_data.get_quadrature(
                                  smearedChargeQuadratureId),
                                atomPointsBin,
                                atomChargesBin,
                                numberGlobalAtomsInBin,
                                imageIdToDomainAtomIdMapCurrentBin,
                                atomsInCurrentBin,
                                mpi_communicator,
                                smearingWidths,
                                enclosingBallCells,
                                bQuadValuesBin,
                                bQuadAtomIdsAllAtoms,
                                bQuadAtomIdsAllAtomsImages,
                                bCellNonTrivialAtomIds,
                                bCellNonTrivialAtomIdsBins[iBin],
                                bCellNonTrivialAtomImageIds,
                                bCellNonTrivialAtomImageIdsBins[iBin],
                                smearedChargeScaling);

        const dftfe::uInt constraintMatrixIdVself = 4 * iBin + offset;
        matrix_free_data.initialize_dof_vector(vselfBinScratch,
                                               constraintMatrixIdVself);
        vselfBinScratch = 0;

        if (!isVselfPerturbationSolve)
          {
            std::map<dealii::types::global_dof_index,
                     dealii::Point<3>>::iterator               iterNodalCoorMap;
            std::map<dealii::types::global_dof_index, double> &vSelfBinNodeMap =
              d_vselfBinField[iBin];

            //
            // set initial guess to vSelfBinScratch
            //
            for (iterNodalCoorMap = supportPoints.begin();
                 iterNodalCoorMap != supportPoints.end();
                 ++iterNodalCoorMap)
              if (vselfBinScratch.in_local_range(iterNodalCoorMap->first) &&
                  !d_vselfBinConstraintMatrices[4 * iBin].is_constrained(
                    iterNodalCoorMap->first))
                {
                  iterMapVal = vSelfBinNodeMap.find(iterNodalCoorMap->first);
                  if (iterMapVal != vSelfBinNodeMap.end())
                    vselfBinScratch(iterNodalCoorMap->first) =
                      iterMapVal->second;
                }

            if (useSmearedCharges)
              for (dftfe::uInt idim = 0; idim < 3; idim++)
                {
                  constraintMatrixIdVselfDerR[idim] =
                    4 * iBin + idim + offset + 1;
                  matrix_free_data.initialize_dof_vector(
                    vselfDerRBinScratch[idim],
                    constraintMatrixIdVselfDerR[idim]);
                  vselfDerRBinScratch[idim] = 0;
                }
          }
        else
          {
            vselfBinScratch = d_vselfFieldBins[iBin];
            d_vselfBinConstraintMatrices[4 * iBin].set_zero(vselfBinScratch);
          }

        MPI_Barrier(d_mpiCommParent);
        init_time = MPI_Wtime() - init_time;
        if (d_dftParams.verbosity >= 4)
          pcout
            << " Time taken for vself field initialization for current bin: "
            << init_time << std::endl;

        double vselfinit_time;
        MPI_Barrier(d_mpiCommParent);
        vselfinit_time = MPI_Wtime();

        //
        // call the poisson solver to compute vSelf in current bin
        //
        if (useSmearedCharges)
          vselfSolverProblem.reinit(
            basisOperationsPtr,
            vselfBinScratch,
            d_vselfBinConstraintMatrices[4 * iBin],
            constraintMatrixIdVself,
            0,
            matrixFreeQuadratureIdAX,
            std::map<dealii::types::global_dof_index, double>(),
            bQuadValuesBin,
            smearedChargeQuadratureId,
            dummy,
            true,
            false,
            true,
            false,
            false,
            0,
            false,
            false,
            true);
        else
          vselfSolverProblem.reinit(basisOperationsPtr,
                                    vselfBinScratch,
                                    d_vselfBinConstraintMatrices[4 * iBin],
                                    constraintMatrixIdVself,
                                    0,
                                    matrixFreeQuadratureIdAX,
                                    d_atomsInBin[iBin],
                                    bQuadValuesBin,
                                    smearedChargeQuadratureId,
                                    dummy,
                                    true,
                                    false,
                                    false,
                                    false,
                                    false,
                                    0,
                                    false,
                                    false,
                                    true);

        MPI_Barrier(d_mpiCommParent);
        vselfinit_time = MPI_Wtime() - vselfinit_time;
        if (d_dftParams.verbosity >= 4)
          pcout << " Time taken for vself solver problem init for current bin: "
                << vselfinit_time << std::endl;

        CGSolver.solve(vselfSolverProblem,
                       d_dftParams.absLinearSolverTolerance,
                       d_dftParams.maxLinearSolverIterations,
                       d_dftParams.verbosity);

        if (useSmearedCharges && !isVselfPerturbationSolve)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            {
              MPI_Barrier(d_mpiCommParent);
              vselfinit_time = MPI_Wtime();
              //
              // call the poisson solver to compute vSelf in current bin
              //
              vselfSolverProblem.reinit(
                basisOperationsPtr,
                vselfDerRBinScratch[idim],
                d_vselfBinConstraintMatrices[4 * iBin + idim + 1],
                constraintMatrixIdVselfDerR[idim],
                0,
                matrixFreeQuadratureIdAX,
                std::map<dealii::types::global_dof_index, double>(),
                bQuadValuesBin,
                smearedChargeQuadratureId,
                dummy,
                true,
                false,
                true,
                false,
                true,
                idim,
                false,
                false,
                true);


              MPI_Barrier(d_mpiCommParent);
              vselfinit_time = MPI_Wtime() - vselfinit_time;
              if (d_dftParams.verbosity >= 4)
                pcout
                  << " Time taken for vself solver problem init for current bin: "
                  << vselfinit_time << std::endl;

              CGSolver.solve(vselfSolverProblem,
                             d_dftParams.absLinearSolverTolerance,
                             d_dftParams.maxLinearSolverIterations,
                             d_dftParams.verbosity);
            }

        //
        // store Vselfs for atoms in bin
        //
        if (!isVselfPerturbationSolve)
          {
            if (useSmearedCharges)
              {
                double selfenergy_time;
                MPI_Barrier(d_mpiCommParent);
                selfenergy_time = MPI_Wtime();

                FEEvaluationWrapperClass<1> fe_eval_sc(
                  -1,
                  1,
                  matrix_free_data,
                  constraintMatrixIdVself,
                  smearedChargeQuadratureId);

                double vselfTimesSmearedChargesIntegralBin = 0.0;

                const dftfe::uInt numQuadPointsSmearedb = fe_eval_sc.n_q_points;
                dealii::AlignedVector<dealii::VectorizedArray<double>>
                  smearedbQuads(numQuadPointsSmearedb,
                                dealii::make_vectorized_array(0.0));
                for (dftfe::uInt macrocell = 0;
                     macrocell < matrix_free_data.n_cell_batches();
                     ++macrocell)
                  {
                    std::fill(smearedbQuads.begin(),
                              smearedbQuads.end(),
                              dealii::make_vectorized_array(0.0));
                    bool              isMacroCellTrivial = true;
                    const dftfe::uInt numSubCells =
                      matrix_free_data.n_active_entries_per_cell_batch(
                        macrocell);
                    for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr = matrix_free_data.get_cell_iterator(
                          macrocell, iSubCell, constraintMatrixIdVself);
                        dealii::CellId             subCellId = subCellPtr->id();
                        const std::vector<double> &tempVec =
                          bQuadValuesBin.find(subCellId)->second;
                        if (tempVec.size() == 0)
                          continue;

                        for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                          smearedbQuads[q][iSubCell] = tempVec[q];

                        isMacroCellTrivial = false;
                      }

                    if (!isMacroCellTrivial)
                      {
                        fe_eval_sc.reinit(macrocell);
                        fe_eval_sc.read_dof_values_plain(vselfBinScratch);
                        fe_eval_sc.evaluate(dealii::EvaluationFlags::values);
                        for (dftfe::uInt q = 0; q < fe_eval_sc.n_q_points; ++q)
                          {
                            fe_eval_sc.submit_value(fe_eval_sc.get_value(q) *
                                                      smearedbQuads[q],
                                                    q);
                          }
                        dealii::VectorizedArray<double> val =
                          fe_eval_sc.integrate_value();

                        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                             ++iSubCell)
                          vselfTimesSmearedChargesIntegralBin += val[iSubCell];
                      }
                  }

                cell = dofHandler.begin_active();
                for (; cell != endc; ++cell)
                  if (cell->is_locally_owned())
                    {
                      std::vector<double> &bQuadValuesBinCell =
                        bQuadValuesBin[cell->id()];
                      std::vector<double> &bQuadValuesAllAtomsCell =
                        bQuadValuesAllAtoms[cell->id()];

                      if (bQuadValuesBinCell.size() == 0)
                        continue;

                      for (dftfe::uInt q = 0; q < n_q_points_sc; ++q)
                        bQuadValuesAllAtomsCell[q] += bQuadValuesBinCell[q];
                    }

                localVselfs[0][0] += vselfTimesSmearedChargesIntegralBin;

                MPI_Barrier(d_mpiCommParent);
                selfenergy_time = MPI_Wtime() - selfenergy_time;
                if (d_dftParams.verbosity >= 4)
                  pcout << " Time taken for vself self energy for current bin: "
                        << selfenergy_time << std::endl;
              }
            else
              {
                for (std::map<dealii::types::global_dof_index, double>::iterator
                       it = d_atomsInBin[iBin].begin();
                     it != d_atomsInBin[iBin].end();
                     ++it)
                  {
                    std::vector<double> temp(2, 0.0);
                    temp[0] = it->second;                 // charge;
                    temp[1] = vselfBinScratch(it->first); // vself
                    if (d_dftParams.verbosity >= 4)
                      std::cout << "(only for debugging: peak value of Vself: "
                                << temp[1] << ")" << std::endl;

                    localVselfs.push_back(temp);
                  }
              }

            //
            // store solved vselfBinScratch field
            //
            d_vselfFieldBins[iBin] = vselfBinScratch;


            if (useSmearedCharges)
              for (dftfe::uInt idim = 0; idim < 3; idim++)
                d_vselfFieldDerRBins[3 * iBin + idim] =
                  vselfDerRBinScratch[idim];
          }
        else
          {
            //
            // store solved vselfBinScratch field
            //
            d_vselfFieldPerturbedBins[iBin] = vselfBinScratch;
          }
      } // bin loop
  }

#ifdef DFTFE_WITH_DEVICE

  void
  vselfBinsManager::solveVselfInBinsDevice(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt mfBaseDofHandlerIndex,
    const dftfe::uInt matrixFreeQuadratureIdAX,
    const dftfe::uInt offset,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &cellGradNIGradNJIntergralDevice,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                            &BLASWrapperPtr,
    const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
    const std::vector<std::vector<double>>  &imagePositions,
    const std::vector<dftfe::Int>           &imageIds,
    const std::vector<double>               &imageCharges,
    std::vector<std::vector<double>>        &localVselfs,
    std::map<dealii::CellId, std::vector<double>>     &bQuadValuesAllAtoms,
    std::map<dealii::CellId, std::vector<dftfe::Int>> &bQuadAtomIdsAllAtoms,
    std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtomsImages,
    std::map<dealii::CellId, std::vector<dftfe::uInt>> &bCellNonTrivialAtomIds,
    std::vector<std::map<dealii::CellId, std::vector<dftfe::uInt>>>
      &bCellNonTrivialAtomIdsBins,
    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      &bCellNonTrivialAtomImageIds,
    std::vector<std::map<dealii::CellId, std::vector<dftfe::uInt>>>
                              &bCellNonTrivialAtomImageIdsBins,
    const std::vector<double> &smearingWidths,
    std::vector<double>       &smearedChargeScaling,
    const dftfe::uInt          smearedChargeQuadratureId,
    const bool                 useSmearedCharges,
    const bool                 isVselfPerturbationSolve)
  {
    auto matrix_free_data = basisOperationsPtr->matrixFreeData();
    if (!isVselfPerturbationSolve)
      d_binsImages = d_bins;
    smearedChargeScaling.clear();
    localVselfs.clear();
    if (!isVselfPerturbationSolve)
      {
        d_vselfFieldBins.clear();
        d_vselfFieldDerRBins.clear();
      }
    else
      {
        d_vselfFieldPerturbedBins.clear();
      }
    bQuadValuesAllAtoms.clear();
    bQuadAtomIdsAllAtoms.clear();
    bQuadAtomIdsAllAtomsImages.clear();
    bCellNonTrivialAtomIds.clear();
    bCellNonTrivialAtomIdsBins.clear();
    bCellNonTrivialAtomImageIds.clear();
    bCellNonTrivialAtomImageIdsBins.clear();

    const dftfe::uInt numberBins          = d_boundaryFlagOnlyChargeId.size();
    const dftfe::uInt numberGlobalCharges = d_atomLocations.size();

    smearedChargeScaling.resize(numberGlobalCharges, 0.0);
    bCellNonTrivialAtomIdsBins.resize(numberBins);
    bCellNonTrivialAtomImageIdsBins.resize(numberBins);

    const dealii::DoFHandler<3> &dofHandler =
      matrix_free_data.get_dof_handler(offset);
    const dealii::Quadrature<3> &quadratureFormula =
      matrix_free_data.get_quadrature();

    const dftfe::uInt n_q_points_sc =
      matrix_free_data.get_quadrature(smearedChargeQuadratureId).size();

    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  dofHandler.begin_active(),
                                                endc = dofHandler.end();

    std::map<dealii::CellId, std::pair<dealii::Point<3>, double>>
      enclosingBallCells;
    if (useSmearedCharges)
      {
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              bQuadValuesAllAtoms[cell->id()].resize(n_q_points_sc, 0.0);
              bQuadAtomIdsAllAtoms[cell->id()].resize(n_q_points_sc, -1);
              bQuadAtomIdsAllAtomsImages[cell->id()].resize(n_q_points_sc, -1);
              enclosingBallCells[cell->id()] = cell->enclosing_ball();
            }

        localVselfs.resize(1, std::vector<double>(1));
        localVselfs[0][0] = 0.0;
      }

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    if (!isVselfPerturbationSolve)
      {
        d_vselfFieldBins.resize(numberBins);
        d_vselfFieldDerRBins.resize(numberBins * 3);

        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          matrix_free_data.initialize_dof_vector(d_vselfFieldBins[iBin],
                                                 4 * iBin + offset);

        if (useSmearedCharges)
          for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              matrix_free_data.initialize_dof_vector(
                d_vselfFieldDerRBins[iBin * 3 + idim],
                4 * iBin + idim + offset + 1);
      }
    else
      {
        d_vselfFieldPerturbedBins.resize(numberBins);

        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          matrix_free_data.initialize_dof_vector(
            d_vselfFieldPerturbedBins[iBin], 4 * iBin + offset);
      }

    const dftfe::uInt localSize = d_vselfFieldBins[0].locally_owned_size();
    const dftfe::uInt numberPoissonSolves =
      (useSmearedCharges && !isVselfPerturbationSolve) ? numberBins * 4 :
                                                         numberBins;
    const dftfe::uInt binStride =
      (useSmearedCharges && !isVselfPerturbationSolve) ? 4 : 1;
    std::vector<double> vselfBinsFieldsFlattened(localSize *
                                                   numberPoissonSolves,
                                                 0.0);

    std::vector<double> rhsFlattened(localSize * numberPoissonSolves, 0.0);

    const dftfe::uInt dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt num_quad_points = quadratureFormula.size();

    std::vector<std::map<dealii::CellId, std::vector<double>>> bQuadValuesBins(
      numberBins);

    MPI_Barrier(d_mpiCommParent);
    double time = MPI_Wtime();

    if (isVselfPerturbationSolve)
      {
        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          {
            distributedCPUVec<double> &vselfField = d_vselfFieldBins[iBin];
            for (dftfe::uInt i = 0; i < localSize; ++i)
              vselfBinsFieldsFlattened[numberPoissonSolves * i +
                                       binStride * iBin] =
                vselfField.local_element(i);
          }
      }

    dealii::Vector<double>                       elementalRhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    //
    // compute rhs for each bin and store in rhsFlattened
    //
    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        double smeared_init_time;
        MPI_Barrier(d_mpiCommParent);
        smeared_init_time = MPI_Wtime();

        std::set<dftfe::Int>   &atomsInBinSet       = d_bins[iBin];
        std::set<dftfe::Int>   &atomsImagesInBinSet = d_binsImages[iBin];
        std::vector<dftfe::Int> atomsInCurrentBin(atomsInBinSet.begin(),
                                                  atomsInBinSet.end());
        const dftfe::uInt numberGlobalAtomsInBin = atomsInCurrentBin.size();

        std::vector<dftfe::Int> imageIdsOfAtomsInCurrentBin;
        std::vector<dftfe::Int> imageChargeIdsOfAtomsInCurrentBin;
        std::vector<dftfe::Int> imageIdToDomainAtomIdMapCurrentBin;
        for (dftfe::Int index = 0; index < numberGlobalAtomsInBin; ++index)
          {
            dftfe::Int globalChargeIdInCurrentBin = atomsInCurrentBin[index];
            for (dftfe::Int iImageAtom = 0; iImageAtom < imageIds.size();
                 ++iImageAtom)
              if (imageIds[iImageAtom] == globalChargeIdInCurrentBin)
                {
                  imageIdsOfAtomsInCurrentBin.push_back(iImageAtom);
                  imageChargeIdsOfAtomsInCurrentBin.push_back(
                    imageIds[iImageAtom]);
                  imageIdToDomainAtomIdMapCurrentBin.push_back(index);
                  if (!isVselfPerturbationSolve)
                    atomsImagesInBinSet.insert(iImageAtom +
                                               numberGlobalCharges);
                }
          }

        std::vector<dealii::Point<3>> atomPointsBin(numberGlobalAtomsInBin);
        std::vector<double>           atomChargesBin(numberGlobalAtomsInBin);
        for (dftfe::uInt i = 0; i < numberGlobalAtomsInBin; ++i)
          {
            atomPointsBin[i][0] = d_atomLocations[atomsInCurrentBin[i]][2];
            atomPointsBin[i][1] = d_atomLocations[atomsInCurrentBin[i]][3];
            atomPointsBin[i][2] = d_atomLocations[atomsInCurrentBin[i]][4];
            if (d_dftParams.isPseudopotential)
              atomChargesBin[i] = d_atomLocations[atomsInCurrentBin[i]][1];
            else
              atomChargesBin[i] = d_atomLocations[atomsInCurrentBin[i]][0];
          }

        for (dftfe::uInt i = 0; i < imageIdsOfAtomsInCurrentBin.size(); ++i)
          {
            dealii::Point<3> imagePoint;
            imagePoint[0] = imagePositions[imageIdsOfAtomsInCurrentBin[i]][0];
            imagePoint[1] = imagePositions[imageIdsOfAtomsInCurrentBin[i]][1];
            imagePoint[2] = imagePositions[imageIdsOfAtomsInCurrentBin[i]][2];
            atomPointsBin.push_back(imagePoint);
            if (d_dftParams.isPseudopotential)
              atomChargesBin.push_back(
                d_atomLocations[imageChargeIdsOfAtomsInCurrentBin[i]][1]);
            else
              atomChargesBin.push_back(
                d_atomLocations[imageChargeIdsOfAtomsInCurrentBin[i]][0]);
            atomsInCurrentBin.push_back(imageIdsOfAtomsInCurrentBin[i] +
                                        numberGlobalCharges);
          }

        if (useSmearedCharges)
          smearedNuclearCharges(dofHandler,
                                matrix_free_data.get_quadrature(
                                  smearedChargeQuadratureId),
                                atomPointsBin,
                                atomChargesBin,
                                numberGlobalAtomsInBin,
                                imageIdToDomainAtomIdMapCurrentBin,
                                atomsInCurrentBin,
                                mpi_communicator,
                                smearingWidths,
                                enclosingBallCells,
                                bQuadValuesBins[iBin],
                                bQuadAtomIdsAllAtoms,
                                bQuadAtomIdsAllAtomsImages,
                                bCellNonTrivialAtomIds,
                                bCellNonTrivialAtomIdsBins[iBin],
                                bCellNonTrivialAtomImageIds,
                                bCellNonTrivialAtomImageIdsBins[iBin],
                                smearedChargeScaling);

        MPI_Barrier(d_mpiCommParent);
        smeared_init_time = MPI_Wtime() - smeared_init_time;
        if (d_dftParams.verbosity >= 4)
          pcout
            << " Time taken for smeared charge initialization for current bin: "
            << smeared_init_time << std::endl;

        // rhs contribution from static condensation of dirichlet boundary
        // conditions
        const dftfe::uInt constraintMatrixId = 4 * iBin + offset;

        distributedCPUVec<double> tempvec;
        matrix_free_data.initialize_dof_vector(tempvec, constraintMatrixId);
        tempvec = 0.0;

        distributedCPUVec<double> rhs;
        rhs.reinit(tempvec);
        rhs = 0;

        dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
          constraintsMatrixDataInfo;
        constraintsMatrixDataInfo.initialize(
          matrix_free_data.get_vector_partitioner(4 * iBin + offset),
          d_vselfBinConstraintMatrices[4 * iBin]);

        tempvec.update_ghost_values();
        constraintsMatrixDataInfo.distribute(tempvec);

        std::map<dealii::CellId, std::vector<double>> &bQuadValuesBin =
          bQuadValuesBins[iBin];
        FEEvaluationWrapperClass<1> fe_eval(
          d_dftParams.finiteElementPolynomialOrderElectrostatics,
          d_dftParams.finiteElementPolynomialOrderElectrostatics + 1,
          matrix_free_data,
          constraintMatrixId,
          matrixFreeQuadratureIdAX);
        FEEvaluationWrapperClass<1> fe_eval_sc(-1,
                                               1,
                                               matrix_free_data,
                                               constraintMatrixId,
                                               smearedChargeQuadratureId);


        dealii::VectorizedArray<double> quarter =
          dealii::make_vectorized_array(1.0 / (4.0 * M_PI));
        for (dftfe::uInt macrocell = 0;
             macrocell < matrix_free_data.n_cell_batches();
             ++macrocell)
          {
            fe_eval.reinit(macrocell);
            fe_eval.read_dof_values_plain(tempvec);
            fe_eval.evaluate(dealii::EvaluationFlags::gradients);
            for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
              {
                fe_eval.submit_gradient(-quarter * fe_eval.get_gradient(q), q);
              }
            fe_eval.integrate(dealii::EvaluationFlags::gradients);
            fe_eval.distribute_local_to_global(rhs);
          }

        // rhs contribution from atomic charge at fem nodes
        if (useSmearedCharges)
          {
            const dftfe::uInt numQuadPointsSmearedb = fe_eval_sc.n_q_points;
            dealii::AlignedVector<dealii::VectorizedArray<double>>
              smearedbQuads(numQuadPointsSmearedb,
                            dealii::make_vectorized_array(0.0));
            for (dftfe::uInt macrocell = 0;
                 macrocell < matrix_free_data.n_cell_batches();
                 ++macrocell)
              {
                std::fill(smearedbQuads.begin(),
                          smearedbQuads.end(),
                          dealii::make_vectorized_array(0.0));
                bool              isMacroCellTrivial = true;
                const dftfe::uInt numSubCells =
                  matrix_free_data.n_active_entries_per_cell_batch(macrocell);
                for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                     ++iSubCell)
                  {
                    subCellPtr =
                      matrix_free_data.get_cell_iterator(macrocell,
                                                         iSubCell,
                                                         constraintMatrixId);
                    dealii::CellId             subCellId = subCellPtr->id();
                    const std::vector<double> &tempVec =
                      bQuadValuesBin.find(subCellId)->second;
                    if (tempVec.size() == 0)
                      continue;

                    for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                      smearedbQuads[q][iSubCell] = tempVec[q];

                    isMacroCellTrivial = false;
                  }

                if (!isMacroCellTrivial)
                  {
                    fe_eval_sc.reinit(macrocell);
                    for (dftfe::uInt q = 0; q < fe_eval_sc.n_q_points; ++q)
                      {
                        fe_eval_sc.submit_value(smearedbQuads[q], q);
                      }
                    fe_eval_sc.integrate(dealii::EvaluationFlags::values);
                    fe_eval_sc.distribute_local_to_global(rhs);
                  }
              }
          }
        else
          for (std::map<dealii::types::global_dof_index, double>::const_iterator
                 it = d_atomsInBin[iBin].begin();
               it != d_atomsInBin[iBin].end();
               ++it)
            {
              std::vector<dealii::AffineConstraints<double>::size_type>
                local_dof_indices_origin(1, it->first); // atomic node
              dealii::Vector<double> cell_rhs_origin(1);
              cell_rhs_origin(0) = -(it->second); // atomic charge

              d_vselfBinConstraintMatrices[4 * iBin].distribute_local_to_global(
                cell_rhs_origin, local_dof_indices_origin, rhs);
            }

        // MPI operation to sync data
        rhs.compress(dealii::VectorOperation::add);

        // FIXME: check if this is really required
        d_vselfBinConstraintMatrices[4 * iBin].set_zero(rhs);

        for (dftfe::uInt i = 0; i < localSize; ++i)
          rhsFlattened[i * numberPoissonSolves + binStride * iBin] =
            rhs.local_element(i);

        if (useSmearedCharges && !isVselfPerturbationSolve)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            {
              const dftfe::uInt constraintMatrixId2 =
                4 * iBin + offset + idim + 1;

              matrix_free_data.initialize_dof_vector(tempvec,
                                                     constraintMatrixId2);
              tempvec = 0.0;

              rhs.reinit(tempvec);
              rhs = 0;

              dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
                constraintsMatrixDataInfo2;
              constraintsMatrixDataInfo2.initialize(
                matrix_free_data.get_vector_partitioner(4 * iBin + idim + 1 +
                                                        offset),
                d_vselfBinConstraintMatrices[4 * iBin + idim + 1]);

              tempvec.update_ghost_values();
              constraintsMatrixDataInfo2.distribute(tempvec);
              FEEvaluationWrapperClass<1> fe_eval2(
                d_dftParams.finiteElementPolynomialOrderElectrostatics,
                d_dftParams.finiteElementPolynomialOrderElectrostatics + 1,
                matrix_free_data,
                constraintMatrixId2,
                matrixFreeQuadratureIdAX);
              FEEvaluationWrapperClass<1> fe_eval_sc2(
                -1,
                1,
                matrix_free_data,
                constraintMatrixId2,
                smearedChargeQuadratureId);


              for (dftfe::uInt macrocell = 0;
                   macrocell < matrix_free_data.n_cell_batches();
                   ++macrocell)
                {
                  fe_eval2.reinit(macrocell);
                  fe_eval2.read_dof_values_plain(tempvec);
                  fe_eval2.evaluate(dealii::EvaluationFlags::gradients);
                  for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
                    {
                      fe_eval2.submit_gradient(-quarter *
                                                 fe_eval2.get_gradient(q),
                                               q);
                    }
                  fe_eval2.integrate(dealii::EvaluationFlags::gradients);
                  fe_eval2.distribute_local_to_global(rhs);
                }

              const dftfe::uInt numQuadPointsSmearedb = fe_eval_sc2.n_q_points;

              dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
              for (dftfe::uInt i = 0; i < 3; i++)
                zeroTensor[i] = dealii::make_vectorized_array(0.0);

              dealii::AlignedVector<
                dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                smearedbQuads(numQuadPointsSmearedb, zeroTensor);
              for (dftfe::uInt macrocell = 0;
                   macrocell < matrix_free_data.n_cell_batches();
                   ++macrocell)
                {
                  std::fill(smearedbQuads.begin(),
                            smearedbQuads.end(),
                            dealii::make_vectorized_array(0.0));
                  bool              isMacroCellTrivial = true;
                  const dftfe::uInt numSubCells =
                    matrix_free_data.n_active_entries_per_cell_batch(macrocell);
                  for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                       ++iSubCell)
                    {
                      subCellPtr =
                        matrix_free_data.get_cell_iterator(macrocell,
                                                           iSubCell,
                                                           constraintMatrixId2);
                      dealii::CellId             subCellId = subCellPtr->id();
                      const std::vector<double> &tempVec =
                        bQuadValuesBin.find(subCellId)->second;
                      if (tempVec.size() == 0)
                        continue;

                      for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                        smearedbQuads[q][idim][iSubCell] = tempVec[q];

                      isMacroCellTrivial = false;
                    }

                  if (!isMacroCellTrivial)
                    {
                      fe_eval_sc2.reinit(macrocell);
                      for (dftfe::uInt q = 0; q < fe_eval_sc2.n_q_points; ++q)
                        {
                          fe_eval_sc2.submit_gradient(smearedbQuads[q], q);
                        }
                      fe_eval_sc2.integrate(dealii::EvaluationFlags::gradients);
                      fe_eval_sc2.distribute_local_to_global(rhs);
                    }
                }

              // MPI operation to sync data
              rhs.compress(dealii::VectorOperation::add);

              // FIXME: check if this is really required
              d_vselfBinConstraintMatrices[4 * iBin + idim + 1].set_zero(rhs);

              for (dftfe::uInt i = 0; i < localSize; ++i)
                rhsFlattened[i * numberPoissonSolves + binStride * iBin + idim +
                             1] = rhs.local_element(i);
            }
      } // bin loop

    //
    // compute diagonal
    //
    distributedCPUVec<double> diagonalA;
    matrix_free_data.initialize_dof_vector(diagonalA, mfBaseDofHandlerIndex);
    diagonalA = 0;


    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadratureFormula,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);
    dealii::Vector<double> elementalDiagonalA(dofs_per_cell);

    cell = dofHandler.begin_active();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          elementalDiagonalA = 0.0;
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              elementalDiagonalA(i) += (1.0 / (4.0 * M_PI)) *
                                       (fe_values.shape_grad(i, q_point) *
                                        fe_values.shape_grad(i, q_point)) *
                                       fe_values.JxW(q_point);

          hangingPeriodicConstraintMatrix.distribute_local_to_global(
            elementalDiagonalA, local_dof_indices, diagonalA);
        }

    diagonalA.compress(dealii::VectorOperation::add);

    for (dealii::types::global_dof_index i = 0; i < diagonalA.size(); ++i)
      if (diagonalA.in_local_range(i))
        if (!hangingPeriodicConstraintMatrix.is_constrained(i))
          diagonalA(i) = 1.0 / diagonalA(i);

    diagonalA.compress(dealii::VectorOperation::insert);

    const dftfe::uInt ghostSize =
      matrix_free_data.get_vector_partitioner(mfBaseDofHandlerIndex)
        ->n_ghost_indices();

    std::vector<double> inhomoIdsColoredVecFlattened((localSize + ghostSize) *
                                                       numberPoissonSolves,
                                                     1.0);
    for (dftfe::uInt i = 0; i < (localSize + ghostSize); ++i)
      {
        const dealii::types::global_dof_index globalNodeId =
          matrix_free_data.get_vector_partitioner(mfBaseDofHandlerIndex)
            ->local_to_global(i);
        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          {
            if (std::abs(
                  d_vselfBinConstraintMatrices[4 * iBin].get_inhomogeneity(
                    globalNodeId)) > 1e-10 &&
                d_vselfBinConstraintMatrices[4 * iBin]
                    .get_constraint_entries(globalNodeId)
                    ->size() == 0)
              {
                inhomoIdsColoredVecFlattened[i * numberPoissonSolves +
                                             binStride * iBin] = 0.0;
                if (i < localSize && isVselfPerturbationSolve)
                  vselfBinsFieldsFlattened[numberPoissonSolves * i +
                                           binStride * iBin] = 0.0;
              }
          }
      }


    if (useSmearedCharges && !isVselfPerturbationSolve)
      for (dftfe::uInt idim = 0; idim < 3; idim++)
        for (dftfe::uInt i = 0; i < (localSize + ghostSize); ++i)
          {
            const dealii::types::global_dof_index globalNodeId =
              matrix_free_data.get_vector_partitioner(mfBaseDofHandlerIndex)
                ->local_to_global(i);
            for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
              {
                if (std::abs(d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                               .get_inhomogeneity(globalNodeId)) > 1e-10 &&
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                        .get_constraint_entries(globalNodeId)
                        ->size() == 0)
                  inhomoIdsColoredVecFlattened[i * numberPoissonSolves +
                                               binStride * iBin + idim + 1] =
                    0.0;
              }
          }

    MPI_Barrier(d_mpiCommParent);
    time = MPI_Wtime() - time;
    if (d_dftParams.verbosity >= 4 && this_mpi_process == 0)
      std::cout
        << "Solve vself in bins: time for smeared charge initialization, compute rhs and diagonal: "
        << time << std::endl;

    MPI_Barrier(d_mpiCommParent);
    time = MPI_Wtime();
    //
    // Device poisson solve
    //
    poissonDevice::solveVselfInBins(
      cellGradNIGradNJIntergralDevice,
      BLASWrapperPtr,
      matrix_free_data,
      mfBaseDofHandlerIndex,
      hangingPeriodicConstraintMatrix,
      &rhsFlattened[0],
      diagonalA.begin(),
      &inhomoIdsColoredVecFlattened[0],
      localSize,
      ghostSize,
      numberPoissonSolves,
      d_mpiCommParent,
      mpi_communicator,
      &vselfBinsFieldsFlattened[0],
      d_dftParams.verbosity,
      d_dftParams.maxLinearSolverIterations,
      d_dftParams.absLinearSolverTolerance,
      d_dftParams.finiteElementPolynomialOrderElectrostatics !=
          d_dftParams.finiteElementPolynomialOrder ?
        true :
        false);

    MPI_Barrier(d_mpiCommParent);
    time = MPI_Wtime() - time;
    if (d_dftParams.verbosity >= 4 && this_mpi_process == 0)
      std::cout
        << "Solve vself in bins: time for poissonDevice::solveVselfInBins : "
        << time << std::endl;

    MPI_Barrier(d_mpiCommParent);
    time = MPI_Wtime();

    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        if (!isVselfPerturbationSolve)
          {
            //
            // store solved vselfBinScratch field
            //
            distributedCPUVec<double> &vselfField = d_vselfFieldBins[iBin];

            for (dftfe::uInt i = 0; i < localSize; ++i)
              vselfField.local_element(i) =
                vselfBinsFieldsFlattened[numberPoissonSolves * i +
                                         binStride * iBin];

            const dftfe::uInt constraintMatrixId = 4 * iBin + offset;

            dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
              constraintsMatrixDataInfo;
            constraintsMatrixDataInfo.initialize(
              matrix_free_data.get_vector_partitioner(constraintMatrixId),
              d_vselfBinConstraintMatrices[4 * iBin]);

            d_vselfFieldBins[iBin].update_ghost_values();
            constraintsMatrixDataInfo.distribute(d_vselfFieldBins[iBin]);

            if (useSmearedCharges)
              for (dftfe::uInt idim = 0; idim < 3; idim++)
                {
                  const dftfe::uInt constraintMatrixId2 =
                    4 * iBin + offset + idim + 1;

                  distributedCPUVec<double> &vselfFieldDerR =
                    d_vselfFieldDerRBins[3 * iBin + idim];

                  for (dftfe::uInt i = 0; i < localSize; ++i)
                    vselfFieldDerR.local_element(i) =
                      vselfBinsFieldsFlattened[numberPoissonSolves * i +
                                               binStride * iBin + idim + 1];

                  dftUtils::constraintMatrixInfo<
                    dftfe::utils::MemorySpace::HOST>
                    constraintsMatrixDataInfo2;
                  constraintsMatrixDataInfo2.initialize(
                    matrix_free_data.get_vector_partitioner(
                      constraintMatrixId2),
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1]);

                  d_vselfFieldDerRBins[3 * iBin + idim].update_ghost_values();
                  constraintsMatrixDataInfo2.distribute(
                    d_vselfFieldDerRBins[3 * iBin + idim]);
                }

            //
            // store Vselfs for atoms in bin
            //
            if (useSmearedCharges)
              {
                double selfenergy_time;
                MPI_Barrier(d_mpiCommParent);
                selfenergy_time = MPI_Wtime();

                FEEvaluationWrapperClass<1> fe_eval_sc(
                  -1,
                  1,
                  matrix_free_data,
                  constraintMatrixId,
                  smearedChargeQuadratureId);

                std::map<dealii::CellId, std::vector<double>> &bQuadValuesBin =
                  bQuadValuesBins[iBin];

                double vselfTimesSmearedChargesIntegralBin = 0.0;

                const dftfe::uInt numQuadPointsSmearedb = fe_eval_sc.n_q_points;
                dealii::AlignedVector<dealii::VectorizedArray<double>>
                  smearedbQuads(numQuadPointsSmearedb,
                                dealii::make_vectorized_array(0.0));
                for (dftfe::uInt macrocell = 0;
                     macrocell < matrix_free_data.n_cell_batches();
                     ++macrocell)
                  {
                    std::fill(smearedbQuads.begin(),
                              smearedbQuads.end(),
                              dealii::make_vectorized_array(0.0));
                    bool              isMacroCellTrivial = true;
                    const dftfe::uInt numSubCells =
                      matrix_free_data.n_active_entries_per_cell_batch(
                        macrocell);
                    for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr = matrix_free_data.get_cell_iterator(
                          macrocell, iSubCell, constraintMatrixId);
                        dealii::CellId             subCellId = subCellPtr->id();
                        const std::vector<double> &tempVec =
                          bQuadValuesBin.find(subCellId)->second;
                        if (tempVec.size() == 0)
                          continue;

                        for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                          smearedbQuads[q][iSubCell] = tempVec[q];

                        isMacroCellTrivial = false;
                      }

                    if (!isMacroCellTrivial)
                      {
                        fe_eval_sc.reinit(macrocell);
                        fe_eval_sc.read_dof_values_plain(
                          d_vselfFieldBins[iBin]);
                        fe_eval_sc.evaluate(dealii::EvaluationFlags::values);
                        for (dftfe::uInt q = 0; q < fe_eval_sc.n_q_points; ++q)
                          {
                            fe_eval_sc.submit_value(fe_eval_sc.get_value(q) *
                                                      smearedbQuads[q],
                                                    q);
                          }
                        dealii::VectorizedArray<double> val =
                          fe_eval_sc.integrate_value();

                        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                             ++iSubCell)
                          vselfTimesSmearedChargesIntegralBin += val[iSubCell];
                      }
                  }


                cell = dofHandler.begin_active();
                for (; cell != endc; ++cell)
                  if (cell->is_locally_owned())
                    {
                      std::vector<double> &bQuadValuesBinCell =
                        bQuadValuesBin[cell->id()];
                      std::vector<double> &bQuadValuesAllAtomsCell =
                        bQuadValuesAllAtoms[cell->id()];

                      if (bQuadValuesBinCell.size() == 0)
                        continue;

                      for (dftfe::uInt q = 0; q < n_q_points_sc; ++q)
                        bQuadValuesAllAtomsCell[q] += bQuadValuesBinCell[q];
                    }


                localVselfs[0][0] += vselfTimesSmearedChargesIntegralBin;

                MPI_Barrier(d_mpiCommParent);
                selfenergy_time = MPI_Wtime() - selfenergy_time;
                if (d_dftParams.verbosity >= 4)
                  pcout << " Time taken for vself self energy for current bin: "
                        << selfenergy_time << std::endl;
              }
            else
              for (std::map<dealii::types::global_dof_index, double>::iterator
                     it = d_atomsInBin[iBin].begin();
                   it != d_atomsInBin[iBin].end();
                   ++it)
                {
                  std::vector<double> temp(2, 0.0);
                  temp[0] = it->second;                        // charge;
                  temp[1] = d_vselfFieldBins[iBin](it->first); // vself
                  if (d_dftParams.verbosity >= 4)
                    std::cout
                      << "(only for debugging: peak value of Vself: " << temp[1]
                      << ")" << std::endl;

                  localVselfs.push_back(temp);
                }
          }
        else
          {
            //
            // store solved vselfBinScratch field
            //
            distributedCPUVec<double> &vselfFieldPerturbed =
              d_vselfFieldPerturbedBins[iBin];
            for (dftfe::uInt i = 0; i < localSize; ++i)
              vselfFieldPerturbed.local_element(i) =
                vselfBinsFieldsFlattened[numberPoissonSolves * i +
                                         binStride * iBin];

            const dftfe::uInt constraintMatrixId = 4 * iBin + offset;

            dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
              constraintsMatrixDataInfo;
            constraintsMatrixDataInfo.initialize(
              matrix_free_data.get_vector_partitioner(constraintMatrixId),
              d_vselfBinConstraintMatrices[4 * iBin]);

            d_vselfFieldPerturbedBins[iBin].update_ghost_values();
            constraintsMatrixDataInfo.distribute(
              d_vselfFieldPerturbedBins[iBin]);
          }

      } // bin loop

    MPI_Barrier(d_mpiCommParent);
    time = MPI_Wtime() - time;
    if (d_dftParams.verbosity >= 4 && this_mpi_process == 0)
      std::cout << "Solve vself in bins: time for updating d_vselfFieldBins : "
                << time << std::endl;
  }
#endif


  void
  vselfBinsManager::solveVselfInBinsPerturbedDomain(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt mfBaseDofHandlerIndex,
    const dftfe::uInt matrixFreeQuadratureIdAX,
    const dftfe::uInt offset,
#ifdef DFTFE_WITH_DEVICE
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &cellGradNIGradNJIntergralDevice,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &BLASWrapperPtr,
#endif
    const dealii::AffineConstraints<double> &hangingPeriodicConstraintMatrix,
    const std::vector<std::vector<double>>  &imagePositions,
    const std::vector<dftfe::Int>           &imageIds,
    const std::vector<double>               &imageCharges,
    const std::vector<double>               &smearingWidths,
    const dftfe::uInt                        smearedChargeQuadratureId,
    const bool                               useSmearedCharges)
  {
    auto matrix_free_data = basisOperationsPtr->matrixFreeData();
    // dummy variables
    std::vector<std::vector<double>>                  localVselfs;
    std::map<dealii::CellId, std::vector<double>>     bQuadValuesAllAtoms;
    std::map<dealii::CellId, std::vector<double>>     gradbQuadValuesAllAtoms;
    std::map<dealii::CellId, std::vector<dftfe::Int>> bQuadAtomIdsAllAtoms;
    std::map<dealii::CellId, std::vector<dftfe::Int>>
      bQuadAtomIdsAllAtomsImages;
    std::map<dealii::CellId, std::vector<dftfe::uInt>> bCellNonTrivialAtomIds;
    std::vector<std::map<dealii::CellId, std::vector<dftfe::uInt>>>
      bCellNonTrivialAtomIdsBins;
    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      bCellNonTrivialAtomImageIds;
    std::vector<std::map<dealii::CellId, std::vector<dftfe::uInt>>>
                        bCellNonTrivialAtomImageIdsBins;
    std::vector<double> smearedChargeScaling;

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParams.useDevice and d_dftParams.vselfGPU)
      solveVselfInBinsDevice(basisOperationsPtr,
                             mfBaseDofHandlerIndex,
                             matrixFreeQuadratureIdAX,
                             offset,
                             cellGradNIGradNJIntergralDevice,
                             BLASWrapperPtr,
                             hangingPeriodicConstraintMatrix,
                             imagePositions,
                             imageIds,
                             imageCharges,
                             localVselfs,
                             bQuadValuesAllAtoms,
                             bQuadAtomIdsAllAtoms,
                             bQuadAtomIdsAllAtomsImages,
                             bCellNonTrivialAtomIds,
                             bCellNonTrivialAtomIdsBins,
                             bCellNonTrivialAtomImageIds,
                             bCellNonTrivialAtomImageIdsBins,
                             smearingWidths,
                             smearedChargeScaling,
                             smearedChargeQuadratureId,
                             useSmearedCharges,
                             true);
    else
      solveVselfInBins(basisOperationsPtr,
                       offset,
                       matrixFreeQuadratureIdAX,
                       hangingPeriodicConstraintMatrix,
                       imagePositions,
                       imageIds,
                       imageCharges,
                       localVselfs,
                       bQuadValuesAllAtoms,
                       bQuadAtomIdsAllAtoms,
                       bQuadAtomIdsAllAtomsImages,
                       bCellNonTrivialAtomIds,
                       bCellNonTrivialAtomIdsBins,
                       bCellNonTrivialAtomImageIds,
                       bCellNonTrivialAtomImageIdsBins,
                       smearingWidths,
                       smearedChargeScaling,
                       smearedChargeQuadratureId,
                       useSmearedCharges,
                       true);
#else
    solveVselfInBins(basisOperationsPtr,
                     offset,
                     matrixFreeQuadratureIdAX,
                     hangingPeriodicConstraintMatrix,
                     imagePositions,
                     imageIds,
                     imageCharges,
                     localVselfs,
                     bQuadValuesAllAtoms,
                     bQuadAtomIdsAllAtoms,
                     bQuadAtomIdsAllAtomsImages,
                     bCellNonTrivialAtomIds,
                     bCellNonTrivialAtomIdsBins,
                     bCellNonTrivialAtomImageIds,
                     bCellNonTrivialAtomImageIdsBins,
                     smearingWidths,
                     smearedChargeScaling,
                     smearedChargeQuadratureId,
                     useSmearedCharges,
                     true);
#endif
  }

} // namespace dftfe
