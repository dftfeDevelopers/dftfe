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

namespace dftfe
{
  struct quadData
  {
    double density;
  };

  namespace internal
  {
    void
    checkTriangulationEqualityAcrossProcessorPools(
      const dealii::parallel::distributed::Triangulation<3>
        &                parallelTriangulation,
      const unsigned int numLocallyOwnedCells,
      const MPI_Comm &   interpool_comm)
    {
      const unsigned int numberGlobalCellsParallelMinPools =
        dealii::Utilities::MPI::min(
          parallelTriangulation.n_global_active_cells(), interpool_comm);
      const unsigned int numberGlobalCellsParallelMaxPools =
        dealii::Utilities::MPI::max(
          parallelTriangulation.n_global_active_cells(), interpool_comm);
      AssertThrow(numberGlobalCellsParallelMinPools ==
                    numberGlobalCellsParallelMaxPools,
                  dealii::ExcMessage(
                    "Number of global cells are different across pools."));

      const unsigned int numberLocalCellsMinPools =
        dealii::Utilities::MPI::min(numLocallyOwnedCells, interpool_comm);
      const unsigned int numberLocalCellsMaxPools =
        dealii::Utilities::MPI::max(numLocallyOwnedCells, interpool_comm);
      AssertThrow(
        numberLocalCellsMinPools == numberLocalCellsMaxPools,
        dealii::ExcMessage(
          "Number of local cells are different across pools or in other words the physical partitions don't have the same ordering across pools."));
    }


    void
    computeMeshMetrics(const dealii::parallel::distributed::Triangulation<3>
                         &                               parallelTriangulation,
                       const std::string &               printCommand,
                       const dealii::ConditionalOStream &pcout,
                       const MPI_Comm &                  mpi_comm,
                       const MPI_Comm &                  interpool_comm1,
                       const MPI_Comm &                  interpool_comm2,
                       const dftParameters &             dftParams)

    {
      //
      // compute some adaptive mesh metrics
      //
      double       minElemLength        = dftParams.meshSizeOuterDomain;
      unsigned int numLocallyOwnedCells = 0;
      typename dealii::parallel::distributed::Triangulation<
        3>::active_cell_iterator cell,
        endc;
      cell = parallelTriangulation.begin_active();
      endc = parallelTriangulation.end();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              numLocallyOwnedCells++;
              if (cell->minimum_vertex_distance() < minElemLength)
                minElemLength = cell->minimum_vertex_distance();
            }
        }

      minElemLength = dealii::Utilities::MPI::min(minElemLength, mpi_comm);

      //
      // print out adaptive mesh metrics
      //
      if (dftParams.verbosity >= 4)
        {
          pcout << printCommand << std::endl
                << " num elements: "
                << parallelTriangulation.n_global_active_cells()
                << ", min element length: " << minElemLength << std::endl;
        }

      checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
                                                     numLocallyOwnedCells,
                                                     interpool_comm1);
      checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
                                                     numLocallyOwnedCells,
                                                     interpool_comm2);
    }

    void
    computeLocalFiniteElementError(
      const dealii::DoFHandler<3> &                         dofHandler,
      const std::vector<const distributedCPUVec<double> *> &eigenVectorsArray,
      std::vector<double> &                                 errorInEachCell,
      const unsigned int                                    FEOrder)
    {
      typename dealii::DoFHandler<3>::active_cell_iterator cell, endc;
      cell = dofHandler.begin_active();
      endc = dofHandler.end();

      errorInEachCell.clear();

      //
      // create some FE data structures
      //
      dealii::QGauss<3>   quadrature(FEOrder + 1);
      dealii::FEValues<3> fe_values(dofHandler.get_fe(),
                                    quadrature,
                                    dealii::update_values |
                                      dealii::update_JxW_values |
                                      dealii::update_3rd_derivatives);
      const unsigned int  num_quad_points = quadrature.size();

      std::vector<dealii::Tensor<3, 3, double>> thirdDerivatives(
        num_quad_points);

      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);

              const dealii::Point<3> center(cell->center());
              double                 currentMeshSize =
                cell->minimum_vertex_distance(); // cell->diameter();
              //
              // Estimate the error for the current mesh
              //
              double derPsiSquare = 0.0;
              for (unsigned int iwave = 0; iwave < eigenVectorsArray.size();
                   ++iwave)
                {
                  fe_values.get_function_third_derivatives(
                    *eigenVectorsArray[iwave], thirdDerivatives);
                  for (unsigned int q_point = 0; q_point < num_quad_points;
                       ++q_point)
                    {
                      double sum = 0.0;
                      for (unsigned int i = 0; i < 3; ++i)
                        {
                          for (unsigned int j = 0; j < 3; ++j)
                            {
                              for (unsigned int k = 0; k < 3; ++k)
                                {
                                  sum += std::abs(
                                           thirdDerivatives[q_point][i][j][k]) *
                                         std::abs(
                                           thirdDerivatives[q_point][i][j][k]);
                                }
                            }
                        }

                      derPsiSquare += sum * fe_values.JxW(q_point);
                    } // q_point
                }     // iwave
              double exponent = 4.0;
              double error    = pow(currentMeshSize, exponent) * derPsiSquare;
              errorInEachCell.push_back(error);
            }
          else
            {
              errorInEachCell.push_back(0.0);
            }
        }
    }


  } // namespace internal

  void triangulationManager::generateCoarseMesh(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation)
  {
    //
    // compute magnitudes of domainBounding Vectors
    //
    const double domainBoundingVectorMag1 =
      sqrt(d_domainBoundingVectors[0][0] * d_domainBoundingVectors[0][0] +
           d_domainBoundingVectors[0][1] * d_domainBoundingVectors[0][1] +
           d_domainBoundingVectors[0][2] * d_domainBoundingVectors[0][2]);
    const double domainBoundingVectorMag2 =
      sqrt(d_domainBoundingVectors[1][0] * d_domainBoundingVectors[1][0] +
           d_domainBoundingVectors[1][1] * d_domainBoundingVectors[1][1] +
           d_domainBoundingVectors[1][2] * d_domainBoundingVectors[1][2]);
    const double domainBoundingVectorMag3 =
      sqrt(d_domainBoundingVectors[2][0] * d_domainBoundingVectors[2][0] +
           d_domainBoundingVectors[2][1] * d_domainBoundingVectors[2][1] +
           d_domainBoundingVectors[2][2] * d_domainBoundingVectors[2][2]);

    unsigned int subdivisions[3];
    subdivisions[0] = 1.0;
    subdivisions[1] = 1.0;
    subdivisions[2] = 1.0;

    std::vector<double> numberIntervalsEachDirection;

    double largestMeshSizeAroundAtom = d_dftParams.meshSizeOuterBall;

    if (d_dftParams.useMeshSizesFromAtomsFile)
      {
        largestMeshSizeAroundAtom = 1e-6;
        for (unsigned int n = 0; n < d_atomPositions.size(); n++)
          {
            if (d_atomPositions[n][5] > largestMeshSizeAroundAtom)
              largestMeshSizeAroundAtom = d_atomPositions[n][5];
          }
      }

    if (d_dftParams.autoAdaptBaseMeshSize)
      {
        double baseMeshSize1, baseMeshSize2, baseMeshSize3;
        if (d_dftParams.periodicX || d_dftParams.periodicY ||
            d_dftParams.periodicZ)
          {
            const double targetBaseMeshSize =
              (std::min(std::min(domainBoundingVectorMag1,
                                 domainBoundingVectorMag2),
                        domainBoundingVectorMag3) > 50.0) ?
                (d_dftParams.reproducible_output ? 7.0 : 4.0) :
                std::max(2.0, largestMeshSizeAroundAtom);
            baseMeshSize1 = std::pow(2,
                                     round(log2(targetBaseMeshSize /
                                                largestMeshSizeAroundAtom))) *
                            largestMeshSizeAroundAtom;
            baseMeshSize2 = std::pow(2,
                                     round(log2(targetBaseMeshSize /
                                                largestMeshSizeAroundAtom))) *
                            largestMeshSizeAroundAtom;
            baseMeshSize3 = std::pow(2,
                                     round(log2(targetBaseMeshSize /
                                                largestMeshSizeAroundAtom))) *
                            largestMeshSizeAroundAtom;
          }
        else
          {
            baseMeshSize1 =
              std::pow(2,
                       round(
                         log2((d_dftParams.reproducible_output ?
                                 std::min(domainBoundingVectorMag1 / 8.0, 8.0) :
                                 4.0) /
                              largestMeshSizeAroundAtom))) *
              largestMeshSizeAroundAtom;
            baseMeshSize2 =
              std::pow(2,
                       round(
                         log2((d_dftParams.reproducible_output ?
                                 std::min(domainBoundingVectorMag2 / 8.0, 8.0) :
                                 4.0) /
                              largestMeshSizeAroundAtom))) *
              largestMeshSizeAroundAtom;
            baseMeshSize3 =
              std::pow(2,
                       round(
                         log2((d_dftParams.reproducible_output ?
                                 std::min(domainBoundingVectorMag3 / 8.0, 8.0) :
                                 4.0) /
                              largestMeshSizeAroundAtom))) *
              largestMeshSizeAroundAtom;
          }

        numberIntervalsEachDirection.push_back(domainBoundingVectorMag1 /
                                               baseMeshSize1);
        numberIntervalsEachDirection.push_back(domainBoundingVectorMag2 /
                                               baseMeshSize2);
        numberIntervalsEachDirection.push_back(domainBoundingVectorMag3 /
                                               baseMeshSize3);
      }
    else
      {
        numberIntervalsEachDirection.push_back(domainBoundingVectorMag1 /
                                               d_dftParams.meshSizeOuterDomain);
        numberIntervalsEachDirection.push_back(domainBoundingVectorMag2 /
                                               d_dftParams.meshSizeOuterDomain);
        numberIntervalsEachDirection.push_back(domainBoundingVectorMag3 /
                                               d_dftParams.meshSizeOuterDomain);
      }

    dealii::Point<3> vector1(d_domainBoundingVectors[0][0],
                             d_domainBoundingVectors[0][1],
                             d_domainBoundingVectors[0][2]);
    dealii::Point<3> vector2(d_domainBoundingVectors[1][0],
                             d_domainBoundingVectors[1][1],
                             d_domainBoundingVectors[1][2]);
    dealii::Point<3> vector3(d_domainBoundingVectors[2][0],
                             d_domainBoundingVectors[2][1],
                             d_domainBoundingVectors[2][2]);

    //
    // Generate coarse mesh
    //
    dealii::Point<3> basisVectors[3] = {vector1, vector2, vector3};


    for (unsigned int i = 0; i < 3; i++)
      {
        const double temp = numberIntervalsEachDirection[i] -
                            std::floor(numberIntervalsEachDirection[i]);
        if (temp >= 0.5)
          subdivisions[i] = std::ceil(numberIntervalsEachDirection[i]);
        else
          subdivisions[i] = std::floor(numberIntervalsEachDirection[i]);
      }


    dealii::GridGenerator::subdivided_parallelepiped<3>(parallelTriangulation,
                                                        subdivisions,
                                                        basisVectors);

    //
    // Translate the main grid so that midpoint is at center
    //
    const dealii::Point<3> translation = 0.5 * (vector1 + vector2 + vector3);
    dealii::GridTools::shift(-translation, parallelTriangulation);

    //
    // collect periodic faces of the first level mesh to set up periodic
    // boundary conditions later
    //
    meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,
                                                 d_domainBoundingVectors,
                                                 d_mpiCommParent,
                                                 d_dftParams);

    if (d_dftParams.verbosity >= 4)
      pcout << std::endl
            << "Coarse triangulation number of elements: "
            << parallelTriangulation.n_global_active_cells() << std::endl;
  }

  bool triangulationManager::refinementAlgorithmA(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
    std::vector<unsigned int> &             locallyOwnedCellsRefineFlags,
    std::map<dealii::CellId, unsigned int> &cellIdToCellRefineFlagMapLocal,
    const bool                              smoothenCellsOnPeriodicBoundary,
    const double                            smootheningFactor)
  {
    //
    // compute magnitudes of domainBounding Vectors
    //
    const double domainBoundingVectorMag1 =
      sqrt(d_domainBoundingVectors[0][0] * d_domainBoundingVectors[0][0] +
           d_domainBoundingVectors[0][1] * d_domainBoundingVectors[0][1] +
           d_domainBoundingVectors[0][2] * d_domainBoundingVectors[0][2]);
    const double domainBoundingVectorMag2 =
      sqrt(d_domainBoundingVectors[1][0] * d_domainBoundingVectors[1][0] +
           d_domainBoundingVectors[1][1] * d_domainBoundingVectors[1][1] +
           d_domainBoundingVectors[1][2] * d_domainBoundingVectors[1][2]);
    const double domainBoundingVectorMag3 =
      sqrt(d_domainBoundingVectors[2][0] * d_domainBoundingVectors[2][0] +
           d_domainBoundingVectors[2][1] * d_domainBoundingVectors[2][1] +
           d_domainBoundingVectors[2][2] * d_domainBoundingVectors[2][2]);

    locallyOwnedCellsRefineFlags.clear();
    cellIdToCellRefineFlagMapLocal.clear();
    typename dealii::parallel::distributed::Triangulation<
      3>::active_cell_iterator cell,
      endc;
    cell = parallelTriangulation.begin_active();
    endc = parallelTriangulation.end();

    std::map<dealii::CellId, unsigned int> cellIdToLocallyOwnedId;
    unsigned int                           locallyOwnedCount = 0;

    bool   isAnyCellRefined           = false;
    double smallestMeshSizeAroundAtom = d_dftParams.meshSizeOuterBall;

    if (d_dftParams.useMeshSizesFromAtomsFile)
      {
        smallestMeshSizeAroundAtom = 1e+6;
        for (unsigned int n = 0; n < d_atomPositions.size(); n++)
          {
            if (d_atomPositions[n][5] < smallestMeshSizeAroundAtom)
              smallestMeshSizeAroundAtom = d_atomPositions[n][5];
          }
      }

    std::vector<double>       atomPointsLocal;
    std::vector<unsigned int> atomIdsLocal;
    std::vector<double>       meshSizeAroundAtomLocalAtoms;
    std::vector<double>       outerAtomBallRadiusLocalAtoms;
    for (unsigned int iAtom = 0;
         iAtom < (d_atomPositions.size() + d_imageAtomPositions.size());
         iAtom++)
      {
        if (iAtom < d_atomPositions.size())
          {
            atomPointsLocal.push_back(d_atomPositions[iAtom][2]);
            atomPointsLocal.push_back(d_atomPositions[iAtom][3]);
            atomPointsLocal.push_back(d_atomPositions[iAtom][4]);
            atomIdsLocal.push_back(iAtom);

            meshSizeAroundAtomLocalAtoms.push_back(
              d_dftParams.useMeshSizesFromAtomsFile ?
                d_atomPositions[iAtom][5] :
                d_dftParams.meshSizeOuterBall);
            outerAtomBallRadiusLocalAtoms.push_back(
              d_dftParams.useMeshSizesFromAtomsFile ?
                d_atomPositions[iAtom][6] :
                d_dftParams.outerAtomBallRadius);
          }
        else
          {
            const unsigned int iImageCharge = iAtom - d_atomPositions.size();
            atomPointsLocal.push_back(d_imageAtomPositions[iImageCharge][0]);
            atomPointsLocal.push_back(d_imageAtomPositions[iImageCharge][1]);
            atomPointsLocal.push_back(d_imageAtomPositions[iImageCharge][2]);
            const unsigned int imageChargeId = d_imageIds[iImageCharge];
            atomIdsLocal.push_back(imageChargeId);

            meshSizeAroundAtomLocalAtoms.push_back(
              d_dftParams.useMeshSizesFromAtomsFile ?
                d_atomPositions[imageChargeId][5] :
                d_dftParams.meshSizeOuterBall);
            outerAtomBallRadiusLocalAtoms.push_back(
              d_dftParams.useMeshSizesFromAtomsFile ?
                d_atomPositions[imageChargeId][6] :
                d_dftParams.outerAtomBallRadius);
          }
      }

    //
    //
    //
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            cellIdToLocallyOwnedId[cell->id()] = locallyOwnedCount;
            locallyOwnedCount++;

            const dealii::Point<3> center(cell->center());
            double currentMeshSize = cell->minimum_vertex_distance();

            //
            // compute projection of the vector joining the center of domain and
            // centroid of cell onto each of the domain bounding vectors
            //
            double projComponent_1 =
              (center[0] * d_domainBoundingVectors[0][0] +
               center[1] * d_domainBoundingVectors[0][1] +
               center[2] * d_domainBoundingVectors[0][2]) /
              domainBoundingVectorMag1;
            double projComponent_2 =
              (center[0] * d_domainBoundingVectors[1][0] +
               center[1] * d_domainBoundingVectors[1][1] +
               center[2] * d_domainBoundingVectors[1][2]) /
              domainBoundingVectorMag2;
            double projComponent_3 =
              (center[0] * d_domainBoundingVectors[2][0] +
               center[1] * d_domainBoundingVectors[2][1] +
               center[2] * d_domainBoundingVectors[2][2]) /
              domainBoundingVectorMag3;


            bool cellRefineFlag = false;


            // loop over all atoms
            double           distanceToClosestAtom = 1e8;
            dealii::Point<3> closestAtom;
            unsigned int     closestId = 0;
            for (unsigned int n = 0; n < atomPointsLocal.size() / 3; n++)
              {
                dealii::Point<3> atom(atomPointsLocal[3 * n],
                                      atomPointsLocal[3 * n + 1],
                                      atomPointsLocal[3 * n + 2]);
                if (center.distance(atom) < distanceToClosestAtom)
                  {
                    distanceToClosestAtom = center.distance(atom);
                    closestAtom           = atom;
                    closestId             = n;
                  }
              }

            if (d_dftParams.autoAdaptBaseMeshSize)
              {
                bool inOuterAtomBall = false;

                if (distanceToClosestAtom <=
                    outerAtomBallRadiusLocalAtoms[closestId])
                  inOuterAtomBall = true;

                if (inOuterAtomBall &&
                    (currentMeshSize >
                     1.2 * meshSizeAroundAtomLocalAtoms[closestId]))
                  cellRefineFlag = true;

                bool inInnerAtomBall = false;

                if (distanceToClosestAtom <= d_dftParams.innerAtomBallRadius)
                  inInnerAtomBall = true;

                if (inInnerAtomBall &&
                    currentMeshSize > 1.2 * d_dftParams.meshSizeInnerBall)
                  cellRefineFlag = true;
              }
            else
              {
                bool inOuterAtomBall = false;

                if (distanceToClosestAtom <=
                    outerAtomBallRadiusLocalAtoms[closestId])
                  inOuterAtomBall = true;

                if (inOuterAtomBall &&
                    (currentMeshSize > meshSizeAroundAtomLocalAtoms[closestId]))
                  cellRefineFlag = true;

                bool inInnerAtomBall = false;

                if (distanceToClosestAtom <= d_dftParams.innerAtomBallRadius)
                  inInnerAtomBall = true;

                if (inInnerAtomBall &&
                    currentMeshSize > d_dftParams.meshSizeInnerBall)
                  cellRefineFlag = true;
              }

            /*
            if (d_dftParams.autoAdaptBaseMeshSize  &&
            !d_dftParams.reproducible_output)
            {
              bool inBiggerAtomBall = false;

              if(distanceToClosestAtom <= 10.0)
                inBiggerAtomBall = true;

              if(inBiggerAtomBall && currentMeshSize > 6.0)
                cellRefineFlag = true;
            }
            */

            dealii::MappingQ1<3, 3> mapping;
            try
              {
                dealii::Point<3> p_cell =
                  mapping.transform_real_to_unit_cell(cell, closestAtom);
                double dist =
                  dealii::GeometryInfo<3>::distance_to_unit_cell(p_cell);

                if (dist < 1e-08 &&
                    ((currentMeshSize > d_dftParams.meshSizeInnerBall) ||
                     (currentMeshSize >
                      1.2 * meshSizeAroundAtomLocalAtoms[closestId])))
                  cellRefineFlag = true;
              }
            catch (dealii::MappingQ1<3>::ExcTransformationFailed)
              {}

            cellRefineFlag =
              dealii::Utilities::MPI::max((unsigned int)cellRefineFlag,
                                          interpoolcomm);
            cellRefineFlag =
              dealii::Utilities::MPI::max((unsigned int)cellRefineFlag,
                                          interBandGroupComm);

            //
            // set refine flags
            if (cellRefineFlag)
              {
                locallyOwnedCellsRefineFlags.push_back(1);
                cellIdToCellRefineFlagMapLocal[cell->id()] = 1;
                cell->set_refine_flag();
                isAnyCellRefined = true;
              }
            else
              {
                cellIdToCellRefineFlagMapLocal[cell->id()] = 0;
                locallyOwnedCellsRefineFlags.push_back(0);
              }
          }
      }


    //
    // refine cells on periodic boundary if their length is greater than
    // mesh size around atom by a factor (set by smootheningFactor)
    //
    if (smoothenCellsOnPeriodicBoundary)
      {
        locallyOwnedCount = 0;
        cell              = parallelTriangulation.begin_active();
        endc              = parallelTriangulation.end();

        const unsigned int faces_per_cell =
          dealii::GeometryInfo<3>::faces_per_cell;

        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                if (cell->at_boundary() &&
                    cell->minimum_vertex_distance() >
                      (d_dftParams.autoAdaptBaseMeshSize ? 1.5 : 1) *
                        smootheningFactor * smallestMeshSizeAroundAtom &&
                    !cell->refine_flag_set())
                  for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
                    if (cell->has_periodic_neighbor(iFace))
                      {
                        cell->set_refine_flag();
                        isAnyCellRefined = true;
                        locallyOwnedCellsRefineFlags
                          [cellIdToLocallyOwnedId[cell->id()]]     = 1;
                        cellIdToCellRefineFlagMapLocal[cell->id()] = 1;
                        break;
                      }
                locallyOwnedCount++;
              }
          }
      }

    return isAnyCellRefined;
  }

  //
  // internal function which sets refinement flags to have consistent refinement
  // across periodic boundary
  //
  bool triangulationManager::consistentPeriodicBoundaryRefinement(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
    std::vector<unsigned int> &             locallyOwnedCellsRefineFlags,
    std::map<dealii::CellId, unsigned int> &cellIdToCellRefineFlagMapLocal)
  {
    locallyOwnedCellsRefineFlags.clear();
    cellIdToCellRefineFlagMapLocal.clear();
    typename dealii::parallel::distributed::Triangulation<
      3>::active_cell_iterator cell,
      endc;
    cell = parallelTriangulation.begin_active();
    endc = parallelTriangulation.end();

    //
    // populate maps refinement flag maps to zero values
    //
    std::map<dealii::CellId, unsigned int> cellIdToLocallyOwnedId;
    unsigned int                           locallyOwnedCount = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cellIdToLocallyOwnedId[cell->id()] = locallyOwnedCount;
          locallyOwnedCellsRefineFlags.push_back(0);
          cellIdToCellRefineFlagMapLocal[cell->id()] = 0;
          locallyOwnedCount++;
        }


    cell = parallelTriangulation.begin_active();
    endc = parallelTriangulation.end();

    //
    // go to each locally owned or ghost cell which has a face on the periodic
    // boundary-> query if cell has a periodic neighbour which is coarser -> if
    // yes and the coarse cell is locally owned set refinement flag on that cell
    //
    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    bool               isAnyCellRefined = false;
    for (; cell != endc; ++cell)
      {
        if ((cell->is_locally_owned() || cell->is_ghost()) &&
            cell->at_boundary())
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            if (cell->has_periodic_neighbor(iFace))
              if (cell->periodic_neighbor_is_coarser(iFace))
                {
                  typename dealii::parallel::distributed::Triangulation<
                    3>::active_cell_iterator periodicCell =
                    cell->periodic_neighbor(iFace);

                  if (periodicCell->is_locally_owned())
                    {
                      locallyOwnedCellsRefineFlags
                        [cellIdToLocallyOwnedId[periodicCell->id()]]     = 1;
                      cellIdToCellRefineFlagMapLocal[periodicCell->id()] = 1;
                      periodicCell->set_refine_flag();

                      isAnyCellRefined = true;
                    }
                }
      }
    return isAnyCellRefined;
  }

  //
  // check that triangulation has consistent refinement across periodic boundary
  //
  bool triangulationManager::checkPeriodicSurfaceRefinementConsistency(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation)
  {
    typename dealii::parallel::distributed::Triangulation<
      3>::active_cell_iterator cell,
      endc;
    cell = parallelTriangulation.begin_active();
    endc = parallelTriangulation.end();

    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;

    unsigned int notConsistent = 0;
    for (; cell != endc; ++cell)
      if ((cell->is_locally_owned() || cell->is_ghost()) && cell->at_boundary())
        for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
          if (cell->has_periodic_neighbor(iFace))
            {
              typename dealii::parallel::distributed::Triangulation<
                3>::active_cell_iterator periodicCell =
                cell->periodic_neighbor(iFace);
              if (periodicCell->is_locally_owned() || cell->is_locally_owned())
                if (cell->periodic_neighbor_is_coarser(iFace) ||
                    periodicCell->has_children())
                  notConsistent = 1;
            }
    notConsistent =
      dealii::Utilities::MPI::max(notConsistent, mpi_communicator);
    return notConsistent == 1 ? false : true;
  }


  //
  // check that FEOrder=1 dofHandler using the triangulation has parallel
  // consistent combined hanging node and periodic constraints
  //
  bool triangulationManager::checkConstraintsConsistency(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation)
  {
    dealii::FESystem<3> FE(
      dealii::FE_Q<3>(dealii::QGaussLobatto<1>(d_FEOrder + 1)), 1);
    // dealii::FESystem<3> FE(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(1+1)),
    // 1);
    dealii::DoFHandler<3> dofHandler;
    dofHandler.reinit(parallelTriangulation);
    dofHandler.distribute_dofs(FE);
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dofHandler,
                                                    locally_relevant_dofs);

    dealii::AffineConstraints<double> constraints;
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dofHandler, constraints);
    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::DoFHandler<3>::cell_iterator>>
      periodicity_vector;

    // create unitVectorsXYZ
    std::vector<std::vector<double>> unitVectorsXYZ;
    unitVectorsXYZ.resize(3);

    for (int i = 0; i < 3; ++i)
      {
        unitVectorsXYZ[i].resize(3, 0.0);
        unitVectorsXYZ[i][i] = 0.0;
      }

    std::vector<dealii::Tensor<1, 3>> offsetVectors;
    // resize offset vectors
    offsetVectors.resize(3);

    for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          {
            offsetVectors[i][j] =
              unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];
          }
      }

    const std::array<int, 3> periodic = {d_dftParams.periodicX,
                                         d_dftParams.periodicY,
                                         d_dftParams.periodicZ};

    std::vector<int> periodicDirectionVector;
    for (unsigned int d = 0; d < 3; ++d)
      {
        if (periodic[d] == 1)
          {
            periodicDirectionVector.push_back(d);
          }
      }

    for (int i = 0; i < std::accumulate(periodic.begin(), periodic.end(), 0);
         ++i)
      {
        dealii::GridTools::collect_periodic_faces(
          dofHandler,
          2 * i + 1,
          2 * i + 2,
          periodicDirectionVector[i],
          periodicity_vector,
          offsetVectors[periodicDirectionVector[i]]);
      }

    dealii::DoFTools::make_periodicity_constraints<3, 3>(periodicity_vector,
                                                         constraints);
    constraints.close();

    dealii::IndexSet locally_active_dofs_debug;
    dealii::DoFTools::extract_locally_active_dofs(dofHandler,
                                                  locally_active_dofs_debug);

    const std::vector<dealii::IndexSet> &locally_owned_dofs_debug =
      dealii::Utilities::MPI::all_gather(mpi_communicator,
                                         dofHandler.locally_owned_dofs());

    return constraints.is_consistent_in_parallel(
      locally_owned_dofs_debug,
      locally_active_dofs_debug,
      mpi_communicator,
      !d_dftParams.reproducible_output);
  }

  //
  // generate mesh based on a-posteriori estimates
  //
  void
  triangulationManager::generateAutomaticMeshApriori(
    const dealii::DoFHandler<3> &                    dofHandler,
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
    const std::vector<distributedCPUVec<double>> &   eigenVectorsArrayIn,
    const unsigned int                               FEOrder)
  {
    double topfrac    = d_dftParams.topfrac;
    double bottomfrac = 0.0;

    //
    // create an array of pointers holding the eigenVectors on starting mesh
    //
    unsigned int numberWaveFunctionsEstimate = eigenVectorsArrayIn.size();
    std::vector<const distributedCPUVec<double> *> eigenVectorsArrayOfPtrsIn(
      numberWaveFunctionsEstimate);
    for (int iWave = 0; iWave < numberWaveFunctionsEstimate; ++iWave)
      {
        eigenVectorsArrayOfPtrsIn[iWave] = &eigenVectorsArrayIn[iWave];
      }

    //
    // create storage for storing errors in each cell
    //
    dealii::Vector<double> estimated_error_per_cell(
      parallelTriangulation.n_active_cells());
    std::vector<double> errorInEachCell;

    //
    // fill in the errors corresponding to each cell
    //

    internal::computeLocalFiniteElementError(dofHandler,
                                             eigenVectorsArrayOfPtrsIn,
                                             errorInEachCell,
                                             FEOrder);

    for (unsigned int i = 0; i < errorInEachCell.size(); ++i)
      estimated_error_per_cell(i) = errorInEachCell[i];

    //
    // print certain error metrics of each cell
    //
    if (d_dftParams.verbosity >= 4)
      {
        double maxErrorIndicator =
          *std::max_element(errorInEachCell.begin(), errorInEachCell.end());
        double globalMaxError =
          dealii::Utilities::MPI::max(maxErrorIndicator, mpi_communicator);
        double errorSum =
          std::accumulate(errorInEachCell.begin(), errorInEachCell.end(), 0.0);
        double globalSum =
          dealii::Utilities::MPI::sum(errorSum, mpi_communicator);
        pcout << " Sum Error of all Cells: " << globalSum
              << " Max Error of all Cells:" << globalMaxError << std::endl;
      }

    //
    // reset moved parallel triangulation to unmoved triangulation
    //
    resetMesh(d_parallelTriangulationUnmoved, parallelTriangulation);

    //
    // prepare all meshes for refinement using estimated errors in each cell
    //
    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number(parallelTriangulation,
                                      estimated_error_per_cell,
                                      topfrac,
                                      bottomfrac);

    parallelTriangulation.prepare_coarsening_and_refinement();
    parallelTriangulation.execute_coarsening_and_refinement();

    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number(d_parallelTriangulationUnmoved,
                                      estimated_error_per_cell,
                                      topfrac,
                                      bottomfrac);

    d_parallelTriangulationUnmoved.prepare_coarsening_and_refinement();
    d_parallelTriangulationUnmoved.execute_coarsening_and_refinement();

    std::string printCommand =
      "Automatic Adaptive Mesh Refinement Based Triangulation Summary";
    internal::computeMeshMetrics(parallelTriangulation,
                                 printCommand,
                                 pcout,
                                 mpi_communicator,
                                 interpoolcomm,
                                 interBandGroupComm,
                                 d_dftParams);
  }


  void triangulationManager::generateMesh(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
    dealii::parallel::distributed::Triangulation<3> &serialTriangulation,
    const bool                                       generateSerialTria)
  {
    generateCoarseMesh(parallelTriangulation);
    if (generateSerialTria)
      {
        generateCoarseMesh(serialTriangulation);
        AssertThrow(
          parallelTriangulation.n_global_active_cells() ==
            serialTriangulation.n_global_active_cells(),
          dealii::ExcMessage(
            "Number of coarse mesh cells are different in serial and parallel triangulations."));
      }

    d_parallelTriaCurrentRefinement.clear();
    if (generateSerialTria)
      d_serialTriaCurrentRefinement.clear();

    //
    // Multilayer refinement. Refinement algorithm is progressively modified
    // if check of parallel consistency of combined periodic
    // and hanging node constraitns fails (Related to
    // https://github.com/dealii/dealii/issues/7053).
    //

    //
    // STAGE0: Call only refinementAlgorithmA. Multilevel refinement is
    // performed until refinementAlgorithmA does not set refinement flags on any
    // cell.
    //
    unsigned int numLevels  = 0;
    bool         refineFlag = true;
    while (refineFlag)
      {
        refineFlag = false;
        std::vector<unsigned int>              locallyOwnedCellsRefineFlags;
        std::map<dealii::CellId, unsigned int> cellIdToCellRefineFlagMapLocal;

        refineFlag = refinementAlgorithmA(parallelTriangulation,
                                          locallyOwnedCellsRefineFlags,
                                          cellIdToCellRefineFlagMapLocal);

        // This sets the global refinement sweep flag
        refineFlag = dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                 mpi_communicator);

        // Refine
        if (refineFlag)
          {
            if (numLevels < d_max_refinement_steps)
              {
                if (d_dftParams.verbosity >= 4)
                  pcout << "refinement in progress, level: " << numLevels
                        << std::endl;

                if (generateSerialTria)
                  {
                    d_serialTriaCurrentRefinement.push_back(
                      std::vector<bool>());

                    // First refine serial mesh
                    refineSerialMesh(cellIdToCellRefineFlagMapLocal,
                                     mpi_communicator,
                                     serialTriangulation,
                                     parallelTriangulation,
                                     d_serialTriaCurrentRefinement[numLevels]);
                  }

                d_parallelTriaCurrentRefinement.push_back(std::vector<bool>());
                parallelTriangulation.save_refine_flags(
                  d_parallelTriaCurrentRefinement[numLevels]);

                parallelTriangulation.execute_coarsening_and_refinement();
                numLevels++;
              }
            else
              {
                refineFlag = false;
              }
          }
      }

    if (!d_dftParams.reproducible_output &&
        !d_dftParams.createConstraintsFromSerialDofhandler)
      {
        //
        // STAGE1: This stage is only activated if combined periodic and hanging
        // node constraints are not consistent in parallel. Call
        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement
        // alternatively. In the call to refinementAlgorithmA there is no
        // additional reduction of adaptivity performed on the periodic
        // boundary. Multilevel refinement is performed until both
        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement do not
        // set refinement flags on any cell.
        //
        if (!checkConstraintsConsistency(parallelTriangulation))
          {
            refineFlag = true;
            while (refineFlag)
              {
                refineFlag = false;
                std::vector<unsigned int> locallyOwnedCellsRefineFlags;
                std::map<dealii::CellId, unsigned int>
                  cellIdToCellRefineFlagMapLocal;
                if (numLevels % 2 == 0)
                  {
                    refineFlag =
                      refinementAlgorithmA(parallelTriangulation,
                                           locallyOwnedCellsRefineFlags,
                                           cellIdToCellRefineFlagMapLocal);

                    // This sets the global refinement sweep flag
                    refineFlag =
                      dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                  mpi_communicator);

                    // try the other type of refinement to prevent while loop
                    // from ending prematurely
                    if (!refineFlag)
                      {
                        // call refinement algorithm  which sets refinement
                        // flags such as to create consistent refinement across
                        // periodic boundary
                        refineFlag = consistentPeriodicBoundaryRefinement(
                          parallelTriangulation,
                          locallyOwnedCellsRefineFlags,
                          cellIdToCellRefineFlagMapLocal);

                        // This sets the global refinement sweep flag
                        refineFlag =
                          dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                      mpi_communicator);
                      }
                  }
                else
                  {
                    // call refinement algorithm  which sets refinement flags
                    // such as to create consistent refinement across periodic
                    // boundary
                    refineFlag = consistentPeriodicBoundaryRefinement(
                      parallelTriangulation,
                      locallyOwnedCellsRefineFlags,
                      cellIdToCellRefineFlagMapLocal);

                    // This sets the global refinement sweep flag
                    refineFlag =
                      dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                  mpi_communicator);

                    // try the other type of refinement to prevent while loop
                    // from ending prematurely
                    if (!refineFlag)
                      {
                        refineFlag =
                          refinementAlgorithmA(parallelTriangulation,
                                               locallyOwnedCellsRefineFlags,
                                               cellIdToCellRefineFlagMapLocal);

                        // This sets the global refinement sweep flag
                        refineFlag =
                          dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                      mpi_communicator);
                      }
                  }

                // Refine
                if (refineFlag)
                  {
                    if (numLevels < d_max_refinement_steps)
                      {
                        if (d_dftParams.verbosity >= 4)
                          pcout
                            << "refinement in progress, level: " << numLevels
                            << std::endl;

                        if (generateSerialTria)
                          {
                            d_serialTriaCurrentRefinement.push_back(
                              std::vector<bool>());

                            // First refine serial mesh
                            refineSerialMesh(
                              cellIdToCellRefineFlagMapLocal,
                              mpi_communicator,
                              serialTriangulation,
                              parallelTriangulation,
                              d_serialTriaCurrentRefinement[numLevels]);
                          }

                        d_parallelTriaCurrentRefinement.push_back(
                          std::vector<bool>());
                        parallelTriangulation.save_refine_flags(
                          d_parallelTriaCurrentRefinement[numLevels]);

                        parallelTriangulation
                          .execute_coarsening_and_refinement();

                        numLevels++;
                      }
                    else
                      {
                        refineFlag = false;
                      }
                  }
              }
          }

        //
        // STAGE2: This stage is only activated if combined periodic and hanging
        // node constraints are still not consistent in parallel. Call
        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement
        // alternatively. In the call to refinementAlgorithmA there is an
        // additional reduction of adaptivity performed on the periodic boundary
        // such that the maximum cell length on the periodic boundary is less
        // than two times the MESH SIZE AROUND ATOM. Multilevel refinement is
        // performed until both refinementAlgorithmAand
        // consistentPeriodicBoundaryRefinement do not set refinement flags on
        // any cell.
        //
        if (!checkConstraintsConsistency(parallelTriangulation))
          {
            refineFlag = true;
            while (refineFlag)
              {
                refineFlag = false;
                std::vector<unsigned int> locallyOwnedCellsRefineFlags;
                std::map<dealii::CellId, unsigned int>
                  cellIdToCellRefineFlagMapLocal;
                if (numLevels % 2 == 0)
                  {
                    refineFlag =
                      refinementAlgorithmA(parallelTriangulation,
                                           locallyOwnedCellsRefineFlags,
                                           cellIdToCellRefineFlagMapLocal,
                                           true,
                                           2.0);

                    // This sets the global refinement sweep flag
                    refineFlag =
                      dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                  mpi_communicator);

                    // try the other type of refinement to prevent while loop
                    // from ending prematurely
                    if (!refineFlag)
                      {
                        // call refinement algorithm  which sets refinement
                        // flags such as to create consistent refinement across
                        // periodic boundary
                        refineFlag = consistentPeriodicBoundaryRefinement(
                          parallelTriangulation,
                          locallyOwnedCellsRefineFlags,
                          cellIdToCellRefineFlagMapLocal);

                        // This sets the global refinement sweep flag
                        refineFlag =
                          dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                      mpi_communicator);
                      }
                  }
                else
                  {
                    // call refinement algorithm  which sets refinement flags
                    // such as to create consistent refinement across periodic
                    // boundary
                    refineFlag = consistentPeriodicBoundaryRefinement(
                      parallelTriangulation,
                      locallyOwnedCellsRefineFlags,
                      cellIdToCellRefineFlagMapLocal);

                    // This sets the global refinement sweep flag
                    refineFlag =
                      dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                  mpi_communicator);

                    // try the other type of refinement to prevent while loop
                    // from ending prematurely
                    if (!refineFlag)
                      {
                        refineFlag =
                          refinementAlgorithmA(parallelTriangulation,
                                               locallyOwnedCellsRefineFlags,
                                               cellIdToCellRefineFlagMapLocal,
                                               true,
                                               2.0);

                        // This sets the global refinement sweep flag
                        refineFlag =
                          dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                      mpi_communicator);
                      }
                  }

                // Refine
                if (refineFlag)
                  {
                    if (numLevels < d_max_refinement_steps)
                      {
                        if (d_dftParams.verbosity >= 4)
                          pcout
                            << "refinement in progress, level: " << numLevels
                            << std::endl;

                        if (generateSerialTria)
                          {
                            d_serialTriaCurrentRefinement.push_back(
                              std::vector<bool>());

                            // First refine serial mesh
                            refineSerialMesh(
                              cellIdToCellRefineFlagMapLocal,
                              mpi_communicator,
                              serialTriangulation,
                              parallelTriangulation,
                              d_serialTriaCurrentRefinement[numLevels]);
                          }

                        d_parallelTriaCurrentRefinement.push_back(
                          std::vector<bool>());
                        parallelTriangulation.save_refine_flags(
                          d_parallelTriaCurrentRefinement[numLevels]);

                        parallelTriangulation
                          .execute_coarsening_and_refinement();
                        numLevels++;
                      }
                    else
                      {
                        refineFlag = false;
                      }
                  }
              }
          }

        //
        // STAGE3: This stage is only activated if combined periodic and hanging
        // node constraints are still not consistent in parallel. Call
        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement
        // alternatively. In the call to refinementAlgorithmA there is an
        // additional reduction of adaptivity performed on the periodic boundary
        // such that the maximum cell length on the periodic boundary is less
        // than MESH SIZE AROUND ATOM essentially ensuring uniform refinement on
        // the periodic boundary in the case of MESH SIZE AROUND ATOM being same
        // as MESH SIZE AT ATOM. Multilevel refinement is performed until both
        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement do not
        // set refinement flags on any cell.
        //
        if (!checkConstraintsConsistency(parallelTriangulation))
          {
            refineFlag = true;
            while (refineFlag)
              {
                refineFlag = false;
                std::vector<unsigned int> locallyOwnedCellsRefineFlags;
                std::map<dealii::CellId, unsigned int>
                  cellIdToCellRefineFlagMapLocal;
                if (numLevels % 2 == 0)
                  {
                    refineFlag =
                      refinementAlgorithmA(parallelTriangulation,
                                           locallyOwnedCellsRefineFlags,
                                           cellIdToCellRefineFlagMapLocal,
                                           true,
                                           1.0);

                    // This sets the global refinement sweep flag
                    refineFlag =
                      dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                  mpi_communicator);

                    // try the other type of refinement to prevent while loop
                    // from ending prematurely
                    if (!refineFlag)
                      {
                        // call refinement algorithm  which sets refinement
                        // flags such as to create consistent refinement across
                        // periodic boundary
                        refineFlag = consistentPeriodicBoundaryRefinement(
                          parallelTriangulation,
                          locallyOwnedCellsRefineFlags,
                          cellIdToCellRefineFlagMapLocal);

                        // This sets the global refinement sweep flag
                        refineFlag =
                          dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                      mpi_communicator);
                      }
                  }
                else
                  {
                    // call refinement algorithm  which sets refinement flags
                    // such as to create consistent refinement across periodic
                    // boundary
                    refineFlag = consistentPeriodicBoundaryRefinement(
                      parallelTriangulation,
                      locallyOwnedCellsRefineFlags,
                      cellIdToCellRefineFlagMapLocal);

                    // This sets the global refinement sweep flag
                    refineFlag =
                      dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                  mpi_communicator);

                    // try the other type of refinement to prevent while loop
                    // from ending prematurely
                    if (!refineFlag)
                      {
                        refineFlag =
                          refinementAlgorithmA(parallelTriangulation,
                                               locallyOwnedCellsRefineFlags,
                                               cellIdToCellRefineFlagMapLocal,
                                               true,
                                               1.0);

                        // This sets the global refinement sweep flag
                        refineFlag =
                          dealii::Utilities::MPI::max((unsigned int)refineFlag,
                                                      mpi_communicator);
                      }
                  }

                // Refine
                if (refineFlag)
                  {
                    if (numLevels < d_max_refinement_steps)
                      {
                        if (d_dftParams.verbosity >= 4)
                          pcout
                            << "refinement in progress, level: " << numLevels
                            << std::endl;

                        if (generateSerialTria)
                          {
                            d_serialTriaCurrentRefinement.push_back(
                              std::vector<bool>());

                            // First refine serial mesh
                            refineSerialMesh(
                              cellIdToCellRefineFlagMapLocal,
                              mpi_communicator,
                              serialTriangulation,
                              parallelTriangulation,
                              d_serialTriaCurrentRefinement[numLevels]);
                          }

                        d_parallelTriaCurrentRefinement.push_back(
                          std::vector<bool>());
                        parallelTriangulation.save_refine_flags(
                          d_parallelTriaCurrentRefinement[numLevels]);

                        parallelTriangulation
                          .execute_coarsening_and_refinement();
                        numLevels++;
                      }
                    else
                      {
                        refineFlag = false;
                      }
                  }
              }
          }

        if (checkConstraintsConsistency(parallelTriangulation))
          {
            if (d_dftParams.verbosity >= 4)
              pcout
                << "Hanging node and periodic constraints parallel consistency achieved."
                << std::endl;
          }
        else
          {
            if (d_dftParams.verbosity >= 4)
              pcout
                << "Hanging node and periodic constraints parallel consistency not achieved."
                << std::endl;

            AssertThrow(
              d_dftParams.createConstraintsFromSerialDofhandler,
              dealii::ExcMessage(
                "DFT-FE error: this is due to a known issue related to hanging node constraints in dealii. Please set CONSTRAINTS FROM SERIAL DOFHANDLER = true under the Boundary conditions subsection in the input parameters file to circumvent this issue."));
          }
      }

    //
    // compute some adaptive mesh metrics
    //
    double minElemLength = d_dftParams.meshSizeOuterDomain;
    double maxElemLength = 0.0;
    typename dealii::parallel::distributed::Triangulation<
      3>::active_cell_iterator cell,
      endc, cellDisp, cellForce;
    cell                              = parallelTriangulation.begin_active();
    endc                              = parallelTriangulation.end();
    unsigned int numLocallyOwnedCells = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            numLocallyOwnedCells++;
            if (cell->minimum_vertex_distance() < minElemLength)
              minElemLength = cell->minimum_vertex_distance();

            if (cell->minimum_vertex_distance() > maxElemLength)
              maxElemLength = cell->minimum_vertex_distance();
          }
      }

    minElemLength =
      dealii::Utilities::MPI::min(minElemLength, mpi_communicator);
    maxElemLength =
      dealii::Utilities::MPI::max(maxElemLength, mpi_communicator);

    //
    // print out adaptive mesh metrics and check mesh generation synchronization
    // across pools
    //
    if (d_dftParams.verbosity >= 4)
      {
        pcout << "Triangulation generation summary: " << std::endl
              << " num elements: "
              << parallelTriangulation.n_global_active_cells()
              << ", num refinement levels: " << numLevels
              << ", min element length: " << minElemLength
              << ", max element length: " << maxElemLength << std::endl;
      }

    internal::checkTriangulationEqualityAcrossProcessorPools(
      parallelTriangulation, numLocallyOwnedCells, interpoolcomm);
    internal::checkTriangulationEqualityAcrossProcessorPools(
      parallelTriangulation, numLocallyOwnedCells, interBandGroupComm);

    if (generateSerialTria)
      {
        const unsigned int numberGlobalCellsParallel =
          parallelTriangulation.n_global_active_cells();
        const unsigned int numberGlobalCellsSerial =
          serialTriangulation.n_global_active_cells();

        if (d_dftParams.verbosity >= 4)
          pcout << " numParallelCells: " << numberGlobalCellsParallel
                << ", numSerialCells: " << numberGlobalCellsSerial << std::endl;

        AssertThrow(
          numberGlobalCellsParallel == numberGlobalCellsSerial,
          dealii::ExcMessage(
            "Number of cells are different for parallel and serial triangulations"));
      }
    else
      {
        const unsigned int numberGlobalCellsParallel =
          parallelTriangulation.n_global_active_cells();

        if (d_dftParams.verbosity >= 4)
          pcout << " numParallelCells: " << numberGlobalCellsParallel
                << std::endl;
      }
  }


  //
  void
  triangulationManager::refineSerialMesh(
    const std::map<dealii::CellId, unsigned int>
      &             cellIdToCellRefineFlagMapLocal,
    const MPI_Comm &mpi_comm,
    dealii::parallel::distributed::Triangulation<3> &serialTriangulation,
    const dealii::parallel::distributed::Triangulation<3>
      &                parallelTriangulation,
    std::vector<bool> &serialTriaCurrentRefinement)

  {
    const unsigned int numberGlobalCellsSerial =
      serialTriangulation.n_global_active_cells();
    std::vector<unsigned int> refineFlagsSerialCells(numberGlobalCellsSerial,
                                                     0);

    dealii::BoundingBox<3> boundingBoxParallelTria =
      dealii::GridTools::compute_bounding_box(parallelTriangulation);

    unsigned int count = 0;
    for (auto cell : serialTriangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (boundingBoxParallelTria.point_inside(cell->center()))
            {
              std::map<dealii::CellId, unsigned int>::const_iterator iter =
                cellIdToCellRefineFlagMapLocal.find(cell->id());
              if (iter != cellIdToCellRefineFlagMapLocal.end())
                refineFlagsSerialCells[count] = iter->second;
            }
          count++;
        }

    MPI_Allreduce(MPI_IN_PLACE,
                  &refineFlagsSerialCells[0],
                  numberGlobalCellsSerial,
                  MPI_UNSIGNED,
                  MPI_SUM,
                  mpi_comm);

    count = 0;
    for (auto cell : serialTriangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          if (refineFlagsSerialCells[count] == 1)
            cell->set_refine_flag();
          count++;
        }

    serialTriangulation.save_refine_flags(serialTriaCurrentRefinement);
    serialTriangulation.execute_coarsening_and_refinement();
  }


} // namespace dftfe
