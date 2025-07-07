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
//  @author Nikhil Kodali


#include "groupSymmetry.h"
#include "linearAlgebraOperations.h"
#include <spglib.h>

namespace dftfe
{

  groupSymmetryClass::groupSymmetryClass(const MPI_Comm &mpi_comm_parent,
                                         const MPI_Comm &mpi_comm_domain,
                                         const bool      isGroupSymmetry,
                                         const bool      isTimeReversal)
    : d_mpiCommParent(mpi_comm_parent)
    , d_mpiCommDomain(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , d_isGroupSymmetry(isGroupSymmetry)
    , d_isTimeReversal(isTimeReversal)
  {}


  void
  groupSymmetryClass::initGroupSymmetry(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                     &BLASWrapperPtrHost,
    std::vector<std::vector<double>> &atomLocationsFractional,
    std::vector<std::vector<double>> &domainBoundingVectors,
    std::vector<bool>                &periodicBoundaryConditions,
    const bool                        isCollinearSpin)
  {
    d_numAtoms = atomLocationsFractional.size();
    d_atomicCoordsCart.clear();
    d_atomicCoordsFrac.clear();
    d_domainBoundingVectors.clear();
    d_domainBoundingVectorsInverse.clear();
    d_periodicBoundaryConditions.clear();
    d_atomicCoordsCart.resize(3 * d_numAtoms, 0.0);
    d_atomicCoordsFrac.resize(3 * d_numAtoms, 0.0);
    d_domainBoundingVectors.resize(9, 0.0);
    d_domainBoundingVectorsInverse.resize(9, 0.0);
    d_periodicBoundaryConditions = periodicBoundaryConditions;
    for (dftfe::uInt iAtom = 0; iAtom < d_numAtoms; ++iAtom)
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        d_atomicCoordsFrac[3 * iAtom + iDim] =
          atomLocationsFractional[iAtom][2 + iDim] -
          std::floor(atomLocationsFractional[iAtom][2 + iDim]);

    for (dftfe::uInt iVec = 0; iVec < 3; ++iVec)
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        d_domainBoundingVectors[3 * iVec + iDim] =
          domainBoundingVectors[iVec][iDim];
    auto inv3 = [](auto const &m) {
      auto cross = [](std::array<double, 3> a, std::array<double, 3> b) {
        return std::array<double, 3>{a[1] * b[2] - a[2] * b[1],
                                     a[2] * b[0] - a[0] * b[2],
                                     a[0] * b[1] - a[1] * b[0]};
      };
      auto   c0  = cross({m[3], m[4], m[5]}, {m[6], m[7], m[8]});
      auto   c1  = cross({m[6], m[7], m[8]}, {m[0], m[1], m[2]});
      auto   c2  = cross({m[0], m[1], m[2]}, {m[3], m[4], m[5]});
      double det = m[0] * c0[0] + m[1] * c0[1] + m[2] * c0[2];
      return std::vector<double>{c0[0] / det,
                                 c1[0] / det,
                                 c2[0] / det,
                                 c0[1] / det,
                                 c1[1] / det,
                                 c2[1] / det,
                                 c0[2] / det,
                                 c1[2] / det,
                                 c2[2] / det};
    };
    d_domainBoundingVectorsInverse.resize(9, 0.0);
    d_domainBoundingVectorsInverse.copyFrom(inv3(d_domainBoundingVectors));
    const double scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
    BLASWrapperPtrHost->xgemm('N',
                              'N',
                              3,
                              d_numAtoms,
                              3,
                              &scalarCoeffAlpha,
                              &d_domainBoundingVectors[0],
                              3,
                              &d_atomicCoordsFrac[0],
                              3,
                              &scalarCoeffBeta,
                              &d_atomicCoordsCart[0],
                              3);

    if (d_isGroupSymmetry)
      {
        const dftfe::Int max_size = 500;
        int              rotation[max_size][3][3];
        double           translation[max_size][3];
        double           lattice[3][3];
        double           position[d_numAtoms][3];
        int              types[d_numAtoms];
        for (dftfe::uInt iVec = 0; iVec < 3; ++iVec)
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            lattice[iVec][jDim] = d_domainBoundingVectors[3 * iVec + jDim];
        for (dftfe::uInt iAtom = 0; iAtom < d_numAtoms; ++iAtom)
          {
            types[iAtom] = atomLocationsFractional[iAtom][0];
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              position[iAtom][jDim] = d_atomicCoordsFrac[iAtom * 3 + jDim];
          }
        if (!isCollinearSpin)
          d_numSymm = spg_get_symmetry(rotation,
                                       translation,
                                       max_size,
                                       lattice,
                                       position,
                                       types,
                                       d_numAtoms,
                                       1e-5);
        else
          {
            int    equivalent_atoms[d_numAtoms];
            double spins[d_numAtoms];
            for (dftfe::uInt iAtom = 0; iAtom < d_numAtoms; ++iAtom)
              spins[iAtom] = atomLocationsFractional[iAtom].size() == 6 ?
                               atomLocationsFractional[iAtom][5] :
                               0.0;
            spg_get_symmetry_with_collinear_spin(rotation,
                                                 translation,
                                                 equivalent_atoms,
                                                 max_size,
                                                 lattice,
                                                 position,
                                                 types,
                                                 spins,
                                                 d_numAtoms,
                                                 1e-5);
          }
        d_symmMat.reserve(d_numSymm);
        d_translation.reserve(d_numSymm);
        dftfe::uInt numSymm = 0;
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          if (std::abs(translation[iSymm][0]) < 1e-8 &&
              std::abs(translation[iSymm][1]) < 1e-8 &&
              std::abs(translation[iSymm][2]) < 1e-8)
            {
              d_symmMat.push_back(std::vector<double>(9, 0.0));
              d_translation.push_back(std::vector<double>(3, 0.0));
              for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                  d_symmMat.back()[jDim * 3 + kDim] =
                    static_cast<double>(rotation[iSymm][jDim][kDim]);
            }
        d_symmMat.shrink_to_fit();
        d_translation.shrink_to_fit();
        d_numSymm = d_symmMat.size();
      }
    else
      {
        d_numSymm = 1;
        d_symmMat.resize(d_numSymm, std::vector<double>(9, 0.0));
        d_translation.resize(d_numSymm, std::vector<double>(3, 0.0));
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            {
              d_translation[iSymm][jDim] = 0.0;
              for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                d_symmMat[iSymm][jDim * 3 + kDim] = jDim == kDim ? 1.0 : 0.0;
            }
      }
    computePointMapsFromGlobalFractionalCoordinates(
      d_atomicCoordsFrac, dftfe::pointSet::atomicCoord, false);
  }



  void
  groupSymmetryClass::reduceKPointGrid(
    std::vector<double> &kPointCoordinatesFrac,
    std::vector<double> &kPointWeights) const
  {
    dftfe::uInt             numKPoints = kPointCoordinatesFrac.size() / 3;
    std::vector<dftfe::Int> kPointSymmetryMap(numKPoints, -1);
    auto                    wrap = [](double x) {
      double r = std::remainder(x, 1.0);
      return (r <= -0.5 ? 0.5 : r);
    };
    auto periodicDist = [](double a, double b) noexcept {
      double d = std::fabs(a - b);
      return (d <= 0.5 ? d : 1.0 - d);
    };
    for (dftfe::uInt iKPoint = 0; iKPoint < numKPoints; ++iKPoint)
      if (kPointSymmetryMap[iKPoint] == -1)
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          {
            std::vector<double> transformedKPoint = {0.0, 0.0, 0.0};
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              transformedKPoint[jDim] +=
                d_symmMat[iSymm][0 * 3 + jDim] *
                  kPointCoordinatesFrac[3 * iKPoint + 0] +
                d_symmMat[iSymm][1 * 3 + jDim] *
                  kPointCoordinatesFrac[3 * iKPoint + 1] +
                d_symmMat[iSymm][2 * 3 + jDim] *
                  kPointCoordinatesFrac[3 * iKPoint + 2];
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              transformedKPoint[jDim] = wrap(transformedKPoint[jDim]);
            for (dftfe::uInt jKPoint = iKPoint + 1; jKPoint < numKPoints;
                 ++jKPoint)
              if (periodicDist(transformedKPoint[0],
                               kPointCoordinatesFrac[3 * jKPoint + 0]) < 1e-8 &&
                  periodicDist(transformedKPoint[1],
                               kPointCoordinatesFrac[3 * jKPoint + 1]) < 1e-8 &&
                  periodicDist(transformedKPoint[2],
                               kPointCoordinatesFrac[3 * jKPoint + 2]) < 1e-8)
                {
                  kPointSymmetryMap[jKPoint] = iKPoint;
                  break;
                }
            if (d_isTimeReversal)
              {
                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                  transformedKPoint[jDim] = wrap(-transformedKPoint[jDim]);
                for (dftfe::uInt jKPoint = iKPoint + 1; jKPoint < numKPoints;
                     ++jKPoint)
                  if (periodicDist(transformedKPoint[0],
                                   kPointCoordinatesFrac[3 * jKPoint + 0]) <
                        1e-8 &&
                      periodicDist(transformedKPoint[1],
                                   kPointCoordinatesFrac[3 * jKPoint + 1]) <
                        1e-8 &&
                      periodicDist(transformedKPoint[2],
                                   kPointCoordinatesFrac[3 * jKPoint + 2]) <
                        1e-8)
                    {
                      kPointSymmetryMap[jKPoint] = iKPoint;
                      break;
                    }
              }
          }
    std::vector<double> kPointCoordinatesFracReduced;
    std::vector<double> kPointWeightsReduced;
    kPointCoordinatesFracReduced.reserve(kPointCoordinatesFrac.size());
    kPointWeightsReduced.reserve(kPointWeights.size());
    for (dftfe::uInt iKPoint = 0; iKPoint < numKPoints; ++iKPoint)
      if (kPointSymmetryMap[iKPoint] == -1)
        {
          kPointCoordinatesFracReduced.push_back(
            kPointCoordinatesFrac[3 * iKPoint + 0]);
          kPointCoordinatesFracReduced.push_back(
            kPointCoordinatesFrac[3 * iKPoint + 1]);
          kPointCoordinatesFracReduced.push_back(
            kPointCoordinatesFrac[3 * iKPoint + 2]);
          kPointWeightsReduced.push_back(kPointWeights[iKPoint]);
          for (dftfe::uInt jKPoint = 0; jKPoint < numKPoints; ++jKPoint)
            if (kPointSymmetryMap[jKPoint] == iKPoint)
              kPointWeightsReduced.back() += kPointWeights[jKPoint];
        }
    kPointCoordinatesFracReduced.shrink_to_fit();
    kPointWeightsReduced.shrink_to_fit();
    kPointCoordinatesFrac = std::move(kPointCoordinatesFracReduced);
    kPointWeights         = std::move(kPointWeightsReduced);
  }

  void
  groupSymmetryClass::computePointMapFromLocalCartesianCoordinates(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &BLASWrapperPtrHost,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                         &localPointCoords,
    const dftfe::pointSet pointSetType,
    const bool            cellOrdered)
  {
    int numLocalCoords = localPointCoords.size();
    // Compute the fractional coordinates of the local points
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                 localPointCoordsFrac(numLocalCoords, 0.0);
    const double scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
    BLASWrapperPtrHost->xgemm('N',
                              'N',
                              3,
                              numLocalCoords / 3,
                              3,
                              &scalarCoeffAlpha,
                              &d_domainBoundingVectorsInverse[0],
                              3,
                              &localPointCoords[0],
                              3,
                              &scalarCoeffBeta,
                              &localPointCoordsFrac[0],
                              3);
    for (dftfe::uInt iPoint = 0; iPoint < numLocalCoords / 3; ++iPoint)
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        localPointCoordsFrac[3 * iPoint + iDim] =
          localPointCoordsFrac[3 * iPoint + iDim] -
          std::floor(localPointCoordsFrac[3 * iPoint + iDim]);

    // Gather the local cell centroids from all MPI processes
    std::vector<int> numCoordsPerMPITask(n_mpi_processes, 0);
    numCoordsPerMPITask[this_mpi_process] = numLocalCoords;
    MPI_Gather(&numLocalCoords,
               1,
               dftfe::dataTypes::mpi_type_id(&numLocalCoords),
               numCoordsPerMPITask.data(),
               1,
               dftfe::dataTypes::mpi_type_id(numCoordsPerMPITask.data()),
               0,
               d_mpiCommDomain);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      globalPointCoordsFrac;
    if (this_mpi_process == 0)
      {
        int totalNumCoords = std::accumulate(numCoordsPerMPITask.begin(),
                                             numCoordsPerMPITask.end(),
                                             0);
        globalPointCoordsFrac.resize(totalNumCoords, 0.0);
        std::vector<int> MPIDisplacements(n_mpi_processes, 0);
        for (dftfe::Int i = 1; i < n_mpi_processes; i++)
          MPIDisplacements[i] =
            numCoordsPerMPITask[i - 1] + MPIDisplacements[i - 1];
        MPI_Gatherv(localPointCoordsFrac.data(),
                    numLocalCoords,
                    dftfe::dataTypes::mpi_type_id(localPointCoordsFrac.data()),
                    globalPointCoordsFrac.data(),
                    numCoordsPerMPITask.data(),
                    MPIDisplacements.data(),
                    dftfe::dataTypes::mpi_type_id(globalPointCoordsFrac.data()),
                    0,
                    d_mpiCommDomain);
      }
    else
      MPI_Gatherv(localPointCoordsFrac.data(),
                  numLocalCoords,
                  dftfe::dataTypes::mpi_type_id(localPointCoordsFrac.data()),
                  nullptr,
                  nullptr,
                  nullptr,
                  dftfe::dataTypes::mpi_type_id(globalPointCoordsFrac.data()),
                  0,
                  d_mpiCommDomain);
    computePointMapsFromGlobalFractionalCoordinates(globalPointCoordsFrac,
                                                    pointSetType,
                                                    cellOrdered);
  }


  void
  groupSymmetryClass::computePointMapsFromGlobalFractionalCoordinates(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                         &globalPointCoords,
    const dftfe::pointSet pointSetType,
    const bool            cellOrdered)
  {
    bool allPointsFound = true;
    auto periodicDist   = [](double a, double b) noexcept {
      double d = std::fabs(a - b);
      return (d <= 0.5 ? d : 1.0 - d);
    };
    if (this_mpi_process == 0 && cellOrdered)
      {
        std::vector<std::vector<dftfe::uInt>> &cellMapForSymmetry =
          d_pointMapsForSymmetry.find(dftfe::pointSet::cellCentroids)->second;
        std::vector<std::vector<dftfe::uInt>> &pointMapForSymmetry =
          d_pointMapsForSymmetry[pointSetType];
        pointMapForSymmetry.clear();
        const dftfe::uInt numPoints        = globalPointCoords.size() / 3;
        const dftfe::uInt numCells         = cellMapForSymmetry[0].size();
        const dftfe::uInt numPointsPerCell = numPoints / numCells;
        pointMapForSymmetry.resize(d_numSymm,
                                   std::vector<dftfe::uInt>(numPoints, 0));

        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          for (dftfe::uInt iCell = 0; iCell < numCells; ++iCell)
            for (dftfe::uInt iPoint = iCell * numPointsPerCell;
                 iPoint < (iCell + 1) * numPointsPerCell;
                 ++iPoint)
              {
                std::vector<double> transformedPoint = d_translation[iSymm];
                for (dftfe::uInt j = 0; j < 3; ++j)
                  transformedPoint[j] += d_symmMat[iSymm][0 * 3 + j] *
                                           globalPointCoords[3 * iPoint + 0] +
                                         d_symmMat[iSymm][1 * 3 + j] *
                                           globalPointCoords[3 * iPoint + 1] +
                                         d_symmMat[iSymm][2 * 3 + j] *
                                           globalPointCoords[3 * iPoint + 2];
                for (dftfe::uInt j = 0; j < 3; ++j)
                  transformedPoint[j] =
                    transformedPoint[j] - std::floor(transformedPoint[j]);
                const dftfe::uInt mappedCell = cellMapForSymmetry[iSymm][iCell];
                bool              pointFound = false;
                for (dftfe::uInt jPoint = mappedCell * numPointsPerCell;
                     jPoint < (mappedCell + 1) * numPointsPerCell;
                     ++jPoint)
                  if (periodicDist(transformedPoint[0],
                                   globalPointCoords[3 * jPoint + 0]) < 1e-8 &&
                      periodicDist(transformedPoint[1],
                                   globalPointCoords[3 * jPoint + 1]) < 1e-8 &&
                      periodicDist(transformedPoint[2],
                                   globalPointCoords[3 * jPoint + 2]) < 1e-8)
                    {
                      pointFound                         = true;
                      pointMapForSymmetry[iSymm][iPoint] = jPoint;
                      break;
                    }
                if (!pointFound)
                  std::cout << "Symmetry class " << iSymm << " point " << iPoint
                            << " found: " << pointFound << " "
                            << pointMapForSymmetry[iSymm][iPoint] << " "
                            << transformedPoint[0] << " " << transformedPoint[1]
                            << " " << transformedPoint[2] << std::endl;
                allPointsFound = allPointsFound && pointFound;
              }
      }
    else if (this_mpi_process == 0)
      {
        std::vector<std::vector<dftfe::uInt>> &pointMapForSymmetry =
          d_pointMapsForSymmetry[pointSetType];
        pointMapForSymmetry.clear();
        const dftfe::uInt numPoints = globalPointCoords.size() / 3;
        pointMapForSymmetry.resize(d_numSymm,
                                   std::vector<dftfe::uInt>(numPoints, 0));
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          for (dftfe::uInt iPoint = 0; iPoint < numPoints; ++iPoint)
            {
              std::vector<double> transformedPoint = d_translation[iSymm];
              for (dftfe::uInt j = 0; j < 3; ++j)
                transformedPoint[j] += d_symmMat[iSymm][0 * 3 + j] *
                                         globalPointCoords[3 * iPoint + 0] +
                                       d_symmMat[iSymm][1 * 3 + j] *
                                         globalPointCoords[3 * iPoint + 1] +
                                       d_symmMat[iSymm][2 * 3 + j] *
                                         globalPointCoords[3 * iPoint + 2];
              for (dftfe::uInt j = 0; j < 3; ++j)
                transformedPoint[j] =
                  transformedPoint[j] - std::floor(transformedPoint[j]);
              bool pointFound = false;
              for (dftfe::uInt jPoint = 0; jPoint < numPoints; ++jPoint)
                if (periodicDist(transformedPoint[0],
                                 globalPointCoords[3 * jPoint + 0]) < 1e-8 &&
                    periodicDist(transformedPoint[1],
                                 globalPointCoords[3 * jPoint + 1]) < 1e-8 &&
                    periodicDist(transformedPoint[2],
                                 globalPointCoords[3 * jPoint + 2]) < 1e-8)
                  {
                    pointFound                         = true;
                    pointMapForSymmetry[iSymm][iPoint] = jPoint;
                    break;
                  }
              if (!pointFound)
                std::cout << "Symmetry class " << iSymm << " point " << iPoint
                          << " found: " << pointFound << " "
                          << pointMapForSymmetry[iSymm][iPoint] << " "
                          << transformedPoint[0] << " " << transformedPoint[1]
                          << " " << transformedPoint[2] << std::endl;
              allPointsFound = allPointsFound && pointFound;
            }
      }
    int allPointsFoundCheck = allPointsFound ? 1 : 0;
    MPI_Allreduce(
      MPI_IN_PLACE, &allPointsFoundCheck, 1, MPI_INT, MPI_MIN, d_mpiCommDomain);
    if (allPointsFoundCheck == 0)
      {
        pcout << "Not all points found in symmetry class. " << std::endl;
        throw std::runtime_error("Not all points found in symmetry class.");
      }
  }

  void
  groupSymmetryClass::symmetrizeScalarFieldFromLocalValues(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                         &scalarFieldValues,
    const dftfe::pointSet pointSetType) const
  {
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     globalScalarFieldValues;
    int              numLocalPoints = scalarFieldValues.size();
    std::vector<int> numPointsPerMPITask(n_mpi_processes, 0);
    std::vector<int> MPIDisplacements(n_mpi_processes, 0);
    numPointsPerMPITask[this_mpi_process] = numLocalPoints;
    MPI_Gather(&numLocalPoints,
               1,
               dftfe::dataTypes::mpi_type_id(&numLocalPoints),
               numPointsPerMPITask.data(),
               1,
               dftfe::dataTypes::mpi_type_id(numPointsPerMPITask.data()),
               0,
               d_mpiCommDomain);
    if (this_mpi_process == 0)
      {
        int totalNumPoints = std::accumulate(numPointsPerMPITask.begin(),
                                             numPointsPerMPITask.end(),
                                             0);
        globalScalarFieldValues.resize(totalNumPoints, 0.0);
        for (dftfe::Int i = 1; i < n_mpi_processes; i++)
          MPIDisplacements[i] =
            numPointsPerMPITask[i - 1] + MPIDisplacements[i - 1];
        MPI_Gatherv(scalarFieldValues.data(),
                    numLocalPoints,
                    dftfe::dataTypes::mpi_type_id(scalarFieldValues.data()),
                    globalScalarFieldValues.data(),
                    numPointsPerMPITask.data(),
                    MPIDisplacements.data(),
                    dftfe::dataTypes::mpi_type_id(
                      globalScalarFieldValues.data()),
                    0,
                    d_mpiCommDomain);
      }
    else
      {
        MPI_Gatherv(scalarFieldValues.data(),
                    numLocalPoints,
                    dftfe::dataTypes::mpi_type_id(scalarFieldValues.data()),
                    nullptr,
                    nullptr,
                    nullptr,
                    dftfe::dataTypes::mpi_type_id(
                      globalScalarFieldValues.data()),
                    0,
                    d_mpiCommDomain);
      }
    if (this_mpi_process == 0)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          symmetrizedGlobalScalarFieldValues = globalScalarFieldValues;
        symmetrizedGlobalScalarFieldValues.setValue(0.0);
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          {
            const std::vector<dftfe::uInt> &pointMap =
              d_pointMapsForSymmetry.find(pointSetType)->second[iSymm];
            for (dftfe::uInt iPoint = 0;
                 iPoint < globalScalarFieldValues.size();
                 ++iPoint)
              {
                dftfe::uInt mappedPoint = pointMap[iPoint];
                symmetrizedGlobalScalarFieldValues[mappedPoint] +=
                  globalScalarFieldValues[iPoint] / d_numSymm;
              }
          }
        MPI_Scatterv(symmetrizedGlobalScalarFieldValues.data(),
                     numPointsPerMPITask.data(),
                     MPIDisplacements.data(),
                     dftfe::dataTypes::mpi_type_id(
                       symmetrizedGlobalScalarFieldValues.data()),
                     scalarFieldValues.data(),
                     numLocalPoints,
                     dftfe::dataTypes::mpi_type_id(scalarFieldValues.data()),
                     0,
                     d_mpiCommDomain);
      }
    else
      {
        MPI_Scatterv(nullptr,
                     nullptr,
                     nullptr,
                     dftfe::dataTypes::mpi_type_id(scalarFieldValues.data()),
                     scalarFieldValues.data(),
                     numLocalPoints,
                     dftfe::dataTypes::mpi_type_id(scalarFieldValues.data()),
                     0,
                     d_mpiCommDomain);
      }
  }
  void
  groupSymmetryClass::symmetrizeVectorFieldFromLocalValues(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                         &vectorFieldValues,
    const dftfe::pointSet pointSetType) const
  {
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     globalVectorFieldValues;
    int              numLocalPoints = vectorFieldValues.size();
    std::vector<int> numPointsPerMPITask(n_mpi_processes, 0);
    std::vector<int> MPIDisplacements(n_mpi_processes, 0);
    numPointsPerMPITask[this_mpi_process] = numLocalPoints;
    MPI_Gather(&numLocalPoints,
               1,
               dftfe::dataTypes::mpi_type_id(&numLocalPoints),
               numPointsPerMPITask.data(),
               1,
               dftfe::dataTypes::mpi_type_id(numPointsPerMPITask.data()),
               0,
               d_mpiCommDomain);
    if (this_mpi_process == 0)
      {
        int totalNumPoints = std::accumulate(numPointsPerMPITask.begin(),
                                             numPointsPerMPITask.end(),
                                             0);
        globalVectorFieldValues.resize(totalNumPoints, 0.0);
        for (dftfe::Int i = 1; i < n_mpi_processes; i++)
          MPIDisplacements[i] =
            numPointsPerMPITask[i - 1] + MPIDisplacements[i - 1];
        MPI_Gatherv(vectorFieldValues.data(),
                    numLocalPoints,
                    dftfe::dataTypes::mpi_type_id(vectorFieldValues.data()),
                    globalVectorFieldValues.data(),
                    numPointsPerMPITask.data(),
                    MPIDisplacements.data(),
                    dftfe::dataTypes::mpi_type_id(
                      globalVectorFieldValues.data()),
                    0,
                    d_mpiCommDomain);
      }
    else
      {
        MPI_Gatherv(vectorFieldValues.data(),
                    numLocalPoints,
                    dftfe::dataTypes::mpi_type_id(vectorFieldValues.data()),
                    nullptr,
                    nullptr,
                    nullptr,
                    dftfe::dataTypes::mpi_type_id(
                      globalVectorFieldValues.data()),
                    0,
                    d_mpiCommDomain);
      }
    if (this_mpi_process == 0)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          symmetrizedGlobalVectorFieldValues = globalVectorFieldValues;
        symmetrizedGlobalVectorFieldValues.setValue(0.0);
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          {
            const std::vector<dftfe::uInt> &pointMap =
              d_pointMapsForSymmetry.find(pointSetType)->second[iSymm];
            for (dftfe::uInt iPoint = 0;
                 iPoint < globalVectorFieldValues.size() / 3;
                 ++iPoint)
              {
                dftfe::uInt mappedPoint = pointMap[iPoint];
                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    symmetrizedGlobalVectorFieldValues[mappedPoint * 3 +
                                                       jDim] +=
                      d_symmMat[iSymm][iDim * 3 + jDim] *
                      globalVectorFieldValues[iPoint * 3 + iDim] / d_numSymm;
              }
          }
        MPI_Scatterv(symmetrizedGlobalVectorFieldValues.data(),
                     numPointsPerMPITask.data(),
                     MPIDisplacements.data(),
                     dftfe::dataTypes::mpi_type_id(
                       symmetrizedGlobalVectorFieldValues.data()),
                     vectorFieldValues.data(),
                     numLocalPoints,
                     dftfe::dataTypes::mpi_type_id(vectorFieldValues.data()),
                     0,
                     d_mpiCommDomain);
      }
    else
      {
        MPI_Scatterv(nullptr,
                     nullptr,
                     nullptr,
                     dftfe::dataTypes::mpi_type_id(vectorFieldValues.data()),
                     vectorFieldValues.data(),
                     numLocalPoints,
                     dftfe::dataTypes::mpi_type_id(vectorFieldValues.data()),
                     0,
                     d_mpiCommDomain);
      }
  }


  void
  groupSymmetryClass::symmetrizeVectorFieldFromGlobalValues(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                         &vectorFieldValues,
    const dftfe::pointSet pointSetType) const
  {
    if (this_mpi_process == 0)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          symmetrizedGlobalVectorFieldValues = vectorFieldValues;
        symmetrizedGlobalVectorFieldValues.setValue(0.0);
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          {
            const std::vector<dftfe::uInt> &pointMap =
              d_pointMapsForSymmetry.find(pointSetType)->second[iSymm];
            for (dftfe::uInt iPoint = 0; iPoint < vectorFieldValues.size() / 3;
                 ++iPoint)
              {
                dftfe::uInt mappedPoint = pointMap[iPoint];
                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    symmetrizedGlobalVectorFieldValues[mappedPoint * 3 +
                                                       jDim] +=
                      d_symmMat[iSymm][iDim * 3 + jDim] *
                      vectorFieldValues[iPoint * 3 + iDim] / d_numSymm;
              }
          }
        vectorFieldValues = std::move(symmetrizedGlobalVectorFieldValues);
      }
    MPI_Bcast(vectorFieldValues.data(),
              vectorFieldValues.size(),
              MPI_DOUBLE,
              0,
              d_mpiCommDomain);
  }


  void
  groupSymmetryClass::symmetrizeRank2Tensor(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &tensorValues) const
  {
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      symmetrizedTensorValues = tensorValues;
    symmetrizedTensorValues.setValue(0.0);
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      {
        for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
              for (dftfe::uInt lDim = 0; lDim < 3; ++lDim)
                symmetrizedTensorValues[iDim * 3 + jDim] +=
                  d_symmMat[iSymm][kDim * 3 + iDim] *
                  d_symmMat[iSymm][lDim * 3 + jDim] *
                  tensorValues[lDim * 3 + kDim] / d_numSymm;
      }
    tensorValues = std::move(symmetrizedTensorValues);
  }
} // namespace dftfe
