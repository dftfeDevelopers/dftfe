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
    std::vector<std::vector<double>> &atomLocationsFractional,
    std::vector<std::vector<double>> &domainBoundingVectors,
    std::vector<bool>                &periodicBoundaryConditions,
    const bool                        isCollinearSpin,
    const bool                        isNonCollinearSpin)
  {
    d_numAtoms = atomLocationsFractional.size();
    d_atomicCoordsFrac.clear();
    d_domainBoundingVectors.clear();
    d_domainBoundingVectorsInverse.clear();
    d_periodicBoundaryConditions.clear();
    d_atomicCoordsFrac.resize(3 * d_numAtoms, 0.0);
    d_domainBoundingVectors.resize(9, 0.0);
    d_domainBoundingVectorsInverse.resize(9, 0.0);
    d_periodicBoundaryConditions = periodicBoundaryConditions;
    d_symmMatCart.clear();
    d_symmMat.clear();
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
    if (d_isGroupSymmetry)
      {
        const dftfe::Int max_size = 2000;
        int              rotation[max_size][3][3];
        double           translation[max_size][3];
        double           lattice[3][3];
        double           position[d_numAtoms][3];
        int              types[d_numAtoms];
        for (dftfe::uInt iVec = 0; iVec < 3; ++iVec)
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            lattice[jDim][iVec] = d_domainBoundingVectors[3 * iVec + jDim];
        for (dftfe::uInt iAtom = 0; iAtom < d_numAtoms; ++iAtom)
          {
            types[iAtom] = atomLocationsFractional[iAtom][0];
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              position[iAtom][jDim] = d_atomicCoordsFrac[iAtom * 3 + jDim];
          }
        if (!isCollinearSpin && !isNonCollinearSpin)
          d_numSymm = spg_get_symmetry(rotation,
                                       translation,
                                       max_size,
                                       lattice,
                                       position,
                                       types,
                                       d_numAtoms,
                                       1e-8);
        else if (isCollinearSpin)
          {
            int    equivalent_atoms[d_numAtoms];
            double spins[d_numAtoms];
            for (dftfe::uInt iAtom = 0; iAtom < d_numAtoms; ++iAtom)
              spins[iAtom] = atomLocationsFractional[iAtom].size() >= 6 ?
                               atomLocationsFractional[iAtom][5] :
                               0.0;
            d_numSymm = spg_get_symmetry_with_collinear_spin(rotation,
                                                             translation,
                                                             equivalent_atoms,
                                                             max_size,
                                                             lattice,
                                                             position,
                                                             types,
                                                             spins,
                                                             d_numAtoms,
                                                             1e-8);
          }
        else if (isNonCollinearSpin)
          {
            int    equivalent_atoms[d_numAtoms];
            int    spin_flips[d_numAtoms];
            double primitive_lattice[3][3];
            double tensors[d_numAtoms * 3];
            for (dftfe::uInt iAtom = 0; iAtom < d_numAtoms; ++iAtom)
              {
                if (atomLocationsFractional[iAtom].size() >= 8)
                  {
                    tensors[3 * iAtom + 0] =
                      atomLocationsFractional[iAtom][5] *
                      std::sin(M_PI / 180.0 *
                               atomLocationsFractional[iAtom][6]) *
                      std::cos(M_PI / 180.0 *
                               atomLocationsFractional[iAtom][7]);
                    tensors[3 * iAtom + 1] =
                      atomLocationsFractional[iAtom][5] *
                      std::sin(M_PI / 180.0 *
                               atomLocationsFractional[iAtom][6]) *
                      std::sin(M_PI / 180.0 *
                               atomLocationsFractional[iAtom][7]);
                    tensors[3 * iAtom + 2] =
                      atomLocationsFractional[iAtom][5] *
                      std::cos(M_PI / 180.0 *
                               atomLocationsFractional[iAtom][6]);
                  }
                else
                  {
                    tensors[3 * iAtom + 0] = 0.0;
                    tensors[3 * iAtom + 1] = 0.0;
                    tensors[3 * iAtom + 2] = 0.0;
                  }
              }
            d_numSymm = spg_get_symmetry_with_site_tensors(rotation,
                                                           translation,
                                                           equivalent_atoms,
                                                           primitive_lattice,
                                                           spin_flips,
                                                           max_size,
                                                           lattice,
                                                           position,
                                                           types,
                                                           tensors,
                                                           1,
                                                           d_numAtoms,
                                                           0,
                                                           1,
                                                           1e-8);
          }
        d_symmMat.reserve(d_numSymm);
        d_symmMatInverse.reserve(d_numSymm);
        d_translation.reserve(d_numSymm);
        dftfe::uInt numSymm = 0;
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          {
            d_symmMat.push_back(std::vector<double>(9, 0.0));
            d_translation.push_back(std::vector<double>(3, 0.0));
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              {
                d_translation.back()[jDim] = translation[iSymm][jDim];
                for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                  d_symmMat.back()[kDim * 3 + jDim] =
                    static_cast<double>(rotation[iSymm][jDim][kDim]);
              }
            d_symmMatInverse.push_back(inv3(d_symmMat.back()));
          }
        d_symmMat.shrink_to_fit();
        d_symmMatInverse.shrink_to_fit();
        d_translation.shrink_to_fit();
        d_numSymm = d_symmMat.size();
      }
    else
      {
        d_numSymm = 1;
        d_symmMat.resize(d_numSymm, std::vector<double>(9, 0.0));
        d_symmMatInverse.resize(d_numSymm, std::vector<double>(9, 0.0));
        d_translation.resize(d_numSymm, std::vector<double>(3, 0.0));
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            {
              d_translation[iSymm][jDim] = 0.0;
              for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                d_symmMat[iSymm][jDim * 3 + kDim] = jDim == kDim ? 1.0 : 0.0;
              for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
                d_symmMatInverse[iSymm][jDim * 3 + kDim] =
                  jDim == kDim ? 1.0 : 0.0;
            }
      }
    d_symmMatCart.resize(d_numSymm, std::vector<double>(9, 0.0));
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      {
        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
              for (dftfe::uInt pDim = 0; pDim < 3; ++pDim)
                d_symmMatCart[iSymm][pDim + 3 * jDim] +=
                  d_domainBoundingVectors[pDim + kDim * 3] *
                  d_symmMat[iSymm][kDim + 3 * iDim] *
                  d_domainBoundingVectorsInverse[iDim + jDim * 3];
      }
    if (!computeAtomIdMapsFromGlobalFractionalCoordinates(d_atomicCoordsFrac))
      {
        pcout << "Not all atoms found in symmetry class. " << std::endl;
        throw std::runtime_error("Not all atoms found in symmetry class.");
      }
  }

  void
  groupSymmetryClass::reinitGroupSymmetry(
    std::vector<std::vector<double>> &atomLocationsFractional,
    std::vector<std::vector<double>> &domainBoundingVectors)
  {
    d_atomicCoordsFrac.clear();
    d_domainBoundingVectors.clear();
    d_domainBoundingVectorsInverse.clear();
    d_atomicCoordsFrac.resize(3 * d_numAtoms, 0.0);
    d_domainBoundingVectors.resize(9, 0.0);
    d_domainBoundingVectorsInverse.resize(9, 0.0);
    d_symmMatCart.clear();
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

    d_symmMatCart.resize(d_numSymm, std::vector<double>(9, 0.0));
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      {
        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            for (dftfe::uInt kDim = 0; kDim < 3; ++kDim)
              for (dftfe::uInt pDim = 0; pDim < 3; ++pDim)
                d_symmMatCart[iSymm][pDim + 3 * jDim] +=
                  d_domainBoundingVectors[pDim + kDim * 3] *
                  d_symmMat[iSymm][kDim + 3 * iDim] *
                  d_domainBoundingVectorsInverse[iDim + jDim * 3];
      }
    auto isOrthogonal = [](const std::vector<double> &m) -> bool {
      for (int iDim = 0; iDim < 3; ++iDim)
        {
          for (int jDim = iDim; jDim < 3; ++jDim)
            {
              double dot = 0.0;
              for (int kDim = 0; kDim < 3; ++kDim)
                dot += m[iDim * 3 + kDim] * m[jDim * 3 + kDim];
              double orthoVal = iDim == jDim ? 1.0 : 0.0;
              if (std::fabs(dot - orthoVal) > 1e-8)
                return false;
            }
        }
      return true;
    };
    bool areSymmetriesOrthogonal = true;
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      areSymmetriesOrthogonal =
        areSymmetriesOrthogonal && isOrthogonal(d_symmMatCart[iSymm]);
    if (!areSymmetriesOrthogonal)
      {
        pcout << "Cell symmetries are not valid in symmetry class."
              << std::endl;
        throw std::runtime_error(
          "Cell symmetries are not valid in symmetry class.");
      }
    if (!computeAtomIdMapsFromGlobalFractionalCoordinates(d_atomicCoordsFrac))
      {
        pcout << "Not all atoms found in symmetry class." << std::endl;
        throw std::runtime_error("Not all atoms found in symmetry class.");
      }
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
      return (r >= 0.5 ? r - 1.0 : r);
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
                d_symmMat[iSymm][jDim * 3 + 0] *
                  kPointCoordinatesFrac[3 * iKPoint + 0] +
                d_symmMat[iSymm][jDim * 3 + 1] *
                  kPointCoordinatesFrac[3 * iKPoint + 1] +
                d_symmMat[iSymm][jDim * 3 + 2] *
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
  groupSymmetryClass::setupCommPatternForNodalField(
    const dealii::DoFHandler<3> &dofHandler)
  {
    const dealii::IndexSet &locallyOwnedNodes = dofHandler.locally_owned_dofs();

    auto nodalCoordinates =
      dealii::DoFTools::map_dofs_to_support_points(mapping, dofHandler);
    requiredPointCoordinates.clear();
    requiredPointCoordinates.reserve(d_numSymm * nodalCoordinates.size());
    localDoFIndexToPointIndexMap.clear();
    localDoFIndexToPointIndexMap.resize(
      d_numSymm, std::vector<dftfe::uInt>(dofHandler.n_locally_owned_dofs()));
    std::map<std::array<std::int64_t, 3>, dftfe::uInt> pointToPointIndexMap;
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      for (dealii::IndexSet::ElementIterator it = locallyOwnedNodes.begin();
           it != locallyOwnedNodes.end();
           it++)
        {
          const dealii::Point<3> &currentNodeCoordinatesCart =
            nodalCoordinates.find(*it)->second;

          dealii::Point<3> currentNodeCoordinatesFrac(0, 0, 0);
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              currentNodeCoordinatesFrac[iDim] +=
                d_domainBoundingVectorsInverse[3 * jDim + iDim] *
                currentNodeCoordinatesCart[jDim];
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            currentNodeCoordinatesFrac[iDim] =
              currentNodeCoordinatesFrac[iDim] + 0.5 -
              std::floor(currentNodeCoordinatesFrac[iDim] + 0.5);

          dealii::Point<3> transformedNodeCoordinatesFrac(0, 0, 0);
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              transformedNodeCoordinatesFrac[iDim] +=
                d_symmMatInverse[iSymm][jDim * 3 + iDim] *
                (currentNodeCoordinatesFrac[jDim] - d_translation[iSymm][jDim]);
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            transformedNodeCoordinatesFrac[iDim] =
              transformedNodeCoordinatesFrac[iDim] -
              std::floor(transformedNodeCoordinatesFrac[iDim]) - 0.5;

          dealii::Point<3> transformedNodeCoordinatesCart(0, 0, 0);
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              transformedNodeCoordinatesCart[iDim] +=
                d_domainBoundingVectors[3 * jDim + iDim] *
                transformedNodeCoordinatesFrac[jDim];
          std::array<std::int64_t, 3> roundedCoords;
          roundedCoords[0] =
            std::llround(transformedNodeCoordinatesCart[0] * 1e8);
          roundedCoords[1] =
            std::llround(transformedNodeCoordinatesCart[1] * 1e8);
          roundedCoords[2] =
            std::llround(transformedNodeCoordinatesCart[2] * 1e8);
          auto pointIterator = pointToPointIndexMap.find(roundedCoords);
          if (pointIterator != pointToPointIndexMap.end())
            {
              localDoFIndexToPointIndexMap[iSymm][locallyOwnedNodes
                                                    .index_within_set(*it)] =
                pointIterator->second;
            }
          else
            {
              requiredPointCoordinates.push_back(
                transformedNodeCoordinatesCart);
              localDoFIndexToPointIndexMap[iSymm][locallyOwnedNodes
                                                    .index_within_set(*it)] =
                requiredPointCoordinates.size() - 1;
              pointToPointIndexMap[roundedCoords] =
                requiredPointCoordinates.size() - 1;
            }
        }
    requiredPointCoordinates.shrink_to_fit();
    remotePointCache.reinit(requiredPointCoordinates,
                            dofHandler.get_triangulation(),
                            mapping);
    if (!remotePointCache.all_points_found())
      {
        pcout << "Not all points found in remotePointCache." << std::endl;
        throw std::runtime_error("Not all points found in remotePointCache.");
      }
  }

  bool
  groupSymmetryClass::computeAtomIdMapsFromGlobalFractionalCoordinates(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &globalPointCoords)
  {
    bool allPointsFound = true;
    auto periodicDist   = [](double a, double b) noexcept {
      double d = std::fabs(a - b);
      return (d <= 0.5 ? d : 1.0 - d);
    };
    d_pointMapsForSymmetry.clear();
    const dftfe::uInt numPoints = globalPointCoords.size() / 3;
    d_pointMapsForSymmetry.resize(d_numSymm,
                                  std::vector<dftfe::uInt>(numPoints, 0));
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      for (dftfe::uInt iPoint = 0; iPoint < numPoints; ++iPoint)
        {
          std::vector<double> transformedPoint = {0.0, 0.0, 0.0};
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            transformedPoint[jDim] =
              d_symmMatInverse[iSymm][0 * 3 + jDim] *
                (globalPointCoords[3 * iPoint + 0] - d_translation[iSymm][0]) +
              d_symmMatInverse[iSymm][1 * 3 + jDim] *
                (globalPointCoords[3 * iPoint + 1] - d_translation[iSymm][1]) +
              d_symmMatInverse[iSymm][2 * 3 + jDim] *
                (globalPointCoords[3 * iPoint + 2] - d_translation[iSymm][2]);
          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
            transformedPoint[jDim] =
              transformedPoint[jDim] - std::floor(transformedPoint[jDim]);
          bool pointFound = false;
          for (dftfe::uInt jPoint = 0; jPoint < numPoints; ++jPoint)
            if (periodicDist(transformedPoint[0],
                             globalPointCoords[3 * jPoint + 0]) < 1e-8 &&
                periodicDist(transformedPoint[1],
                             globalPointCoords[3 * jPoint + 1]) < 1e-8 &&
                periodicDist(transformedPoint[2],
                             globalPointCoords[3 * jPoint + 2]) < 1e-8)
              {
                pointFound                            = true;
                d_pointMapsForSymmetry[iSymm][iPoint] = jPoint;
                break;
              }
          allPointsFound = allPointsFound && pointFound;
        }
    int allPointsFoundCheck = allPointsFound ? 1 : 0;
    MPI_Allreduce(
      MPI_IN_PLACE, &allPointsFoundCheck, 1, MPI_INT, MPI_MIN, d_mpiCommDomain);
    return (allPointsFoundCheck == 1);
  }

  void
  groupSymmetryClass::symmetrizeScalarFieldFromLocalValues(
    distributedCPUVec<double>   &scalarField,
    const dealii::DoFHandler<3> &dofHandler)
  {
    const std::vector<double> &pointValues =
      dealii::VectorTools::point_values<1>(remotePointCache,
                                           dofHandler,
                                           scalarField);
    scalarField *= 0;
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      for (dftfe::uInt iDoF = 0; iDoF < scalarField.locally_owned_size();
           ++iDoF)
        scalarField.local_element(iDoF) +=
          pointValues[localDoFIndexToPointIndexMap[iSymm][iDoF]] / d_numSymm;
  }

  void
  groupSymmetryClass::symmetrizeVectorFieldFromLocalValues(
    distributedCPUVec<double>   &vectorFieldComponentx,
    distributedCPUVec<double>   &vectorFieldComponenty,
    distributedCPUVec<double>   &vectorFieldComponentz,
    const dealii::DoFHandler<3> &dofHandler)
  {
    const std::vector<double> &pointValuesx =
      dealii::VectorTools::point_values<1>(remotePointCache,
                                           dofHandler,
                                           vectorFieldComponentx);
    const std::vector<double> &pointValuesy =
      dealii::VectorTools::point_values<1>(remotePointCache,
                                           dofHandler,
                                           vectorFieldComponenty);
    const std::vector<double> &pointValuesz =
      dealii::VectorTools::point_values<1>(remotePointCache,
                                           dofHandler,
                                           vectorFieldComponentz);
    vectorFieldComponentx *= 0;
    vectorFieldComponenty *= 0;
    vectorFieldComponentz *= 0;
    for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
      for (dftfe::uInt iDoF = 0;
           iDoF < vectorFieldComponentz.locally_owned_size();
           ++iDoF)
        {
          vectorFieldComponentx.local_element(iDoF) +=
            (d_symmMatCart[iSymm][0 * 3 + 0] *
               pointValuesx[localDoFIndexToPointIndexMap[iSymm][iDoF]] +
             d_symmMatCart[iSymm][1 * 3 + 0] *
               pointValuesy[localDoFIndexToPointIndexMap[iSymm][iDoF]] +
             d_symmMatCart[iSymm][2 * 3 + 0] *
               pointValuesz[localDoFIndexToPointIndexMap[iSymm][iDoF]]) /
            d_numSymm;
          vectorFieldComponenty.local_element(iDoF) +=
            (d_symmMatCart[iSymm][0 * 3 + 1] *
               pointValuesx[localDoFIndexToPointIndexMap[iSymm][iDoF]] +
             d_symmMatCart[iSymm][1 * 3 + 1] *
               pointValuesy[localDoFIndexToPointIndexMap[iSymm][iDoF]] +
             d_symmMatCart[iSymm][2 * 3 + 1] *
               pointValuesz[localDoFIndexToPointIndexMap[iSymm][iDoF]]) /
            d_numSymm;
          vectorFieldComponentz.local_element(iDoF) +=
            (d_symmMatCart[iSymm][0 * 3 + 2] *
               pointValuesx[localDoFIndexToPointIndexMap[iSymm][iDoF]] +
             d_symmMatCart[iSymm][1 * 3 + 2] *
               pointValuesy[localDoFIndexToPointIndexMap[iSymm][iDoF]] +
             d_symmMatCart[iSymm][2 * 3 + 2] *
               pointValuesz[localDoFIndexToPointIndexMap[iSymm][iDoF]]) /
            d_numSymm;
        }
  }

  void
  groupSymmetryClass::symmetrizeForce(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &vectorFieldValues) const
  {
    if (this_mpi_process == 0)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          symmetrizedGlobalVectorFieldValues = vectorFieldValues;
        symmetrizedGlobalVectorFieldValues.setValue(0.0);
        for (dftfe::uInt iSymm = 0; iSymm < d_numSymm; ++iSymm)
          {
            const std::vector<dftfe::uInt> &pointMap =
              d_pointMapsForSymmetry[iSymm];
            for (dftfe::uInt iPoint = 0; iPoint < vectorFieldValues.size() / 3;
                 ++iPoint)
              {
                dftfe::uInt mappedPoint = pointMap[iPoint];
                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    symmetrizedGlobalVectorFieldValues[iPoint * 3 + jDim] +=
                      d_symmMatCart[iSymm][iDim * 3 + jDim] *
                      vectorFieldValues[mappedPoint * 3 + iDim] / d_numSymm;
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
  groupSymmetryClass::symmetrizeStress(
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
                symmetrizedTensorValues[jDim * 3 + iDim] +=
                  d_symmMatCart[iSymm][kDim * 3 + iDim] *
                  d_symmMatCart[iSymm][lDim * 3 + jDim] *
                  tensorValues[lDim * 3 + kDim] / d_numSymm;
      }
    tensorValues = std::move(symmetrizedTensorValues);
  }
} // namespace dftfe
