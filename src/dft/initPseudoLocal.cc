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
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <vectorUtilities.h>
#include <feevaluationWrapper.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initAnalyticSmearedLoadData()
  {
    const dftfe::uInt numberGlobalAtoms = atomLocations.size();

    d_bQuadValuesAllAtoms.clear();
    d_gradbQuadValuesAllAtoms.clear();
    d_bQuadAtomIdsAllAtoms.clear();
    d_bQuadAtomIdsAllAtomsImages.clear();
    d_bCellNonTrivialAtomIds.clear();
    d_bCellNonTrivialAtomIdsBins.clear();
    d_bCellNonTrivialAtomImageIds.clear();
    d_bCellNonTrivialAtomImageIdsBins.clear();
    d_smearedChargeMoments.clear();
    d_smearedChargeMoments.resize(13, 0.0);
    d_smearedChargeMomentsComputed = false;
    d_localVselfs.clear();
    d_localVselfs.resize(1, std::vector<double>(1, 0.0));

    const double referenceSmearedChargeWidth =
      std::min(0.7, std::max(1.0e-8, 0.5 * d_minDist - 0.3));

    d_smearedChargeWidths.assign(numberGlobalAtoms,
                                 d_dftParamsPtr->analyticSmearedLoadRadius);
    d_smearedChargeScaling.assign(numberGlobalAtoms, 1.0);

    if (d_dftParamsPtr->analyticSmearedLoadRadius <= 0.0)
      {
        constexpr double              fluxChargeRelativeTolerance = 1.0e-3;
        constexpr double              meshResolutionFactor        = 2.5;
        constexpr dftfe::uInt         nRadiusSamples              = 2000;
        std::map<dftfe::uInt, double> atomTypeToPspCoreWidth;
        std::map<dftfe::uInt, double> atomTypeToMeshFloor;
        std::map<dftfe::uInt, double> atomTypeToWidth;

        auto getAtomMeshSize = [&](const dftfe::uInt atomId) {
          if (d_dftParamsPtr->meshSizesFile != "" &&
              atomId < d_meshSizes.size() && !d_meshSizes[atomId].empty() &&
              d_meshSizes[atomId][0] > 0.0)
            return d_meshSizes[atomId][0];
          return d_dftParamsPtr->meshSizeOuterBall;
        };

        auto estimateCoreWidthFromLocalPotential =
          [&](const dftfe::uInt atomicNumber, const double atomCharge) {
            if (!d_dftParamsPtr->isPseudopotential || !d_oncvClassPtr ||
                atomCharge <= 0.0)
              return referenceSmearedChargeWidth;

            const double maxLocalPotentialRadius =
              d_oncvClassPtr->getRmaxLocalPot(atomicNumber);
            if (!std::isfinite(maxLocalPotentialRadius) ||
                maxLocalPotentialRadius <= referenceSmearedChargeWidth)
              return referenceSmearedChargeWidth;

            std::vector<double> radii(nRadiusSamples, 0.0);
            std::vector<double> residuals(nRadiusSamples,
                                          std::numeric_limits<double>::max());
            const double sampleMax = maxLocalPotentialRadius * (1.0 - 1.0e-8);
            for (dftfe::uInt iSample = 1; iSample < nRadiusSamples; ++iSample)
              {
                const double r = sampleMax * static_cast<double>(iSample) /
                                 static_cast<double>(nRadiusSamples - 1);
                double h = 1.0e-4 * std::max(1.0, r);
                h        = std::min(h, 0.45 * r);
                h        = std::min(h, 0.45 * (maxLocalPotentialRadius - r));
                if (h <= 0.0)
                  continue;

                const double vPlus =
                  d_oncvClassPtr->getRadialLocalPseudo(atomicNumber, r + h);
                const double vMinus =
                  d_oncvClassPtr->getRadialLocalPseudo(atomicNumber, r - h);
                const double dVdr           = (vPlus - vMinus) / (2.0 * h);
                const double enclosedCharge = r * r * dVdr;
                radii[iSample]              = r;
                residuals[iSample] =
                  std::abs(enclosedCharge - atomCharge) / atomCharge;
              }

            dftfe::Int firstWithin = -1;
            for (dftfe::uInt iSample = 1; iSample < nRadiusSamples; ++iSample)
              if (residuals[iSample] <= fluxChargeRelativeTolerance)
                {
                  firstWithin = iSample;
                  break;
                }

            double coreWidth = firstWithin > 0 ? radii[firstWithin] :
                                                 referenceSmearedChargeWidth;
            if (!std::isfinite(coreWidth) || coreWidth <= 0.0)
              coreWidth = referenceSmearedChargeWidth;
            return std::max(coreWidth, referenceSmearedChargeWidth);
          };

        for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
          {
            const dftfe::uInt atomicNumber =
              static_cast<dftfe::uInt>(std::round(atomLocations[iAtom][0]));
            const double atomMeshFloor =
              meshResolutionFactor * getAtomMeshSize(iAtom);
            auto meshFloorIt = atomTypeToMeshFloor.find(atomicNumber);
            if (meshFloorIt == atomTypeToMeshFloor.end())
              atomTypeToMeshFloor[atomicNumber] = atomMeshFloor;
            else
              meshFloorIt->second =
                std::max(meshFloorIt->second, atomMeshFloor);
          }

        for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
          {
            const dftfe::uInt atomicNumber =
              static_cast<dftfe::uInt>(std::round(atomLocations[iAtom][0]));
            const double atomCharge = d_dftParamsPtr->isPseudopotential ?
                                        atomLocations[iAtom][1] :
                                        atomLocations[iAtom][0];
            auto         widthIt    = atomTypeToWidth.find(atomicNumber);
            if (widthIt == atomTypeToWidth.end())
              {
                auto coreIt = atomTypeToPspCoreWidth.find(atomicNumber);
                if (coreIt == atomTypeToPspCoreWidth.end())
                  coreIt =
                    atomTypeToPspCoreWidth
                      .insert({atomicNumber,
                               estimateCoreWidthFromLocalPotential(atomicNumber,
                                                                   atomCharge)})
                      .first;

                const double meshFloor =
                  std::max(referenceSmearedChargeWidth,
                           atomTypeToMeshFloor[atomicNumber]);
                const double selectedWidth =
                  std::max(coreIt->second, meshFloor);
                widthIt =
                  atomTypeToWidth.insert({atomicNumber, selectedWidth}).first;
              }
            d_smearedChargeWidths[iAtom] = widthIt->second;
          }

        if (numberGlobalAtoms > 0 && d_dftParamsPtr->verbosity >= 2)
          {
            pcout << "ASL automatic smeared charge widths by atom type:";
            for (const auto &typeWidth : atomTypeToWidth)
              pcout << " Z=" << typeWidth.first << ":" << typeWidth.second
                    << "(core=" << atomTypeToPspCoreWidth[typeWidth.first]
                    << ", mesh=" << atomTypeToMeshFloor[typeWidth.first] << ")";
            pcout << std::endl;
          }
      }

    const bool hasPeriodicDirection = d_dftParamsPtr->periodicX ||
                                      d_dftParamsPtr->periodicY ||
                                      d_dftParamsPtr->periodicZ;
    const std::array<bool, 3> periodicDirection = {
      static_cast<bool>(d_dftParamsPtr->periodicX),
      static_cast<bool>(d_dftParamsPtr->periodicY),
      static_cast<bool>(d_dftParamsPtr->periodicZ)};
    auto latticeVector = [&](const dftfe::uInt iVector) {
      return std::array<double, 3>{d_domainBoundingVectors[iVector][0],
                                   d_domainBoundingVectors[iVector][1],
                                   d_domainBoundingVectors[iVector][2]};
    };
    const std::array<std::array<double, 3>, 3> latticeVectors = {
      latticeVector(0), latticeVector(1), latticeVector(2)};
    auto dot = [](const std::array<double, 3> &a,
                  const std::array<double, 3> &b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto cross = [](const std::array<double, 3> &a,
                    const std::array<double, 3> &b) {
      return std::array<double, 3>{a[1] * b[2] - a[2] * b[1],
                                   a[2] * b[0] - a[0] * b[2],
                                   a[0] * b[1] - a[1] * b[0]};
    };
    auto norm = [&](const std::array<double, 3> &a) {
      return std::sqrt(dot(a, a));
    };
    auto periodicImageRadiusCap = [&]() {
      if (!hasPeriodicDirection)
        return std::numeric_limits<double>::max();

      double minImageDistance = std::numeric_limits<double>::max();
      for (dftfe::Int i0 = -2; i0 <= 2; ++i0)
        for (dftfe::Int i1 = -2; i1 <= 2; ++i1)
          for (dftfe::Int i2 = -2; i2 <= 2; ++i2)
            {
              if ((periodicDirection[0] ? i0 : 0) == 0 &&
                  (periodicDirection[1] ? i1 : 0) == 0 &&
                  (periodicDirection[2] ? i2 : 0) == 0)
                continue;
              const std::array<dftfe::Int, 3> imageIndex = {
                periodicDirection[0] ? i0 : 0,
                periodicDirection[1] ? i1 : 0,
                periodicDirection[2] ? i2 : 0};
              std::array<double, 3> imageVector = {0.0, 0.0, 0.0};
              for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                  imageVector[iDim] +=
                    imageIndex[jDim] * latticeVectors[jDim][iDim];
              const double distance = norm(imageVector);
              if (distance > 0.0)
                minImageDistance = std::min(minImageDistance, distance);
            }
      return 0.5 * minImageDistance;
    };
    const double periodicCap = periodicImageRadiusCap();

    auto atomFractionalCoordinates = [&](const dftfe::uInt atomId) {
      std::array<double, 3> corner = {0.0, 0.0, 0.0};
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
          corner[iDim] -= 0.5 * latticeVectors[jDim][iDim];

      const std::array<double, 3> rhs = {atomLocations[atomId][2] - corner[0],
                                         atomLocations[atomId][3] - corner[1],
                                         atomLocations[atomId][4] - corner[2]};
      const std::array<double, 3> c0  = latticeVectors[0];
      const std::array<double, 3> c1  = latticeVectors[1];
      const std::array<double, 3> c2  = latticeVectors[2];
      const double                det = dot(c0, cross(c1, c2));
      AssertThrow(
        std::abs(det) > 1.0e-14,
        dealii::ExcMessage(
          "DFT-FE Error: invalid domain bounding vectors for ASL radius cap."));
      return std::array<double, 3>{dot(rhs, cross(c1, c2)) / det,
                                   dot(c0, cross(rhs, c2)) / det,
                                   dot(c0, cross(c1, rhs)) / det};
    };

    auto boundaryRadiusCap = [&](const dftfe::uInt atomId) {
      if (periodicDirection[0] && periodicDirection[1] && periodicDirection[2])
        return std::numeric_limits<double>::max();

      const std::array<double, 3> frac = atomFractionalCoordinates(atomId);
      double boundaryCap               = std::numeric_limits<double>::max();
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        if (!periodicDirection[iDim])
          {
            const std::array<double, 3> normal =
              cross(latticeVectors[(iDim + 1) % 3],
                    latticeVectors[(iDim + 2) % 3]);
            const double normalNorm = norm(normal);
            AssertThrow(
              normalNorm > 1.0e-14,
              dealii::ExcMessage(
                "DFT-FE Error: invalid domain face normal for ASL radius cap."));
            const double cellHeight =
              std::abs(dot(latticeVectors[iDim], normal)) / normalNorm;
            boundaryCap =
              std::min(boundaryCap,
                       std::min(frac[iDim], 1.0 - frac[iDim]) * cellHeight);
          }
      return boundaryCap;
    };

    bool   smearedChargeWidthCapped = false;
    double minRadiusCap             = std::numeric_limits<double>::max();
    double maxRadiusReduction       = 0.0;
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
      {
        const double atomRadiusCap =
          std::min(periodicCap, boundaryRadiusCap(iAtom));
        AssertThrow(
          atomRadiusCap > 1.0e-10,
          dealii::ExcMessage(
            "DFT-FE Error: ASL smeared charge radius cap is non-positive. Check atom positions and boundary conditions."));
        minRadiusCap = std::min(minRadiusCap, atomRadiusCap);
        if (d_smearedChargeWidths[iAtom] > atomRadiusCap)
          {
            maxRadiusReduction =
              std::max(maxRadiusReduction,
                       d_smearedChargeWidths[iAtom] - atomRadiusCap);
            d_smearedChargeWidths[iAtom] = atomRadiusCap;
            smearedChargeWidthCapped     = true;
          }
      }
    if (numberGlobalAtoms > 0 && d_dftParamsPtr->verbosity >= 2)
      {
        const auto minMaxWidth =
          std::minmax_element(d_smearedChargeWidths.begin(),
                              d_smearedChargeWidths.end());
        pcout << "ASL smeared charge radius cap min: " << minRadiusCap
              << ", width min/max after cap: " << *minMaxWidth.first << " "
              << *minMaxWidth.second;
        if (smearedChargeWidthCapped)
          pcout << ", max reduction: " << maxRadiusReduction;
        pcout << std::endl;
      }

    const auto getAtomCharge = [&](const dftfe::uInt atomId) {
      return d_dftParamsPtr->isPseudopotential ? atomLocations[atomId][1] :
                                                 atomLocations[atomId][0];
    };

    const auto getAtomPoint = [&](const dftfe::uInt atomId) {
      return dealii::Point<3>(atomLocations[atomId][2],
                              atomLocations[atomId][3],
                              atomLocations[atomId][4]);
    };
    std::vector<dealii::Point<3>> atomPoints(numberGlobalAtoms);
    std::vector<double>           atomCharges(numberGlobalAtoms, 0.0);
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
      {
        atomPoints[iAtom]  = getAtomPoint(iAtom);
        atomCharges[iAtom] = getAtomCharge(iAtom);
      }

    std::vector<dealii::Point<3>> imagePoints(d_imagePositionsTrunc.size());
    std::vector<dftfe::uInt>      imageAtomIds(d_imagePositionsTrunc.size());
    std::vector<double> imageWidths(d_imagePositionsTrunc.size(), 0.0);
    for (dftfe::uInt iImage = 0; iImage < d_imagePositionsTrunc.size();
         ++iImage)
      {
        imagePoints[iImage] =
          dealii::Point<3>(d_imagePositionsTrunc[iImage][0],
                           d_imagePositionsTrunc[iImage][1],
                           d_imagePositionsTrunc[iImage][2]);
        imageAtomIds[iImage] = d_imageIdsTrunc[iImage];
        imageWidths[iImage]  = d_smearedChargeWidths[imageAtomIds[iImage]];
      }
    const dealii::Quadrature<3> &smearedChargeQuadrature =
      d_matrixFreeDataPRefined.get_quadrature(
        d_smearedChargeQuadratureIdElectro);
    dealii::FEValues<3> feValuesSmearedCharge(d_dofHandlerPRefined.get_fe(),
                                              smearedChargeQuadrature,
                                              dealii::update_quadrature_points |
                                                dealii::update_JxW_values);
    const dftfe::uInt   n_q_points_smeared_charge =
      smearedChargeQuadrature.size();

    std::vector<double> smearedChargeIntegral(numberGlobalAtoms, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerPRefined.begin_active(),
      endc = d_dofHandlerPRefined.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const std::pair<dealii::Point<3>, double> enclosingBallCell =
            cell->enclosing_ball();
          const dealii::Point<3> &enclosingBallCellCenter =
            enclosingBallCell.first;
          const double enclosingBallCellRadius = enclosingBallCell.second;

          std::vector<dftfe::uInt> physicalAtomCandidates;
          std::vector<dftfe::uInt> imageCandidates;
          for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
            {
              const double atomWidth = d_smearedChargeWidths[iAtom];
              if (enclosingBallCellCenter.distance(atomPoints[iAtom]) <=
                  enclosingBallCellRadius + atomWidth)
                physicalAtomCandidates.push_back(iAtom);
            }
          for (dftfe::uInt iImage = 0; iImage < d_imagePositionsTrunc.size();
               ++iImage)
            {
              if (enclosingBallCellCenter.distance(imagePoints[iImage]) <=
                  enclosingBallCellRadius + imageWidths[iImage])
                imageCandidates.push_back(iImage);
            }

          if (physicalAtomCandidates.empty() && imageCandidates.empty())
            continue;

          feValuesSmearedCharge.reinit(cell);
          for (dftfe::uInt q = 0; q < n_q_points_smeared_charge; ++q)
            {
              const dealii::Point<3> &quadPoint =
                feValuesSmearedCharge.quadrature_point(q);
              const double jxw = feValuesSmearedCharge.JxW(q);

              for (const dftfe::uInt iAtom : physicalAtomCandidates)
                {
                  const double atomWidth = d_smearedChargeWidths[iAtom];
                  const double r = quadPoint.distance(atomPoints[iAtom]);
                  if (r <= atomWidth)
                    smearedChargeIntegral[iAtom] +=
                      dftUtils::smearedCharge(r, atomWidth) * jxw;
                }
              for (const dftfe::uInt iImage : imageCandidates)
                {
                  const dftfe::uInt imageAtomId = imageAtomIds[iImage];
                  const double      imageWidth  = imageWidths[iImage];
                  const double      r = quadPoint.distance(imagePoints[iImage]);
                  if (r <= imageWidth)
                    smearedChargeIntegral[imageAtomId] +=
                      dftUtils::smearedCharge(r, imageWidth) * jxw;
                }
            }
        }

    if (numberGlobalAtoms > 0)
      MPI_Allreduce(MPI_IN_PLACE,
                    smearedChargeIntegral.data(),
                    numberGlobalAtoms,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
      {
        AssertThrow(
          smearedChargeIntegral[iAtom] > 1.0e-14,
          dealii::ExcMessage(
            "DFT-FE Error: analytic smeared-load charge normalization integral is zero."));
        d_smearedChargeScaling[iAtom] = 1.0 / smearedChargeIntegral[iAtom];
      }
    if (numberGlobalAtoms > 0 && d_dftParamsPtr->verbosity >= 2)
      {
        const auto minMaxScaling =
          std::minmax_element(d_smearedChargeScaling.begin(),
                              d_smearedChargeScaling.end());
        pcout << "ASL smeared charge scaling min/max: " << *minMaxScaling.first
              << " " << *minMaxScaling.second << std::endl;
      }

    std::vector<dftfe::Int> atomLowHighPlusOneIndices;
    if (numberGlobalAtoms > 0)
      dftUtils::createKpointParallelizationIndices(mpi_communicator,
                                                   numberGlobalAtoms,
                                                   atomLowHighPlusOneIndices);

    double            analyticCorrectionEnergy = 0.0;
    const dftfe::uInt mpiTaskId =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const dftfe::uInt atomBegin =
      numberGlobalAtoms > 0 ? atomLowHighPlusOneIndices[2 * mpiTaskId] : 0;
    const dftfe::uInt atomEnd =
      numberGlobalAtoms > 0 ? atomLowHighPlusOneIndices[2 * mpiTaskId + 1] : 0;
    for (dftfe::uInt iAtom = atomBegin; iAtom < atomEnd; ++iAtom)
      {
        const double           atomChargeI = atomCharges[iAtom];
        const double           widthI      = d_smearedChargeWidths[iAtom];
        const dealii::Point<3> pointI      = atomPoints[iAtom];
        analyticCorrectionEnergy +=
          0.5 * atomChargeI * atomChargeI *
          dftUtils::smearedPairInteraction(widthI, widthI, 0.0);
        for (dftfe::uInt jAtom = 0; jAtom < numberGlobalAtoms; ++jAtom)
          if (jAtom != iAtom)
            {
              const dealii::Point<3> pointJ     = atomPoints[jAtom];
              const double           separation = pointI.distance(pointJ);
              analyticCorrectionEnergy +=
                0.5 * atomChargeI * atomCharges[jAtom] *
                dftUtils::smearedPairInteractionDifference(
                  widthI,
                  d_smearedChargeWidths[jAtom],
                  referenceSmearedChargeWidth,
                  referenceSmearedChargeWidth,
                  separation);
            }
        for (dftfe::uInt iImage = 0; iImage < d_imagePositionsTrunc.size();
             ++iImage)
          {
            const double separation = pointI.distance(imagePoints[iImage]);
            analyticCorrectionEnergy +=
              0.5 * atomChargeI * d_imageChargesTrunc[iImage] *
              dftUtils::smearedPairInteractionDifference(
                widthI,
                imageWidths[iImage],
                referenceSmearedChargeWidth,
                referenceSmearedChargeWidth,
                separation);
          }
      }
    d_localVselfs[0][0] = 2.0 * analyticCorrectionEnergy;
    cell                = d_dofHandlerPRefined.begin_active();
    std::vector<double> cellQuadPoints(3 * n_q_points_smeared_charge, 0.0);
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          std::vector<double>   bQuadValuesCell(n_q_points_smeared_charge, 0.0);
          std::set<dftfe::uInt> nonTrivialAtomIdsCell;
          std::set<dftfe::uInt> nonTrivialAtomImageIdsCell;
          std::vector<dftfe::uInt>                  physicalAtomCandidates;
          std::vector<dftfe::uInt>                  imageCandidates;
          const std::pair<dealii::Point<3>, double> enclosingBallCell =
            cell->enclosing_ball();
          const dealii::Point<3> &enclosingBallCellCenter =
            enclosingBallCell.first;
          const double enclosingBallCellRadius = enclosingBallCell.second;
          for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
            {
              const double atomWidth = d_smearedChargeWidths[iAtom];
              if (enclosingBallCellCenter.distance(atomPoints[iAtom]) <=
                  enclosingBallCellRadius + atomWidth)
                physicalAtomCandidates.push_back(iAtom);
            }
          for (dftfe::uInt iImage = 0; iImage < d_imagePositionsTrunc.size();
               ++iImage)
            {
              if (enclosingBallCellCenter.distance(imagePoints[iImage]) <=
                  enclosingBallCellRadius + imageWidths[iImage])
                imageCandidates.push_back(iImage);
            }

          if (!physicalAtomCandidates.empty() || !imageCandidates.empty())
            {
              feValuesSmearedCharge.reinit(cell);
              for (dftfe::uInt q = 0; q < n_q_points_smeared_charge; ++q)
                {
                  const dealii::Point<3> &quadPoint =
                    feValuesSmearedCharge.quadrature_point(q);
                  cellQuadPoints[3 * q + 0] = quadPoint[0];
                  cellQuadPoints[3 * q + 1] = quadPoint[1];
                  cellQuadPoints[3 * q + 2] = quadPoint[2];
                }
              for (const dftfe::uInt iAtom : physicalAtomCandidates)
                {
                  const dealii::Point<3> atomPoint  = atomPoints[iAtom];
                  const double           atomCharge = atomCharges[iAtom];
                  const double atomWidth       = d_smearedChargeWidths[iAtom];
                  const double atomWidthSquare = atomWidth * atomWidth;
                  bool         atomTouchesCell = false;
                  for (dftfe::uInt q = 0; q < n_q_points_smeared_charge; ++q)
                    {
                      const double dx =
                        cellQuadPoints[3 * q + 0] - atomPoint[0];
                      const double dy =
                        cellQuadPoints[3 * q + 1] - atomPoint[1];
                      const double dz =
                        cellQuadPoints[3 * q + 2] - atomPoint[2];
                      const double distanceSquared =
                        dx * dx + dy * dy + dz * dz;
                      if (distanceSquared > atomWidthSquare)
                        continue;
                      const double distanceToAtom = std::sqrt(distanceSquared);

                      const double chargeValue =
                        -atomCharge * d_smearedChargeScaling[iAtom] *
                        dftUtils::smearedCharge(distanceToAtom, atomWidth);
                      bQuadValuesCell[q] += chargeValue;
                      atomTouchesCell = true;
                    }

                  if (atomTouchesCell)
                    {
                      nonTrivialAtomIdsCell.insert(iAtom);
                    }
                }
              for (const dftfe::uInt iImage : imageCandidates)
                {
                  const dftfe::uInt      imageAtomId = imageAtomIds[iImage];
                  const dealii::Point<3> imagePoint  = imagePoints[iImage];
                  const double           imageWidth  = imageWidths[iImage];
                  const double imageWidthSquare      = imageWidth * imageWidth;
                  bool         atomTouchesCell       = false;
                  for (dftfe::uInt q = 0; q < n_q_points_smeared_charge; ++q)
                    {
                      const double dx =
                        cellQuadPoints[3 * q + 0] - imagePoint[0];
                      const double dy =
                        cellQuadPoints[3 * q + 1] - imagePoint[1];
                      const double dz =
                        cellQuadPoints[3 * q + 2] - imagePoint[2];
                      const double distanceSquared =
                        dx * dx + dy * dy + dz * dz;
                      if (distanceSquared > imageWidthSquare)
                        continue;
                      const double distanceToImage = std::sqrt(distanceSquared);

                      const double chargeValue =
                        -d_imageChargesTrunc[iImage] *
                        d_smearedChargeScaling[imageAtomId] *
                        dftUtils::smearedCharge(distanceToImage, imageWidth);
                      bQuadValuesCell[q] += chargeValue;
                      atomTouchesCell = true;
                    }

                  if (atomTouchesCell)
                    {
                      nonTrivialAtomIdsCell.insert(imageAtomId);
                      nonTrivialAtomImageIdsCell.insert(iImage);
                    }
                }
            }
          d_bQuadValuesAllAtoms[cell->id()] = bQuadValuesCell;
          d_bCellNonTrivialAtomIds[cell->id()] =
            std::vector<dftfe::uInt>(nonTrivialAtomIdsCell.begin(),
                                     nonTrivialAtomIdsCell.end());
          d_bCellNonTrivialAtomImageIds[cell->id()] =
            std::vector<dftfe::uInt>(nonTrivialAtomImageIdsCell.begin(),
                                     nonTrivialAtomImageIdsCell.end());
        }
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
    const bool analyticSmearedLoadRoute =
      d_dftParamsPtr->smearedNuclearChargePathway == "ANALYTIC_SMEARED_LOAD";
    double maxSmearedChargeWidth = 0.0;
    if (analyticSmearedLoadRoute && !d_smearedChargeWidths.empty())
      maxSmearedChargeWidth = *std::max_element(d_smearedChargeWidths.begin(),
                                                d_smearedChargeWidths.end());
    const double cutOffForPsp =
      analyticSmearedLoadRoute ?
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
    if (!analyticSmearedLoadRoute)
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
#pragma omp parallel for num_threads(d_nOMPThreads)
    for (dftfe::uInt macrocell = 0;
         macrocell < _matrix_free_data.n_cell_batches();
         ++macrocell)
      {
        if (macrocell <
              kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId + 1] &&
            macrocell >= kptGroupLowHighPlusOneIndicesStep2[2 * kptGroupTaskId])
          {
            dealii::Point<3> atom;
            dftfe::Int       atomicNumber;
            double           atomCharge;


            for (dftfe::uInt iSubCell = 0;
                 iSubCell <
                 _matrix_free_data.n_active_entries_per_cell_batch(macrocell);
                 ++iSubCell)
              {
                dealii::DoFHandler<3>::active_cell_iterator subCellPtr =
                  _matrix_free_data.get_cell_iterator(macrocell,
                                                      iSubCell,
                                                      _phiExtDofHandlerIndex);
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
                    const dealii::Point<3> quadPoint(quadPointPtr[q * 3],
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
                        diffx =
                          quadPoint[0] - atomsImagesPositions[iAtom * 3 + 0];
                        diffy =
                          quadPoint[1] - atomsImagesPositions[iAtom * 3 + 1];
                        diffz =
                          quadPoint[2] - atomsImagesPositions[iAtom * 3 + 2];

                        atomCharge = atomsImagesCharges[iAtom];

                        distanceToAtom = std::sqrt(
                          diffx * diffx + diffy * diffy + diffz * diffz);
                        distanceToAtomInv = 1.0 / distanceToAtom;

                        dftfe::Int chargeId;
                        if (iAtom < numberGlobalCharges)
                          {
                            chargeId     = iAtom;
                            atomicNumber = std::round(atomLocations[iAtom][0]);
                          }
                        else
                          {
                            const dftfe::uInt iImageCharge =
                              iAtom - numberGlobalCharges;
                            chargeId     = d_imageIds[iImageCharge];
                            atomicNumber = std::round(
                              atomLocations[d_imageIds[iImageCharge]][0]);
                          }

                        if (distanceToAtom <= maxTail)
                          {
                            if (distanceToAtom <=
                                outerMostDataPoint[atomicNumber])
                              {
                                if (d_dftParamsPtr->isPseudopotential)
                                  value = d_oncvClassPtr->getRadialLocalPseudo(
                                    atomicNumber, distanceToAtom);
                                else
                                  value = -atomCharge * distanceToAtomInv;
                              }
                            else
                              {
                                value = -atomCharge * distanceToAtomInv;
                              }
                          }
                        else
                          {
                            value = -atomCharge * distanceToAtomInv;
                          }

                        if (analyticSmearedLoadRoute)
                          value -=
                            distanceToAtom > d_smearedChargeWidths[chargeId] ?
                              -atomCharge / distanceToAtom :
                              -atomCharge * d_smearedChargeScaling[chargeId] *
                                dftUtils::smearedPot(
                                  distanceToAtom,
                                  d_smearedChargeWidths[chargeId]);

                        tempVal += value;
                      } // atom loop
                    pseudoVLoc[q] = tempVal;
                  } // quad loop
              }     // subcell loop
          }         // intercomm paral
      }             // cell loop

    if (!analyticSmearedLoadRoute)
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

    std::vector<double> pseudoVLocAtom(n_q_points);
#pragma omp parallel for num_threads(d_nOMPThreads) firstprivate(pseudoVLocAtom)
    for (dftfe::uInt iCell = 0;
         iCell < d_basisOperationsPtrElectroHost->nCells();
         ++iCell)
      {
        if ((iCell <
               kptGroupLowHighPlusOneIndicesStep3[2 * kptGroupTaskId + 1] &&
             iCell >= kptGroupLowHighPlusOneIndicesStep3[2 * kptGroupTaskId]))
          {
            // compute values for the current elements

            dealii::Point<3> atom;
            dftfe::Int       atomicNumber;
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
                    atom[0] = atomLocations[iAtom][2];
                    atom[1] = atomLocations[iAtom][3];
                    atom[2] = atomLocations[iAtom][4];
                    if (d_dftParamsPtr->isPseudopotential)
                      atomCharge = atomLocations[iAtom][1];
                    else
                      atomCharge = atomLocations[iAtom][0];
                    atomicNumber = std::round(atomLocations[iAtom][0]);
                  }
                else
                  {
                    const dftfe::uInt iImageCharge =
                      iAtom - numberGlobalCharges;
                    atom[0] = d_imagePositionsTrunc[iImageCharge][0];
                    atom[1] = d_imagePositionsTrunc[iImageCharge][1];
                    atom[2] = d_imagePositionsTrunc[iImageCharge][2];
                    if (d_dftParamsPtr->isPseudopotential)
                      atomCharge =
                        atomLocations[d_imageIdsTrunc[iImageCharge]][1];
                    else
                      atomCharge =
                        atomLocations[d_imageIdsTrunc[iImageCharge]][0];
                    atomicNumber = std::round(
                      atomLocations[d_imageIdsTrunc[iImageCharge]][0]);
                  }

                std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
                  boundaryPoints(atom - tempDisp, atom + tempDisp);

                dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

                if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) ==
                    dealii::NeighborType::not_neighbors)
                  continue;
                bool         isPseudoDataInCell = false;
                double       value, distanceToAtom;
                const double cutoff = outerMostDataPoint[atomicNumber];
                // loop over quad points
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    const dealii::Point<3> quadPoint(quadPointPtr[q * 3],
                                                     quadPointPtr[q * 3 + 1],
                                                     quadPointPtr[q * 3 + 2]);
                    distanceToAtom = quadPoint.distance(atom);
                    if (distanceToAtom <= cutoff)
                      {
                        if (d_dftParamsPtr->isPseudopotential)
                          {
                            value = d_oncvClassPtr->getRadialLocalPseudo(
                              atomicNumber, distanceToAtom);
                          }
                        else
                          {
                            value = -atomCharge / distanceToAtom;
                          }
                      }
                    else
                      {
                        value = -atomCharge / distanceToAtom;
                      }

                    if (distanceToAtom <= cutOffForPsp)
                      isPseudoDataInCell = true;

                    pseudoVLocAtom[q] = value;
                  } // loop over quad points
                if (isPseudoDataInCell)
                  {
#pragma omp critical(pseudovalsatoms)
                    _pseudoValuesAtoms[iAtom][d_basisOperationsPtrElectroHost
                                                ->cellID(iCell)] =
                      pseudoVLocAtom;
                  }
              } // loop over atoms
          }     // kpt paral loop
      }         // cell loop

    if (d_basisOperationsPtrElectroHost->nCells() > 0 && numberKptGroups > 1)
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
#include "dft.inst.cc"
} // namespace dftfe
