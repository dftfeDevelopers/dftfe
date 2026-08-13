// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// ---------------------------------------------------------------------

#include <dftfe/analyticSmearedLoadManager.h>
#include <dftfe/dftParameters.h>
#include <dftfe/dftUtils.h>
#include <dftfe/oncvClass.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  analyticSmearedLoadManager<memorySpace>::initialize(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<double>              &imageCharges,
    const std::vector<std::vector<double>> &imagePositions,
    const std::vector<std::vector<double>> &meshSizes,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const double                            minDist,
    const double                            pspCutOffTrunc,
    const std::shared_ptr<oncvClass<dataTypes::number, memorySpace>>
                                        &oncvClassPtr,
    const dealii::DoFHandler<3>         &dofHandlerPRefined,
    const dealii::MatrixFree<3, double> &matrixFreeDataPRefined,
    const dftfe::uInt                    smearedChargeQuadratureId,
    const MPI_Comm                      &mpiCommunicator,
    const dftParameters                 &dftParams,
    dealii::ConditionalOStream          &pcout)
  {
    const dftfe::uInt numberGlobalAtoms = atomLocations.size();

    AssertThrow(
      imagePositions.size() == imageIds.size() &&
        imageCharges.size() == imageIds.size(),
      dealii::ExcMessage(
        "DFT-FE Error: inconsistent truncated-image data in ASL initialization."));
    for (const dftfe::Int atomId : imageIds)
      AssertThrow(
        atomId >= 0 && static_cast<dftfe::uInt>(atomId) < numberGlobalAtoms,
        dealii::ExcMessage(
          "DFT-FE Error: invalid truncated-image owner in ASL initialization."));

    d_bQuadValuesAllAtoms.clear();
    d_bCellNonTrivialAtomIds.clear();
    d_bCellNonTrivialAtomImageIds.clear();
    d_physicalCandidatesByCell.clear();
    d_imageCandidatesByCell.clear();
    d_localVselfs.clear();
    d_localVselfs.resize(1, std::vector<double>(1, 0.0));

    d_referenceSmearedChargeWidth =
      std::min(0.7, std::max(1.0e-8, 0.5 * minDist - 0.3));

    d_smearedChargeWidths.assign(numberGlobalAtoms,
                                 dftParams.analyticSmearedLoadRadius);
    d_smearedChargeScaling.assign(numberGlobalAtoms, 1.0);

    if (dftParams.analyticSmearedLoadRadius <= 0.0)
      {
        constexpr double              fluxChargeRelativeTolerance = 1.0e-3;
        constexpr double              meshResolutionFactor        = 2.5;
        constexpr dftfe::uInt         nRadiusSamples              = 2000;
        std::map<dftfe::uInt, double> atomTypeToPspCoreWidth;
        std::map<dftfe::uInt, double> atomTypeToMeshFloor;
        std::map<dftfe::uInt, double> atomTypeToWidth;

        auto getAtomMeshSize = [&](const dftfe::uInt atomId) {
          if (dftParams.meshSizesFile != "" && atomId < meshSizes.size() &&
              !meshSizes[atomId].empty() && meshSizes[atomId][0] > 0.0)
            return meshSizes[atomId][0];
          return dftParams.meshSizeOuterBall;
        };

        auto estimateCoreWidthFromLocalPotential =
          [&](const dftfe::uInt atomicNumber, const double atomCharge) {
            if (!dftParams.isPseudopotential || !oncvClassPtr ||
                atomCharge <= 0.0)
              return d_referenceSmearedChargeWidth;

            const double maxLocalPotentialRadius =
              oncvClassPtr->getRmaxLocalPot(atomicNumber);
            if (!std::isfinite(maxLocalPotentialRadius) ||
                maxLocalPotentialRadius <= d_referenceSmearedChargeWidth)
              return d_referenceSmearedChargeWidth;

            std::vector<double> residuals(
              nRadiusSamples, std::numeric_limits<double>::quiet_NaN());
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
                  oncvClassPtr->getRadialLocalPseudo(atomicNumber, r + h);
                const double vMinus =
                  oncvClassPtr->getRadialLocalPseudo(atomicNumber, r - h);
                if (!std::isfinite(vPlus) || !std::isfinite(vMinus))
                  continue;
                const double dVdr           = (vPlus - vMinus) / (2.0 * h);
                const double enclosedCharge = r * r * dVdr;
                const double residual =
                  std::abs(enclosedCharge - atomCharge) / atomCharge;
                if (std::isfinite(residual))
                  residuals[iSample] = residual;
              }

            double     tailMaximum = 0.0;
            dftfe::Int firstStable = -1;
            for (dftfe::Int iSample =
                   static_cast<dftfe::Int>(residuals.size()) - 1;
                 iSample >= 1;
                 --iSample)
              {
                if (!std::isfinite(residuals[iSample]))
                  continue;
                tailMaximum = std::max(tailMaximum, residuals[iSample]);
                if (tailMaximum <= fluxChargeRelativeTolerance)
                  firstStable = iSample;
              }

            double coreWidth = maxLocalPotentialRadius;
            if (firstStable > 0)
              coreWidth = sampleMax * static_cast<double>(firstStable) /
                          static_cast<double>(nRadiusSamples - 1);
            else if (dftParams.verbosity >= 1)
              pcout << "Warning: no stable Coulombic tail was identified for Z="
                    << atomicNumber
                    << "; using the full local-potential radius "
                    << maxLocalPotentialRadius << "." << std::endl;
            if (!std::isfinite(coreWidth) || coreWidth <= 0.0)
              coreWidth = d_referenceSmearedChargeWidth;
            return std::max(coreWidth, d_referenceSmearedChargeWidth);
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
            const double atomCharge = dftParams.isPseudopotential ?
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
                  std::max(d_referenceSmearedChargeWidth,
                           atomTypeToMeshFloor[atomicNumber]);
                const double selectedWidth =
                  std::max(coreIt->second, meshFloor);
                widthIt =
                  atomTypeToWidth.insert({atomicNumber, selectedWidth}).first;
              }
            d_smearedChargeWidths[iAtom] = widthIt->second;
          }

        if (numberGlobalAtoms > 0 && dftParams.verbosity >= 2)
          {
            pcout << "ASL automatic smeared charge widths by atom type:";
            for (const auto &typeWidth : atomTypeToWidth)
              pcout << " Z=" << typeWidth.first << ":" << typeWidth.second
                    << "(core=" << atomTypeToPspCoreWidth[typeWidth.first]
                    << ", mesh=" << atomTypeToMeshFloor[typeWidth.first] << ")";
            pcout << std::endl;
          }
      }

    const bool hasPeriodicDirection =
      dftParams.periodicX || dftParams.periodicY || dftParams.periodicZ;
    const std::array<bool, 3> periodicDirection = {
      static_cast<bool>(dftParams.periodicX),
      static_cast<bool>(dftParams.periodicY),
      static_cast<bool>(dftParams.periodicZ)};
    auto latticeVector = [&](const dftfe::uInt iVector) {
      return std::array<double, 3>{domainBoundingVectors[iVector][0],
                                   domainBoundingVectors[iVector][1],
                                   domainBoundingVectors[iVector][2]};
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
    // Half the truncated-image cutoff guarantees that any smeared support
    // reaching the domain is represented by the retained periodic images.
    const double periodicCap = hasPeriodicDirection ?
                                 0.5 * pspCutOffTrunc :
                                 std::numeric_limits<double>::max();

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
    if (numberGlobalAtoms > 0 && dftParams.verbosity >= 2)
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

    d_smearedPairRadiusCoefficients.clear();
    d_smearedPairRadiusCoefficients.reserve(numberGlobalAtoms);
    for (const double width : d_smearedChargeWidths)
      d_smearedPairRadiusCoefficients.emplace_back(width);

    const auto getAtomCharge = [&](const dftfe::uInt atomId) {
      return dftParams.isPseudopotential ? atomLocations[atomId][1] :
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

    std::vector<dealii::Point<3>> imagePoints(imagePositions.size());
    std::vector<dftfe::uInt>      imageAtomIds(imagePositions.size());
    std::vector<double>           imageWidths(imagePositions.size(), 0.0);
    for (dftfe::uInt iImage = 0; iImage < imagePositions.size(); ++iImage)
      {
        imagePoints[iImage]  = dealii::Point<3>(imagePositions[iImage][0],
                                               imagePositions[iImage][1],
                                               imagePositions[iImage][2]);
        imageAtomIds[iImage] = imageIds[iImage];
        imageWidths[iImage]  = d_smearedChargeWidths[imageAtomIds[iImage]];
      }
    const dealii::Quadrature<3> &smearedChargeQuadrature =
      matrixFreeDataPRefined.get_quadrature(smearedChargeQuadratureId);
    dealii::FEValues<3> feValuesSmearedCharge(dofHandlerPRefined.get_fe(),
                                              smearedChargeQuadrature,
                                              dealii::update_quadrature_points |
                                                dealii::update_JxW_values);
    const dftfe::uInt   n_q_points_smeared_charge =
      smearedChargeQuadrature.size();

    std::vector<double> smearedChargeIntegral(numberGlobalAtoms, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandlerPRefined.begin_active(),
      endc = dofHandlerPRefined.end();
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
          for (dftfe::uInt iImage = 0; iImage < imagePositions.size(); ++iImage)
            {
              if (enclosingBallCellCenter.distance(imagePoints[iImage]) <=
                  enclosingBallCellRadius + imageWidths[iImage])
                imageCandidates.push_back(iImage);
            }

          const dealii::CellId cellId  = cell->id();
          const auto physicalInsertion = d_physicalCandidatesByCell.emplace(
            cellId, std::move(physicalAtomCandidates));
          const auto imageInsertion =
            d_imageCandidatesByCell.emplace(cellId, std::move(imageCandidates));
          const std::vector<dftfe::uInt> &physicalCandidates =
            physicalInsertion.first->second;
          const std::vector<dftfe::uInt> &cellImageCandidates =
            imageInsertion.first->second;

          if (physicalCandidates.empty() && cellImageCandidates.empty())
            continue;

          feValuesSmearedCharge.reinit(cell);
          for (dftfe::uInt q = 0; q < n_q_points_smeared_charge; ++q)
            {
              const dealii::Point<3> &quadPoint =
                feValuesSmearedCharge.quadrature_point(q);
              const double jxw = feValuesSmearedCharge.JxW(q);

              for (const dftfe::uInt iAtom : physicalCandidates)
                {
                  const double atomWidth = d_smearedChargeWidths[iAtom];
                  const double r = quadPoint.distance(atomPoints[iAtom]);
                  if (r <= atomWidth)
                    smearedChargeIntegral[iAtom] +=
                      dftUtils::smearedCharge(r, atomWidth) * jxw;
                }
              for (const dftfe::uInt iImage : cellImageCandidates)
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
                    mpiCommunicator);
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; ++iAtom)
      {
        AssertThrow(
          smearedChargeIntegral[iAtom] > 1.0e-14,
          dealii::ExcMessage(
            "DFT-FE Error: analytic smeared-load charge normalization integral is zero."));
        d_smearedChargeScaling[iAtom] = 1.0 / smearedChargeIntegral[iAtom];
      }
    if (numberGlobalAtoms > 0 && dftParams.verbosity >= 2)
      {
        const auto minMaxScaling =
          std::minmax_element(d_smearedChargeScaling.begin(),
                              d_smearedChargeScaling.end());
        pcout << "ASL smeared charge scaling min/max: " << *minMaxScaling.first
              << " " << *minMaxScaling.second << std::endl;
      }

    double            analyticCorrectionEnergy = 0.0;
    const dftfe::uInt mpiTaskId =
      dealii::Utilities::MPI::this_mpi_process(mpiCommunicator);
    const dftfe::uInt nMpiTasks =
      dealii::Utilities::MPI::n_mpi_processes(mpiCommunicator);
    const dftfe::uInt atomBegin = numberGlobalAtoms * mpiTaskId / nMpiTasks;
    const dftfe::uInt atomEnd = numberGlobalAtoms * (mpiTaskId + 1) / nMpiTasks;
    for (dftfe::uInt iAtom = atomBegin; iAtom < atomEnd; ++iAtom)
      {
        const double           atomChargeI = atomCharges[iAtom];
        const double           widthI      = d_smearedChargeWidths[iAtom];
        const dealii::Point<3> pointI      = atomPoints[iAtom];
        analyticCorrectionEnergy +=
          0.5 * atomChargeI * atomChargeI *
          analyticSmearedLoadManager<memorySpace>::smearedPairInteraction(
            widthI, widthI, 0.0);
        for (dftfe::uInt jAtom = 0; jAtom < numberGlobalAtoms; ++jAtom)
          if (jAtom != iAtom)
            {
              const dealii::Point<3> pointJ     = atomPoints[jAtom];
              const double           separation = pointI.distance(pointJ);
              analyticCorrectionEnergy +=
                0.5 * atomChargeI * atomCharges[jAtom] *
                analyticSmearedLoadManager<memorySpace>::
                  smearedPairInteractionDifference(
                    widthI,
                    d_smearedChargeWidths[jAtom],
                    d_referenceSmearedChargeWidth,
                    d_referenceSmearedChargeWidth,
                    separation,
                    d_smearedPairRadiusCoefficients[iAtom],
                    d_smearedPairRadiusCoefficients[jAtom]);
            }
        for (dftfe::uInt iImage = 0; iImage < imagePositions.size(); ++iImage)
          {
            const double separation = pointI.distance(imagePoints[iImage]);
            analyticCorrectionEnergy +=
              0.5 * atomChargeI * imageCharges[iImage] *
              analyticSmearedLoadManager<memorySpace>::
                smearedPairInteractionDifference(
                  widthI,
                  imageWidths[iImage],
                  d_referenceSmearedChargeWidth,
                  d_referenceSmearedChargeWidth,
                  separation,
                  d_smearedPairRadiusCoefficients[iAtom],
                  d_smearedPairRadiusCoefficients[imageAtomIds[iImage]]);
          }
      }
    // Downstream energy assembly applies 0.5 * (phi - vSelf), so store twice
    // the analytic correction here to preserve its full contribution.
    d_localVselfs[0][0] = 2.0 * analyticCorrectionEnergy;
    cell                = dofHandlerPRefined.begin_active();
    std::vector<double> cellQuadPoints(3 * n_q_points_smeared_charge, 0.0);
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          std::vector<double>   bQuadValuesCell(n_q_points_smeared_charge, 0.0);
          std::set<dftfe::uInt> nonTrivialAtomIdsCell;
          std::set<dftfe::uInt> nonTrivialAtomImageIdsCell;
          const auto            physicalCandidatesIt =
            d_physicalCandidatesByCell.find(cell->id());
          const auto imageCandidatesIt =
            d_imageCandidatesByCell.find(cell->id());
          AssertThrow(physicalCandidatesIt !=
                          d_physicalCandidatesByCell.end() &&
                        imageCandidatesIt != d_imageCandidatesByCell.end(),
                      dealii::ExcMessage(
                        "DFT-FE Error: missing cached ASL cell candidates."));
          const std::vector<dftfe::uInt> &physicalAtomCandidates =
            physicalCandidatesIt->second;
          const std::vector<dftfe::uInt> &imageCandidates =
            imageCandidatesIt->second;

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
                        -imageCharges[iImage] *
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

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<double> &
  analyticSmearedLoadManager<memorySpace>::smearedChargeWidths() const
  {
    return d_smearedChargeWidths;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<double> &
  analyticSmearedLoadManager<memorySpace>::smearedChargeScaling() const
  {
    return d_smearedChargeScaling;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  analyticSmearedLoadManager<memorySpace>::pairInteractionDerivativeDifference(
    const dftfe::uInt atomIdA,
    const dftfe::uInt atomIdB,
    const double      separation) const
  {
    AssertIndexRange(atomIdA, d_smearedChargeWidths.size());
    AssertIndexRange(atomIdB, d_smearedChargeWidths.size());
    return smearedPairInteractionDerDifference(
      d_smearedChargeWidths[atomIdA],
      d_smearedChargeWidths[atomIdB],
      d_referenceSmearedChargeWidth,
      d_referenceSmearedChargeWidth,
      separation,
      d_smearedPairRadiusCoefficients[atomIdA],
      d_smearedPairRadiusCoefficients[atomIdB]);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  analyticSmearedLoadManager<memorySpace>::localVselfs() const
  {
    return d_localVselfs;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<double>> &
  analyticSmearedLoadManager<memorySpace>::bQuadValuesAllAtoms() const
  {
    return d_bQuadValuesAllAtoms;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::map<dealii::CellId, std::vector<double>> &
  analyticSmearedLoadManager<memorySpace>::bQuadValuesAllAtoms()
  {
    return d_bQuadValuesAllAtoms;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<dftfe::uInt>> &
  analyticSmearedLoadManager<memorySpace>::bCellNonTrivialAtomIds() const
  {
    return d_bCellNonTrivialAtomIds;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<dftfe::uInt>> &
  analyticSmearedLoadManager<memorySpace>::bCellNonTrivialAtomImageIds() const
  {
    return d_bCellNonTrivialAtomImageIds;
  }

  template class analyticSmearedLoadManager<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class analyticSmearedLoadManager<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
