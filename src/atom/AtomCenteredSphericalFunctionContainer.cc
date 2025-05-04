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

#include "AtomCenteredSphericalFunctionContainer.h"

namespace dftfe
{
  const std::vector<dftfe::Int> &
  AtomCenteredSphericalFunctionContainer::getAtomIdsInElement(dftfe::uInt iElem)
  {
    return (d_AtomIdsInElement[iElem]);
  }

  const dftfe::uInt
  AtomCenteredSphericalFunctionContainer::getOffsetLocation(
    const dftfe::uInt iAtom)
  {
    AssertThrow(iAtom < d_AtomIdsInCurrentProcess.size(),
                dealii::ExcMessage(
                  "DFT-FE Error: Inconsistent iAtom index used to get OffSet"));
    return (d_offsetLocation[iAtom]);
  }

  void
  AtomCenteredSphericalFunctionContainer::init(
    const std::vector<dftfe::uInt> &atomicNumbers,
    const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                   std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      &listOfSphericalFunctions)
  {
    // std::cout << "Initialising Container Class: " << std::endl;
    d_atomicNumbers               = atomicNumbers;
    d_sphericalFunctionsContainer = listOfSphericalFunctions;
    std::map<dftfe::uInt, dftfe::uInt> startIndexLocation;
    for (const auto &[key, value] : listOfSphericalFunctions)
      {
        dftfe::uInt atomicNumber = key.first;
        dftfe::uInt alpha        = key.second;
        dftfe::uInt lIndex       = value->getQuantumNumberl();
        if (auto atomNumSize = d_numRadialSphericalFunctions.find(atomicNumber);
            atomNumSize != d_numRadialSphericalFunctions.end())
          {
            d_numRadialSphericalFunctions[atomicNumber] =
              d_numRadialSphericalFunctions[atomicNumber] + 1;
            d_numSphericalFunctions[atomicNumber] =
              d_numSphericalFunctions[atomicNumber] + (2 * lIndex + 1);
            d_totalSphericalFunctionIndexStart[atomicNumber].push_back(
              startIndexLocation[atomicNumber]);
            startIndexLocation[atomicNumber] += 2 * lIndex + 1;
          }
        else
          {
            d_numRadialSphericalFunctions[atomicNumber] = 1;
            d_numSphericalFunctions[atomicNumber]       = (2 * lIndex + 1);
            startIndexLocation[atomicNumber]            = 0;
            d_totalSphericalFunctionIndexStart[atomicNumber].push_back(
              startIndexLocation[atomicNumber]);
            startIndexLocation[atomicNumber] += 2 * lIndex + 1;
          }
      }
  }
  void
  AtomCenteredSphericalFunctionContainer::initaliseCoordinates(
    const std::vector<double>              &atomCoords,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<dftfe::Int>          &imageIds)
  {
    d_atomCoords = atomCoords;
    setImageCoordinates(imageIds, periodicCoords);


    // AssertChecks
    AssertThrow(
      d_atomicNumbers.size() == d_atomCoords.size() / 3,
      dealii::ExcMessage(
        "DFT-FE Error: Number of atom Coordinates if interest is differnt from number of atomic Numbers"));
  }

  void
  AtomCenteredSphericalFunctionContainer::setImageCoordinates(
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords)
  {
    d_periodicImageCoord.clear();
    // std::cout << "PeriodicCoords original Size: " << periodicCoords.size()
    //           << std::endl;
    // std::cout << "ImageIds original Size: " << imageIds.size() << std::endl;
    for (dftfe::uInt iAtom = 0; iAtom < d_atomicNumbers.size(); iAtom++)
      {
        d_periodicImageCoord[iAtom].push_back(d_atomCoords[3 * iAtom + 0]);
        d_periodicImageCoord[iAtom].push_back(d_atomCoords[3 * iAtom + 1]);
        d_periodicImageCoord[iAtom].push_back(d_atomCoords[3 * iAtom + 2]);
      }

    for (dftfe::Int iImageId = 0; iImageId < imageIds.size(); iImageId++)
      {
        const dftfe::Int chargeId = imageIds[iImageId];
        d_periodicImageCoord[chargeId].push_back(periodicCoords[iImageId][0]);
        d_periodicImageCoord[chargeId].push_back(periodicCoords[iImageId][1]);
        d_periodicImageCoord[chargeId].push_back(periodicCoords[iImageId][2]);
      }

    // for (dftfe::Int iCharge = 0; iCharge < d_periodicImageCoord.size();
    // iCharge++)
    //   {
    //     dftfe::Int size = d_periodicImageCoord[iCharge].size() / 3;
    //     for (dftfe::Int i = 0; i < size; i++)
    //       {
    //         std::cout << "Processor charges and locations: " << iCharge << "
    //         "
    //                   << d_periodicImageCoord[iCharge][3 * i + 0] << " "
    //                   << d_periodicImageCoord[iCharge][3 * i + 1] << " "
    //                   << d_periodicImageCoord[iCharge][3 * i + 2] <<
    //                   std::endl;
    //       }
    //   }
  }

  dftfe::uInt
  AtomCenteredSphericalFunctionContainer::getNumAtomCentersSize()
  {
    return d_atomicNumbers.size();
  }

  const std::vector<double> &
  AtomCenteredSphericalFunctionContainer::getAtomCoordinates() const
  {
    return d_atomCoords;
  }

  const std::map<dftfe::uInt, std::vector<double>> &
  AtomCenteredSphericalFunctionContainer::getPeriodicImageCoordinatesList()
    const
  {
    return d_periodicImageCoord;
  }

  dftfe::uInt
  AtomCenteredSphericalFunctionContainer::
    getTotalNumberOfRadialSphericalFunctionsPerAtom(dftfe::uInt atomicNumber)
  {
    if (auto atomNumSize = d_numRadialSphericalFunctions.find(atomicNumber);
        atomNumSize != d_numRadialSphericalFunctions.end())
      {
        return atomNumSize->second;
      }
    else
      return 0;
  }

  dftfe::uInt
  AtomCenteredSphericalFunctionContainer::
    getTotalNumberOfSphericalFunctionsPerAtom(dftfe::uInt atomicNumber)
  {
    if (auto atomNumSize = d_numSphericalFunctions.find(atomicNumber);
        atomNumSize != d_numSphericalFunctions.end())
      {
        return atomNumSize->second;
      }
    else
      return 0;
  }



  dftfe::uInt
  AtomCenteredSphericalFunctionContainer::
    getTotalNumberOfSphericalFunctionsInCurrentProcessor()
  {
    dftfe::uInt totalShapeFns = 0;
    for (dftfe::Int iAtom = 0; iAtom < d_AtomIdsInCurrentProcess.size();
         iAtom++)
      {
        dftfe::uInt atomId = d_AtomIdsInCurrentProcess[iAtom];
        dftfe::uInt Znum   = d_atomicNumbers[atomId];
        totalShapeFns += d_numSphericalFunctions.find(Znum)->second;
      }

    return (totalShapeFns);
  }

  dftfe::uInt
  AtomCenteredSphericalFunctionContainer::getMaximumNumberOfSphericalFunctions()
  {
    dftfe::uInt maxShapeFns = 0;
    for (std::map<dftfe::uInt, dftfe::uInt>::const_iterator it =
           d_numSphericalFunctions.begin();
         it != d_numSphericalFunctions.end();
         ++it)
      {
        if (it->second > maxShapeFns)
          maxShapeFns = it->second;
      }

    return (maxShapeFns);
  }


  void
  AtomCenteredSphericalFunctionContainer::
    getTotalAtomsAndNonLocalElementsInCurrentProcessor(
      dftfe::uInt              &totalAtomsInCurrentProcessor,
      dftfe::uInt              &totalNonLocalElements,
      std::vector<dftfe::uInt> &numberCellsForEachAtom,
      std::vector<dftfe::uInt> &numberCellsAccumNonLocalAtoms,
      std::vector<dftfe::uInt> &iElemNonLocalToElemIndexMap)
  {
    totalAtomsInCurrentProcessor = d_AtomIdsInCurrentProcess.size();
    numberCellsAccumNonLocalAtoms.clear();
    numberCellsAccumNonLocalAtoms.resize(totalAtomsInCurrentProcessor, 0);
    numberCellsForEachAtom.clear();
    numberCellsForEachAtom.resize(totalAtomsInCurrentProcessor, 0);
    totalNonLocalElements = 0;
    d_offsetLocation.clear();
    d_offsetLocation.resize(totalAtomsInCurrentProcessor, 0);
    dftfe::uInt offset = 0;
    for (dftfe::uInt iAtom = 0; iAtom < totalAtomsInCurrentProcessor; iAtom++)
      {
        dftfe::uInt atomId = d_AtomIdsInCurrentProcess[iAtom];

        d_offsetLocation[iAtom] = offset;
        offset +=
          getTotalNumberOfSphericalFunctionsPerAtom(d_atomicNumbers[atomId]);
        const dftfe::uInt numberElementsInCompactSupport =
          d_elementIndexesInAtomCompactSupport[atomId].size();
        numberCellsAccumNonLocalAtoms[iAtom] = totalNonLocalElements;
        totalNonLocalElements += numberElementsInCompactSupport;
        numberCellsForEachAtom[iAtom] = numberElementsInCompactSupport;
      }
    iElemNonLocalToElemIndexMap.clear();
    iElemNonLocalToElemIndexMap.resize(totalNonLocalElements, 0);
    offset = 0;
    for (dftfe::uInt iAtom = 0; iAtom < totalAtomsInCurrentProcessor; iAtom++)
      {
        dftfe::uInt atomId = d_AtomIdsInCurrentProcess[iAtom];

        const dftfe::uInt numberElementsInCompactSupport =
          d_elementIndexesInAtomCompactSupport[atomId].size();
        for (dftfe::Int iElem = 0; iElem < numberElementsInCompactSupport;
             iElem++)
          {
            iElemNonLocalToElemIndexMap[offset] =
              d_elementIndexesInAtomCompactSupport[atomId][iElem];
            offset++;
          }
      }
  }

  const dftfe::uInt
  AtomCenteredSphericalFunctionContainer::getTotalSphericalFunctionIndexStart(
    dftfe::uInt Znum,
    dftfe::uInt alpha)
  {
    std::vector<dftfe::uInt> beta = d_totalSphericalFunctionIndexStart[Znum];
    if (alpha < getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum))
      return beta[alpha];
    else
      {
        std::cout
          << "Warning Illegal Access in Line 132 of AtomCenteredContainerClass"
          << std::endl;
        std::exit(0);
      }
  }


  const std::vector<dftfe::uInt> &
  AtomCenteredSphericalFunctionContainer::getAtomicNumbers() const
  {
    return (d_atomicNumbers);
  }
  const std::vector<dftfe::uInt> &
  AtomCenteredSphericalFunctionContainer::getAtomIdsInCurrentProcess() const
  {
    return (d_AtomIdsInCurrentProcess);
  }


  const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                 std::shared_ptr<AtomCenteredSphericalFunctionBase>> &
  AtomCenteredSphericalFunctionContainer::getSphericalFunctions() const
  {
    return d_sphericalFunctionsContainer;
  }

  void
  AtomCenteredSphericalFunctionContainer::getDataForSparseStructure(
    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &sparsityPattern,
    const std::vector<std::vector<dealii::CellId>>
      &elementIdsInAtomCompactSupport,
    const std::vector<std::vector<dftfe::uInt>>
                                   &elementIndexesInAtomCompactSupport,
    const std::vector<dftfe::uInt> &atomIdsInCurrentProcess,
    dftfe::uInt                     numberElements)
  {
    d_sparsityPattern.clear();
    d_elementIdsInAtomCompactSupport.clear();
    d_elementIndexesInAtomCompactSupport.clear();
    d_AtomIdsInCurrentProcess.clear();

    d_sparsityPattern                    = sparsityPattern;
    d_elementIdsInAtomCompactSupport     = elementIdsInAtomCompactSupport;
    d_elementIndexesInAtomCompactSupport = elementIndexesInAtomCompactSupport;
    d_AtomIdsInCurrentProcess            = atomIdsInCurrentProcess;
    d_AtomIdsInElement.clear();
    d_AtomIdsInElement.resize(numberElements);

    for (dftfe::Int iCell = 0; iCell < numberElements; ++iCell)
      {
        for (dftfe::Int iAtom = 0; iAtom < d_AtomIdsInCurrentProcess.size();
             iAtom++)
          {
            if (d_sparsityPattern[d_AtomIdsInCurrentProcess[iAtom]][iCell] >= 0)
              {
                d_AtomIdsInElement[iCell].push_back(
                  d_AtomIdsInCurrentProcess[iAtom]);
              }
          }
      }
  }

  template <typename NumberType>
  void
  AtomCenteredSphericalFunctionContainer::computeSparseStructure(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt quadratureIndex,
    const double      cutOffVal,
    const dftfe::uInt cutOffType)
  {
    //
    // get the number of non-local atoms
    //
    dftfe::Int numberAtomsOfInterest = d_atomicNumbers.size(); //

    // std::cout<<" numberAtomsOfInterest = "<<numberAtomsOfInterest<<"\n";

    //     //
    //     // pre-allocate data structures that stores the sparsity of deltaVl
    //     //
    d_sparsityPattern.clear();
    d_elementIdsInAtomCompactSupport.clear();
    d_elementIndexesInAtomCompactSupport.clear();

    // d_sparsityPattern.resize(numberAtomsOfInterest);
    d_elementIdsInAtomCompactSupport.resize(numberAtomsOfInterest);
    d_elementIndexesInAtomCompactSupport.resize(numberAtomsOfInterest);
    d_AtomIdsInCurrentProcess.clear();

    //
    // loop over nonlocal atoms
    //
    dftfe::uInt       sparseFlag         = 0;
    dftfe::Int        cumulativeSplineId = 0;
    dftfe::Int        waveFunctionId;
    const dftfe::uInt totalLocallyOwnedCells = basisOperationsPtr->nCells();

    basisOperationsPtr->reinit(0, 0, quadratureIndex);
    const dftfe::uInt numberQuadraturePoints =
      basisOperationsPtr->nQuadsPerCell();

    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadraturePointsVector = basisOperationsPtr->quadPoints();
    //
    // get number of global charges
    //
    dftfe::uInt       numberGlobalCharges = d_atomicNumbers.size();
    const dftfe::uInt numberElements      = totalLocallyOwnedCells;

    std::vector<dftfe::Int> sparsityPattern(numberElements, -1);

    for (dftfe::Int iAtom = 0; iAtom < numberAtomsOfInterest; ++iAtom)
      {
        //
        // temp variables
        //
        dftfe::Int  matCount            = 0;
        bool        isAtomIdInProcessor = false;
        dftfe::uInt Znum                = d_atomicNumbers[iAtom];
        //
        //
        dftfe::Int numberSphericalFunctions =
          d_numRadialSphericalFunctions[Znum];

        // std::cout<<" iAtom = "<<iAtom <<" numberSphericalFunctions =
        // "<<numberSphericalFunctions<<"\n";
        //
        // get the global charge Id of the current nonlocal atom
        //

        // std::cout<<" totalLocallyOwnedCells = "<<totalLocallyOwnedCells<<"
        // numberQuadraturePoints = "<<numberQuadraturePoints<<"\n";

        dftfe::uInt imageIdsSize = d_periodicImageCoord[iAtom].size() / 3;

        //
        // resize the data structure corresponding to sparsity pattern
        //

        std::fill(sparsityPattern.begin(), sparsityPattern.end(), -1);
        //
        // parallel loop over all elements
        //

        for (dftfe::Int iCell = 0; iCell < totalLocallyOwnedCells; iCell++)
          {
            double              maxR = 0.0;
            std::vector<double> quadPoints(numberQuadraturePoints * 3, 0.0);
            for (dftfe::Int iQuad = 0; iQuad < numberQuadraturePoints; iQuad++)
              {
                quadPoints[iQuad * 3 + 0] =
                  quadraturePointsVector[iCell * (numberQuadraturePoints * 3) +
                                         iQuad * 3 + 0];
                quadPoints[iQuad * 3 + 1] =
                  quadraturePointsVector[iCell * (numberQuadraturePoints * 3) +
                                         iQuad * 3 + 1];
                quadPoints[iQuad * 3 + 2] =
                  quadraturePointsVector[iCell * (numberQuadraturePoints * 3) +
                                         iQuad * 3 + 2];
              }
            sparseFlag = 0;
            for (dftfe::Int iImageAtomCount = 0; iImageAtomCount < imageIdsSize;
                 ++iImageAtomCount)
              {
                std::vector<double> x(3, 0.0);
                dealii::Point<3>    chargePoint(0.0, 0.0, 0.0);
                if (iImageAtomCount == 0)
                  {
                    chargePoint[0] = d_atomCoords[3 * iAtom + 0];
                    chargePoint[1] = d_atomCoords[3 * iAtom + 1];
                    chargePoint[2] = d_atomCoords[3 * iAtom + 2];
                  }
                else
                  {
                    chargePoint[0] =
                      d_periodicImageCoord[iAtom][3 * iImageAtomCount + 0];
                    chargePoint[1] =
                      d_periodicImageCoord[iAtom][3 * iImageAtomCount + 1];
                    chargePoint[2] =
                      d_periodicImageCoord[iAtom][3 * iImageAtomCount + 2];
                  }

                // if(iCell == 0)
                //   std::cout<<"DEBUG coordinates: "<<iAtom<<"
                //   "<<chargePoint[0]<<" "<<chargePoint[1]<<"
                //   "<<chargePoint[2]<<std::endl;

                for (dftfe::uInt iPsp = 0; iPsp < numberSphericalFunctions;
                     ++iPsp)
                  {
                    std::shared_ptr<AtomCenteredSphericalFunctionBase>
                      SphericalFunction =
                        d_sphericalFunctionsContainer[std::make_pair(Znum,
                                                                     iPsp)];
                    double radialProjVal;
                    for (dftfe::Int iQuadPoint = 0;
                         iQuadPoint < numberQuadraturePoints;
                         ++iQuadPoint)
                      {
                        x[0] = quadPoints[3 * iQuadPoint] - chargePoint[0];
                        x[1] = quadPoints[3 * iQuadPoint + 1] - chargePoint[1];
                        x[2] = quadPoints[3 * iQuadPoint + 2] - chargePoint[2];
                        const double r =
                          std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

                        if (cutOffType == 0)
                          {
                            double RadVal =
                              SphericalFunction->getRadialValue(r);

                            //                            std::cout
                            //                              << "DEBUG: iAtom
                            //                              RadVal projIndex
                            //                              Cell: "
                            //                              << iAtom << " " << r
                            //                              << " "
                            //                              << std::fabs(RadVal)
                            //                              << " " << iPsp << "
                            //                              "
                            //                              << iCell <<
                            //                              std::endl;

                            if (std::fabs(RadVal) >= cutOffVal)
                              {
                                sparseFlag = 1;
                                if (r > maxR)
                                  maxR = r;

                                break;
                              }
                          }
                        else if (cutOffType == 1 && r < cutOffVal)
                          {
                            sparseFlag = 1;
                            break;
                          }
                      } // quadrature loop
                    if (sparseFlag == 1)
                      break;

                  } // iPsp loop ("l" loop)

                if (sparseFlag == 1)
                  break;


              } // image atom loop

            if (sparseFlag == 1)
              {
                dealii::CellId cell    = basisOperationsPtr->cellID(iCell);
                sparsityPattern[iCell] = matCount;
                // std::cout<<"Debug: iAtom iCell cellid maxR: "<<iAtom<<"
                // "<<iCell<<" "<<cell<<" "<<maxR<<std::endl;
                d_elementIdsInAtomCompactSupport[iAtom].push_back(cell);
                d_elementIndexesInAtomCompactSupport[iAtom].push_back(iCell);
                matCount += 1;
                isAtomIdInProcessor = true;
              }

          } // iCell


#ifdef DEBUG
        std::cout << "No.of non zero elements in the compact support of atom "
                  << iAtom << " is "
                  << d_elementIndexesInAtomCompactSupport[iAtom].size()
                  << std::endl;
#endif

        if (isAtomIdInProcessor)
          {
            d_AtomIdsInCurrentProcess.push_back(iAtom);
            d_sparsityPattern[iAtom] = sparsityPattern;
          }

      } // atom loop

    d_AtomIdsInElement.clear();
    d_AtomIdsInElement.resize(numberElements);

    for (dftfe::Int iCell = 0; iCell < numberElements; ++iCell)
      {
        for (dftfe::Int iAtom = 0; iAtom < d_AtomIdsInCurrentProcess.size();
             iAtom++)
          {
            if (d_sparsityPattern[d_AtomIdsInCurrentProcess[iAtom]][iCell] >= 0)
              {
                d_AtomIdsInElement[iCell].push_back(
                  d_AtomIdsInCurrentProcess[iAtom]);
              }
          }
      }
  }
  template void
  AtomCenteredSphericalFunctionContainer::computeSparseStructure(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt quadratureIndex,
    const double      cutOffVal,
    const dftfe::uInt cutOffType);
#ifdef USE_COMPLEX
  template void
  AtomCenteredSphericalFunctionContainer::computeSparseStructure(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<std::complex<double>,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt quadratureIndex,
    const double      cutOffVal,
    const dftfe::uInt cutOffType);

#endif
  const std::map<dftfe::uInt, std::vector<dftfe::Int>> &
  AtomCenteredSphericalFunctionContainer::getSparsityPattern()
  {
    return (d_sparsityPattern);
  }

  bool
  AtomCenteredSphericalFunctionContainer::atomSupportInElement(
    dftfe::uInt iElem)
  {
    return (d_AtomIdsInElement[iElem].size() > 0 ? true : false);
  }

} // end of namespace dftfe
