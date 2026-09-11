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
#include <dftfe/fileReaders.h>
#include <dftfe/vectorUtilities.h>
#include <dftfe/atomicInitKernels.h>
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>
#include <dftfe/atomTypeSplineTables.h>
#ifdef DFTFE_WITH_DEVICE
#  include <dftfe/AtomCenteredSphericalFunctionDeviceKernels.h>
#endif
#include <type_traits>
#include <random>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::loadPSIFiles(dftfe::uInt  Z,
                                      dftfe::uInt  n,
                                      dftfe::uInt  l,
                                      dftfe::uInt &fileReadFlag)
  {
    if (radValues[Z][n].count(l) > 0)
      {
        fileReadFlag = 1;
        return;
      }


    //
    // set the paths for the Single-Atom wavefunction data
    //
    char psiFile[256];

    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->pseudoTestsFlag)
      // if(d_dftParamsPtr->pseudoProjector==2)
      // sprintf(psiFile,
      // "%s/data/electronicStructure/pseudoPotential/z%u/oncv/singleAtomData/psi%u%u.inp",
      // DFTFE_PATH, Z, n, l); else
      sprintf(
        psiFile,
        "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/psi%u%u.inp",
        DFTFE_PATH,
        Z,
        n,
        l);
    else if (d_dftParamsPtr->isPseudopotential &&
             !d_dftParamsPtr->pseudoTestsFlag)
      {
        strcpy(psiFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(Z) + "/" +
                "psi" + std::to_string(n) + std::to_string(l) + ".inp")
                 .c_str());
        std::vector<std::vector<double>> values;
        fileReadFlag = dftUtils::readPsiFile(2, values, psiFile);
        if (fileReadFlag == 0)
          {
            sprintf(
              psiFile,
              "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/psi%u%u.inp",
              DFTFE_PATH,
              Z,
              n,
              l);
          }
      }
    else
      sprintf(
        psiFile,
        "%s/data/electronicStructure/allElectron/z%u/singleAtomData/psi%u%u.inp",
        DFTFE_PATH,
        Z,
        n,
        l);

    std::vector<std::vector<double>> values;
    fileReadFlag = dftUtils::readPsiFile(2, values, psiFile);

    const double truncationTol =
      d_dftParamsPtr->reproducible_output ? 1e-10 : 1e-8;
    //
    // spline fitting for single-atom wavefunctions
    //
    if (fileReadFlag > 0)
      {
        double      maxTruncationRadius = 0.0;
        dftfe::uInt truncRowId          = 0;
        if (!d_dftParamsPtr->reproducible_output)
          pcout << "reading data from file: " << psiFile << std::endl;

        dftfe::Int          numRows = values.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        // x
        for (dftfe::Int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = values[irow][0];
          }
        outerValues[Z][n][l] = xData[numRows - 1];
        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);

        // y
        for (dftfe::Int irow = 0; irow < numRows; ++irow)
          {
            yData[irow] = values[irow][1];

            if (std::fabs(yData[irow]) > truncationTol)
              truncRowId = irow;
          }
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type = 0;
        alglib::spline1dbuildcubic(x,
                                   y,
                                   numRows,
                                   natural_bound_type,
                                   0.0,
                                   natural_bound_type,
                                   0.0,
                                   radValues[Z][n][l]);


        maxTruncationRadius = xData[truncRowId];
        if (maxTruncationRadius > d_wfcInitTruncation)
          d_wfcInitTruncation = maxTruncationRadius;
      }
  }

  //
  // determine orbital ordering
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::determineOrbitalFilling()
  {
    //
    // create a stencil following orbital filling order
    //
    std::vector<dftfe::uInt>              level;
    std::vector<std::vector<dftfe::uInt>> stencil;

    // 1s
    level.clear();
    level.push_back(1);
    level.push_back(0);
    stencil.push_back(level);
    // 2s
    level.clear();
    level.push_back(2);
    level.push_back(0);
    stencil.push_back(level);
    // 2p
    level.clear();
    level.push_back(2);
    level.push_back(1);
    stencil.push_back(level);
    // 3s
    level.clear();
    level.push_back(3);
    level.push_back(0);
    stencil.push_back(level);
    // 3p
    level.clear();
    level.push_back(3);
    level.push_back(1);
    stencil.push_back(level);
    // 4s
    level.clear();
    level.push_back(4);
    level.push_back(0);
    stencil.push_back(level);
    // 3d
    level.clear();
    level.push_back(3);
    level.push_back(2);
    stencil.push_back(level);
    // 4p
    level.clear();
    level.push_back(4);
    level.push_back(1);
    stencil.push_back(level);
    // 5s
    level.clear();
    level.push_back(5);
    level.push_back(0);
    stencil.push_back(level);
    // 4d
    level.clear();
    level.push_back(4);
    level.push_back(2);
    stencil.push_back(level);
    // 5p
    level.clear();
    level.push_back(5);
    level.push_back(1);
    stencil.push_back(level);
    // 6s
    level.clear();
    level.push_back(6);
    level.push_back(0);
    stencil.push_back(level);
    // 4f
    level.clear();
    level.push_back(4);
    level.push_back(3);
    stencil.push_back(level);
    // 5d
    level.clear();
    level.push_back(5);
    level.push_back(2);
    stencil.push_back(level);
    // 6p
    level.clear();
    level.push_back(6);
    level.push_back(1);
    stencil.push_back(level);
    // 7s
    level.clear();
    level.push_back(7);
    level.push_back(0);
    stencil.push_back(level);
    // 5f
    level.clear();
    level.push_back(5);
    level.push_back(3);
    stencil.push_back(level);
    // 6d
    level.clear();
    level.push_back(6);
    level.push_back(2);
    stencil.push_back(level);
    // 7p
    level.clear();
    level.push_back(7);
    level.push_back(1);
    stencil.push_back(level);
    // 8s
    level.clear();
    level.push_back(8);
    level.push_back(0);
    stencil.push_back(level);



    const dftfe::uInt numberGlobalAtoms  = atomLocations.size();
    const dftfe::Int  numberImageCharges = d_imageIds.size();
    const dftfe::Int  totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
    const dftfe::uInt numWfcComponents =
      (d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC ? 2 : 1);
    dftfe::uInt errorReadFile            = 0;
    dftfe::uInt fileReadFlag             = 0;
    dftfe::uInt waveFunctionCount        = 0;
    dftfe::uInt totalNumberWaveFunctions = d_numEigenValues;

    for (std::vector<std::vector<dftfe::uInt>>::iterator it = stencil.begin();
         it < stencil.end();
         it++)
      {
        dftfe::uInt n = (*it)[0], l = (*it)[1];

        for (dftfe::Int m = -l; m <= (dftfe::Int)l; m++)
          {
            for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
              {
                dftfe::uInt Z = atomLocations[iAtom][0];

                //
                // fill levels
                //
                if (radValues.count(Z) == 0)
                  {
                    pcout << "Z:" << Z << std::endl;
                  }

                //
                // load PSI files
                //
                loadPSIFiles(Z, n, l, fileReadFlag);

                if (fileReadFlag > 0)
                  {
                    orbital temp;
                    temp.atomID = iAtom;
                    temp.Z      = Z;
                    temp.n      = n;
                    temp.l      = l;
                    temp.m      = m;
                    temp.psi    = radValues[Z][n][l];
                    temp.waveID = waveFunctionCount;
                    waveFunctionsVector.push_back(temp);
                    waveFunctionCount++;
                    if (waveFunctionCount >=
                          d_numEigenValues / numWfcComponents &&
                        waveFunctionCount >= numberGlobalAtoms)
                      break;
                  }
              }

            if (waveFunctionCount >= d_numEigenValues / numWfcComponents &&
                waveFunctionCount >= numberGlobalAtoms)
              break;
          }

        if (waveFunctionCount >= d_numEigenValues / numWfcComponents &&
            waveFunctionCount >= numberGlobalAtoms)
          break;

        if (fileReadFlag == 0)
          errorReadFile += 1;
      }


    if (waveFunctionsVector.size() > d_numEigenValues / numWfcComponents)
      {
        d_numEigenValues = numWfcComponents * waveFunctionsVector.size();
      }

    pcout
      << "============================================================================================================================="
      << std::endl;
    pcout << "number of electrons: " << numElectrons << std::endl;
    pcout << "number of eigen values: " << d_numEigenValues << std::endl;

    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << "number of wavefunctions computed using single atom data to be used as initial guess for starting the SCF: "
        << waveFunctionCount << std::endl;

    if (errorReadFile == stencil.size())
      {
        // std::cerr<< "Error: Require single-atom wavefunctions as initial
        // guess for starting the SCF."<< std::endl; std::cerr<< "Error: Could
        // not find single-atom wavefunctions for any atom: "<< std::endl;
        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "CAUTION: Could not find single-atom wavefunctions for any atom- the starting guess for all wavefunctions will be random."
            << std::endl;
        // exit(-1);
      }
    pcout
      << "============================================================================================================================="
      << std::endl;
  }

  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::readPSIRadialValues()
  {
#ifdef DFTFE_WITH_DEVICE
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      if (d_dftParamsPtr->startingWFCType == "RANDOM" &&
          !d_dftParamsPtr->reproducible_output &&
          d_dftParamsPtr->solverMode == "GS")
        {
          readPSIRandomValuesDevice();
          return;
        }
#endif
    const dealii::IndexSet &locallyOwnedSet = dofHandler.locally_owned_dofs();
    std::vector<dealii::IndexSet::size_type> locallyOwnedDOFs =
      locallyOwnedSet.get_index_vector();
    dftfe::uInt       numberDofs = locallyOwnedDOFs.size();
    const dftfe::uInt numWfcComponents =
      (d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC ? 2 : 1);

    // sized here rather than by the caller, so that the device generator
    // above can leave the host storage unallocated. BANDS reuses one spin
    // and k-point block.
    const dftfe::uInt numWfcBlocks =
      d_dftParamsPtr->solverMode == "BANDS" ?
        1 :
        (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
    d_eigenVectorsFlattenedHost.resize(d_numEigenValues * numberDofs *
                                         numWfcComponents * numWfcBlocks,
                                       dataTypes::number(0.0));
    std::fill(d_eigenVectorsFlattenedHost.begin(),
              d_eigenVectorsFlattenedHost.end(),
              0.0);

    const dftfe::uInt numberGlobalAtoms = atomLocations.size();

    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << "Number of wavefunctions generated randomly to be used as initial guess for starting the SCF : "
        << d_numEigenValues - numWfcComponents * waveFunctionsVector.size()
        << std::endl;
    //
    // loop over nodes
    //

    std::vector<orbital>   waveFunctionsVectorTruncated;
    dealii::BoundingBox<3> boundingBoxTria(
      vectorTools::createBoundingBoxTriaLocallyOwned(dofHandler));
    dealii::Tensor<1, 3, double> tempDisp;
    tempDisp[0] = d_wfcInitTruncation;
    tempDisp[1] = d_wfcInitTruncation;
    tempDisp[2] = d_wfcInitTruncation;

    // An orbital contributes only if one of its centers, the master atom or
    // any of its periodic images, lies within d_wfcInitTruncation of the
    // locally owned cells.
    for (std::vector<orbital>::iterator it = waveFunctionsVector.begin();
         it < waveFunctionsVector.end();
         it++)
      {
        std::vector<dftfe::Int> imageIdsListFilter;
        if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
            d_dftParamsPtr->periodicZ)
          imageIdsListFilter = d_globalChargeIdToImageIdMap[it->atomID];
        else
          imageIdsListFilter.push_back(it->atomID);

        bool anyCenterInRange = false;
        for (dftfe::uInt iImage = 0;
             iImage < imageIdsListFilter.size() && !anyCenterInRange;
             ++iImage)
          {
            const dftfe::Int chargeId = imageIdsListFilter[iImage];
            dealii::Point<3> atomCoord;

            if (chargeId < numberGlobalAtoms)
              {
                atomCoord[0] = atomLocations[chargeId][2];
                atomCoord[1] = atomLocations[chargeId][3];
                atomCoord[2] = atomLocations[chargeId][4];
              }
            else
              {
                atomCoord[0] =
                  d_imagePositions[chargeId - numberGlobalAtoms][0];
                atomCoord[1] =
                  d_imagePositions[chargeId - numberGlobalAtoms][1];
                atomCoord[2] =
                  d_imagePositions[chargeId - numberGlobalAtoms][2];
              }

            std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
              boundaryPoints;
            boundaryPoints.first  = atomCoord - tempDisp;
            boundaryPoints.second = atomCoord + tempDisp;
            dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

            if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) !=
                dealii::NeighborType::not_neighbors)
              anyCenterInRange = true;
          }

        if (anyCenterInRange)
          waveFunctionsVectorTruncated.push_back(*it);
      }


    // per-orbital center lists: the master atom and its periodic images
    const dftfe::uInt nOrbitalsTrunc = waveFunctionsVectorTruncated.size();
    std::vector<dftfe::uInt> orbitalCenterStart(nOrbitalsTrunc + 1, 0);
    std::vector<double>      orbitalCenterCoords;
    for (dftfe::uInt iOrb = 0; iOrb < nOrbitalsTrunc; ++iOrb)
      {
        std::vector<dftfe::Int> imageIdsList;
        if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
            d_dftParamsPtr->periodicZ)
          imageIdsList =
            d_globalChargeIdToImageIdMap[waveFunctionsVectorTruncated[iOrb]
                                           .atomID];
        else
          imageIdsList.push_back(waveFunctionsVectorTruncated[iOrb].atomID);

        for (dftfe::uInt iImage = 0; iImage < imageIdsList.size(); ++iImage)
          {
            const dftfe::Int chargeId = imageIdsList[iImage];
            if (chargeId < numberGlobalAtoms)
              {
                orbitalCenterCoords.push_back(atomLocations[chargeId][2]);
                orbitalCenterCoords.push_back(atomLocations[chargeId][3]);
                orbitalCenterCoords.push_back(atomLocations[chargeId][4]);
              }
            else
              {
                orbitalCenterCoords.push_back(
                  d_imagePositions[chargeId - numberGlobalAtoms][0]);
                orbitalCenterCoords.push_back(
                  d_imagePositions[chargeId - numberGlobalAtoms][1]);
                orbitalCenterCoords.push_back(
                  d_imagePositions[chargeId - numberGlobalAtoms][2]);
              }
          }
        orbitalCenterStart[iOrb + 1] = orbitalCenterCoords.size() / 3;
      }


    // One thread or iteration per (unconstrained node, orbital), centers
    // looped inside the kernel; a device run dispatches to the device
    // specialization and a CPU run to the host one. The loop further down
    // covers the cases this cannot handle.
    bool wfcComputed = false;
    if (nOrbitalsTrunc > 0 && numberDofs > 0)
      {
        splineHostTableUtils::SplineTableGroup psiTables;
        std::vector<dftfe::Int>                orbitalLHost(nOrbitalsTrunc),
          orbitalMHost(nOrbitalsTrunc);
        std::vector<dftfe::uInt> orbitalSplineIdHost(nOrbitalsTrunc);
        for (dftfe::uInt iOrb = 0; iOrb < nOrbitalsTrunc; ++iOrb)
          {
            const orbital &orb = waveFunctionsVectorTruncated[iOrb];
            orbitalLHost[iOrb] = orb.l;
            orbitalMHost[iOrb] = orb.m;
            orbitalSplineIdHost[iOrb] =
              psiTables.addSpline(orb.psi, d_wfcInitTruncation, 0.0, 1.0);
          }
        AtomTypeSplineTables<memorySpace> psiTablesMem;
        psiTablesMem.stage(psiTables);

        // compacted so the kernel needs no constraint handling
        std::vector<double>      pointCoordsHost;
        std::vector<dftfe::uInt> pointDofIndex;
        pointCoordsHost.reserve(3 * numberDofs);
        pointDofIndex.reserve(numberDofs);
        for (dftfe::uInt dof = 0; dof < numberDofs; ++dof)
          {
            const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
            if (constraintsNone.is_constrained(dofID))
              continue;
            const dealii::Point<3> &node = d_supportPoints[dofID];
            pointCoordsHost.push_back(node[0]);
            pointCoordsHost.push_back(node[1]);
            pointCoordsHost.push_back(node[2]);
            pointDofIndex.push_back(dof);
          }
        const dftfe::uInt nPointsWfc = pointDofIndex.size();

        if (nPointsWfc > 0)
          {
            dftfe::utils::MemoryStorage<double, memorySpace> dPoints, dCenters,
              dOut;
            dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> dCenterStart,
              dSplineIdWfc;
            dftfe::utils::MemoryStorage<dftfe::Int, memorySpace> dOrbL, dOrbM;
            stageHostVector(orbitalCenterCoords, dCenters);
            stageHostVector(orbitalCenterStart, dCenterStart);
            stageHostVector(orbitalSplineIdHost, dSplineIdWfc);
            stageHostVector(orbitalLHost, dOrbL);
            stageHostVector(orbitalMHost, dOrbM);

            const bool isNoncolinWfc =
              d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC;
            dftfe::uInt numKpointsWfc =
              (d_dftParamsPtr->reproducible_output ?
                 ((1 + d_dftParamsPtr->spinPolarized) *
                  d_kPointWeights.size()) :
                 (1 + d_dftParamsPtr->spinPolarized));
            if (d_dftParamsPtr->solverMode == "BANDS")
              numKpointsWfc = 1;

            // cap on the transient output buffer of one block, in bytes
            constexpr std::size_t maxBlockBytes = 256000000;
            const dftfe::uInt     pointsBlockSize =
              std::max<dftfe::uInt>(1,
                                    maxBlockBytes /
                                      (std::size_t)((nOrbitalsTrunc + 3) *
                                                    sizeof(double)));
            std::vector<double> hOut;

            for (dftfe::uInt pointBlockStart = 0; pointBlockStart < nPointsWfc;
                 pointBlockStart += pointsBlockSize)
              {
                const dftfe::uInt nPointsInBlock =
                  std::min(pointsBlockSize, nPointsWfc - pointBlockStart);
                dPoints.resize((std::size_t)nPointsInBlock * 3);
                dPoints.template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  pointCoordsHost.data() + (std::size_t)pointBlockStart * 3);
                dOut.resize((std::size_t)nPointsInBlock * nOrbitalsTrunc);

                atomicInitKernels::atomicWavefunctionSuperposition<memorySpace>(
                  nPointsInBlock,
                  nOrbitalsTrunc,
                  isNoncolinWfc,
                  d_wfcInitTruncation,
                  dPoints.data(),
                  dCenterStart.data(),
                  dCenters.data(),
                  dOrbL.data(),
                  dOrbM.data(),
                  dSplineIdWfc.data(),
                  psiTablesMem.knotX.data(),
                  psiTablesMem.knotY.data(),
                  psiTablesMem.coefB.data(),
                  psiTablesMem.coefC.data(),
                  psiTablesMem.coefD.data(),
                  psiTablesMem.offsets.data(),
                  psiTablesMem.nKnots.data(),
                  psiTablesMem.gpParams.data(),
                  dOut.data());

                hOut.resize((std::size_t)nPointsInBlock * nOrbitalsTrunc);
                dOut.copyTo(hOut);

                for (dftfe::uInt iPoint = 0; iPoint < nPointsInBlock; ++iPoint)
                  {
                    const dftfe::uInt dof =
                      pointDofIndex[pointBlockStart + iPoint];
                    for (dftfe::uInt iOrb = 0; iOrb < nOrbitalsTrunc; ++iOrb)
                      {
                        const double acc =
                          hOut[(std::size_t)iPoint * nOrbitalsTrunc + iOrb];
                        const dftfe::uInt waveId =
                          waveFunctionsVectorTruncated[iOrb].waveID;
                        for (dftfe::uInt kPoint = 0; kPoint < numKpointsWfc;
                             ++kPoint)
                          if (isNoncolinWfc)
                            {
                              const dftfe::uInt baseNC =
                                kPoint * d_numEigenValues * numberDofs * 2 +
                                2 * dof * d_numEigenValues;
                              d_eigenVectorsFlattenedHost[baseNC +
                                                          2 * waveId] +=
                                dataTypes::number(acc);
                              d_eigenVectorsFlattenedHost[baseNC +
                                                          d_numEigenValues +
                                                          2 * waveId] +=
                                dataTypes::number(acc);
                              d_eigenVectorsFlattenedHost[baseNC + 2 * waveId +
                                                          1] +=
                                dataTypes::number(acc);
                              d_eigenVectorsFlattenedHost[baseNC +
                                                          d_numEigenValues +
                                                          2 * waveId + 1] +=
                                -dataTypes::number(acc);
                            }
                          else
                            {
                              const dftfe::uInt baseCO =
                                kPoint * d_numEigenValues * numberDofs +
                                dof * d_numEigenValues;
                              d_eigenVectorsFlattenedHost[baseCO + waveId] +=
                                dataTypes::number(acc);
                            }
                      }
                  }
              }

            wfcComputed = true;
          }
      }

    bool pp = false;
#pragma omp parallel num_threads(d_nOMPThreads) \
  firstprivate(waveFunctionsVectorTruncated)
    {
#pragma omp for
      for (dftfe::uInt dof = 0; dof < numberDofs; dof++)
        {
          const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
          dealii::Point<3>                      node  = d_supportPoints[dofID];
          if (!constraintsNone.is_constrained(dofID))
            {
              //
              // loop over wave functions
              //
              dftfe::uInt numKpoints = (d_dftParamsPtr->reproducible_output ?
                                          ((1 + d_dftParamsPtr->spinPolarized) *
                                           d_kPointWeights.size()) :
                                          (1 + d_dftParamsPtr->spinPolarized));
              if (d_dftParamsPtr->solverMode == "BANDS")
                numKpoints = 1;
              // The guess depends only on the node and the orbital; the
              // k-point index enters only the destination offset, so each
              // orbital's contribution is accumulated once and replicated
              // across the slices.
              for (dftfe::uInt iOrb = 0;
                   iOrb < (wfcComputed ? 0 : nOrbitalsTrunc);
                   ++iOrb)
                {
                  const orbital    &orb    = waveFunctionsVectorTruncated[iOrb];
                  const dftfe::uInt waveId = orb.waveID;
                  const bool        isNoncolin =
                    d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC;

                  double acc = 0.0;
                  for (dftfe::uInt iAtom = orbitalCenterStart[iOrb];
                       iAtom < orbitalCenterStart[iOrb + 1];
                       ++iAtom)
                    {
                      const double x =
                        node[0] - orbitalCenterCoords[iAtom * 3 + 0];
                      const double y =
                        node[1] - orbitalCenterCoords[iAtom * 3 + 1];
                      const double z =
                        node[2] - orbitalCenterCoords[iAtom * 3 + 2];

                      const double r = std::sqrt(x * x + y * y + z * z);

                      // the angles and the harmonic are only needed inside
                      // the truncation radius
                      if (r > d_wfcInitTruncation)
                        continue;

                      double theta = std::acos(z / r);
                      double phi   = std::atan2(y, x);
                      if (r == 0)
                        {
                          theta = 0;
                          phi   = 0;
                        }

                      // radial part
                      const double R = alglib::spline1dcalc((orb.psi), r);

                      const double Yval =
                        orb.m >= 0 ? boost::math::spherical_harmonic_r(orb.l,
                                                                       orb.m,
                                                                       theta,
                                                                       phi) :
                                     boost::math::spherical_harmonic_i(orb.l,
                                                                       -(orb.m),
                                                                       theta,
                                                                       phi);

                      acc +=
                        isNoncolin ?
                          (orb.m == 0 ? R * Yval / std::sqrt(2.0) : R * Yval) :
                          (orb.m == 0 ? R * Yval : R * std::sqrt(2) * Yval);
                    }

                  for (dftfe::Int kPoint = 0; kPoint < numKpoints; ++kPoint)
                    if (isNoncolin)
                      {
                        const dftfe::uInt baseNC =
                          kPoint * d_numEigenValues * numberDofs * 2 +
                          2 * dof * d_numEigenValues;
                        d_eigenVectorsFlattenedHost[baseNC + 2 * waveId] +=
                          dataTypes::number(acc);
                        d_eigenVectorsFlattenedHost[baseNC + d_numEigenValues +
                                                    2 * waveId] +=
                          dataTypes::number(acc);
                        d_eigenVectorsFlattenedHost[baseNC + 2 * waveId + 1] +=
                          dataTypes::number(acc);
                        d_eigenVectorsFlattenedHost[baseNC + d_numEigenValues +
                                                    2 * waveId + 1] +=
                          -dataTypes::number(acc);
                      }
                    else
                      {
                        const dftfe::uInt baseCO =
                          kPoint * d_numEigenValues * numberDofs +
                          dof * d_numEigenValues;
                        d_eigenVectorsFlattenedHost[baseCO + waveId] +=
                          dataTypes::number(acc);
                      }
                }

              for (dftfe::Int kPoint = 0; kPoint < numKpoints; ++kPoint)
                {
                  d_nonAtomicWaveFunctions = 0;
                  if (waveFunctionsVector.size() <
                      d_numEigenValues / numWfcComponents)
                    {
                      d_nonAtomicWaveFunctions =
                        d_numEigenValues -
                        numWfcComponents * waveFunctionsVector.size();

                      //
                      // Counter-based generator keyed on the global dof
                      // index, so the guess is the same on the host and the
                      // device and is independent of the MPI and thread
                      // decomposition.
                      dataTypes::number *temp =
                        d_eigenVectorsFlattenedHost.data() +
                        kPoint * d_numEigenValues * numberDofs *
                          numWfcComponents;
                      const unsigned long long globalDofId =
                        static_cast<unsigned long long>(dofID);
                      for (dftfe::uInt iWave =
                             waveFunctionsVector.size() * numWfcComponents;
                           iWave < d_numEigenValues;
                           ++iWave)
                        for (dftfe::uInt iSpinor = 0;
                             iSpinor < numWfcComponents;
                             ++iSpinor)
                          {
                            const double u1 =
                              atomCenteredSphericalFunctionDeviceUtils::
                                uniformRandomCentered(
                                  atomCenteredSphericalFunctionDeviceUtils::
                                    randomWavefunctionSeed,
                                  globalDofId * numWfcComponents + iSpinor,
                                  iWave,
                                  0ULL);
                            const double u2 =
                              atomCenteredSphericalFunctionDeviceUtils::
                                uniformRandomCentered(
                                  atomCenteredSphericalFunctionDeviceUtils::
                                    randomWavefunctionSeed,
                                  globalDofId * numWfcComponents + iSpinor,
                                  iWave,
                                  1ULL);
                            const double z     = 3.0 * u1;
                            const double value = std::exp(-0.5 * z * z) *
                                                 0.3989422804014327 *
                                                 (u2 < 0.0 ? -1.0 : 1.0);
                            temp[numWfcComponents * dof * d_numEigenValues +
                                 iSpinor * d_numEigenValues + iWave] =
                              dataTypes::number(value);
                          }
                    }
                }
            }
        }
    }

    if (!d_dftParamsPtr->reproducible_output)
      {
        dftfe::uInt numKpoints =
          (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
        if (d_dftParamsPtr->solverMode == "BANDS")
          numKpoints = 1;
        for (dftfe::uInt kPoint = 1; kPoint < numKpoints; ++kPoint)
          {
            dataTypes::number *temp1 =
              d_eigenVectorsFlattenedHost.data() +
              kPoint * d_numEigenValues * numberDofs *
                ((d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC) ? 2 : 1);

            dataTypes::number *temp2 = d_eigenVectorsFlattenedHost.data();

            for (dftfe::uInt idof = 0;
                 idof <
                 numberDofs *
                   ((d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC) ? 2 :
                                                                           1);
                 idof++)
              for (dftfe::uInt iwave = 0; iwave < d_numEigenValues; iwave++)
                temp1[idof * d_numEigenValues + iwave] =
                  temp2[idof * d_numEigenValues + iwave];
          }
      }

    if (d_dftParamsPtr->startingWFCType == "RANDOM")
      {
        pcout
          << "============================================================================================================================="
          << std::endl;
        pcout << "number of electrons: " << numElectrons << std::endl;
        pcout << "number of eigen values: " << d_numEigenValues << std::endl;
        pcout
          << "============================================================================================================================="
          << std::endl;
      }
  }

  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::readPSI()
  {
    computing_timer.enter_subsection("initialize wave functions");
    readPSIRadialValues();
    computing_timer.leave_subsection("initialize wave functions");
  }
#ifdef DFTFE_WITH_DEVICE
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::readPSIRandomValuesDevice()
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        const dealii::IndexSet &locallyOwnedSet =
          dofHandler.locally_owned_dofs();
        std::vector<dealii::IndexSet::size_type> locallyOwnedDOFs =
          locallyOwnedSet.get_index_vector();
        const dftfe::uInt numberDofs = locallyOwnedDOFs.size();
        const dftfe::uInt numSpinorComponents =
          (d_dftParamsPtr->noncolin || d_dftParamsPtr->hasSOC) ? 2 : 1;
        const dftfe::uInt numBlocks =
          (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
        const dftfe::uInt blockSize =
          d_numEigenValues * numberDofs * numSpinorComponents;

        d_eigenVectorsFlattenedDevice.resize(blockSize * numBlocks,
                                             dataTypes::number(0.0));

        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "Number of wavefunctions generated randomly to be used as initial guess for starting the SCF : "
            << d_numEigenValues << std::endl;

        d_nonAtomicWaveFunctions = d_numEigenValues;

        // constrained dofs stay zero, matching the host generator
        std::vector<dftfe::uInt> constrainedMaskHost(numberDofs, 0);
        for (dftfe::uInt dof = 0; dof < numberDofs; dof++)
          if (constraintsNone.is_constrained(locallyOwnedDOFs[dof]))
            constrainedMaskHost[dof] = 1;
        dftfe::utils::MemoryStorage<dftfe::uInt,
                                    dftfe::utils::MemorySpace::DEVICE>
          constrainedMaskDevice(numberDofs, 0);
        constrainedMaskDevice.copyFrom(constrainedMaskHost);

        // Fill the first (k-point, spin) block; the keying on the global
        // dof id makes the guess independent of the parallel decomposition.
        // The kernel derives its key as globalDofStartIndex + iNode, which
        // equals the host key locallyOwnedDOFs[dof] * numSpinorComponents +
        // iSpinor only while the locally owned set is one contiguous range.
        // deal.II distributed dof handlers guarantee that; assert it so a
        // future change cannot silently make host and device disagree.
        AssertThrow(
          numberDofs == 0 || locallyOwnedSet.is_contiguous(),
          dealii::ExcMessage(
            "DFT-FE Error: random wavefunction device path requires a contiguous locally owned dof range."));
        const unsigned long long globalDofStartIndex =
          numberDofs > 0 ?
            static_cast<unsigned long long>(locallyOwnedDOFs[0]) *
              numSpinorComponents :
            0ULL;
        const dftfe::uInt numRealsPerCoeff =
          std::is_same<dataTypes::number, std::complex<double>>::value ? 2 : 1;
        atomCenteredSphericalFunctionDeviceKernels::fillRandomWavefunctions(
          numberDofs * numSpinorComponents,
          d_numEigenValues,
          numRealsPerCoeff,
          numSpinorComponents,
          globalDofStartIndex,
          0ULL,
          atomCenteredSphericalFunctionDeviceUtils::randomWavefunctionSeed,
          constrainedMaskDevice.data(),
          reinterpret_cast<double *>(d_eigenVectorsFlattenedDevice.data()));

        // replicate the first block to all other (k-point, spin) blocks,
        // matching the host generator's behavior
        for (dftfe::uInt iBlock = 1; iBlock < numBlocks; iBlock++)
          d_eigenVectorsFlattenedDevice.copyFrom(d_eigenVectorsFlattenedDevice,
                                                 blockSize,
                                                 0,
                                                 iBlock * blockSize);

        pcout
          << "============================================================================================================================="
          << std::endl;
        pcout << "number of electrons: " << numElectrons << std::endl;
        pcout << "number of eigen values: " << d_numEigenValues << std::endl;
        pcout
          << "============================================================================================================================="
          << std::endl;
      }
  }
#endif

#include "dft.inst.cc"
} // namespace dftfe
