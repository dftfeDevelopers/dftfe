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
// @author Sambit Das
//

// source file for restart functionality in dftClass

//
//
#include <dft.h>
#include <fileReaders.h>
#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <sys/stat.h>

namespace dftfe
{
  namespace internal
  {
    std::vector<double>
    getFractionalCoordinates(const std::vector<double> &latticeVectors,
                             const dealii::Point<3>    &point,
                             const dealii::Point<3>    &corner);
    std::vector<double>
    wrapAtomsAcrossPeriodicBc(const dealii::Point<3>    &cellCenteredCoord,
                              const dealii::Point<3>    &corner,
                              const std::vector<double> &latticeVectors,
                              const std::vector<bool>   &periodicBc);
  } // namespace internal

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::saveTriaInfoAndRhoNodalData()
  {
    d_basisOperationsPtrElectroHost->reinit(0,
                                            0,
                                            d_densityQuadratureIdElectro,
                                            false);
    dftfe::uInt nQuadsPerCell =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();
    std::vector<const distributedCPUVec<double> *> solutionVectors;



    //
    // compute nodal electron-density from quad data through l2 projection
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                            d_constraintsRhoNodal,
                            d_densityDofHandlerIndexElectro,
                            d_densityQuadratureIdElectro,
                            d_densityOutQuadValues[0],
                            rhoNodalField);

    distributedCPUVec<double> magNodalField;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        magNodalField.reinit(rhoNodalField);
        magNodalField = 0;
        l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                d_constraintsRhoNodal,
                                d_densityDofHandlerIndexElectro,
                                d_densityQuadratureIdElectro,
                                d_densityOutQuadValues[1],
                                magNodalField);
      }

    solutionVectors.push_back(&rhoNodalField);

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        solutionVectors.push_back(&magNodalField);
      }

    pcout << "Checkpointing tria info and rho data in progress..." << std::endl;

    d_mesh.saveTriangulationsSolutionVectors(
      d_dftParamsPtr->restartFolder,
      d_dftParamsPtr->finiteElementPolynomialOrderRhoNodal,
      1,
      solutionVectors,
      interpoolcomm,
      interBandGroupComm);

    pcout << "...checkpointing done." << std::endl;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::loadTriaInfoAndRhoNodalData()
  {
    pcout << "Reading tria info and rho data from checkpoint in progress..."
          << std::endl;
    // read rho data from checkpoint file

    std::vector<distributedCPUVec<double> *> solutionVectors;

    solutionVectors.push_back(&d_rhoInNodalValuesRead);

    if (d_dftParamsPtr->spinPolarized == 1 &&
        !d_dftParamsPtr->restartSpinFromNoSpin)
      {
        solutionVectors.push_back(&d_magInNodalValuesRead);
      }

    d_mesh.loadTriangulationsSolutionVectors(
      d_dftParamsPtr->restartFolder,
      d_dftParamsPtr->finiteElementPolynomialOrderRhoNodal,
      1,
      solutionVectors);

    pcout << "...Reading from checkpoint done." << std::endl;

    if (d_dftParamsPtr->spinPolarized == 1 &&
        d_dftParamsPtr->restartSpinFromNoSpin)
      {
        d_magInNodalValuesRead.reinit(d_rhoInNodalValuesRead);

        d_magInNodalValuesRead = 0;

        for (dftfe::uInt i = 0; i < d_rhoInNodalValuesRead.locally_owned_size();
             i++)
          {
            d_magInNodalValuesRead.local_element(i) =
              (d_dftParamsPtr->tot_magnetization) *
              d_rhoInNodalValuesRead.local_element(i);
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::writeDomainAndAtomCoordinates()
  {
    dftUtils::writeDataIntoFile(d_domainBoundingVectors,
                                "domainBoundingVectorsCurrent.chk",
                                d_mpiCommParent);

    std::vector<std::vector<double>> atomLocationsFractionalCurrent;
    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        atomLocationsFractionalCurrent        = atomLocationsFractional;
        const dftfe::Int    numberGlobalAtoms = atomLocations.size();
        std::vector<double> latticeVectorsFlattened(9, 0.0);
        std::vector<std::vector<double>> atomFractionalCoordinates;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
            latticeVectorsFlattened[3 * idim + jdim] =
              d_domainBoundingVectors[idim][jdim];
        dealii::Point<3> corner;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            corner[idim] = 0;
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              corner[idim] -= d_domainBoundingVectors[jdim][idim] / 2.0;
          }

        std::vector<bool> periodicBc(3, false);
        periodicBc[0] = d_dftParamsPtr->periodicX;
        periodicBc[1] = d_dftParamsPtr->periodicY;
        periodicBc[2] = d_dftParamsPtr->periodicZ;

        if (!d_dftParamsPtr->floatingNuclearCharges)
          {
            for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
              {
                dealii::Point<3> atomCoor;
                dftfe::Int       atomId = iAtom;
                atomCoor[0]             = d_atomLocationsAutoMesh[iAtom][0];
                atomCoor[1]             = d_atomLocationsAutoMesh[iAtom][1];
                atomCoor[2]             = d_atomLocationsAutoMesh[iAtom][2];

                std::vector<double> newFracCoord =
                  dftfe::internal::wrapAtomsAcrossPeriodicBc(
                    atomCoor, corner, latticeVectorsFlattened, periodicBc);
                // for synchrozination
                MPI_Bcast(
                  &(newFracCoord[0]), 3, MPI_DOUBLE, 0, d_mpiCommParent);

                atomLocationsFractional[iAtom][2] = newFracCoord[0];
                atomLocationsFractional[iAtom][3] = newFracCoord[1];
                atomLocationsFractional[iAtom][4] = newFracCoord[2];
              }
          }
        else
          {
            for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
              {
                dealii::Point<3> atomCoor;
                dftfe::Int       atomId = iAtom;
                atomCoor[0]             = atomLocations[iAtom][2];
                atomCoor[1]             = atomLocations[iAtom][3];
                atomCoor[2]             = atomLocations[iAtom][4];

                std::vector<double> newFracCoord =
                  internal::wrapAtomsAcrossPeriodicBc(atomCoor,
                                                      corner,
                                                      latticeVectorsFlattened,
                                                      periodicBc);
                // for synchrozination
                MPI_Bcast(
                  &(newFracCoord[0]), 3, MPI_DOUBLE, 0, d_mpiCommParent);

                atomLocationsFractionalCurrent[iAtom][2] = newFracCoord[0];
                atomLocationsFractionalCurrent[iAtom][3] = newFracCoord[1];
                atomLocationsFractionalCurrent[iAtom][4] = newFracCoord[2];
              }
          }
      }

    std::vector<std::vector<double>> atomLocationsAutoMesh = atomLocations;
    if (!d_dftParamsPtr->floatingNuclearCharges)
      for (dftfe::uInt iAtom = 0; iAtom < d_atomLocationsAutoMesh.size();
           iAtom++)
        {
          atomLocationsAutoMesh[iAtom][2] = d_atomLocationsAutoMesh[iAtom][0];
          atomLocationsAutoMesh[iAtom][3] = d_atomLocationsAutoMesh[iAtom][1];
          atomLocationsAutoMesh[iAtom][4] = d_atomLocationsAutoMesh[iAtom][2];
        }
#ifdef USE_COMPLEX
    if (!d_dftParamsPtr->floatingNuclearCharges)
      dftUtils::writeDataIntoFile(atomLocationsFractional,
                                  "atomsFracCoordAutomesh.chk",
                                  d_mpiCommParent);

    dftUtils::writeDataIntoFile(atomLocationsFractionalCurrent,
                                "atomsFracCoordCurrent.chk",
                                d_mpiCommParent);
#else
    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        if (!d_dftParamsPtr->floatingNuclearCharges)
          dftUtils::writeDataIntoFile(atomLocationsFractional,
                                      "atomsFracCoordAutomesh.chk",
                                      d_mpiCommParent);

        dftUtils::writeDataIntoFile(atomLocationsFractionalCurrent,
                                    "atomsFracCoordCurrent.chk",
                                    d_mpiCommParent);
      }
    else
      {
        if (!d_dftParamsPtr->floatingNuclearCharges)
          dftUtils::writeDataIntoFile(atomLocationsAutoMesh,
                                      "atomsCartCoordAutomesh.chk",
                                      d_mpiCommParent);

        dftUtils::writeDataIntoFile(atomLocations,
                                    "atomsCartCoordCurrent.chk",
                                    d_mpiCommParent);
      }
#endif

    if (!d_dftParamsPtr->floatingNuclearCharges)
      {
        if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
            d_dftParamsPtr->periodicZ)
          {
            atomLocationsFractional = atomLocationsFractionalCurrent;
          }

        //
        // write Gaussian atomic displacements
        //
        std::vector<std::vector<double>> atomsDisplacementsGaussian(
          d_atomLocationsAutoMesh.size(), std::vector<double>(3, 0.0));
        for (dftfe::Int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
          for (dftfe::Int j = 0; j < 3; ++j)
            atomsDisplacementsGaussian[i][j] =
              d_gaussianMovementAtomsNetDisplacements[i][j];

        dftUtils::writeDataIntoFile(atomsDisplacementsGaussian,
                                    "atomsGaussianDispCoord.chk",
                                    d_mpiCommParent);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::writeDomainAndAtomCoordinates(
    const std::string Path) const
  {
    dftUtils::writeDataIntoFile(d_domainBoundingVectors,
                                Path + "domainBoundingVectorsCurrent.chk",
                                d_mpiCommParent);

    std::vector<std::vector<double>> atomLocationsFractionalCurrent;
    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        atomLocationsFractionalCurrent        = atomLocationsFractional;
        const dftfe::Int    numberGlobalAtoms = atomLocations.size();
        std::vector<double> latticeVectorsFlattened(9, 0.0);
        std::vector<std::vector<double>> atomFractionalCoordinates;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
            latticeVectorsFlattened[3 * idim + jdim] =
              d_domainBoundingVectors[idim][jdim];
        dealii::Point<3> corner;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            corner[idim] = 0;
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              corner[idim] -= d_domainBoundingVectors[jdim][idim] / 2.0;
          }

        std::vector<bool> periodicBc(3, false);
        periodicBc[0] = d_dftParamsPtr->periodicX;
        periodicBc[1] = d_dftParamsPtr->periodicY;
        periodicBc[2] = d_dftParamsPtr->periodicZ;



        for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
          {
            dealii::Point<3> atomCoor;
            dftfe::Int       atomId = iAtom;
            atomCoor[0]             = atomLocations[iAtom][2];
            atomCoor[1]             = atomLocations[iAtom][3];
            atomCoor[2]             = atomLocations[iAtom][4];

            std::vector<double> newFracCoord =
              internal::wrapAtomsAcrossPeriodicBc(atomCoor,
                                                  corner,
                                                  latticeVectorsFlattened,
                                                  periodicBc);
            // for synchrozination
            MPI_Bcast(&(newFracCoord[0]), 3, MPI_DOUBLE, 0, d_mpiCommParent);

            atomLocationsFractionalCurrent[iAtom][2] = newFracCoord[0];
            atomLocationsFractionalCurrent[iAtom][3] = newFracCoord[1];
            atomLocationsFractionalCurrent[iAtom][4] = newFracCoord[2];
          }
      }


    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        dftUtils::writeDataIntoFile(atomLocationsFractionalCurrent,
                                    Path + "atomsFracCoordCurrent.chk",
                                    d_mpiCommParent);
      }
    else
      {
        dftUtils::writeDataIntoFile(atomLocations,
                                    Path + "atomsCartCoordCurrent.chk",
                                    d_mpiCommParent);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::writeStructureEnergyForcesDataPostProcess(
    const std::string Path) const
  {
    const dftfe::Int                 numberGlobalAtoms = atomLocations.size();
    std::vector<std::vector<double>> data(
      4 + numberGlobalAtoms + 2 +
        (d_dftParamsPtr->isIonForce ? numberGlobalAtoms : 0) +
        (d_dftParamsPtr->isCellStress ? 3 : 0),
      std::vector<double>(1, 0));

    data[0][0] = numberGlobalAtoms;
    data[1]    = getCell()[0];
    data[2]    = getCell()[1];
    data[3]    = getCell()[2];

    if (getParametersObject().periodicX || getParametersObject().periodicY ||
        getParametersObject().periodicZ)
      {
        std::vector<std::vector<double>> atomsFrac = getAtomLocationsFrac();
        for (dftfe::uInt i = 0; i < numberGlobalAtoms; ++i)
          {
            data[4 + i]    = std::vector<double>(4, 0);
            data[4 + i][0] = atomsFrac[i][0];
            data[4 + i][1] = atomsFrac[i][2];
            data[4 + i][2] = atomsFrac[i][3];
            data[4 + i][3] = atomsFrac[i][4];
          }
      }
    else
      {
        std::vector<std::vector<double>> atomsCart = getAtomLocationsCart();
        for (dftfe::uInt i = 0; i < numberGlobalAtoms; ++i)
          {
            data[4 + i]    = std::vector<double>(4, 0);
            data[4 + i][0] = atomsCart[i][0];
            data[4 + i][1] = atomsCart[i][2];
            data[4 + i][2] = atomsCart[i][3];
            data[4 + i][3] = atomsCart[i][4];
          }
      }

    data[4 + numberGlobalAtoms][0] = getFreeEnergy();
    data[5 + numberGlobalAtoms][0] = getInternalEnergy();
    if (d_dftParamsPtr->isIonForce)
      {
        for (dftfe::uInt i = 0; i < numberGlobalAtoms; ++i)
          {
            data[6 + numberGlobalAtoms + i]    = std::vector<double>(3, 0);
            data[6 + numberGlobalAtoms + i][0] = -getForceonAtoms()[3 * i];
            data[6 + numberGlobalAtoms + i][1] = -getForceonAtoms()[3 * i + 1];
            data[6 + numberGlobalAtoms + i][2] = -getForceonAtoms()[3 * i + 2];
          }
      }


    if (d_dftParamsPtr->isCellStress)
      {
        for (dftfe::uInt i = 0; i < 3; ++i)
          {
            data[6 + 2 * numberGlobalAtoms + i] = std::vector<double>(3, 0);
            for (dftfe::uInt j = 0; j < 3; ++j)
              data[6 + 2 * numberGlobalAtoms + i][j] = -getCellStress()[i][j];
          }
      }


    dftUtils::writeDataIntoFile(data, Path, d_mpiCommParent);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::loadQuadratureData(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt quadratureId,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                      &quadratureValueData,
    const dftfe::uInt  fieldDimension,
    const std::string &fieldName,
    const std::string &folderPath,
    const MPI_Comm    &mpi_comm_parent,
    const MPI_Comm    &mpi_comm_domain,
    const MPI_Comm    &interpoolcomm,
    const MPI_Comm    &interBandGroupComm)
  {
    pcout << "Reading Quad data from checkpoint in progress..." << std::endl;
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const dftfe::uInt nCells        = basisOperationsPtr->nCells();
    const dftfe::uInt totalTarget   = nCells * nQuadsPerCell;
    const dealii::DoFHandler<3> &dofHandlerTemp =
      basisOperationsPtr->getDofHandler();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
               &quadPoints = basisOperationsPtr->quadPoints();
    std::string masterFileName =
      folderPath + "/MasterFile_" + fieldName + "_.chk";

    std::vector<double>      centroidX, centroidY, centroidZ;
    std::vector<dftfe::uInt> startIndex;
    std::vector<std::string> fileNames;

    std::ifstream inMasterFile(masterFileName);
    if (inMasterFile.is_open())
      {
        std::string line;
        while (std::getline(inMasterFile, line))
          {
            std::istringstream iss(line);
            dftfe::uInt        start;
            double             x, y, z;
            std::string        fileName;
            iss >> x >> y >> z >> fileName >> start;
            centroidX.push_back(x);
            centroidY.push_back(y);
            centroidZ.push_back(z);
            fileNames.push_back(fileName);
            startIndex.push_back(start);
          }
        inMasterFile.close();
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage("DFT-FE Error: Master file not found"));
      }
    dftfe::uInt              count = 0;
    std::vector<dftfe::uInt> countPerThread(d_nOMPThreads,
                                            0); // for each thread
    if (nCells > 0)
      {
        typename dealii::DoFHandler<3>::active_cell_iterator cell =
          basisOperationsPtr->getCellIterator(0);

        // search for fileName and startLocation
        std::string                      fileName;
        dftfe::uInt                      startLocation = 0;
        std::vector<std::vector<double>> dataInput;
        if (cell->is_locally_owned())
          {
            for (dftfe::uInt index = 0; index < startIndex.size(); ++index)
              {
                if (std::fabs(cell->center()[0] - centroidX[index]) < 1e-6 &&
                    std::fabs(cell->center()[1] - centroidY[index]) < 1e-6 &&
                    std::fabs(cell->center()[2] - centroidZ[index]) < 1e-6)
                  {
                    fileName      = folderPath + "/" + fileNames[index];
                    startLocation = startIndex[index];
                    break;
                  }
              }

            dftUtils::readFile(dataInput, fileName);
            for (dftfe::uInt q = 0; q < nQuadsPerCell; ++q)
              {
                for (dftfe::Int iField = 0; iField < fieldDimension; ++iField)
                  {
                    quadratureValueData[q * fieldDimension + iField] =
                      dataInput[startLocation + q][3 + iField];
                  }
                count++;
              }
          }


        std::string fileNameOld = fileName;
        dftfe::uInt iCell       = 1;

#pragma omp parallel for num_threads(d_nOMPThreads) \
  firstprivate(fileNameOld, fileName, startLocation, dataInput, cell)
        for (iCell = 1; iCell < nCells; ++iCell)
          {
            cell = basisOperationsPtr->getCellIterator(iCell);

            if (cell->is_locally_owned())
              {
                for (dftfe::uInt index = 0; index < startIndex.size(); ++index)
                  {
                    if (std::fabs(cell->center()[0] - centroidX[index]) <
                          1e-6 &&
                        std::fabs(cell->center()[1] - centroidY[index]) <
                          1e-6 &&
                        std::fabs(cell->center()[2] - centroidZ[index]) < 1e-6)
                      {
                        fileName      = folderPath + "/" + fileNames[index];
                        startLocation = startIndex[index];
                        break;
                      }
                  }
                if (fileName != fileNameOld)
                  {
                    dataInput.clear();
                    dataInput.resize(0);
                    dftUtils::readFile(dataInput, fileName);
                    fileNameOld = fileName;
                  }
                for (dftfe::uInt q = 0; q < nQuadsPerCell; ++q)
                  {
                    for (dftfe::Int iField = 0; iField < fieldDimension;
                         ++iField)
                      {
                        quadratureValueData[iCell * nQuadsPerCell *
                                              fieldDimension +
                                            q * fieldDimension + iField] =
                          dataInput[startLocation + q][3 + iField];
                      }
                    countPerThread[omp_get_thread_num()]++;
                  }
              }
          } // iCell
      }
    for (dftfe::Int i = 0; i < d_nOMPThreads; ++i)
      {
        count += countPerThread[i];
      }
    if (count < totalTarget)
      {
        AssertThrow(false,
                    dealii::ExcMessage(std::string(
                      "All quadrature data not filled. Check restart files!")));
      }
    pcout << "Reading Quad data done..." << std::endl;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::saveQuadratureData(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                     &basisOperationsPtr,
    const dftfe::uInt quadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                      &quadratureValueData,
    const dftfe::uInt  fieldDimension,
    const std::string &fieldName,
    const std::string &folderPath,
    const MPI_Comm    &mpi_comm_parent,
    const MPI_Comm    &mpi_comm_domain,
    const MPI_Comm    &interpoolcomm,
    const MPI_Comm    &interBandGroupComm)
  {
    pcout << "Saving Quad data in progress..." << std::endl;
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const dftfe::uInt nCells        = basisOperationsPtr->nCells();
    const dealii::DoFHandler<3> &dofHandlerTemp =
      basisOperationsPtr->getDofHandler();
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPoints = basisOperationsPtr->quadPoints();
    if (dealii::Utilities::MPI::this_mpi_process(interpoolcomm) == 0 &&
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm) == 0)
      {
        const dftfe::uInt this_process =
          dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain);
        const dftfe::uInt n_mpi_processes =
          dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain);
        std::vector<dftfe::uInt> nCellsPerTask(n_mpi_processes, 0);
        nCellsPerTask[this_process] = nCells;
        MPI_Allreduce(MPI_IN_PLACE,
                      &nCellsPerTask[0],
                      n_mpi_processes,
                      dftfe::dataTypes::mpi_type_id(nCellsPerTask.data()),
                      MPI_SUM,
                      mpi_comm_domain);
        std::vector<std::vector<double>> quadratureData(
          nCellsPerTask[this_process] * nQuadsPerCell,
          std::vector<double>(fieldDimension + 3, 0.0));
        std::vector<double> centroidLocations(3 * nCellsPerTask[this_process],
                                              0.0);
        std::vector<dftfe::uInt> startLocations(nCellsPerTask[this_process], 0);
        // Try openmp parallelization Here
        dftfe::uInt iCell = 0;
#pragma omp parallel for num_threads(d_nOMPThreads)
        for (iCell = 0; iCell < nCells; ++iCell)
          {
            typename dealii::DoFHandler<3>::active_cell_iterator cell =
              basisOperationsPtr->getCellIterator(iCell);
            if (cell->is_locally_owned())
              {
                for (dftfe::uInt q = 0; q < nQuadsPerCell; ++q)
                  {
                    quadratureData[iCell * nQuadsPerCell + q][0] =
                      quadPoints[3 * iCell * nQuadsPerCell + 3 * q + 0];
                    quadratureData[iCell * nQuadsPerCell + q][1] =
                      quadPoints[3 * iCell * nQuadsPerCell + 3 * q + 1];
                    quadratureData[iCell * nQuadsPerCell + q][2] =
                      quadPoints[3 * iCell * nQuadsPerCell + 3 * q + 2];
                    for (dftfe::uInt i = 0; i < fieldDimension; ++i)
                      {
                        quadratureData[iCell * nQuadsPerCell + q][3 + i] =
                          quadratureValueData[iCell * nQuadsPerCell *
                                                fieldDimension +
                                              q * fieldDimension + i];
                      }
                  } // QuadPoint Loop
                centroidLocations[3 * iCell + 0] = cell->center()[0];
                centroidLocations[3 * iCell + 1] = cell->center()[1];
                centroidLocations[3 * iCell + 2] = cell->center()[2];
                startLocations[iCell]            = iCell * nQuadsPerCell;
              }

          } // iCell

        // Save QuadPointData to File
        std::string quadFileName = folderPath + "/MPITask_" +
                                   std::to_string(this_process) + "_" +
                                   fieldName + "_quadPoints.chk";
        dftUtils::writeDataIntoFile(quadratureData, quadFileName);
        dftfe::uInt startLocation = 0;
        dftfe::uInt totalSize     = 0;
        for (dftfe::uInt i = 0; i < n_mpi_processes; ++i)
          {
            totalSize += nCellsPerTask[i];
            if (i < this_process)
              startLocation += nCellsPerTask[i];
          }
        std::vector<double>      centroidData(3 * totalSize, 0.0);
        std::vector<dftfe::uInt> startLocationsData(totalSize, 0);
        for (dftfe::uInt i = 0; i < nCellsPerTask[this_process]; ++i)
          {
            centroidData[3 * (startLocation + i) + 0] =
              centroidLocations[3 * i + 0];
            centroidData[3 * (startLocation + i) + 1] =
              centroidLocations[3 * i + 1];
            centroidData[3 * (startLocation + i) + 2] =
              centroidLocations[3 * i + 2];
            startLocationsData[startLocation + i] = startLocations[i];
          }
        MPI_Allreduce(MPI_IN_PLACE,
                      &centroidData[0],
                      3 * totalSize,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_comm_domain);
        MPI_Allreduce(MPI_IN_PLACE,
                      &startLocationsData[0],
                      totalSize,
                      dftfe::dataTypes::mpi_type_id(startLocationsData.data()),
                      MPI_SUM,
                      mpi_comm_domain);
        std::string masterFileName =
          folderPath + "/MasterFile_" + fieldName + "_.chk";
        if (this_process == 0)
          {
            if (std::ifstream(masterFileName))
              dftfe::dftUtils::moveFile(masterFileName,
                                        masterFileName + ".old");
          }
        dftfe::uInt   index = 0;
        std::ofstream outFile(masterFileName);
        if (outFile.is_open())
          {
            for (dftfe::uInt i = 0; i < n_mpi_processes; ++i)
              {
                const dftfe::uInt totalCells = nCellsPerTask[i];
                const std::string tempFile   = "MPITask_" + std::to_string(i) +
                                             "_" + fieldName +
                                             "_quadPoints.chk";
                for (dftfe::uInt j = 0; j < totalCells; j++)
                  {
                    outFile << std::setprecision(
                                 std::numeric_limits<double>::max_digits10)
                            << centroidData[3 * index + 0] << " "
                            << centroidData[3 * index + 1] << " "
                            << centroidData[3 * index + 2] << " " << tempFile
                            << " " << startLocationsData[index] << std::endl;
                    index++;
                  }
              }
          }
        outFile.close();

      } // Pool ==0 and bandGroup == 0
    pcout << "Saving Quad data completed..." << std::endl;
  }

#include "dft.inst.cc"
} // namespace dftfe
