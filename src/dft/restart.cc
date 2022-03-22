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
// @author Sambit Das
//

// source file for restart functionality in dftClass

//
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::saveTriaInfoAndRhoData()
{
  pcout << "Checkpointing tria info and rho data in progress..." << std::endl;
  std::vector<const std::map<dealii::CellId, std::vector<double>> *>
    cellQuadDataContainerIn;


  for (auto it = rhoInVals.cbegin(); it != rhoInVals.cend(); it++)
    cellQuadDataContainerIn.push_back(&(*it));

  for (auto it = rhoOutVals.cbegin(); it != rhoOutVals.cend(); it++)
    cellQuadDataContainerIn.push_back(&(*it));

  if (dftParameters::xcFamilyType == "GGA")
    {
      for (auto it = gradRhoInVals.cbegin(); it != gradRhoInVals.cend(); it++)
        cellQuadDataContainerIn.push_back(&(*it));

      for (auto it = gradRhoOutVals.cbegin(); it != gradRhoOutVals.cend(); it++)
        cellQuadDataContainerIn.push_back(&(*it));
    }

  if (dftParameters::spinPolarized == 1)
    {
      for (auto it = rhoInValsSpinPolarized.cbegin();
           it != rhoInValsSpinPolarized.cend();
           it++)
        cellQuadDataContainerIn.push_back(&(*it));

      for (auto it = rhoOutValsSpinPolarized.cbegin();
           it != rhoOutValsSpinPolarized.cend();
           it++)
        cellQuadDataContainerIn.push_back(&(*it));
    }

  if (dftParameters::xcFamilyType == "GGA" && dftParameters::spinPolarized == 1)
    {
      for (auto it = gradRhoInValsSpinPolarized.cbegin();
           it != gradRhoInValsSpinPolarized.cend();
           it++)
        cellQuadDataContainerIn.push_back(&(*it));

      for (auto it = gradRhoOutValsSpinPolarized.cbegin();
           it != gradRhoOutValsSpinPolarized.cend();
           it++)
        cellQuadDataContainerIn.push_back(&(*it));
    }

  d_mesh.saveTriangulationsCellQuadData(cellQuadDataContainerIn,
                                        interpoolcomm,
                                        interBandGroupComm);

  // write size of current mixing history into an additional .txt file
  const std::string extraInfoFileName = "rhoDataExtraInfo.chk";
  if (std::ifstream(extraInfoFileName) &&
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    dftUtils::moveFile(extraInfoFileName, extraInfoFileName + ".old");
  std::ofstream extraInfoFile(extraInfoFileName);
  if (extraInfoFile.is_open())
    {
      extraInfoFile << rhoInVals.size();
      extraInfoFile.close();
    }

  pcout << "...checkpointing done." << std::endl;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::saveTriaInfoAndRhoNodalData()
{
  std::vector<const distributedCPUVec<double> *> solutionVectors;

  dealii::IndexSet locally_relevant_dofs_;
  dealii::DoFTools::extract_locally_relevant_dofs(d_dofHandlerRhoNodal,
                                                  locally_relevant_dofs_);

  const dealii::IndexSet &locally_owned_dofs_ =
    d_dofHandlerRhoNodal.locally_owned_dofs();
  dealii::IndexSet ghost_indices_ = locally_relevant_dofs_;
  ghost_indices_.subtract_set(locally_owned_dofs_);

  distributedCPUVec<double> tempVec =
    distributedCPUVec<double>(locally_owned_dofs_,
                              ghost_indices_,
                              mpi_communicator);

  distributedCPUVec<double> tempVecSpin0, tempVecSpin1;

  //
  // compute nodal electron-density from quad data through l2 projection
  //
  distributedCPUVec<double> rhoNodalField;
  d_matrixFreeDataPRefined.initialize_dof_vector(
    rhoNodalField, d_densityDofHandlerIndexElectro);
  rhoNodalField = 0;
  std::function<
    double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
           const unsigned int                                          q)>
    funcRho =
      [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
          const unsigned int                                          q) {
        return (*rhoOutValues).find(cell->id())->second[q];
      };
  dealii::VectorTools::project<3, distributedCPUVec<double>>(
    dealii::MappingQ1<3, 3>(),
    d_dofHandlerRhoNodal,
    d_constraintsRhoNodal,
    d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
    funcRho,
    rhoNodalField);
  rhoNodalField.update_ghost_values();

  distributedCPUVec<double> rhoNodalFieldSpin0;
  distributedCPUVec<double> rhoNodalFieldSpin1;
  if (dftParameters::spinPolarized == 1)
    {
      rhoNodalFieldSpin0.reinit(rhoNodalField);
      rhoNodalFieldSpin0 = 0;
      std::function<
        double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
               const unsigned int                                          q)>
        funcRhoSpin0 =
          [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
              const unsigned int                                          q) {
            return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2 * q];
          };
      dealii::VectorTools::project<3, distributedCPUVec<double>>(
        dealii::MappingQ1<3, 3>(),
        d_dofHandlerRhoNodal,
        d_constraintsRhoNodal,
        d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
        funcRhoSpin0,
        rhoNodalFieldSpin0);
      rhoNodalFieldSpin0.update_ghost_values();


      rhoNodalFieldSpin1.reinit(rhoNodalField);
      rhoNodalFieldSpin1 = 0;
      std::function<
        double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
               const unsigned int                                          q)>
        funcRhoSpin1 =
          [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
              const unsigned int                                          q) {
            return (*rhoOutValuesSpinPolarized)
              .find(cell->id())
              ->second[2 * q + 1];
          };
      dealii::VectorTools::project<3, distributedCPUVec<double>>(
        dealii::MappingQ1<3, 3>(),
        d_dofHandlerRhoNodal,
        d_constraintsRhoNodal,
        d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
        funcRhoSpin1,
        rhoNodalFieldSpin1);
      rhoNodalFieldSpin1.update_ghost_values();
    }

  tempVec = 0.0;
  for (unsigned int i = 0; i < rhoNodalField.local_size(); i++)
    tempVec.local_element(i) = rhoNodalField.local_element(i);

  tempVec.update_ghost_values();
  solutionVectors.push_back(&tempVec);

  if (dftParameters::spinPolarized == 1)
    {
      tempVecSpin0.reinit(tempVec);
      tempVecSpin1.reinit(tempVec);

      tempVecSpin0 = 0.0;
      tempVecSpin1 = 0.0;

      for (unsigned int i = 0; i < rhoNodalFieldSpin0.local_size(); i++)
        {
          tempVecSpin0.local_element(i) = rhoNodalFieldSpin0.local_element(i);
          tempVecSpin1.local_element(i) = rhoNodalFieldSpin1.local_element(i);
        }

      tempVecSpin0.update_ghost_values();
      tempVecSpin1.update_ghost_values();

      solutionVectors.push_back(&tempVecSpin0);
      solutionVectors.push_back(&tempVecSpin1);
    }

  pcout << "Checkpointing tria info and rho data in progress..." << std::endl;

  d_mesh.saveTriangulationsSolutionVectors(
    C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
    1,
    solutionVectors,
    interpoolcomm,
    interBandGroupComm);

  pcout << "...checkpointing done." << std::endl;
}

//
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::loadTriaInfoAndRhoData()
{
  pcout << "Reading tria info and rho data from checkpoint in progress..."
        << std::endl;
  // read mixing history size of the rhoData to be read in the next step
  unsigned int      mixingHistorySize;
  const std::string extraInfoFileName = "rhoDataExtraInfo.chk";
  dftUtils::verifyCheckpointFileExists(extraInfoFileName);
  std::ifstream extraInfoFile(extraInfoFileName);
  if (extraInfoFile.is_open())
    {
      extraInfoFile >> mixingHistorySize;
      extraInfoFile.close();
    }
  else
    AssertThrow(false, ExcMessage("Unable to find rhoDataExtraInfo.txt"));

  Assert(mixingHistorySize > 1, ExcInternalError());

  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  const unsigned int num_quad_points = quadrature.size();

  // Fill input data for the load function call
  std::vector<unsigned int> cellDataSizeContainer;
  std::vector<std::map<dealii::CellId, std::vector<double>>>
    cellQuadDataContainerOut;

  for (unsigned int i = 0; i < mixingHistorySize; i++)
    {
      cellDataSizeContainer.push_back(num_quad_points);
      cellQuadDataContainerOut.push_back(
        std::map<dealii::CellId, std::vector<double>>());
    }

  for (unsigned int i = 0; i < mixingHistorySize; i++)
    {
      cellDataSizeContainer.push_back(num_quad_points);
      cellQuadDataContainerOut.push_back(
        std::map<dealii::CellId, std::vector<double>>());
    }

  if (dftParameters::xcFamilyType == "GGA")
    {
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          cellDataSizeContainer.push_back(3 * num_quad_points);
          cellQuadDataContainerOut.push_back(
            std::map<dealii::CellId, std::vector<double>>());
        }
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          cellDataSizeContainer.push_back(3 * num_quad_points);
          cellQuadDataContainerOut.push_back(
            std::map<dealii::CellId, std::vector<double>>());
        }
    }

  if (dftParameters::spinPolarized == 1)
    {
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          cellDataSizeContainer.push_back(2 * num_quad_points);
          cellQuadDataContainerOut.push_back(
            std::map<dealii::CellId, std::vector<double>>());
        }
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          cellDataSizeContainer.push_back(2 * num_quad_points);
          cellQuadDataContainerOut.push_back(
            std::map<dealii::CellId, std::vector<double>>());
        }
    }

  if (dftParameters::xcFamilyType == "GGA" && dftParameters::spinPolarized == 1)
    {
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          cellDataSizeContainer.push_back(6 * num_quad_points);
          cellQuadDataContainerOut.push_back(
            std::map<dealii::CellId, std::vector<double>>());
        }
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          cellDataSizeContainer.push_back(6 * num_quad_points);
          cellQuadDataContainerOut.push_back(
            std::map<dealii::CellId, std::vector<double>>());
        }
    }

  // read rho data from checkpoint file
  d_mesh.loadTriangulationsCellQuadData(cellQuadDataContainerOut,
                                        cellDataSizeContainer);

  // Fill appropriate data structure using the read rho data
  clearRhoData();
  unsigned int count = 0;
  for (unsigned int i = 0; i < mixingHistorySize; i++)
    {
      rhoInVals.push_back(cellQuadDataContainerOut[count]);
      count++;
    }
  rhoInValues = &(rhoInVals.back());
  for (unsigned int i = 0; i < mixingHistorySize; i++)
    {
      rhoOutVals.push_back(cellQuadDataContainerOut[count]);
      count++;
    }
  rhoOutValues = &(rhoOutVals.back());

  if (dftParameters::xcFamilyType == "GGA")
    {
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          gradRhoInVals.push_back(cellQuadDataContainerOut[count]);
          count++;
        }
      gradRhoInValues = &(gradRhoInVals.back());
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          gradRhoOutVals.push_back(cellQuadDataContainerOut[count]);
          count++;
        }
      gradRhoOutValues = &(gradRhoOutVals.back());
    }

  if (dftParameters::spinPolarized == 1)
    {
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          rhoInValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
          count++;
        }
      rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          rhoOutValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
          count++;
        }
      rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());
    }

  if (dftParameters::xcFamilyType == "GGA" && dftParameters::spinPolarized == 1)
    {
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          gradRhoInValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
          count++;
        }
      gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
      for (unsigned int i = 0; i < mixingHistorySize; i++)
        {
          gradRhoOutValsSpinPolarized.push_back(
            cellQuadDataContainerOut[count]);
          count++;
        }
      gradRhoOutValuesSpinPolarized = &(gradRhoOutValsSpinPolarized.back());
    }
  pcout << "...Reading from checkpoint done." << std::endl;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::loadTriaInfoAndRhoNodalData()
{
  pcout << "Reading tria info and rho data from checkpoint in progress..."
        << std::endl;
  // read rho data from checkpoint file

  std::vector<distributedCPUVec<double> *> solutionVectors;

  solutionVectors.push_back(&d_rhoInNodalValuesRead);

  if (dftParameters::spinPolarized == 1 &&
      !dftParameters::restartSpinFromNoSpin)
    {
      solutionVectors.push_back(&d_rhoInSpin0NodalValuesRead);
      solutionVectors.push_back(&d_rhoInSpin1NodalValuesRead);
    }

  d_mesh.loadTriangulationsSolutionVectors(
    C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(), 1, solutionVectors);

  pcout << "...Reading from checkpoint done." << std::endl;

  if (dftParameters::spinPolarized == 1 && dftParameters::restartSpinFromNoSpin)
    {
      d_rhoInSpin0NodalValuesRead.reinit(d_rhoInNodalValuesRead);
      d_rhoInSpin1NodalValuesRead.reinit(d_rhoInNodalValuesRead);

      d_rhoInSpin0NodalValuesRead = 0;
      d_rhoInSpin1NodalValuesRead = 0;

      for (unsigned int i = 0; i < d_rhoInNodalValuesRead.local_size(); i++)
        {
          d_rhoInSpin0NodalValuesRead.local_element(i) =
            (0.5 - dftParameters::start_magnetization) *
            d_rhoInNodalValuesRead.local_element(i);
          d_rhoInSpin1NodalValuesRead.local_element(i) =
            (0.5 + dftParameters::start_magnetization) *
            d_rhoInNodalValuesRead.local_element(i);
        }
    }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::writeDomainAndAtomCoordinates()
{
  dftUtils::writeDataIntoFile(d_domainBoundingVectors,
                              "domainBoundingVectorsCurrent.chk");

  std::vector<std::vector<double>> atomLocationsFractionalCurrent;
  if (dftParameters::periodicX || dftParameters::periodicY ||
      dftParameters::periodicZ)
    {
      atomLocationsFractionalCurrent = atomLocationsFractional;
      const int                        numberGlobalAtoms = atomLocations.size();
      std::vector<double>              latticeVectorsFlattened(9, 0.0);
      std::vector<std::vector<double>> atomFractionalCoordinates;
      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          latticeVectorsFlattened[3 * idim + jdim] =
            d_domainBoundingVectors[idim][jdim];
      Point<3> corner;
      for (unsigned int idim = 0; idim < 3; idim++)
        {
          corner[idim] = 0;
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            corner[idim] -= d_domainBoundingVectors[jdim][idim] / 2.0;
        }

      std::vector<bool> periodicBc(3, false);
      periodicBc[0] = dftParameters::periodicX;
      periodicBc[1] = dftParameters::periodicY;
      periodicBc[2] = dftParameters::periodicZ;

      if (!dftParameters::floatingNuclearCharges)
        {
          for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
            {
              Point<3> atomCoor;
              int      atomId = iAtom;
              atomCoor[0]     = d_atomLocationsAutoMesh[iAtom][0];
              atomCoor[1]     = d_atomLocationsAutoMesh[iAtom][1];
              atomCoor[2]     = d_atomLocationsAutoMesh[iAtom][2];

              std::vector<double> newFracCoord =
                internal::wrapAtomsAcrossPeriodicBc(atomCoor,
                                                    corner,
                                                    latticeVectorsFlattened,
                                                    periodicBc);
              // for synchrozination
              MPI_Bcast(&(newFracCoord[0]), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

              atomLocationsFractional[iAtom][2] = newFracCoord[0];
              atomLocationsFractional[iAtom][3] = newFracCoord[1];
              atomLocationsFractional[iAtom][4] = newFracCoord[2];
            }
        }
      else
        {
          for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
            {
              Point<3> atomCoor;
              int      atomId = iAtom;
              atomCoor[0]     = atomLocations[iAtom][2];
              atomCoor[1]     = atomLocations[iAtom][3];
              atomCoor[2]     = atomLocations[iAtom][4];

              std::vector<double> newFracCoord =
                internal::wrapAtomsAcrossPeriodicBc(atomCoor,
                                                    corner,
                                                    latticeVectorsFlattened,
                                                    periodicBc);
              // for synchrozination
              MPI_Bcast(&(newFracCoord[0]), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

              atomLocationsFractionalCurrent[iAtom][2] = newFracCoord[0];
              atomLocationsFractionalCurrent[iAtom][3] = newFracCoord[1];
              atomLocationsFractionalCurrent[iAtom][4] = newFracCoord[2];
            }
        }
    }

  std::vector<std::vector<double>> atomLocationsAutoMesh = atomLocations;
  if (!dftParameters::floatingNuclearCharges)
    for (unsigned int iAtom = 0; iAtom < d_atomLocationsAutoMesh.size();
         iAtom++)
      {
        atomLocationsAutoMesh[iAtom][2] = d_atomLocationsAutoMesh[iAtom][0];
        atomLocationsAutoMesh[iAtom][3] = d_atomLocationsAutoMesh[iAtom][1];
        atomLocationsAutoMesh[iAtom][4] = d_atomLocationsAutoMesh[iAtom][2];
      }
#ifdef USE_COMPLEX
  if (!dftParameters::floatingNuclearCharges)
    dftUtils::writeDataIntoFile(atomLocationsFractional,
                                "atomsFracCoordAutomesh.chk");

  dftUtils::writeDataIntoFile(atomLocationsFractionalCurrent,
                              "atomsFracCoordCurrent.chk");
#else
  if (dftParameters::periodicX || dftParameters::periodicY ||
      dftParameters::periodicZ)
    {
      if (!dftParameters::floatingNuclearCharges)
        dftUtils::writeDataIntoFile(atomLocationsFractional,
                                    "atomsFracCoordAutomesh.chk");

      dftUtils::writeDataIntoFile(atomLocationsFractionalCurrent,
                                  "atomsFracCoordCurrent.chk");
    }
  else
    {
      if (!dftParameters::floatingNuclearCharges)
        dftUtils::writeDataIntoFile(atomLocationsAutoMesh,
                                    "atomsCartCoordAutomesh.chk");

      dftUtils::writeDataIntoFile(atomLocations, "atomsCartCoordCurrent.chk");
    }
#endif

  if (!dftParameters::floatingNuclearCharges)
    {
      if (dftParameters::periodicX || dftParameters::periodicY ||
          dftParameters::periodicZ)
        {
          atomLocationsFractional = atomLocationsFractionalCurrent;
        }

      //
      // write Gaussian atomic displacements
      //
      std::vector<std::vector<double>> atomsDisplacementsGaussian(
        d_atomLocationsAutoMesh.size(), std::vector<double>(3, 0.0));
      for (int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
        for (int j = 0; j < 3; ++j)
          atomsDisplacementsGaussian[i][j] =
            d_gaussianMovementAtomsNetDisplacements[i][j];

      dftUtils::writeDataIntoFile(atomsDisplacementsGaussian,
                                  "atomsGaussianDispCoord.chk");
    }
}
