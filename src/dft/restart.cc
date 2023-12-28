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
#include <dft.h>
#include <fileReaders.h>
#include <dftUtils.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  namespace internal
  {
    std::vector<double>
    getFractionalCoordinates(const std::vector<double> &latticeVectors,
                             const dealii::Point<3> &   point,
                             const dealii::Point<3> &   corner);
    std::vector<double>
    wrapAtomsAcrossPeriodicBc(const dealii::Point<3> &   cellCenteredCoord,
                              const dealii::Point<3> &   corner,
                              const std::vector<double> &latticeVectors,
                              const std::vector<bool> &  periodicBc);
  } // namespace internal

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::saveTriaInfoAndRhoNodalData()
  {
    basisOperationsPtrElectroHost->reinit(0,
                                          0,
                                          d_densityQuadratureIdElectro,
                                          false);
    unsigned int nQuadsPerCell = basisOperationsPtrElectroHost->nQuadsPerCell();
    std::vector<const distributedCPUVec<double> *> solutionVectors;



    //
    // compute nodal electron-density from quad data through l2 projection
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    l2ProjectionQuadToNodal(basisOperationsPtrElectroHost,
                            d_constraintsRhoNodal,
                            d_densityDofHandlerIndexElectro,
                            d_densityQuadratureIdElectro,
                            d_densityOutQuadValues[0],
                            rhoNodalField);
    rhoNodalField.update_ghost_values();

    distributedCPUVec<double> magNodalField;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        magNodalField.reinit(rhoNodalField);
        magNodalField = 0;
        l2ProjectionQuadToNodal(basisOperationsPtrElectroHost,
                                d_constraintsRhoNodal,
                                d_densityDofHandlerIndexElectro,
                                d_densityQuadratureIdElectro,
                                d_densityOutQuadValues[1],
                                magNodalField);
        magNodalField.update_ghost_values();
      }

    solutionVectors.push_back(&rhoNodalField);

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        solutionVectors.push_back(&magNodalField);
      }

    pcout << "Checkpointing tria info and rho data in progress..." << std::endl;

    d_mesh.saveTriangulationsSolutionVectors(
      d_dftParamsPtr->restartFolder,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      1,
      solutionVectors,
      interpoolcomm,
      interBandGroupComm);

    pcout << "...checkpointing done." << std::endl;
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

    if (d_dftParamsPtr->spinPolarized == 1 &&
        !d_dftParamsPtr->restartSpinFromNoSpin)
      {
        solutionVectors.push_back(&d_magInNodalValuesRead);
      }

    d_mesh.loadTriangulationsSolutionVectors(
      d_dftParamsPtr->restartFolder,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      1,
      solutionVectors);

    pcout << "...Reading from checkpoint done." << std::endl;

    if (d_dftParamsPtr->spinPolarized == 1 &&
        d_dftParamsPtr->restartSpinFromNoSpin)
      {
        d_magInNodalValuesRead.reinit(d_rhoInNodalValuesRead);

        d_magInNodalValuesRead = 0;

        for (unsigned int i = 0; i < d_rhoInNodalValuesRead.local_size(); i++)
          {
            d_magInNodalValuesRead.local_element(i) =
              -2.0 * (d_dftParamsPtr->start_magnetization) *
              d_rhoInNodalValuesRead.local_element(i);
          }
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::writeDomainAndAtomCoordinates()
  {
    dftUtils::writeDataIntoFile(d_domainBoundingVectors,
                                "domainBoundingVectorsCurrent.chk",
                                d_mpiCommParent);

    std::vector<std::vector<double>> atomLocationsFractionalCurrent;
    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        atomLocationsFractionalCurrent        = atomLocationsFractional;
        const int           numberGlobalAtoms = atomLocations.size();
        std::vector<double> latticeVectorsFlattened(9, 0.0);
        std::vector<std::vector<double>> atomFractionalCoordinates;
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            latticeVectorsFlattened[3 * idim + jdim] =
              d_domainBoundingVectors[idim][jdim];
        dealii::Point<3> corner;
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            corner[idim] = 0;
            for (unsigned int jdim = 0; jdim < 3; jdim++)
              corner[idim] -= d_domainBoundingVectors[jdim][idim] / 2.0;
          }

        std::vector<bool> periodicBc(3, false);
        periodicBc[0] = d_dftParamsPtr->periodicX;
        periodicBc[1] = d_dftParamsPtr->periodicY;
        periodicBc[2] = d_dftParamsPtr->periodicZ;

        if (!d_dftParamsPtr->floatingNuclearCharges)
          {
            for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
              {
                dealii::Point<3> atomCoor;
                int              atomId = iAtom;
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
            for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
              {
                dealii::Point<3> atomCoor;
                int              atomId = iAtom;
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
      for (unsigned int iAtom = 0; iAtom < d_atomLocationsAutoMesh.size();
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
        for (int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
          for (int j = 0; j < 3; ++j)
            atomsDisplacementsGaussian[i][j] =
              d_gaussianMovementAtomsNetDisplacements[i][j];

        dftUtils::writeDataIntoFile(atomsDisplacementsGaussian,
                                    "atomsGaussianDispCoord.chk",
                                    d_mpiCommParent);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::writeDomainAndAtomCoordinates(
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
        const int           numberGlobalAtoms = atomLocations.size();
        std::vector<double> latticeVectorsFlattened(9, 0.0);
        std::vector<std::vector<double>> atomFractionalCoordinates;
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            latticeVectorsFlattened[3 * idim + jdim] =
              d_domainBoundingVectors[idim][jdim];
        dealii::Point<3> corner;
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            corner[idim] = 0;
            for (unsigned int jdim = 0; jdim < 3; jdim++)
              corner[idim] -= d_domainBoundingVectors[jdim][idim] / 2.0;
          }

        std::vector<bool> periodicBc(3, false);
        periodicBc[0] = d_dftParamsPtr->periodicX;
        periodicBc[1] = d_dftParamsPtr->periodicY;
        periodicBc[2] = d_dftParamsPtr->periodicZ;



        for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
          {
            dealii::Point<3> atomCoor;
            int              atomId = iAtom;
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

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::writeStructureEnergyForcesDataPostProcess(
    const std::string Path) const
  {
    const int                        numberGlobalAtoms = atomLocations.size();
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
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
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
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
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
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            data[6 + numberGlobalAtoms + i]    = std::vector<double>(3, 0);
            data[6 + numberGlobalAtoms + i][0] = -getForceonAtoms()[3 * i];
            data[6 + numberGlobalAtoms + i][1] = -getForceonAtoms()[3 * i + 1];
            data[6 + numberGlobalAtoms + i][2] = -getForceonAtoms()[3 * i + 2];
          }
      }


    if (d_dftParamsPtr->isCellStress)
      {
        for (unsigned int i = 0; i < 3; ++i)
          {
            data[6 + 2 * numberGlobalAtoms + i] = std::vector<double>(3, 0);
            for (unsigned int j = 0; j < 3; ++j)
              data[6 + 2 * numberGlobalAtoms + i][j] = -getCellStress()[i][j];
          }
      }


    dftUtils::writeDataIntoFile(data, Path, d_mpiCommParent);
  }

#include "dft.inst.cc"
} // namespace dftfe
