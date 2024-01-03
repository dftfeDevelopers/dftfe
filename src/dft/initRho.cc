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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

//
// Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftParameters.h>
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <vectorUtilities.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::clearRhoData()
  {
    for (unsigned int iComp = 0; iComp < d_mixingSchemePtrs.size(); ++iComp)
      d_mixingSchemePtrs[iComp]->clearHistory();

    // related to low rank jacobian inverse scf preconditioning
    d_vcontainerVals.clear();
    d_fvcontainerVals.clear();
    d_vSpin0containerVals.clear();
    d_fvSpin0containerVals.clear();
    d_vSpin1containerVals.clear();
    d_fvSpin1containerVals.clear();
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initRho()
  {
    computingTimerStandard.enter_subsection("initialize density");

    // clear existing data
    clearRhoData();

    // Reading single atom rho initial guess
    pcout << std::endl
          << "Reading initial guess for electron-density....." << std::endl;
    std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
    std::map<unsigned int, std::vector<std::vector<double>>>
                                   singleAtomElectronDensity;
    std::map<unsigned int, double> outerMostPointDen;
    const double                   truncationTol = 1e-10;
    double                         maxRhoTail    = 0.0;

    // loop over atom types
    for (std::set<unsigned int>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        char densityFile[256];
        if (d_dftParamsPtr->isPseudopotential)
          {
            strcpy(densityFile,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/density.inp")
                     .c_str());
          }
        else
          {
            sprintf(
              densityFile,
              "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp",
              DFTFE_PATH,
              *it);
          }

        dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
        unsigned int        numRows = singleAtomElectronDensity[*it].size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        unsigned int maxRowId = 0;
        for (unsigned int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = singleAtomElectronDensity[*it][irow][0];
            yData[irow] = singleAtomElectronDensity[*it][irow][1];

            if (yData[irow] > truncationTol)
              maxRowId = irow;
          }

        if (d_dftParamsPtr->isPseudopotential)
          yData[0] = yData[1];

        // interpolate rho
        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type_L = 1;
        alglib::ae_int_t natural_bound_type_R = 1;
        spline1dbuildcubic(x,
                           y,
                           numRows,
                           natural_bound_type_L,
                           0.0,
                           natural_bound_type_R,
                           0.0,
                           denSpline[*it]);
        outerMostPointDen[*it] = xData[maxRowId];

        if (outerMostPointDen[*it] > maxRhoTail)
          maxRhoTail = outerMostPointDen[*it];
      }

    // Initialize electron density table storage for rhoIn
    basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId, false);
    const unsigned int n_q_points = basisOperationsPtrHost->nQuadsPerCell();
    const unsigned int nCells     = basisOperationsPtrHost->nCells();
    d_densityInQuadValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    for (unsigned int iComp = 0; iComp < d_densityInQuadValues.size(); ++iComp)
      d_densityInQuadValues[iComp].resize(n_q_points * nCells);


    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        d_gradDensityInQuadValues.resize(
          d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
        for (unsigned int iComp = 0; iComp < d_densityInQuadValues.size();
             ++iComp)
          d_gradDensityInQuadValues[iComp].resize(3 * n_q_points * nCells);
      }

    // Initialize electron density table storage for rhoOut only for Anderson
    // with Kerker for other mixing schemes it is done in density.cc as we need
    // to do this initialization every SCF
    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      {
        d_densityOutQuadValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 :
                                                                           1);


        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          {
            d_gradDensityOutQuadValues.resize(
              d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
          }
      }



    //
    // get number of image charges used only for periodic
    //
    const int numberImageCharges  = d_imageIdsTrunc.size();
    const int numberGlobalCharges = atomLocations.size();


    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      {
        const dealii::IndexSet &locallyOwnedSet =
          d_dofHandlerRhoNodal.locally_owned_dofs();
        std::vector<dealii::IndexSet::size_type> locallyOwnedDOFs;
        locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
        unsigned int numberDofs = locallyOwnedDOFs.size();
        std::map<dealii::types::global_dof_index, dealii::Point<3>>
          supportPointsRhoNodal;
        dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                     d_dofHandlerRhoNodal,
                                                     supportPointsRhoNodal);

        dealii::BoundingBox<3> boundingBoxTria(
          vectorTools::createBoundingBoxTriaLocallyOwned(d_dofHandlerRhoNodal));
        dealii::Tensor<1, 3, double> tempDisp;
        tempDisp[0] = maxRhoTail;
        tempDisp[1] = maxRhoTail;
        tempDisp[2] = maxRhoTail;

        std::vector<double> atomsImagesPositions;
        std::vector<double> atomsImagesChargeIds;
        for (unsigned int iAtom = 0;
             iAtom < numberGlobalCharges + numberImageCharges;
             iAtom++)
          {
            dealii::Point<3> atomCoord;
            int              chargeId;
            if (iAtom < numberGlobalCharges)
              {
                atomCoord[0] = atomLocations[iAtom][2];
                atomCoord[1] = atomLocations[iAtom][3];
                atomCoord[2] = atomLocations[iAtom][4];
                chargeId     = iAtom;
              }
            else
              {
                const unsigned int iImageCharge = iAtom - numberGlobalCharges;
                atomCoord[0] = d_imagePositionsTrunc[iImageCharge][0];
                atomCoord[1] = d_imagePositionsTrunc[iImageCharge][1];
                atomCoord[2] = d_imagePositionsTrunc[iImageCharge][2];
                chargeId     = d_imageIdsTrunc[iImageCharge];
              }

            std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
              boundaryPoints;
            boundaryPoints.first  = atomCoord - tempDisp;
            boundaryPoints.second = atomCoord + tempDisp;
            dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

            if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) !=
                dealii::NeighborType::not_neighbors)
              ;
            {
              atomsImagesPositions.push_back(atomCoord[0]);
              atomsImagesPositions.push_back(atomCoord[1]);
              atomsImagesPositions.push_back(atomCoord[2]);
              atomsImagesChargeIds.push_back(chargeId);
            }
          }

        // kpoint group parallelization data structures
        const unsigned int numberKptGroups =
          dealii::Utilities::MPI::n_mpi_processes(interpoolcomm);

        const unsigned int kptGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
        std::vector<int> kptGroupLowHighPlusOneIndices;

        if (numberDofs > 0)
          dftUtils::createKpointParallelizationIndices(
            interpoolcomm, numberDofs, kptGroupLowHighPlusOneIndices);
        for (unsigned int iComp = 0; iComp < d_densityInNodalValues.size();
             ++iComp)
          d_densityInNodalValues[iComp] = 0;
#pragma omp parallel for num_threads(d_nOMPThreads) firstprivate(denSpline)
        for (unsigned int dof = 0; dof < numberDofs; ++dof)
          {
            if (dof < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                dof >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                const dealii::types::global_dof_index dofID =
                  locallyOwnedDOFs[dof];
                const dealii::Point<3> &nodalCoor =
                  supportPointsRhoNodal[dofID];
                if (!d_constraintsRhoNodal.is_constrained(dofID))
                  {
                    // loop over atoms and superimpose electron-density at a
                    // given dof from all atoms
                    double rhoNodalValue = 0.0;
                    int    chargeId;
                    double distanceToAtom;
                    double diffx;
                    double diffy;
                    double diffz;

                    for (unsigned int iAtom = 0;
                         iAtom < atomsImagesChargeIds.size();
                         ++iAtom)
                      {
                        diffx =
                          nodalCoor[0] - atomsImagesPositions[iAtom * 3 + 0];
                        diffy =
                          nodalCoor[1] - atomsImagesPositions[iAtom * 3 + 1];
                        diffz =
                          nodalCoor[2] - atomsImagesPositions[iAtom * 3 + 2];

                        distanceToAtom = std::sqrt(
                          diffx * diffx + diffy * diffy + diffz * diffz);

                        chargeId = atomsImagesChargeIds[iAtom];

                        if (distanceToAtom <=
                            outerMostPointDen[atomLocations[chargeId][0]])
                          rhoNodalValue += alglib::spline1dcalc(
                            denSpline[atomLocations[chargeId][0]],
                            distanceToAtom);
                      }

                    d_densityInNodalValues[0].local_element(dof) =
                      std::abs(rhoNodalValue);
                  }
              }
          }

        if (numberDofs > 0 && numberKptGroups > 1)
          MPI_Allreduce(MPI_IN_PLACE,
                        d_densityInNodalValues[0].begin(),
                        numberDofs,
                        MPI_DOUBLE,
                        MPI_SUM,
                        interpoolcomm);
        MPI_Barrier(interpoolcomm);

        d_densityInNodalValues[0].update_ghost_values();

        // normalize rho
        const double charge =
          totalCharge(d_matrixFreeDataPRefined, d_densityInNodalValues[0]);


        const double scalingFactor = ((double)numElectrons) / charge;

        // scale nodal vector with scalingFactor
        d_densityInNodalValues[0] *= scalingFactor;

        if (d_dftParamsPtr->verbosity >= 3)
          {
            pcout << "Total Charge before Normalizing nodal Rho:  " << charge
                  << std::endl;
            pcout << "Total Charge after Normalizing nodal Rho: "
                  << totalCharge(d_matrixFreeDataPRefined,
                                 d_densityInNodalValues[0])
                  << std::endl;
          }

        interpolateDensityNodalDataToQuadratureDataGeneral(
          basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_densityInNodalValues[0],
          d_densityInQuadValues[0],
          d_gradDensityInQuadValues[0],
          d_gradDensityInQuadValues[0],
          d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA);

        if (d_dftParamsPtr->spinPolarized == 1)
          {
#pragma omp parallel for num_threads(d_nOMPThreads)
            for (unsigned int dof = 0; dof < numberDofs; ++dof)
              {
                const dealii::types::global_dof_index dofID =
                  locallyOwnedDOFs[dof];
                const dealii::Point<3> &nodalCoor =
                  supportPointsRhoNodal[dofID];
                if (!d_constraintsRhoNodal.is_constrained(dofID))
                  {
                    d_densityInNodalValues[1].local_element(dof) =
                      -2.0 * d_dftParamsPtr->start_magnetization *
                      d_densityInNodalValues[0].local_element(dof);
                  }
              }

            d_densityInNodalValues[1].update_ghost_values();

            interpolateDensityNodalDataToQuadratureDataGeneral(
              basisOperationsPtrElectroHost,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_densityInNodalValues[1],
              d_densityInQuadValues[1],
              d_gradDensityInQuadValues[1],
              d_gradDensityInQuadValues[1],
              d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);
          }

        normalizeRhoInQuadValues();
      }
    else
      {
        // loop over elements
#pragma omp parallel for num_threads(d_nOMPThreads) firstprivate(denSpline)
        for (auto iCell = 0; iCell < nCells; ++iCell)
          {
            auto    cellid = basisOperationsPtrHost->cellID(iCell);
            double *rhoInValuesPtr =
              &(d_densityInQuadValues[0][iCell * n_q_points]);

            double *magInValuesPtr;
            if (d_dftParamsPtr->spinPolarized == 1)
              {
                magInValuesPtr =
                  &(d_densityInQuadValues[1][iCell * n_q_points]);
              }
            const double *quadPointPtr =
              basisOperationsPtrHost->quadPoints().data() +
              iCell * n_q_points * 3;
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const dealii::Point<3> quadPoint(quadPointPtr[q * 3],
                                                 quadPointPtr[q * 3 + 1],
                                                 quadPointPtr[q * 3 + 2]);
                double                 rhoValueAtQuadPt = 0.0;

                // loop over atoms
                for (unsigned int n = 0; n < atomLocations.size(); n++)
                  {
                    dealii::Point<3> atom(atomLocations[n][2],
                                          atomLocations[n][3],
                                          atomLocations[n][4]);
                    double           distanceToAtom = quadPoint.distance(atom);
                    if (distanceToAtom <=
                        outerMostPointDen[atomLocations[n][0]])
                      {
                        rhoValueAtQuadPt +=
                          alglib::spline1dcalc(denSpline[atomLocations[n][0]],
                                               distanceToAtom);
                      }
                    else
                      {
                        rhoValueAtQuadPt += 0.0;
                      }
                  }

                // loop over image charges
                for (int iImageCharge = 0; iImageCharge < numberImageCharges;
                     ++iImageCharge)
                  {
                    dealii::Point<3> imageAtom(
                      d_imagePositionsTrunc[iImageCharge][0],
                      d_imagePositionsTrunc[iImageCharge][1],
                      d_imagePositionsTrunc[iImageCharge][2]);
                    double distanceToAtom = quadPoint.distance(imageAtom);
                    int    masterAtomId   = d_imageIdsTrunc[iImageCharge];
                    if (
                      distanceToAtom <=
                      outerMostPointDen
                        [atomLocations
                           [masterAtomId]
                           [0]]) // outerMostPointPseudo[atomLocations[masterAtomId][0]])
                      {
                        rhoValueAtQuadPt += alglib::spline1dcalc(
                          denSpline[atomLocations[masterAtomId][0]],
                          distanceToAtom);
                      }
                  }

                rhoInValuesPtr[q] = std::abs(rhoValueAtQuadPt);
                if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    magInValuesPtr[q] = -2.0 *
                                        (d_dftParamsPtr->start_magnetization) *
                                        (std::abs(rhoValueAtQuadPt));
                  }
              }
          }


        // loop over elements
        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          {
#pragma omp parallel for num_threads(d_nOMPThreads) firstprivate(denSpline)
            for (unsigned int iCell = 0;
                 iCell < basisOperationsPtrHost->nCells();
                 ++iCell)
              {
                auto    cellid = basisOperationsPtrHost->cellID(iCell);
                double *gradRhoInValuesPtr =
                  &(d_gradDensityInQuadValues[0][3 * iCell * n_q_points]);

                double *gradMagInValuesPtr;
                if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    gradMagInValuesPtr =
                      &(d_gradDensityInQuadValues[1][3 * iCell * n_q_points]);
                  }
                const double *quadPointPtr =
                  basisOperationsPtrHost->quadPoints().data() +
                  iCell * n_q_points * 3;
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const dealii::Point<3> quadPoint(quadPointPtr[q * 3],
                                                     quadPointPtr[q * 3 + 1],
                                                     quadPointPtr[q * 3 + 2]);
                    double                 gradRhoXValueAtQuadPt = 0.0;
                    double                 gradRhoYValueAtQuadPt = 0.0;
                    double                 gradRhoZValueAtQuadPt = 0.0;
                    // loop over atoms
                    for (unsigned int n = 0; n < atomLocations.size(); n++)
                      {
                        dealii::Point<3> atom(atomLocations[n][2],
                                              atomLocations[n][3],
                                              atomLocations[n][4]);
                        double distanceToAtom = quadPoint.distance(atom);

                        if (d_dftParamsPtr->floatingNuclearCharges &&
                            distanceToAtom < 1.0e-3)
                          continue;

                        if (distanceToAtom <=
                            outerMostPointDen[atomLocations[n][0]])
                          {
                            // rhoValueAtQuadPt+=alglib::spline1dcalc(denSpline[atomLocations[n][0]],
                            // distanceToAtom);
                            double value, radialDensityFirstDerivative,
                              radialDensitySecondDerivative;
                            alglib::spline1ddiff(denSpline[atomLocations[n][0]],
                                                 distanceToAtom,
                                                 value,
                                                 radialDensityFirstDerivative,
                                                 radialDensitySecondDerivative);

                            gradRhoXValueAtQuadPt +=
                              radialDensityFirstDerivative *
                              ((quadPoint[0] - atomLocations[n][2]) /
                               distanceToAtom);
                            gradRhoYValueAtQuadPt +=
                              radialDensityFirstDerivative *
                              ((quadPoint[1] - atomLocations[n][3]) /
                               distanceToAtom);
                            gradRhoZValueAtQuadPt +=
                              radialDensityFirstDerivative *
                              ((quadPoint[2] - atomLocations[n][4]) /
                               distanceToAtom);
                          }
                      }

                    for (int iImageCharge = 0;
                         iImageCharge < numberImageCharges;
                         ++iImageCharge)
                      {
                        dealii::Point<3> imageAtom(
                          d_imagePositionsTrunc[iImageCharge][0],
                          d_imagePositionsTrunc[iImageCharge][1],
                          d_imagePositionsTrunc[iImageCharge][2]);
                        double distanceToAtom = quadPoint.distance(imageAtom);

                        if (d_dftParamsPtr->floatingNuclearCharges &&
                            distanceToAtom < 1.0e-3)
                          continue;

                        int masterAtomId = d_imageIdsTrunc[iImageCharge];
                        if (
                          distanceToAtom <=
                          outerMostPointDen
                            [atomLocations
                               [masterAtomId]
                               [0]]) // outerMostPointPseudo[atomLocations[masterAtomId][0]])
                          {
                            double value, radialDensityFirstDerivative,
                              radialDensitySecondDerivative;
                            alglib::spline1ddiff(
                              denSpline[atomLocations[masterAtomId][0]],
                              distanceToAtom,
                              value,
                              radialDensityFirstDerivative,
                              radialDensitySecondDerivative);

                            gradRhoXValueAtQuadPt +=
                              radialDensityFirstDerivative *
                              ((quadPoint[0] -
                                d_imagePositionsTrunc[iImageCharge][0]) /
                               distanceToAtom);
                            gradRhoYValueAtQuadPt +=
                              radialDensityFirstDerivative *
                              ((quadPoint[1] -
                                d_imagePositionsTrunc[iImageCharge][1]) /
                               distanceToAtom);
                            gradRhoZValueAtQuadPt +=
                              radialDensityFirstDerivative *
                              ((quadPoint[2] -
                                d_imagePositionsTrunc[iImageCharge][2]) /
                               distanceToAtom);
                          }
                      }

                    int signRho = 0;
                    /*
                       if (std::abs((*rhoInValues)[cellid][q] )
                       > 1.0E-7) signRho =
                       (*rhoInValues)[cellid][q]>0.0?1:-1;
                     */
                    if (std::abs(
                          d_densityInQuadValues[0][iCell * n_q_points + q]) >
                        1.0E-8)
                      signRho =
                        d_densityInQuadValues[0][iCell * n_q_points + q] /
                        std::abs(
                          d_densityInQuadValues[0][iCell * n_q_points + q]);

                    // KG: the fact that we are forcing gradRho to zero
                    // whenever rho is zero is valid. Because rho is always
                    // positive, so whenever it is zero, it must have a
                    // local minima.
                    //
                    gradRhoInValuesPtr[3 * q + 0] =
                      signRho * gradRhoXValueAtQuadPt;
                    gradRhoInValuesPtr[3 * q + 1] =
                      signRho * gradRhoYValueAtQuadPt;
                    gradRhoInValuesPtr[3 * q + 2] =
                      signRho * gradRhoZValueAtQuadPt;
                    if (d_dftParamsPtr->spinPolarized == 1)
                      {
                        gradMagInValuesPtr[3 * q + 0] =
                          -2.0 * (d_dftParamsPtr->start_magnetization) *
                          signRho * gradRhoXValueAtQuadPt;
                        gradMagInValuesPtr[3 * q + 1] =
                          -2.0 * (d_dftParamsPtr->start_magnetization) *
                          signRho * gradRhoYValueAtQuadPt;
                        gradMagInValuesPtr[3 * q + 2] =
                          -2.0 * (d_dftParamsPtr->start_magnetization) *
                          signRho * gradRhoZValueAtQuadPt;
                      }
                  }
              }
          }

        normalizeRhoInQuadValues();
      }
    //
    computingTimerStandard.leave_subsection("initialize density");
  }

  //
  //
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::computeRhoInitialGuessFromPSI(
    std::vector<std::vector<distributedCPUVec<double>>> eigenVectors)

  {
    computingTimerStandard.enter_subsection("initialize density");

    // clear existing data
    clearRhoData();

    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(
      FEEigen, quadrature, dealii::update_values | dealii::update_gradients);
    const unsigned int num_quad_points = quadrature.size();
    const unsigned int numCells        = matrix_free_data.n_physical_cells();

    // Initialize electron density table storage
    d_densityInQuadValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    d_densityInQuadValues[0].resize(numCells * num_quad_points);
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        d_densityInQuadValues[1].resize(numCells * num_quad_points);
      }

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        d_gradDensityInQuadValues.resize(
          d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
        d_gradDensityInQuadValues[0].resize(3 * numCells * num_quad_points);
        //
        if (d_dftParamsPtr->spinPolarized == 1)
          {
            d_gradDensityInQuadValues[1].resize(3 * numCells * num_quad_points);
          }
      }

    // temp arrays
    std::vector<double> rhoTemp(num_quad_points),
      rhoTempSpinPolarized(2 * num_quad_points), rhoIn(num_quad_points),
      rhoInSpinPolarized(2 * num_quad_points);
    std::vector<double> gradRhoTemp(3 * num_quad_points),
      gradRhoTempSpinPolarized(6 * num_quad_points),
      gradRhoIn(3 * num_quad_points),
      gradRhoInSpinPolarized(6 * num_quad_points);

    // loop over locally owned elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = dofHandlerEigen.begin_active(),
      endc             = dofHandlerEigen.end();
    unsigned int iCell = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          std::fill(rhoTemp.begin(), rhoTemp.end(), 0.0);
          std::fill(rhoIn.begin(), rhoIn.end(), 0.0);
          if (d_dftParamsPtr->spinPolarized == 1)
            {
              std::fill(rhoTempSpinPolarized.begin(),
                        rhoTempSpinPolarized.end(),
                        0.0);
            }

#ifdef USE_COMPLEX
          std::vector<dealii::Vector<double>> tempPsi(num_quad_points),
            tempPsi2(num_quad_points);
          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              tempPsi[q_point].reinit(2);
              tempPsi2[q_point].reinit(2);
            }
#else
          std::vector<double> tempPsi(num_quad_points),
            tempPsi2(num_quad_points);
#endif



          if (d_excManagerPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA) // GGA
            {
              std::fill(gradRhoTemp.begin(), gradRhoTemp.end(), 0.0);
              if (d_dftParamsPtr->spinPolarized == 1)
                {
                  std::fill(gradRhoTempSpinPolarized.begin(),
                            gradRhoTempSpinPolarized.end(),
                            0.0);
                }
#ifdef USE_COMPLEX
              std::vector<std::vector<dealii::Tensor<1, 3, double>>>
                tempGradPsi(num_quad_points), tempGradPsi2(num_quad_points);
              for (unsigned int q_point = 0; q_point < num_quad_points;
                   ++q_point)
                {
                  tempGradPsi[q_point].resize(2);
                  tempGradPsi2[q_point].resize(2);
                }
#else
              std::vector<dealii::Tensor<1, 3, double>> tempGradPsi(
                num_quad_points),
                tempGradPsi2(num_quad_points);
#endif


              for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
                {
                  for (unsigned int i = 0; i < d_numEigenValues; ++i)
                    {
                      fe_values.get_function_values(
                        eigenVectors[(1 + d_dftParamsPtr->spinPolarized) *
                                     kPoint][i],
                        tempPsi);
                      if (d_dftParamsPtr->spinPolarized == 1)
                        fe_values.get_function_values(
                          eigenVectors[(1 + d_dftParamsPtr->spinPolarized) *
                                         kPoint +
                                       1][i],
                          tempPsi2);
                      //
                      fe_values.get_function_gradients(
                        eigenVectors[(1 + d_dftParamsPtr->spinPolarized) *
                                     kPoint][i],
                        tempGradPsi);
                      if (d_dftParamsPtr->spinPolarized == 1)
                        fe_values.get_function_gradients(
                          eigenVectors[(1 + d_dftParamsPtr->spinPolarized) *
                                         kPoint +
                                       1][i],
                          tempGradPsi2);

                      for (unsigned int q_point = 0; q_point < num_quad_points;
                           ++q_point)
                        {
                          double factor =
                            (eigenValues[kPoint][i] - fermiEnergy) /
                            (C_kb * d_dftParamsPtr->TVal);
                          double partialOccupancy =
                            (factor >= 0) ?
                              std::exp(-factor) / (1.0 + std::exp(-factor)) :
                              1.0 / (1.0 + std::exp(factor));
                          //
                          factor =
                            (eigenValues[kPoint]
                                        [i + d_dftParamsPtr->spinPolarized *
                                               d_numEigenValues] -
                             fermiEnergy) /
                            (C_kb * d_dftParamsPtr->TVal);
                          double partialOccupancy2 =
                            (factor >= 0) ?
                              std::exp(-factor) / (1.0 + std::exp(-factor)) :
                              1.0 / (1.0 + std::exp(factor));
#ifdef USE_COMPLEX
                          if (d_dftParamsPtr->spinPolarized == 1)
                            {
                              rhoTempSpinPolarized[2 * q_point] +=
                                partialOccupancy * d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) * tempPsi[q_point](0) +
                                 tempPsi[q_point](1) * tempPsi[q_point](1));
                              rhoTempSpinPolarized[2 * q_point + 1] +=
                                partialOccupancy2 * d_kPointWeights[kPoint] *
                                (tempPsi2[q_point](0) * tempPsi2[q_point](0) +
                                 tempPsi2[q_point](1) * tempPsi2[q_point](1));
                              //
                              gradRhoTempSpinPolarized[6 * q_point + 0] +=
                                2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) *
                                   tempGradPsi[q_point][0][0] +
                                 tempPsi[q_point](1) *
                                   tempGradPsi[q_point][1][0]);
                              gradRhoTempSpinPolarized[6 * q_point + 1] +=
                                2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) *
                                   tempGradPsi[q_point][0][1] +
                                 tempPsi[q_point](1) *
                                   tempGradPsi[q_point][1][1]);
                              gradRhoTempSpinPolarized[6 * q_point + 2] +=
                                2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) *
                                   tempGradPsi[q_point][0][2] +
                                 tempPsi[q_point](1) *
                                   tempGradPsi[q_point][1][2]);
                              gradRhoTempSpinPolarized[6 * q_point + 3] +=
                                2.0 * partialOccupancy2 *
                                d_kPointWeights[kPoint] *
                                (tempPsi2[q_point](0) *
                                   tempGradPsi2[q_point][0][0] +
                                 tempPsi2[q_point](1) *
                                   tempGradPsi2[q_point][1][0]);
                              gradRhoTempSpinPolarized[6 * q_point + 4] +=
                                2.0 * partialOccupancy2 *
                                d_kPointWeights[kPoint] *
                                (tempPsi2[q_point](0) *
                                   tempGradPsi2[q_point][0][1] +
                                 tempPsi2[q_point](1) *
                                   tempGradPsi2[q_point][1][1]);
                              gradRhoTempSpinPolarized[6 * q_point + 5] +=
                                2.0 * partialOccupancy2 *
                                d_kPointWeights[kPoint] *
                                (tempPsi2[q_point](0) *
                                   tempGradPsi2[q_point][0][2] +
                                 tempPsi2[q_point](1) *
                                   tempGradPsi2[q_point][1][2]);
                            }
                          else
                            {
                              rhoTemp[q_point] +=
                                2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) * tempPsi[q_point](0) +
                                 tempPsi[q_point](1) * tempPsi[q_point](1));
                              gradRhoTemp[3 * q_point + 0] +=
                                2.0 * 2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) *
                                   tempGradPsi[q_point][0][0] +
                                 tempPsi[q_point](1) *
                                   tempGradPsi[q_point][1][0]);
                              gradRhoTemp[3 * q_point + 1] +=
                                2.0 * 2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) *
                                   tempGradPsi[q_point][0][1] +
                                 tempPsi[q_point](1) *
                                   tempGradPsi[q_point][1][1]);
                              gradRhoTemp[3 * q_point + 2] +=
                                2.0 * 2.0 * partialOccupancy *
                                d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) *
                                   tempGradPsi[q_point][0][2] +
                                 tempPsi[q_point](1) *
                                   tempGradPsi[q_point][1][2]);
                            }
#else
                          if (d_dftParamsPtr->spinPolarized == 1)
                            {
                              rhoTempSpinPolarized[2 * q_point] +=
                                partialOccupancy * tempPsi[q_point] *
                                tempPsi[q_point];
                              rhoTempSpinPolarized[2 * q_point + 1] +=
                                partialOccupancy2 * tempPsi2[q_point] *
                                tempPsi2[q_point];
                              gradRhoTempSpinPolarized[6 * q_point + 0] +=
                                2.0 * partialOccupancy *
                                (tempPsi[q_point] * tempGradPsi[q_point][0]);
                              gradRhoTempSpinPolarized[6 * q_point + 1] +=
                                2.0 * partialOccupancy *
                                (tempPsi[q_point] * tempGradPsi[q_point][1]);
                              gradRhoTempSpinPolarized[6 * q_point + 2] +=
                                2.0 * partialOccupancy *
                                (tempPsi[q_point] * tempGradPsi[q_point][2]);
                              gradRhoTempSpinPolarized[6 * q_point + 3] +=
                                2.0 * partialOccupancy2 *
                                (tempPsi2[q_point] * tempGradPsi2[q_point][0]);
                              gradRhoTempSpinPolarized[6 * q_point + 4] +=
                                2.0 * partialOccupancy2 *
                                (tempPsi2[q_point] * tempGradPsi2[q_point][1]);
                              gradRhoTempSpinPolarized[6 * q_point + 5] +=
                                2.0 * partialOccupancy2 *
                                (tempPsi2[q_point] * tempGradPsi2[q_point][2]);
                            }
                          else
                            {
                              rhoTemp[q_point] +=
                                2.0 * partialOccupancy * tempPsi[q_point] *
                                tempPsi
                                  [q_point]; // std::pow(tempPsi[q_point],2.0);
                              gradRhoTemp[3 * q_point + 0] +=
                                2.0 * 2.0 * partialOccupancy *
                                tempPsi[q_point] * tempGradPsi[q_point][0];
                              gradRhoTemp[3 * q_point + 1] +=
                                2.0 * 2.0 * partialOccupancy *
                                tempPsi[q_point] * tempGradPsi[q_point][1];
                              gradRhoTemp[3 * q_point + 2] +=
                                2.0 * 2.0 * partialOccupancy *
                                tempPsi[q_point] * tempGradPsi[q_point][2];
                            }

#endif
                        }
                    }
                }

              //  gather density from all pools
              int numPoint = num_quad_points;
              MPI_Allreduce(&rhoTemp[0],
                            &rhoIn[0],
                            numPoint,
                            MPI_DOUBLE,
                            MPI_SUM,
                            interpoolcomm);
              MPI_Allreduce(&gradRhoTemp[0],
                            &gradRhoIn[0],
                            3 * numPoint,
                            MPI_DOUBLE,
                            MPI_SUM,
                            interpoolcomm);
              if (d_dftParamsPtr->spinPolarized == 1)
                {
                  MPI_Allreduce(&rhoTempSpinPolarized[0],
                                &rhoInSpinPolarized[0],
                                2 * numPoint,
                                MPI_DOUBLE,
                                MPI_SUM,
                                interpoolcomm);
                  MPI_Allreduce(&gradRhoTempSpinPolarized[0],
                                &gradRhoInSpinPolarized[0],
                                6 * numPoint,
                                MPI_DOUBLE,
                                MPI_SUM,
                                interpoolcomm);
                }

              //


              for (unsigned int q_point = 0; q_point < num_quad_points;
                   ++q_point)
                {
                  if (d_dftParamsPtr->spinPolarized == 1)
                    {
                      d_densityInQuadValues[0][iCell * num_quad_points +
                                               q_point] =
                        rhoInSpinPolarized[2 * q_point] +
                        rhoInSpinPolarized[2 * q_point + 1];
                      d_densityInQuadValues[1][iCell * num_quad_points +
                                               q_point] =
                        rhoInSpinPolarized[2 * q_point] -
                        rhoInSpinPolarized[2 * q_point + 1];
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        d_gradDensityInQuadValues[0][iCell * num_quad_points *
                                                       3 +
                                                     3 * q_point + iDim] =
                          gradRhoInSpinPolarized[6 * q_point + iDim] +
                          gradRhoInSpinPolarized[6 * q_point + iDim + 3];
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        d_gradDensityInQuadValues[1][iCell * num_quad_points *
                                                       3 +
                                                     3 * q_point + iDim] =
                          gradRhoInSpinPolarized[6 * q_point + iDim] -
                          gradRhoInSpinPolarized[6 * q_point + iDim + 3];
                    }
                  else
                    {
                      d_densityInQuadValues[0][iCell * num_quad_points +
                                               q_point] = rhoIn[q_point];
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        d_gradDensityInQuadValues[0][iCell * num_quad_points *
                                                       3 +
                                                     3 * q_point + iDim] =
                          gradRhoIn[3 * q_point + iDim];
                    }
                }
            }
          else
            {
              for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
                {
                  for (unsigned int i = 0; i < d_numEigenValues; ++i)
                    {
                      fe_values.get_function_values(
                        eigenVectors[(1 + d_dftParamsPtr->spinPolarized) *
                                     kPoint][i],
                        tempPsi);
                      if (d_dftParamsPtr->spinPolarized == 1)
                        fe_values.get_function_values(
                          eigenVectors[(1 + d_dftParamsPtr->spinPolarized) *
                                         kPoint +
                                       1][i],
                          tempPsi2);

                      for (unsigned int q_point = 0; q_point < num_quad_points;
                           ++q_point)
                        {
                          double factor =
                            (eigenValues[kPoint][i] - fermiEnergy) /
                            (C_kb * d_dftParamsPtr->TVal);
                          double partialOccupancy =
                            (factor >= 0) ?
                              std::exp(-factor) / (1.0 + std::exp(-factor)) :
                              1.0 / (1.0 + std::exp(factor));
                          //
                          factor =
                            (eigenValues[kPoint]
                                        [i + d_dftParamsPtr->spinPolarized *
                                               d_numEigenValues] -
                             fermiEnergy) /
                            (C_kb * d_dftParamsPtr->TVal);
                          double partialOccupancy2 =
                            (factor >= 0) ?
                              std::exp(-factor) / (1.0 + std::exp(-factor)) :
                              1.0 / (1.0 + std::exp(factor));
#ifdef USE_COMPLEX
                          if (d_dftParamsPtr->spinPolarized == 1)
                            {
                              rhoTempSpinPolarized[2 * q_point] +=
                                partialOccupancy * d_kPointWeights[kPoint] *
                                (tempPsi[q_point](0) * tempPsi[q_point](0) +
                                 tempPsi[q_point](1) * tempPsi[q_point](1));
                              rhoTempSpinPolarized[2 * q_point + 1] +=
                                partialOccupancy2 * d_kPointWeights[kPoint] *
                                (tempPsi2[q_point](0) * tempPsi2[q_point](0) +
                                 tempPsi2[q_point](1) * tempPsi2[q_point](1));
                            }
                          else
                            rhoTemp[q_point] +=
                              2.0 * partialOccupancy * d_kPointWeights[kPoint] *
                              (tempPsi[q_point](0) * tempPsi[q_point](0) +
                               tempPsi[q_point](1) * tempPsi[q_point](1));
#else
                          if (d_dftParamsPtr->spinPolarized == 1)
                            {
                              rhoTempSpinPolarized[2 * q_point] +=
                                partialOccupancy * tempPsi[q_point] *
                                tempPsi[q_point];
                              rhoTempSpinPolarized[2 * q_point + 1] +=
                                partialOccupancy2 * tempPsi2[q_point] *
                                tempPsi2[q_point];
                            }
                          else
                            rhoTemp[q_point] +=
                              2.0 * partialOccupancy * tempPsi[q_point] *
                              tempPsi
                                [q_point]; // std::pow(tempPsi[q_point],2.0);
                                           //
#endif
                        }
                    }
                }
              //  gather density from all pools
              int numPoint = num_quad_points;
              MPI_Allreduce(&rhoTemp[0],
                            &rhoIn[0],
                            numPoint,
                            MPI_DOUBLE,
                            MPI_SUM,
                            interpoolcomm);
              if (d_dftParamsPtr->spinPolarized == 1)
                MPI_Allreduce(&rhoTempSpinPolarized[0],
                              &rhoInSpinPolarized[0],
                              2 * numPoint,
                              MPI_DOUBLE,
                              MPI_SUM,
                              interpoolcomm);
              //
              for (unsigned int q_point = 0; q_point < num_quad_points;
                   ++q_point)
                {
                  if (d_dftParamsPtr->spinPolarized == 1)
                    {
                      d_densityInQuadValues[0][iCell * num_quad_points +
                                               q_point] =
                        rhoInSpinPolarized[2 * q_point] +
                        rhoInSpinPolarized[2 * q_point + 1];
                      d_densityInQuadValues[1][iCell * num_quad_points +
                                               q_point] =
                        rhoInSpinPolarized[2 * q_point] -
                        rhoInSpinPolarized[2 * q_point + 1];
                    }
                  else
                    d_densityInQuadValues[0][iCell * num_quad_points +
                                             q_point] = rhoIn[q_point];
                }
            }
          ++iCell;
        }

    normalizeRhoInQuadValues();
    //
    computingTimerStandard.leave_subsection("initialize density");
  }


  //
  // Normalize rho
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::normalizeRhoInQuadValues()
  {
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int nCells     = matrix_free_data.n_physical_cells();
    const double       charge =
      totalCharge(d_dofHandlerRhoNodal, d_densityInQuadValues[0]);
    const double scaling = ((double)numElectrons) / charge;

    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "initial total charge before normalizing to number of electrons: "
        << charge << std::endl;

    // scaling rho
    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int iComp = 0; iComp < d_densityInQuadValues.size();
                 ++iComp)
              d_densityInQuadValues[iComp][iCell * n_q_points + q] *= scaling;
            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              for (unsigned int iComp = 0;
                   iComp < d_gradDensityInQuadValues.size();
                   ++iComp)
                for (unsigned int idim = 0; idim < 3; ++idim)
                  d_gradDensityInQuadValues[iComp][3 * iCell * n_q_points +
                                                   3 * q + idim] *= scaling;
          }
      }
    double chargeAfterScaling =
      totalCharge(d_dofHandlerRhoNodal, d_densityInQuadValues[0]);

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Initial total charge: " << chargeAfterScaling << std::endl;
  }

  //
  // Normalize rho
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::normalizeRhoOutQuadValues()
  {
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int nCells     = matrix_free_data.n_physical_cells();

    const double charge =
      totalCharge(d_dofHandlerRhoNodal, d_densityOutQuadValues[0]);
    const double scaling = ((double)numElectrons) / charge;

    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Total charge out before normalizing to number of electrons: "
            << charge << std::endl;

    // scaling rho
    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int iComp = 0; iComp < d_densityOutQuadValues.size();
                 ++iComp)
              d_densityOutQuadValues[iComp][iCell * n_q_points + q] *= scaling;
            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              for (unsigned int iComp = 0;
                   iComp < d_gradDensityOutQuadValues.size();
                   ++iComp)
                for (unsigned int idim = 0; idim < 3; ++idim)
                  d_gradDensityOutQuadValues[iComp][3 * iCell * n_q_points +
                                                    3 * q + idim] *= scaling;
          }
      }
    double chargeAfterScaling =
      totalCharge(d_dofHandlerRhoNodal, d_densityOutQuadValues[0]);

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total charge out after scaling: " << chargeAfterScaling
            << std::endl;
  }
#include "dft.inst.cc"
} // namespace dftfe
