// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::clearRhoData()
{
  rhoInVals.clear();
  rhoOutVals.clear();
  gradRhoInVals.clear();
  gradRhoOutVals.clear();
  rhoInValsSpinPolarized.clear();
  rhoOutValsSpinPolarized.clear();
  gradRhoInValsSpinPolarized.clear();
  gradRhoOutValsSpinPolarized.clear();
  dFBroyden.clear();
  graddFBroyden.clear();
  uBroyden.clear();
  gradUBroyden.clear();
  d_rhoInNodalVals.clear();
  d_rhoOutNodalVals.clear();
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::initRho()
{
  computing_timer.enter_section("initialize density");

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
      if (dftParameters::isPseudopotential)
        {
          sprintf(densityFile, "temp/z%u/density.inp", *it);
        }
      else
        {
          sprintf(
            densityFile,
            "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp",
            DFT_PATH,
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

      if (dftParameters::isPseudopotential)
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


  // Initialize rho
  const Quadrature<3> &quadrature_formula =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3> fe_values(FE, quadrature_formula, update_quadrature_points);
  const unsigned int n_q_points = quadrature_formula.size();

  // Initialize electron density table storage for rhoIn

  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());
  if (dftParameters::spinPolarized == 1)
    {
      rhoInValsSpinPolarized.push_back(
        std::map<dealii::CellId, std::vector<double>>());
      rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
    }

  if (dftParameters::xcFamilyType == "GGA")
    {
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
      //
      if (dftParameters::spinPolarized == 1)
        {
          gradRhoInValsSpinPolarized.push_back(
            std::map<dealii::CellId, std::vector<double>>());
          gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
        }
    }

  // Initialize electron density table storage for rhoOut only for Anderson with
  // Kerker for other mixing schemes it is done in density.cc as we need to do
  // this initialization every SCF
  if (dftParameters::mixingMethod == "ANDERSON_WITH_KERKER")
    {
      rhoOutVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      rhoOutValues = &(rhoOutVals.back());

      if (dftParameters::xcFamilyType == "GGA")
        {
          gradRhoOutVals.push_back(
            std::map<dealii::CellId, std::vector<double>>());
          gradRhoOutValues = &(gradRhoOutVals.back());
        }
    }



  //
  // get number of image charges used only for periodic
  //
  const int numberImageCharges  = d_imageIdsTrunc.size();
  const int numberGlobalCharges = atomLocations.size();

  if (dftParameters::mixingMethod == "ANDERSON_WITH_KERKER")
    {
      IndexSet locallyOwnedSet;
      DoFTools::extract_locally_owned_dofs(d_dofHandlerRhoNodal,
                                           locallyOwnedSet);
      std::vector<IndexSet::size_type> locallyOwnedDOFs;
      locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
      unsigned int numberDofs = locallyOwnedDOFs.size();
      std::map<types::global_dof_index, Point<3>> supportPointsRhoNodal;
      DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                           d_dofHandlerRhoNodal,
                                           supportPointsRhoNodal);

      dealii::BoundingBox<3> boundingBoxTria(
        vectorTools::createBoundingBoxTriaLocallyOwned(d_dofHandlerRhoNodal));
      dealii::Tensor<1, 3, double> tempDisp;
      tempDisp[0] = maxRhoTail;
      tempDisp[1] = maxRhoTail;
      tempDisp[2] = maxRhoTail;

      // d_matrixFreeDataPRefined.initialize_dof_vector(d_rhoInNodalValues);
      std::vector<double> atomsImagesPositions;
      std::vector<double> atomsImagesChargeIds;
      for (unsigned int iAtom = 0;
           iAtom < numberGlobalCharges + numberImageCharges;
           iAtom++)
        {
          Point<3> atomCoord;    
          int chargeId;
          if (iAtom < numberGlobalCharges)
            {
              atomCoord[0] = atomLocations[iAtom][2];
              atomCoord[1] = atomLocations[iAtom][3];
              atomCoord[2] = atomLocations[iAtom][4];
              chargeId         = iAtom;
            }
          else
            {
              const unsigned int iImageCharge = iAtom - numberGlobalCharges;
              atomCoord[0] =
                d_imagePositionsTrunc[iImageCharge][0];
              atomCoord[1] =
                d_imagePositionsTrunc[iImageCharge][1];
              atomCoord[2] =
                d_imagePositionsTrunc[iImageCharge][2];
              chargeId = d_imageIdsTrunc[iImageCharge];
            }

            std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
              boundaryPoints;
            boundaryPoints.first  = atomCoord - tempDisp;
            boundaryPoints.second = atomCoord + tempDisp;
            dealii::BoundingBox<3> boundingBoxAroundAtom(boundaryPoints);

            if (boundingBoxTria.get_neighbor_type(boundingBoxAroundAtom) !=
                NeighborType::not_neighbors)
              ;
            {
              atomsImagesPositions.push_back(atomCoord[0]);
              atomsImagesPositions.push_back(atomCoord[1]);
              atomsImagesPositions.push_back(atomCoord[2]);
              atomsImagesChargeIds.push_back(chargeId);              
            }
        }


      for (unsigned int dof = 0; dof < numberDofs; ++dof)
        {
          const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
          Point<3> nodalCoor = supportPointsRhoNodal[dofID];
          if (!d_constraintsRhoNodal.is_constrained(dofID))
            {
              // loop over atoms and superimpose electron-density at a given dof
              // from all atoms
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
                  diffx = nodalCoor[0] - atomsImagesPositions[iAtom * 3 + 0];
                  diffy = nodalCoor[1] - atomsImagesPositions[iAtom * 3 + 1];
                  diffz = nodalCoor[2] - atomsImagesPositions[iAtom * 3 + 2];

                  distanceToAtom =
                    std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);

                  chargeId = atomsImagesChargeIds[iAtom];

                  if (distanceToAtom <=
                      outerMostPointDen[atomLocations[chargeId][0]])
                    rhoNodalValue += alglib::spline1dcalc(
                      denSpline[atomLocations[chargeId][0]], distanceToAtom);
                }

              d_rhoInNodalValues.local_element(dof) = std::abs(rhoNodalValue);
            }
        }

      d_rhoInNodalValues.update_ghost_values();

      // normalize rho
      const double charge =
        totalCharge(d_matrixFreeDataPRefined, d_rhoInNodalValues);


      const double scalingFactor = ((double)numElectrons) / charge;

      // scale nodal vector with scalingFactor
      d_rhoInNodalValues *= scalingFactor;

      // push the rhoIn to deque storing the history of nodal values
      d_rhoInNodalVals.push_back(d_rhoInNodalValues);

      if (dftParameters::verbosity >= 3)
        {
          pcout << "Total Charge before Normalizing nodal Rho:  " << charge
                << std::endl;
          pcout << "Total Charge after Normalizing nodal Rho: "
                << totalCharge(d_matrixFreeDataPRefined, d_rhoInNodalValues)
                << std::endl;
        }

      if (dftParameters::xcFamilyType == "GGA")
        {
          gradRhoInVals.push_back(
            std::map<dealii::CellId, std::vector<double>>());
          gradRhoInValues = &(gradRhoInVals.back());
        }

      interpolateRhoNodalDataToQuadratureDataGeneral(
        d_matrixFreeDataPRefined,
        d_densityDofHandlerIndexElectro,
        d_densityQuadratureIdElectro,
        d_rhoInNodalValues,
        *rhoInValues,
        *gradRhoInValues,
        *gradRhoInValues,
        dftParameters::xcFamilyType == "GGA");
      normalizeRhoInQuadValues();
    }
  // else
  {
    // loop over elements
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dofHandler.begin_active(),
                                                 endc = dofHandler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            (*rhoInValues)[cell->id()] = std::vector<double>(n_q_points);
            double *rhoInValuesPtr     = &((*rhoInValues)[cell->id()][0]);

            double *rhoInValuesSpinPolarizedPtr;
            if (dftParameters::spinPolarized == 1)
              {
                (*rhoInValuesSpinPolarized)[cell->id()] =
                  std::vector<double>(2 * n_q_points);
                rhoInValuesSpinPolarizedPtr =
                  &((*rhoInValuesSpinPolarized)[cell->id()][0]);
              }
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const Point<3> &quadPoint = fe_values.quadrature_point(q);
                double          rhoValueAtQuadPt = 0.0;

                // loop over atoms
                for (unsigned int n = 0; n < atomLocations.size(); n++)
                  {
                    Point<3> atom(atomLocations[n][2],
                                  atomLocations[n][3],
                                  atomLocations[n][4]);
                    double   distanceToAtom = quadPoint.distance(atom);
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
                    Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
                                       d_imagePositionsTrunc[iImageCharge][1],
                                       d_imagePositionsTrunc[iImageCharge][2]);
                    double   distanceToAtom = quadPoint.distance(imageAtom);
                    int      masterAtomId   = d_imageIdsTrunc[iImageCharge];
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
                if (dftParameters::spinPolarized == 1)
                  {
                    rhoInValuesSpinPolarizedPtr[2 * q + 1] =
                      (0.5 + dftParameters::start_magnetization) *
                      (std::abs(rhoValueAtQuadPt));
                    rhoInValuesSpinPolarizedPtr[2 * q] =
                      (0.5 - dftParameters::start_magnetization) *
                      (std::abs(rhoValueAtQuadPt));
                  }
              }
          }
      }


    // loop over elements
    if (dftParameters::xcFamilyType == "GGA")
      {
        //
        cell = dofHandler.begin_active();
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);

                (*gradRhoInValues)[cell->id()] =
                  std::vector<double>(3 * n_q_points, 0.0);
                double *gradRhoInValuesPtr =
                  &((*gradRhoInValues)[cell->id()][0]);

                double *gradRhoInValuesSpinPolarizedPtr;
                if (dftParameters::spinPolarized == 1)
                  {
                    (*gradRhoInValuesSpinPolarized)[cell->id()] =
                      std::vector<double>(6 * n_q_points, 0.0);
                    gradRhoInValuesSpinPolarizedPtr =
                      &((*gradRhoInValuesSpinPolarized)[cell->id()][0]);
                  }
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const Point<3> &quadPoint = fe_values.quadrature_point(q);
                    double          gradRhoXValueAtQuadPt = 0.0;
                    double          gradRhoYValueAtQuadPt = 0.0;
                    double          gradRhoZValueAtQuadPt = 0.0;
                    // loop over atoms
                    for (unsigned int n = 0; n < atomLocations.size(); n++)
                      {
                        Point<3> atom(atomLocations[n][2],
                                      atomLocations[n][3],
                                      atomLocations[n][4]);
                        double   distanceToAtom = quadPoint.distance(atom);

                        if (dftParameters::floatingNuclearCharges &&
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
                        Point<3> imageAtom(
                          d_imagePositionsTrunc[iImageCharge][0],
                          d_imagePositionsTrunc[iImageCharge][1],
                          d_imagePositionsTrunc[iImageCharge][2]);
                        double distanceToAtom = quadPoint.distance(imageAtom);

                        if (dftParameters::floatingNuclearCharges &&
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
                       if (std::abs((*rhoInValues)[cell->id()][q] ) > 1.0E-7)
                       signRho = (*rhoInValues)[cell->id()][q]>0.0?1:-1;
                     */
                    if (std::abs((*rhoInValues)[cell->id()][q]) > 1.0E-8)
                      signRho = (*rhoInValues)[cell->id()][q] /
                                std::abs((*rhoInValues)[cell->id()][q]);

                    // KG: the fact that we are forcing gradRho to zero whenever
                    // rho is zero is valid. Because rho is always positive, so
                    // whenever it is zero, it must have a local minima.
                    //
                    gradRhoInValuesPtr[3 * q + 0] =
                      signRho * gradRhoXValueAtQuadPt;
                    gradRhoInValuesPtr[3 * q + 1] =
                      signRho * gradRhoYValueAtQuadPt;
                    gradRhoInValuesPtr[3 * q + 2] =
                      signRho * gradRhoZValueAtQuadPt;
                    if (dftParameters::spinPolarized == 1)
                      {
                        gradRhoInValuesSpinPolarizedPtr[6 * q + 0] =
                          (0.5 - dftParameters::start_magnetization) * signRho *
                          gradRhoXValueAtQuadPt;
                        gradRhoInValuesSpinPolarizedPtr[6 * q + 1] =
                          (0.5 - dftParameters::start_magnetization) * signRho *
                          gradRhoYValueAtQuadPt;
                        gradRhoInValuesSpinPolarizedPtr[6 * q + 2] =
                          (0.5 - dftParameters::start_magnetization) * signRho *
                          gradRhoZValueAtQuadPt;
                        gradRhoInValuesSpinPolarizedPtr[6 * q + 3] =
                          (0.5 + dftParameters::start_magnetization) * signRho *
                          gradRhoXValueAtQuadPt;
                        gradRhoInValuesSpinPolarizedPtr[6 * q + 4] =
                          (0.5 + dftParameters::start_magnetization) * signRho *
                          gradRhoYValueAtQuadPt;
                        gradRhoInValuesSpinPolarizedPtr[6 * q + 5] =
                          (0.5 + dftParameters::start_magnetization) * signRho *
                          gradRhoZValueAtQuadPt;
                      }
                  }
              }
          }
      }

    normalizeRhoInQuadValues();
  }
  //
  computing_timer.exit_section("initialize density");
}

//
//
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::computeRhoInitialGuessFromPSI(
  std::vector<std::vector<distributedCPUVec<double>>> eigenVectors)

{
  computing_timer.enter_section("initialize density");

  // clear existing data
  clearRhoData();

  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3> fe_values(FEEigen, quadrature, update_values | update_gradients);
  const unsigned int num_quad_points = quadrature.size();

  // Initialize electron density table storage

  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());
  if (dftParameters::spinPolarized == 1)
    {
      rhoInValsSpinPolarized.push_back(
        std::map<dealii::CellId, std::vector<double>>());
      rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
    }

  if (dftParameters::xcFamilyType == "GGA")
    {
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
      //
      if (dftParameters::spinPolarized == 1)
        {
          gradRhoInValsSpinPolarized.push_back(
            std::map<dealii::CellId, std::vector<double>>());
          gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
        }
    }

  // temp arrays
  std::vector<double> rhoTemp(num_quad_points),
    rhoTempSpinPolarized(2 * num_quad_points), rhoIn(num_quad_points),
    rhoInSpinPolarized(2 * num_quad_points);
  std::vector<double> gradRhoTemp(3 * num_quad_points),
    gradRhoTempSpinPolarized(6 * num_quad_points),
    gradRhoIn(3 * num_quad_points), gradRhoInSpinPolarized(6 * num_quad_points);

  // loop over locally owned elements
  typename DoFHandler<3>::active_cell_iterator cell =
                                                 dofHandlerEigen.begin_active(),
                                               endc = dofHandlerEigen.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);

        (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
        std::fill(rhoTemp.begin(), rhoTemp.end(), 0.0);
        std::fill(rhoIn.begin(), rhoIn.end(), 0.0);
        if (dftParameters::spinPolarized == 1)
          {
            (*rhoInValuesSpinPolarized)[cell->id()] =
              std::vector<double>(2 * num_quad_points);
            std::fill(rhoTempSpinPolarized.begin(),
                      rhoTempSpinPolarized.end(),
                      0.0);
          }

#ifdef USE_COMPLEX
        std::vector<Vector<double>> tempPsi(num_quad_points),
          tempPsi2(num_quad_points);
        for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
          {
            tempPsi[q_point].reinit(2);
            tempPsi2[q_point].reinit(2);
          }
#else
        std::vector<double> tempPsi(num_quad_points), tempPsi2(num_quad_points);
#endif



        if (dftParameters::xcFamilyType == "GGA") // GGA
          {
            (*gradRhoInValues)[cell->id()] =
              std::vector<double>(3 * num_quad_points);
            std::fill(gradRhoTemp.begin(), gradRhoTemp.end(), 0.0);
            if (dftParameters::spinPolarized == 1)
              {
                (*gradRhoInValuesSpinPolarized)[cell->id()] =
                  std::vector<double>(6 * num_quad_points);
                std::fill(gradRhoTempSpinPolarized.begin(),
                          gradRhoTempSpinPolarized.end(),
                          0.0);
              }
#ifdef USE_COMPLEX
            std::vector<std::vector<Tensor<1, 3, double>>> tempGradPsi(
              num_quad_points),
              tempGradPsi2(num_quad_points);
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                tempGradPsi[q_point].resize(2);
                tempGradPsi2[q_point].resize(2);
              }
#else
            std::vector<Tensor<1, 3, double>> tempGradPsi(num_quad_points),
              tempGradPsi2(num_quad_points);
#endif


            for (int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
              {
                for (unsigned int i = 0; i < d_numEigenValues; ++i)
                  {
                    fe_values.get_function_values(
                      eigenVectors[(1 + dftParameters::spinPolarized) * kPoint]
                                  [i],
                      tempPsi);
                    if (dftParameters::spinPolarized == 1)
                      fe_values.get_function_values(
                        eigenVectors[(1 + dftParameters::spinPolarized) *
                                       kPoint +
                                     1][i],
                        tempPsi2);
                    //
                    fe_values.get_function_gradients(
                      eigenVectors[(1 + dftParameters::spinPolarized) * kPoint]
                                  [i],
                      tempGradPsi);
                    if (dftParameters::spinPolarized == 1)
                      fe_values.get_function_gradients(
                        eigenVectors[(1 + dftParameters::spinPolarized) *
                                       kPoint +
                                     1][i],
                        tempGradPsi2);

                    for (unsigned int q_point = 0; q_point < num_quad_points;
                         ++q_point)
                      {
                        double factor = (eigenValues[kPoint][i] - fermiEnergy) /
                                        (C_kb * dftParameters::TVal);
                        double partialOccupancy =
                          (factor >= 0) ?
                            std::exp(-factor) / (1.0 + std::exp(-factor)) :
                            1.0 / (1.0 + std::exp(factor));
                        //
                        factor = (eigenValues[kPoint]
                                             [i + dftParameters::spinPolarized *
                                                    d_numEigenValues] -
                                  fermiEnergy) /
                                 (C_kb * dftParameters::TVal);
                        double partialOccupancy2 =
                          (factor >= 0) ?
                            std::exp(-factor) / (1.0 + std::exp(-factor)) :
                            1.0 / (1.0 + std::exp(factor));
#ifdef USE_COMPLEX
                        if (dftParameters::spinPolarized == 1)
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
                              2.0 * partialOccupancy * d_kPointWeights[kPoint] *
                              (tempPsi[q_point](0) *
                                 tempGradPsi[q_point][0][0] +
                               tempPsi[q_point](1) *
                                 tempGradPsi[q_point][1][0]);
                            gradRhoTempSpinPolarized[6 * q_point + 1] +=
                              2.0 * partialOccupancy * d_kPointWeights[kPoint] *
                              (tempPsi[q_point](0) *
                                 tempGradPsi[q_point][0][1] +
                               tempPsi[q_point](1) *
                                 tempGradPsi[q_point][1][1]);
                            gradRhoTempSpinPolarized[6 * q_point + 2] +=
                              2.0 * partialOccupancy * d_kPointWeights[kPoint] *
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
                              2.0 * partialOccupancy * d_kPointWeights[kPoint] *
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
                        if (dftParameters::spinPolarized == 1)
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
                              2.0 * 2.0 * partialOccupancy * tempPsi[q_point] *
                              tempGradPsi[q_point][0];
                            gradRhoTemp[3 * q_point + 1] +=
                              2.0 * 2.0 * partialOccupancy * tempPsi[q_point] *
                              tempGradPsi[q_point][1];
                            gradRhoTemp[3 * q_point + 2] +=
                              2.0 * 2.0 * partialOccupancy * tempPsi[q_point] *
                              tempGradPsi[q_point][2];
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
            if (dftParameters::spinPolarized == 1)
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


            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                if (dftParameters::spinPolarized == 1)
                  {
                    (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
                      rhoInSpinPolarized[2 * q_point];
                    (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                      rhoInSpinPolarized[2 * q_point + 1];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                0] =
                      gradRhoInSpinPolarized[6 * q_point + 0];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                1] =
                      gradRhoInSpinPolarized[6 * q_point + 1];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                2] =
                      gradRhoInSpinPolarized[6 * q_point + 2];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                3] =
                      gradRhoInSpinPolarized[6 * q_point + 3];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                4] =
                      gradRhoInSpinPolarized[6 * q_point + 4];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                5] =
                      gradRhoInSpinPolarized[6 * q_point + 5];
                    //
                    (*rhoInValues)[cell->id()][q_point] =
                      rhoInSpinPolarized[2 * q_point] +
                      rhoInSpinPolarized[2 * q_point + 1];
                    (*gradRhoInValues)[cell->id()][3 * q_point + 0] =
                      gradRhoInSpinPolarized[6 * q_point + 0] +
                      gradRhoInSpinPolarized[6 * q_point + 3];
                    (*gradRhoInValues)[cell->id()][3 * q_point + 1] =
                      gradRhoInSpinPolarized[6 * q_point + 1] +
                      gradRhoInSpinPolarized[6 * q_point + 4];
                    (*gradRhoInValues)[cell->id()][3 * q_point + 2] =
                      gradRhoInSpinPolarized[6 * q_point + 2] +
                      gradRhoInSpinPolarized[6 * q_point + 5];
                  }
                else
                  {
                    (*rhoInValues)[cell->id()][q_point] = rhoIn[q_point];
                    (*gradRhoInValues)[cell->id()][3 * q_point + 0] =
                      gradRhoIn[3 * q_point + 0];
                    (*gradRhoInValues)[cell->id()][3 * q_point + 1] =
                      gradRhoIn[3 * q_point + 1];
                    (*gradRhoInValues)[cell->id()][3 * q_point + 2] =
                      gradRhoIn[3 * q_point + 2];
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
                      eigenVectors[(1 + dftParameters::spinPolarized) * kPoint]
                                  [i],
                      tempPsi);
                    if (dftParameters::spinPolarized == 1)
                      fe_values.get_function_values(
                        eigenVectors[(1 + dftParameters::spinPolarized) *
                                       kPoint +
                                     1][i],
                        tempPsi2);

                    for (unsigned int q_point = 0; q_point < num_quad_points;
                         ++q_point)
                      {
                        double factor = (eigenValues[kPoint][i] - fermiEnergy) /
                                        (C_kb * dftParameters::TVal);
                        double partialOccupancy =
                          (factor >= 0) ?
                            std::exp(-factor) / (1.0 + std::exp(-factor)) :
                            1.0 / (1.0 + std::exp(factor));
                        //
                        factor = (eigenValues[kPoint]
                                             [i + dftParameters::spinPolarized *
                                                    d_numEigenValues] -
                                  fermiEnergy) /
                                 (C_kb * dftParameters::TVal);
                        double partialOccupancy2 =
                          (factor >= 0) ?
                            std::exp(-factor) / (1.0 + std::exp(-factor)) :
                            1.0 / (1.0 + std::exp(factor));
#ifdef USE_COMPLEX
                        if (dftParameters::spinPolarized == 1)
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
                        if (dftParameters::spinPolarized == 1)
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
                            tempPsi[q_point]; // std::pow(tempPsi[q_point],2.0);
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
            if (dftParameters::spinPolarized == 1)
              MPI_Allreduce(&rhoTempSpinPolarized[0],
                            &rhoInSpinPolarized[0],
                            2 * numPoint,
                            MPI_DOUBLE,
                            MPI_SUM,
                            interpoolcomm);
            //
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                if (dftParameters::spinPolarized == 1)
                  {
                    (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
                      rhoInSpinPolarized[2 * q_point];
                    (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                      rhoInSpinPolarized[2 * q_point + 1];
                    (*rhoInValues)[cell->id()][q_point] =
                      rhoInSpinPolarized[2 * q_point] +
                      rhoInSpinPolarized[2 * q_point + 1];
                  }
                else
                  (*rhoInValues)[cell->id()][q_point] = rhoIn[q_point];
              }
          }
      }

  normalizeRhoInQuadValues();
  //
  computing_timer.exit_section("initialize density");
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::computeNodalRhoFromQuadData()
{
  //
  // compute nodal electron-density from cell quadrature data
  //
  matrix_free_data.initialize_dof_vector(d_rhoNodalField,
                                         d_densityDofHandlerIndex);
  d_rhoNodalField = 0;

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
    dofHandler,
    constraintsNone,
    matrix_free_data.get_quadrature(d_densityQuadratureId),
    funcRho,
    d_rhoNodalField);


  d_rhoNodalField.update_ghost_values();


  if (dftParameters::spinPolarized == 1)
    {
      matrix_free_data.initialize_dof_vector(d_rhoNodalFieldSpin0,
                                             d_densityDofHandlerIndex);
      d_rhoNodalFieldSpin0 = 0;

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
        dofHandler,
        constraintsNone,
        matrix_free_data.get_quadrature(d_densityQuadratureId),
        funcRhoSpin0,
        d_rhoNodalFieldSpin0);


      d_rhoNodalFieldSpin0.update_ghost_values();

      matrix_free_data.initialize_dof_vector(d_rhoNodalFieldSpin1,
                                             d_densityDofHandlerIndex);
      d_rhoNodalFieldSpin1 = 0;

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
        dofHandler,
        constraintsNone,
        matrix_free_data.get_quadrature(d_densityQuadratureId),
        funcRhoSpin1,
        d_rhoNodalFieldSpin1);

      d_rhoNodalFieldSpin1.update_ghost_values();
    }
}


//
// Normalize rho
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::normalizeRhoInQuadValues()
{
  const Quadrature<3> &quadrature_formula =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  const unsigned int n_q_points = quadrature_formula.size();

  const double charge  = totalCharge(d_dofHandlerRhoNodal, rhoInValues);
  const double scaling = ((double)numElectrons) / charge;

  if (dftParameters::verbosity >= 2)
    pcout << "initial total charge before normalizing to number of electrons: "
          << charge << std::endl;

  // scaling rho
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              (*rhoInValues)[cell->id()][q] *= scaling;

              if (dftParameters::xcFamilyType == "GGA")
                for (unsigned int idim = 0; idim < 3; ++idim)
                  (*gradRhoInValues)[cell->id()][3 * q + idim] *= scaling;
              if (dftParameters::spinPolarized == 1)
                {
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q + 1] *= scaling;
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q] *= scaling;
                  if (dftParameters::xcFamilyType == "GGA")
                    for (unsigned int idim = 0; idim < 3; ++idim)
                      {
                        (*gradRhoInValuesSpinPolarized)[cell->id()]
                                                       [6 * q + idim] *=
                          scaling;
                        (*gradRhoInValuesSpinPolarized)[cell->id()]
                                                       [6 * q + 3 + idim] *=
                          scaling;
                      }
                }
            }
        }
    }
  double chargeAfterScaling = totalCharge(d_dofHandlerRhoNodal, rhoInValues);

  if (dftParameters::verbosity >= 1)
    pcout << "Initial total charge: " << chargeAfterScaling << std::endl;
}

//
// Normalize rho
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::normalizeRhoOutQuadValues()
{
  const Quadrature<3> &quadrature_formula =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  const unsigned int n_q_points = quadrature_formula.size();

  const double charge  = totalCharge(d_dofHandlerRhoNodal, rhoOutValues);
  const double scaling = ((double)numElectrons) / charge;

  if (dftParameters::verbosity >= 2)
    pcout << "Total charge out before normalizing to number of electrons: "
          << charge << std::endl;

  // scaling rho
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              (*rhoOutValues)[cell->id()][q] *= scaling;

              if (dftParameters::xcFamilyType == "GGA")
                for (unsigned int idim = 0; idim < 3; ++idim)
                  (*gradRhoOutValues)[cell->id()][3 * q + idim] *= scaling;
              if (dftParameters::spinPolarized == 1)
                {
                  (*rhoOutValuesSpinPolarized)[cell->id()][2 * q + 1] *=
                    scaling;
                  (*rhoOutValuesSpinPolarized)[cell->id()][2 * q] *= scaling;
                  if (dftParameters::xcFamilyType == "GGA")
                    for (unsigned int idim = 0; idim < 3; ++idim)
                      {
                        (*gradRhoOutValuesSpinPolarized)[cell->id()]
                                                        [6 * q + idim] *=
                          scaling;
                        (*gradRhoOutValuesSpinPolarized)[cell->id()]
                                                        [6 * q + 3 + idim] *=
                          scaling;
                      }
                }
            }
        }
    }
  double chargeAfterScaling = totalCharge(d_dofHandlerRhoNodal, rhoOutValues);

  if (dftParameters::verbosity >= 1)
    pcout << "Total charge out after scaling: " << chargeAfterScaling
          << std::endl;
}
