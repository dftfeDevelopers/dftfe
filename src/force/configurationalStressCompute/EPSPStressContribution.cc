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

// compute EPSP contribution stress (local pseudopotential)
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::addEPSPStressContribution(
  FEValues<3> &    feValues,
  FEFaceValues<3> &feFaceValues,
  FEEvaluation<3, 1, C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(), 3>
    &                          forceEval,
  const MatrixFree<3, double> &matrixFreeData,
  const unsigned int           phiTotDofHandlerIndexElectro,
  const unsigned int           cell,
  const dealii::AlignedVector<VectorizedArray<double>> &rhoQuads,
  const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
    &gradRhoQuads,
  const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    &                                              pseudoVLocAtoms,
  const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManager,
  const std::vector<std::map<dealii::CellId, unsigned int>>
    &cellsVselfBallsClosestAtomIdDofHandler)
{
  Tensor<1, 3, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);

  Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  Tensor<2, 3, double> zeroTensor2Nonvect;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      zeroTensor2Nonvect[idim][jdim] = 0.0;

  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  const unsigned int dofs_per_cell =
    matrixFreeData.get_dof_handler(0).get_fe().dofs_per_cell;

  const unsigned int faces_per_cell    = GeometryInfo<3>::faces_per_cell;
  const unsigned int numFaceQuadPoints = feFaceValues.get_quadrature().size();

  dealii::AlignedVector<Tensor<2, 3, double>> surfaceIntegralSubcells(
    numSubCells);
  std::vector<double> rhoFaceQuads(numFaceQuadPoints);
  dealii::AlignedVector<VectorizedArray<double>> vselfQuads(
    numQuadPoints, make_vectorized_array(0.0));
  dealii::AlignedVector<VectorizedArray<double>> pseudoVLocAtomsQuads(
    numQuadPoints, make_vectorized_array(0.0));
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>> gradVselfQuads(
    numQuadPoints, zeroTensor1);
  dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
    totalContribution(numQuadPoints, zeroTensor2);
  dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
                                             vselfFDStrainQuads(numQuadPoints, zeroTensor2);
  std::vector<std::vector<dealii::Point<3>>> quadPointsSubCells(
    numSubCells, std::vector<dealii::Point<3>>(numQuadPoints));

  DoFHandler<3>::active_cell_iterator subCellPtr;

  for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
    {
      subCellPtr =
        matrixFreeData.get_cell_iterator(cell,
                                         iSubCell,
                                         phiTotDofHandlerIndexElectro);
      feValues.reinit(subCellPtr);

      std::vector<dealii::Point<3>> &temp = quadPointsSubCells[iSubCell];
      for (unsigned int q = 0; q < numQuadPoints; ++q)
        temp[q] = feValues.quadrature_point(q);
    }


  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      bool isLocalDomainOutsideVselfBall = false;
      bool isLocalDomainOutsidePspTail   = false;
      if (pseudoVLocAtoms.find(iAtom) == pseudoVLocAtoms.end())
        isLocalDomainOutsidePspTail = true;

      double       atomCharge;
      unsigned int atomId = iAtom;
      Point<3>     atomLocation;
      if (iAtom < numberGlobalAtoms)
        {
          atomLocation[0] = dftPtr->atomLocations[iAtom][2];
          atomLocation[1] = dftPtr->atomLocations[iAtom][3];
          atomLocation[2] = dftPtr->atomLocations[iAtom][4];
          if (dftParameters::isPseudopotential)
            atomCharge = dftPtr->atomLocations[iAtom][1];
          else
            atomCharge = dftPtr->atomLocations[iAtom][0];
        }
      else
        {
          const int imageId = iAtom - numberGlobalAtoms;
          atomId            = dftPtr->d_imageIdsTrunc[imageId];
          atomCharge        = dftPtr->d_imageChargesTrunc[imageId];
          atomLocation[0]   = dftPtr->d_imagePositionsTrunc[imageId][0];
          atomLocation[1]   = dftPtr->d_imagePositionsTrunc[imageId][1];
          atomLocation[2]   = dftPtr->d_imagePositionsTrunc[imageId][2];
        }

      unsigned int                                         binIdiAtom;
      std::map<unsigned int, unsigned int>::const_iterator it1 =
        vselfBinsManager.getAtomIdBinIdMapLocalAllImages().find(atomId);
      if (it1 == vselfBinsManager.getAtomIdBinIdMapLocalAllImages().end())
        isLocalDomainOutsideVselfBall = true;
      else
        binIdiAtom = it1->second;

      // Assuming psp tail is larger than vself ball
      if (isLocalDomainOutsidePspTail && isLocalDomainOutsideVselfBall)
        continue;

      std::fill(surfaceIntegralSubcells.begin(),
                surfaceIntegralSubcells.end(),
                zeroTensor2Nonvect);
      std::fill(vselfQuads.begin(),
                vselfQuads.end(),
                make_vectorized_array(0.0));
      std::fill(pseudoVLocAtomsQuads.begin(),
                pseudoVLocAtomsQuads.end(),
                make_vectorized_array(0.0));
      std::fill(gradVselfQuads.begin(), gradVselfQuads.end(), zeroTensor1);
      std::fill(totalContribution.begin(),
                totalContribution.end(),
                zeroTensor2);
      std::fill(vselfFDStrainQuads.begin(),
                vselfFDStrainQuads.end(),
                zeroTensor2);

      bool isTrivial = true;
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr =
            matrixFreeData.get_cell_iterator(cell,
                                             iSubCell,
                                             phiTotDofHandlerIndexElectro);
          dealii::CellId subCellId = subCellPtr->id();


          // get derivative R vself for iAtom
          bool isCellOutsideVselfBall = true;
          if (!isLocalDomainOutsideVselfBall)
            {
              std::map<dealii::CellId, unsigned int>::const_iterator it2 =
                cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].find(
                  subCellId);
              if (it2 !=
                  cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].end())
                {
                  Point<3>           closestAtomLocation;
                  const unsigned int closestAtomId = it2->second;
                  if (it2->second >= numberGlobalAtoms)
                    {
                      const unsigned int imageIdTrunc =
                        closestAtomId - numberGlobalAtoms;
                      closestAtomLocation[0] =
                        dftPtr->d_imagePositionsTrunc[imageIdTrunc][0];
                      closestAtomLocation[1] =
                        dftPtr->d_imagePositionsTrunc[imageIdTrunc][1];
                      closestAtomLocation[2] =
                        dftPtr->d_imagePositionsTrunc[imageIdTrunc][2];
                    }
                  else
                    {
                      closestAtomLocation[0] =
                        dftPtr->atomLocations[closestAtomId][2];
                      closestAtomLocation[1] =
                        dftPtr->atomLocations[closestAtomId][3];
                      closestAtomLocation[2] =
                        dftPtr->atomLocations[closestAtomId][4];
                    }

                  if (atomLocation.distance(closestAtomLocation) < 1e-5)
                    {
                      feValues.reinit(subCellPtr);
                      isCellOutsideVselfBall = false;

                      std::vector<double> vselfFDStrainQuadsSubCell(
                        numQuadPoints);
                      unsigned int flattenedIdCount = 0;
                      for (unsigned int idim = 0; idim < 3; ++idim)
                        for (unsigned int jdim = 0; jdim <= idim; jdim++)
                          {
                            feValues.get_function_values(
                              dftPtr->d_vselfFieldGateauxDerStrainFDBins
                                [6 * binIdiAtom + flattenedIdCount],
                              vselfFDStrainQuadsSubCell);
                            for (unsigned int q = 0; q < numQuadPoints; ++q)
                              {
                                vselfFDStrainQuads[q][idim][jdim][iSubCell] =
                                  vselfFDStrainQuadsSubCell[q];
                                vselfFDStrainQuads[q][jdim][idim][iSubCell] =
                                  vselfFDStrainQuadsSubCell[q];
                              }

                            flattenedIdCount += 1;
                          }

                      std::vector<double> vselfQuadsSubCell(numQuadPoints);
                      feValues.get_function_values(
                        vselfBinsManager.getVselfFieldBins()[binIdiAtom],
                        vselfQuadsSubCell);
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        vselfQuads[q][iSubCell] = vselfQuadsSubCell[q];

                      std::vector<Tensor<1, 3, double>> gradVselfQuadsSubCell(
                        numQuadPoints);
                      feValues.get_function_gradients(
                        vselfBinsManager.getVselfFieldBins()[binIdiAtom],
                        gradVselfQuadsSubCell);
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          gradVselfQuads[q][0][iSubCell] =
                            gradVselfQuadsSubCell[q][0];
                          gradVselfQuads[q][1][iSubCell] =
                            gradVselfQuadsSubCell[q][1];
                          gradVselfQuads[q][2][iSubCell] =
                            gradVselfQuadsSubCell[q][2];
                        }
                    }
                }
            }

          // get grad pseudo VLoc for iAtom
          bool isCellOutsidePspTail = true;
          if (!isLocalDomainOutsidePspTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator it =
                pseudoVLocAtoms.find(iAtom)->second.find(subCellId);
              if (it != pseudoVLocAtoms.find(iAtom)->second.end())
                {
                  isCellOutsidePspTail = false;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    pseudoVLocAtomsQuads[q][iSubCell] = (it->second)[q];
                }
            }
          else if (!isCellOutsideVselfBall)
            {
              std::vector<dealii::Point<3>> &temp =
                quadPointsSubCells[iSubCell];
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  Tensor<1, 3, double> dispAtom     = temp[q] - atomLocation;
                  const double         dist         = dispAtom.norm();
                  pseudoVLocAtomsQuads[q][iSubCell] = -atomCharge / dist;
                }
            }

          if (isCellOutsideVselfBall && !isCellOutsidePspTail)
            {
              std::vector<dealii::Point<3>> &temp =
                quadPointsSubCells[iSubCell];
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  Tensor<1, 3, double> dispAtom = temp[q] - atomLocation;
                  const double         dist     = dispAtom.norm();
                  vselfQuads[q][iSubCell]       = -atomCharge / dist;
                }
            }

          if (!isCellOutsideVselfBall)
            {
              Tensor<2, 3, double> &surfaceIntegral =
                surfaceIntegralSubcells[iSubCell];

              const std::map<DoFHandler<3>::active_cell_iterator,
                             std::vector<unsigned int>>
                &cellsVselfBallSurfacesDofHandler =
                  d_cellFacesVselfBallSurfacesDofHandlerElectro[binIdiAtom];

              if (cellsVselfBallSurfacesDofHandler.find(subCellPtr) !=
                  cellsVselfBallSurfacesDofHandler.end())
                {
                  const std::vector<unsigned int> &dirichletFaceIds =
                    cellsVselfBallSurfacesDofHandler.find(subCellPtr)->second;
                  for (unsigned int index = 0; index < dirichletFaceIds.size();
                       index++)
                    {
                      const unsigned int faceId = dirichletFaceIds[index];

                      feFaceValues.reinit(
                        d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro
                          .find(subCellId)
                          ->second,
                        faceId);
                      feFaceValues.get_function_values(
                        dftPtr->d_rhoOutNodalValuesDistributed, rhoFaceQuads);
                      for (unsigned int qPoint = 0; qPoint < numFaceQuadPoints;
                           ++qPoint)
                        {
                          const Point<3> quadPoint =
                            feFaceValues.quadrature_point(qPoint);
                          const Tensor<1, 3, double> dispClosestAtom =
                            quadPoint - atomLocation;
                          const double dist = dispClosestAtom.norm();
                          const double vselfFaceQuadExact = -atomCharge / dist;

                          // note sign of normal vector which is outer to the
                          // vself ball, but for applying the surface integral
                          // requires it to be opposite
                          surfaceIntegral +=
                            outer_product(rhoFaceQuads[qPoint] *
                                            vselfFaceQuadExact *
                                            feFaceValues.normal_vector(qPoint),
                                          dispClosestAtom) *
                            feFaceValues.JxW(qPoint);
                        } // q point loop
                    }     // face loop
                }         // surface cells
            }             // inside or intersecting vself ball

          if (isCellOutsideVselfBall && !isCellOutsidePspTail)
            {
              isTrivial = false;

              std::vector<dealii::Point<3>> &quadPointsSubCell =
                quadPointsSubCells[iSubCell];

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  const Tensor<1, 3, double> dispAtom =
                    quadPointsSubCell[q] - atomLocation;

                  for (unsigned int idim = 0; idim < 3; idim++)
                    for (unsigned int jdim = 0; jdim < 3; jdim++)
                      totalContribution[q][idim][jdim][iSubCell] =
                        (gradRhoQuads[q][idim][iSubCell] *
                           vselfQuads[q][iSubCell] -
                         gradRhoQuads[q][idim][iSubCell] *
                           pseudoVLocAtomsQuads[q][iSubCell]) *
                        dispAtom[jdim];
                }
            }
          else if (!isCellOutsideVselfBall)
            {
              isTrivial = false;

              std::vector<dealii::Point<3>> &quadPointsSubCell =
                quadPointsSubCells[iSubCell];

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  const Tensor<1, 3, double> dispAtom =
                    quadPointsSubCell[q] - atomLocation;

                  for (unsigned int idim = 0; idim < 3; idim++)
                    for (unsigned int jdim = 0; jdim < 3; jdim++)
                      {
                        totalContribution[q][idim][jdim][iSubCell] =
                          -rhoQuads[q][iSubCell] *
                            vselfFDStrainQuads[q][idim][jdim][iSubCell] -
                          gradRhoQuads[q][idim][iSubCell] *
                            pseudoVLocAtomsQuads[q][iSubCell] * dispAtom[jdim];

                        if (idim == jdim)
                          {
                            totalContribution[q][idim][jdim][iSubCell] -=
                              rhoQuads[q][iSubCell] * (vselfQuads[q][iSubCell]);
                          }
                      }
                }
            }
        } // subCell loop

      if (isTrivial)
        continue;

      Tensor<2, 3, VectorizedArray<double>> EPSPStressContribution;
      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          EPSPStressContribution[idim][jdim] = make_vectorized_array(0.0);

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        {
          EPSPStressContribution += totalContribution[q] * forceEval.JxW(q);
        }

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            d_stress[idim][jdim] +=
              EPSPStressContribution[idim][jdim][iSubCell] +
              surfaceIntegralSubcells[iSubCell][idim][jdim];
    } // iAtom loop
}
