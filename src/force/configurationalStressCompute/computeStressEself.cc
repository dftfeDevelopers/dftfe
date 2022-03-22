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

// compute stress contribution from nuclear self energy
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::computeStressEself(
  const DoFHandler<3> &                            dofHandlerElectro,
  const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
  const MatrixFree<3, double> &                    matrixFreeDataElectro,
  const unsigned int                               smearedChargeQuadratureId)
{
#ifdef DEBUG
  double               dummyTest = 0;
  Tensor<1, 3, double> dummyVec;
  Tensor<2, 3, double> dummyTensor;
#endif
  const std::vector<std::vector<double>> &atomLocations = dftPtr->atomLocations;
  const std::vector<std::vector<double>> &imagePositions =
    dftPtr->d_imagePositionsTrunc;
  const std::vector<double> &imageCharges      = dftPtr->d_imageChargesTrunc;
  const unsigned int         numberGlobalAtoms = atomLocations.size();
  //
  // First add configurational stress contribution from the volume integral
  //
  QGauss<3> quadrature(
    C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>());
  FEValues<3>        feVselfValues(dofHandlerElectro.get_fe(),
                            quadrature,
                            update_gradients | update_JxW_values);
  const unsigned int numQuadPoints = quadrature.size();
  const unsigned int numberBins =
    vselfBinsManagerElectro.getAtomIdsBins().size();

  std::vector<Tensor<1, 3, double>> gradVselfQuad(numQuadPoints);

  for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
    {
      const std::vector<DoFHandler<3>::active_cell_iterator>
        &cellsVselfBallDofHandler = d_cellsVselfBallsDofHandlerElectro[iBin];
      const distributedCPUVec<double> &iBinVselfField =
        vselfBinsManagerElectro.getVselfFieldBins()[iBin];
      std::vector<DoFHandler<3>::active_cell_iterator>::const_iterator iter1;
      for (iter1 = cellsVselfBallDofHandler.begin();
           iter1 != cellsVselfBallDofHandler.end();
           ++iter1)
        {
          DoFHandler<3>::active_cell_iterator cell = *iter1;
          feVselfValues.reinit(cell);
          feVselfValues.get_function_gradients(iBinVselfField, gradVselfQuad);

          for (unsigned int qPoint = 0; qPoint < numQuadPoints; ++qPoint)
            {
              d_stress += eshelbyTensor::getVselfBallEshelbyTensor(
                            gradVselfQuad[qPoint]) *
                          feVselfValues.JxW(qPoint);
            } // q point loop
        }     // cell loop
    }         // bin loop


  //
  // second add configurational stress contribution from the surface integral
  //
  QGauss<3 - 1> faceQuadrature(
    C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>());
  FEFaceValues<3>    feVselfFaceValues(dofHandlerElectro.get_fe(),
                                    faceQuadrature,
                                    update_gradients | update_JxW_values |
                                      update_normal_vectors |
                                      update_quadrature_points);
  const unsigned int faces_per_cell    = GeometryInfo<3>::faces_per_cell;
  const unsigned int numFaceQuadPoints = faceQuadrature.size();


  for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
    {
      const std::map<DoFHandler<3>::active_cell_iterator,
                     std::vector<unsigned int>>
        &cellsVselfBallSurfacesDofHandler =
          d_cellFacesVselfBallSurfacesDofHandlerElectro[iBin];
      const distributedCPUVec<double> &iBinVselfField =
        vselfBinsManagerElectro.getVselfFieldBins()[iBin];
      std::map<DoFHandler<3>::active_cell_iterator,
               std::vector<unsigned int>>::const_iterator iter1;
      for (iter1 = cellsVselfBallSurfacesDofHandler.begin();
           iter1 != cellsVselfBallSurfacesDofHandler.end();
           ++iter1)
        {
          DoFHandler<3>::active_cell_iterator cell = iter1->first;
          const int                           closestAtomId =
            d_cellsVselfBallsClosestAtomIdDofHandlerElectro[iBin][cell->id()];
          double   closestAtomCharge;
          Point<3> closestAtomLocation;
          if (closestAtomId < numberGlobalAtoms)
            {
              closestAtomLocation[0] = atomLocations[closestAtomId][2];
              closestAtomLocation[1] = atomLocations[closestAtomId][3];
              closestAtomLocation[2] = atomLocations[closestAtomId][4];
              if (dftParameters::isPseudopotential)
                closestAtomCharge = atomLocations[closestAtomId][1];
              else
                closestAtomCharge = atomLocations[closestAtomId][0];
            }
          else
            {
              const int imageId      = closestAtomId - numberGlobalAtoms;
              closestAtomCharge      = imageCharges[imageId];
              closestAtomLocation[0] = imagePositions[imageId][0];
              closestAtomLocation[1] = imagePositions[imageId][1];
              closestAtomLocation[2] = imagePositions[imageId][2];
            }

          const std::vector<unsigned int> &dirichletFaceIds = iter1->second;
          for (unsigned int index = 0; index < dirichletFaceIds.size(); index++)
            {
              const unsigned int faceId = dirichletFaceIds[index];
              feVselfFaceValues.reinit(cell, faceId);

              for (unsigned int qPoint = 0; qPoint < numFaceQuadPoints;
                   ++qPoint)
                {
                  const Point<3> quadPoint =
                    feVselfFaceValues.quadrature_point(qPoint);
                  const Tensor<1, 3, double> dispClosestAtom =
                    quadPoint - closestAtomLocation;
                  const double               dist = dispClosestAtom.norm();
                  const Tensor<1, 3, double> gradVselfFaceQuadExact =
                    closestAtomCharge * dispClosestAtom / dist / dist / dist;
                  d_stress -=
                    outer_product(dispClosestAtom,
                                  eshelbyTensor::getVselfBallEshelbyTensor(
                                    gradVselfFaceQuadExact) *
                                    feVselfFaceValues.normal_vector(qPoint)) *
                    feVselfFaceValues.JxW(qPoint);
#ifdef DEBUG
                  dummyTest +=
                    scalar_product(gradVselfFaceQuadExact,
                                   feVselfFaceValues.normal_vector(qPoint)) *
                    feVselfFaceValues.JxW(qPoint);
                  dummyVec += feVselfFaceValues.normal_vector(qPoint) *
                              feVselfFaceValues.JxW(qPoint);
                  dummyTensor +=
                    outer_product(gradVselfFaceQuadExact,
                                  feVselfFaceValues.normal_vector(qPoint)) *
                    feVselfFaceValues.JxW(qPoint);
#endif

                } // q point loop
            }     // face loop
        }         // cell loop
    }             // bin loop

  //
  // Add stress due to smeared charges
  //
  if (dftParameters::smearedNuclearCharges)
    {
      const std::map<int, std::set<int>> &atomImageIdsBins =
        vselfBinsManagerElectro.getAtomImageIdsBins();

      FEEvaluation<3, -1, 1, 3> forceEvalSmearedCharge(
        matrixFreeDataElectro,
        d_forceDofHandlerIndexElectro,
        smearedChargeQuadratureId);

      DoFHandler<3>::active_cell_iterator subCellPtr;
      const unsigned int                  numQuadPointsSmearedb =
        forceEvalSmearedCharge.n_q_points;

      Tensor<1, 3, VectorizedArray<double>> zeroTensor;
      for (unsigned int idim = 0; idim < 3; idim++)
        {
          zeroTensor[idim] = make_vectorized_array(0.0);
        }

      Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
      for (unsigned int idim = 0; idim < 3; idim++)
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          {
            zeroTensor2[idim][jdim] = make_vectorized_array(0.0);
          }

      dealii::AlignedVector<VectorizedArray<double>> smearedbQuads(
        numQuadPointsSmearedb, make_vectorized_array(0.0));
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
        gradVselfSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);

      for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
        {
          FEEvaluation<3, -1> vselfEvalSmearedCharge(
            matrixFreeDataElectro,
            dftPtr->d_binsStartDofHandlerIndexElectro + 4 * iBin,
            smearedChargeQuadratureId);

          const std::set<int> &atomImageIdsInBin =
            atomImageIdsBins.find(iBin)->second;
          for (unsigned int cell = 0;
               cell < matrixFreeDataElectro.n_macro_cells();
               ++cell)
            {
              std::set<unsigned int>
                                 nonTrivialSmearedChargeAtomImageIdsMacroCell;
              const unsigned int numSubCells =
                matrixFreeDataElectro.n_components_filled(cell);
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr =
                    matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                  dealii::CellId                   subCellId = subCellPtr->id();
                  const std::vector<unsigned int> &temp =
                    dftPtr->d_bCellNonTrivialAtomImageIdsBins[iBin]
                      .find(subCellId)
                      ->second;
                  for (int i = 0; i < temp.size(); i++)
                    nonTrivialSmearedChargeAtomImageIdsMacroCell.insert(
                      temp[i]);
                }

              if (nonTrivialSmearedChargeAtomImageIdsMacroCell.size() == 0)
                continue;

              std::fill(smearedbQuads.begin(),
                        smearedbQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(gradVselfSmearedChargeQuads.begin(),
                        gradVselfSmearedChargeQuads.end(),
                        zeroTensor);

              bool isCellNonTrivial = false;
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr =
                    matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                  dealii::CellId subCellId = subCellPtr->id();

                  const std::vector<int> &bQuadAtomIdsCell =
                    dftPtr->d_bQuadAtomIdsAllAtoms.find(subCellId)->second;
                  const std::vector<double> &bQuadValuesCell =
                    dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;

                  for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                    {
                      if (atomImageIdsInBin.find(bQuadAtomIdsCell[q]) !=
                          atomImageIdsInBin.end())
                        {
                          isCellNonTrivial           = true;
                          smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                        }
                    } // quad loop
                }     // subcell loop

              if (!isCellNonTrivial)
                continue;

              forceEvalSmearedCharge.reinit(cell);
              vselfEvalSmearedCharge.reinit(cell);
              vselfEvalSmearedCharge.read_dof_values_plain(
                vselfBinsManagerElectro.getVselfFieldBins()[iBin]);
              vselfEvalSmearedCharge.evaluate(false, true);

              for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                {
                  gradVselfSmearedChargeQuads[q] =
                    vselfEvalSmearedCharge.get_gradient(q);
                }

              addEVselfSmearedStressContribution(
                forceEvalSmearedCharge,
                matrixFreeDataElectro,
                cell,
                gradVselfSmearedChargeQuads,
                std::vector<unsigned int>(
                  nonTrivialSmearedChargeAtomImageIdsMacroCell.begin(),
                  nonTrivialSmearedChargeAtomImageIdsMacroCell.end()),
                dftPtr->d_bQuadAtomIdsAllAtomsImages,
                smearedbQuads);
            } // macrocell loop
        }     // bin loop
    }
}
