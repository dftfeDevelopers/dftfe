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

// compute nonlinear core correction contribution to stress
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::
  addENonlinearCoreCorrectionStressContribution(
    FEEvaluation<3,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 3> &                                     forceEval,
    const MatrixFree<3, double> &                         matrixFreeData,
    const unsigned int                                    cell,
    const dealii::AlignedVector<VectorizedArray<double>> &vxcQuads,
    const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
      &derExcGradRho,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &hessianRhoCoreAtoms)
{
  Tensor<1, 3, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);

  Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  DoFHandler<3>::active_cell_iterator subCellPtr;

  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>> xMinusAtomLoc(
    numQuadPoints, zeroTensor1);


  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
        gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);
      dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
        hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor2);

      double       atomCharge;
      unsigned int atomId = iAtom;
      Point<3>     atomLocation;
      if (iAtom < numberGlobalAtoms)
        {
          atomLocation[0] = dftPtr->atomLocations[iAtom][2];
          atomLocation[1] = dftPtr->atomLocations[iAtom][3];
          atomLocation[2] = dftPtr->atomLocations[iAtom][4];
          if (d_dftParams.isPseudopotential)
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

      bool isLocalDomainOutsideCoreRhoTail = false;
      if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
        isLocalDomainOutsideCoreRhoTail = true;

      if (isLocalDomainOutsideCoreRhoTail)
        continue;

      bool isCellOutsideCoreRhoTail = true;

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();

          std::map<dealii::CellId, std::vector<double>>::const_iterator it =
            gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
          if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
            {
              isCellOutsideCoreRhoTail        = false;
              const std::vector<double> &temp = it->second;
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  gradRhoCoreAtomsQuads[q][0][iSubCell] = temp[q * 3];
                  gradRhoCoreAtomsQuads[q][1][iSubCell] = temp[q * 3 + 1];
                  gradRhoCoreAtomsQuads[q][2][iSubCell] = temp[q * 3 + 2];
                }
            }

          if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA && !isCellOutsideCoreRhoTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator
                it2 = hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

              if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                {
                  const std::vector<double> &temp = it2->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        for (unsigned int jDim = 0; jDim < 3; ++jDim)
                          hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] =
                            temp[q * 3 * 3 + 3 * iDim + jDim];
                    }
                }
            }

          if (!isCellOutsideCoreRhoTail)
            for (unsigned int q = 0; q < numQuadPoints; ++q)
              {
                const Point<3, VectorizedArray<double>> &quadPointVectorized =
                  forceEval.quadrature_point(q);
                Point<3> quadPoint;
                quadPoint[0] = quadPointVectorized[0][iSubCell];
                quadPoint[1] = quadPointVectorized[1][iSubCell];
                quadPoint[2] = quadPointVectorized[2][iSubCell];
                const Tensor<1, 3, double> dispAtom = quadPoint - atomLocation;
                for (unsigned int idim = 0; idim < 3; idim++)
                  {
                    xMinusAtomLoc[q][idim][iSubCell] = dispAtom[idim];
                  }
              }
        } // subCell loop

      if (isCellOutsideCoreRhoTail)
        continue;


      Tensor<2, 3, VectorizedArray<double>> stressContribution = zeroTensor2;

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        {
          stressContribution +=
            outer_product(eshelbyTensor::getFNonlinearCoreCorrection(
                            vxcQuads[q], gradRhoCoreAtomsQuads[q]) +
                            eshelbyTensor::getFNonlinearCoreCorrection(
                              derExcGradRho[q], hessianRhoCoreAtomsQuads[q]),
                          xMinusAtomLoc[q]) *
            forceEval.JxW(q);
        }

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            d_stress[idim][jdim] += stressContribution[idim][jdim][iSubCell];
    } // iAtom loop
}

// compute nonlinear core correction contribution to stress
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::
  addENonlinearCoreCorrectionStressContributionSpinPolarized(
    FEEvaluation<3,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 3> &                                     forceEval,
    const MatrixFree<3, double> &                         matrixFreeData,
    const unsigned int                                    cell,
    const dealii::AlignedVector<VectorizedArray<double>> &vxcQuadsSpin0,
    const dealii::AlignedVector<VectorizedArray<double>> &vxcQuadsSpin1,
    const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
      &derExcGradRhoSpin0,
    const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
      &derExcGradRhoSpin1,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &        hessianRhoCoreAtoms,
    const bool isXCGGA)
{
  Tensor<1, 3, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);

  Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  DoFHandler<3>::active_cell_iterator subCellPtr;

  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>> xMinusAtomLoc(
    numQuadPoints, zeroTensor1);


  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
        gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);
      dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
        hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor2);

      double       atomCharge;
      unsigned int atomId = iAtom;
      Point<3>     atomLocation;
      if (iAtom < numberGlobalAtoms)
        {
          atomLocation[0] = dftPtr->atomLocations[iAtom][2];
          atomLocation[1] = dftPtr->atomLocations[iAtom][3];
          atomLocation[2] = dftPtr->atomLocations[iAtom][4];
          if (d_dftParams.isPseudopotential)
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

      bool isLocalDomainOutsideCoreRhoTail = false;
      if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
        isLocalDomainOutsideCoreRhoTail = true;

      if (isLocalDomainOutsideCoreRhoTail)
        continue;

      bool isCellOutsideCoreRhoTail = true;

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();

          std::map<dealii::CellId, std::vector<double>>::const_iterator it =
            gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
          if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
            {
              isCellOutsideCoreRhoTail        = false;
              const std::vector<double> &temp = it->second;
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  gradRhoCoreAtomsQuads[q][0][iSubCell] = temp[q * 3] / 2.0;
                  gradRhoCoreAtomsQuads[q][1][iSubCell] = temp[q * 3 + 1] / 2.0;
                  gradRhoCoreAtomsQuads[q][2][iSubCell] = temp[q * 3 + 2] / 2.0;
                }
            }

          if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() == densityFamilyType::GGA && !isCellOutsideCoreRhoTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator
                it2 = hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

              if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                {
                  const std::vector<double> &temp2 = it2->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        for (unsigned int jDim = 0; jDim < 3; ++jDim)
                          hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] =
                            temp2[q * 3 * 3 + 3 * iDim + jDim] / 2.0;
                    }
                }
            }

          if (!isCellOutsideCoreRhoTail)
            for (unsigned int q = 0; q < numQuadPoints; ++q)
              {
                const Point<3, VectorizedArray<double>> &quadPointVectorized =
                  forceEval.quadrature_point(q);
                Point<3> quadPoint;
                quadPoint[0] = quadPointVectorized[0][iSubCell];
                quadPoint[1] = quadPointVectorized[1][iSubCell];
                quadPoint[2] = quadPointVectorized[2][iSubCell];
                const Tensor<1, 3, double> dispAtom = quadPoint - atomLocation;
                for (unsigned int idim = 0; idim < 3; idim++)
                  {
                    xMinusAtomLoc[q][idim][iSubCell] = dispAtom[idim];
                  }
              }
        } // subCell loop

      if (isCellOutsideCoreRhoTail)
        continue;


      Tensor<2, 3, VectorizedArray<double>> stressContribution = zeroTensor2;

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        stressContribution +=
          outer_product(eshelbyTensorSP::getFNonlinearCoreCorrection(
                          vxcQuadsSpin0[q],
                          vxcQuadsSpin1[q],
                          derExcGradRhoSpin0[q],
                          derExcGradRhoSpin1[q],
                          gradRhoCoreAtomsQuads[q],
                          hessianRhoCoreAtomsQuads[q],
                          isXCGGA),
                        xMinusAtomLoc[q]) *
          forceEval.JxW(q);

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            d_stress[idim][jdim] += stressContribution[idim][jdim][iSubCell];
    } // iAtom loop
}
