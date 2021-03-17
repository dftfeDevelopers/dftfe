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
// @author Phani Motamarri, Sambit Das (2020)
//

//(locally used function) compute FNonlinearCoreCorrection contibution due to
// Gamma(Rj) for given set of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FNonlinearCoreCorrectionGammaAtomsElementalContribution(
    std::map<unsigned int, std::vector<double>>
      &                  forceContributionFNonlinearCoreCorrectionGammaAtoms,
    FEEvaluation<C_DIM,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 C_DIM> &forceEval,
    const MatrixFree<3, double> &               matrixFreeData,
    const unsigned int                          cell,
    const std::vector<VectorizedArray<double>> &vxcQuads,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms)
{
  Tensor<1, C_DIM, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);
  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
        gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);

      unsigned int atomId = iAtom;
      if (iAtom >= numberGlobalAtoms)
        {
          const int imageId = iAtom - numberGlobalAtoms;
          atomId            = dftPtr->d_imageIdsTrunc[imageId];
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

          // get grad rho for iAtom
          if (!isLocalDomainOutsideCoreRhoTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator it =
                gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
              if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
                {
                  isCellOutsideCoreRhoTail        = false;
                  const std::vector<double> &temp = it->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      gradRhoCoreAtomsQuads[q][0][iSubCell] = temp[q * C_DIM];
                      gradRhoCoreAtomsQuads[q][1][iSubCell] =
                        temp[q * C_DIM + 1];
                      gradRhoCoreAtomsQuads[q][2][iSubCell] =
                        temp[q * C_DIM + 2];
                    }
                }
            }
        } // subCell loop

      if (isCellOutsideCoreRhoTail)
        continue;

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        {
          forceEval.submit_value(-eshelbyTensor::getFNonlinearCoreCorrection(
                                   vxcQuads[q], gradRhoCoreAtomsQuads[q]),
                                 q);
        }
      Tensor<1, C_DIM, VectorizedArray<double>>
        forceContributionFNonlinearCoreCorrectionGammaiAtomCells =
          forceEval.integrate_value();

      if (forceContributionFNonlinearCoreCorrectionGammaAtoms.find(atomId) ==
          forceContributionFNonlinearCoreCorrectionGammaAtoms.end())
        forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId] =
          std::vector<double>(C_DIM, 0.0);
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < C_DIM; idim++)
          {
            forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId][idim] +=
              forceContributionFNonlinearCoreCorrectionGammaiAtomCells
                [idim][iSubCell];
          }
    } // iAtom loop
}


//(locally used function) compute FNonlinearCoreCorrection contibution due to
// Gamma(Rj) for given set of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FNonlinearCoreCorrectionGammaAtomsElementalContribution(
    std::map<unsigned int, std::vector<double>>
      &                  forceContributionFNonlinearCoreCorrectionGammaAtoms,
    FEEvaluation<C_DIM,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 C_DIM> &forceEval,
    const MatrixFree<3, double> &matrixFreeData,
    const unsigned int           cell,
    const std::vector<Tensor<1, C_DIM, VectorizedArray<double>>> &derExcGradRho,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &hessianRhoCoreAtoms)
{
  Tensor<2, C_DIM, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    {
      for (unsigned int jdim = 0; jdim < C_DIM; jdim++)
        {
          zeroTensor1[idim][jdim] = make_vectorized_array(0.0);
        }
    }

  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      std::vector<Tensor<2, C_DIM, VectorizedArray<double>>>
        hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);

      unsigned int atomId = iAtom;
      if (iAtom >= numberGlobalAtoms)
        {
          const int imageId = iAtom - numberGlobalAtoms;
          atomId            = dftPtr->d_imageIdsTrunc[imageId];
        }

      bool isLocalDomainOutsideCoreRhoTail = false;
      if (hessianRhoCoreAtoms.find(iAtom) == hessianRhoCoreAtoms.end())
        isLocalDomainOutsideCoreRhoTail = true;

      if (isLocalDomainOutsideCoreRhoTail)
        continue;

      bool isCellOutsideCoreRhoTail = true;

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();

          // get grad rho for iAtom
          if (!isLocalDomainOutsideCoreRhoTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator it =
                hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

              if (it != hessianRhoCoreAtoms.find(iAtom)->second.end())
                {
                  isCellOutsideCoreRhoTail        = false;
                  const std::vector<double> &temp = it->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        for (unsigned int jDim = 0; jDim < 3; ++jDim)
                          hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] =
                            temp[q * C_DIM * C_DIM + C_DIM * iDim + jDim];
                    }
                }
            }
        } // subCell loop

      if (isCellOutsideCoreRhoTail)
        continue;

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        {
          forceEval.submit_value(-eshelbyTensor::getFNonlinearCoreCorrection(
                                   derExcGradRho[q],
                                   hessianRhoCoreAtomsQuads[q]),
                                 q);
        }
      Tensor<1, C_DIM, VectorizedArray<double>>
        forceContributionFNonlinearCoreCorrectionGammaiAtomCells =
          forceEval.integrate_value();

      if (forceContributionFNonlinearCoreCorrectionGammaAtoms.find(atomId) ==
          forceContributionFNonlinearCoreCorrectionGammaAtoms.end())
        forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId] =
          std::vector<double>(C_DIM, 0.0);

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < C_DIM; idim++)
          {
            forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId][idim] +=
              forceContributionFNonlinearCoreCorrectionGammaiAtomCells
                [idim][iSubCell];
          }
    } // iAtom loop
}

//(locally used function) compute FNonlinearCoreCorrection contibution due to
// Gamma(Rj) for given set of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FNonlinearCoreCorrectionGammaAtomsElementalContributionSpinPolarized(
    std::map<unsigned int, std::vector<double>>
      &                  forceContributionFNonlinearCoreCorrectionGammaAtoms,
    FEEvaluation<C_DIM,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 C_DIM> &forceEval,
    const MatrixFree<3, double> &               matrixFreeData,
    const unsigned int                          cell,
    const std::vector<VectorizedArray<double>> &vxcQuadsSpin0,
    const std::vector<VectorizedArray<double>> &vxcQuadsSpin1,
    const std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
      &derExcGradRhoSpin0,
    const std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
      &derExcGradRhoSpin1,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &        hessianRhoCoreAtoms,
    const bool isXCGGA)
{
  Tensor<1, C_DIM, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);

  Tensor<2, C_DIM, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < C_DIM; idim++)
    for (unsigned int jdim = 0; jdim < C_DIM; jdim++)
      zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      std::vector<Tensor<1, C_DIM, VectorizedArray<double>>>
        gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);
      std::vector<Tensor<2, C_DIM, VectorizedArray<double>>>
        hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor2);

      unsigned int atomId = iAtom;
      if (iAtom >= numberGlobalAtoms)
        {
          const int imageId = iAtom - numberGlobalAtoms;
          atomId            = dftPtr->d_imageIdsTrunc[imageId];
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

          // get grad rho for iAtom
          if (!isLocalDomainOutsideCoreRhoTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator it =
                gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
              if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
                {
                  isCellOutsideCoreRhoTail        = false;
                  const std::vector<double> &temp = it->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      gradRhoCoreAtomsQuads[q][0][iSubCell] =
                        temp[q * C_DIM] / 2.0;
                      gradRhoCoreAtomsQuads[q][1][iSubCell] =
                        temp[q * C_DIM + 1] / 2.0;
                      gradRhoCoreAtomsQuads[q][2][iSubCell] =
                        temp[q * C_DIM + 2] / 2.0;
                    }
                }

              if (isXCGGA && !isCellOutsideCoreRhoTail)
                {
                  std::map<dealii::CellId, std::vector<double>>::const_iterator
                    it2 =
                      hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

                  if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                    {
                      const std::vector<double> &temp2 = it2->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          for (unsigned int iDim = 0; iDim < 3; ++iDim)
                            for (unsigned int jDim = 0; jDim < 3; ++jDim)
                              hessianRhoCoreAtomsQuads
                                [q][iDim][jDim][iSubCell] =
                                  temp2[q * C_DIM * C_DIM + C_DIM * iDim +
                                        jDim] /
                                  2.0;
                        }
                    }
                }
            }
        } // subCell loop

      if (isCellOutsideCoreRhoTail)
        continue;

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        forceEval.submit_value(-eshelbyTensorSP::getFNonlinearCoreCorrection(
                                 vxcQuadsSpin0[q],
                                 vxcQuadsSpin1[q],
                                 derExcGradRhoSpin0[q],
                                 derExcGradRhoSpin1[q],
                                 gradRhoCoreAtomsQuads[q],
                                 hessianRhoCoreAtomsQuads[q],
                                 isXCGGA),
                               q);

      Tensor<1, C_DIM, VectorizedArray<double>>
        forceContributionFNonlinearCoreCorrectionGammaiAtomCells =
          forceEval.integrate_value();

      if (forceContributionFNonlinearCoreCorrectionGammaAtoms.find(atomId) ==
          forceContributionFNonlinearCoreCorrectionGammaAtoms.end())
        forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId] =
          std::vector<double>(C_DIM, 0.0);
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < C_DIM; idim++)
          {
            forceContributionFNonlinearCoreCorrectionGammaAtoms[atomId][idim] +=
              forceContributionFNonlinearCoreCorrectionGammaiAtomCells
                [idim][iSubCell];
          }
    } // iAtom loop
}
