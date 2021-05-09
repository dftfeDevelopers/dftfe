// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE
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

//(locally used function) compute FShadowLocal contibution due to Gamma(Rj) for
// given set of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FShadowLocalGammaAtomsElementalContributionElectronic(
    std::map<unsigned int, std::vector<double>>
      &                          forceContributionLocalGammaAtoms,
    FEEvaluation<3,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 3> &            forceEval,
    const MatrixFree<3, double> &matrixFreeData,
    const unsigned int           cell,
    const dealii::AlignedVector<VectorizedArray<double>>
      &derVxcWithRhoTimesRhoDiffQuads,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoAtomsQuadsSeparate,
    const dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
      &der2ExcWithGradRhoQuads,
    const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
      &derVxcWithGradRhoQuads,
    const dealii::AlignedVector<VectorizedArray<double>>
      &shadowKSRhoMinMinusRhoQuads,
    const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
      &shadowKSGradRhoMinMinusGradRhoQuads,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &hessianRhoAtomsQuadsSeparate,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &        hessianRhoCoreAtoms,
    const bool isAtomicRhoSplitting,
    const bool isXCGGA,
    const bool isNLCC)
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

  if (isAtomicRhoSplitting)
    for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
          gradRhoAtomQuads(numQuadPoints, zeroTensor1);
        dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
          hessianRhoAtomQuads(numQuadPoints, zeroTensor2);

        unsigned int atomId = iAtom;
        if (iAtom >= numberGlobalAtoms)
          {
            const int imageId = iAtom - numberGlobalAtoms;
            atomId            = dftPtr->d_imageIdsTrunc[imageId];
          }

        bool isLocalDomainOutsideRhoTail = false;
        if (gradRhoAtomsQuadsSeparate.find(iAtom) ==
            gradRhoAtomsQuadsSeparate.end())
          isLocalDomainOutsideRhoTail = true;

        if (isLocalDomainOutsideRhoTail)
          continue;

        bool isCellOutsideRhoTail = true;
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            // get grad rho for iAtom
            if (!isLocalDomainOutsideRhoTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it = gradRhoAtomsQuadsSeparate.find(iAtom)->second.find(
                    subCellId);
                if (it != gradRhoAtomsQuadsSeparate.find(iAtom)->second.end())
                  {
                    isCellOutsideRhoTail            = false;
                    const std::vector<double> &temp = it->second;
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        gradRhoAtomQuads[q][0][iSubCell] = temp[q * 3];
                        gradRhoAtomQuads[q][1][iSubCell] = temp[q * 3 + 1];
                        gradRhoAtomQuads[q][2][iSubCell] = temp[q * 3 + 2];
                      }
                  }

                if (isXCGGA && !isCellOutsideRhoTail)
                  {
                    std::map<dealii::CellId,
                             std::vector<double>>::const_iterator it2 =
                      hessianRhoAtomsQuadsSeparate.find(iAtom)->second.find(
                        subCellId);
                    if (it2 !=
                        hessianRhoAtomsQuadsSeparate.find(iAtom)->second.end())
                      {
                        const std::vector<double> &temp = it2->second;
                        for (unsigned int q = 0; q < numQuadPoints; ++q)
                          {
                            for (unsigned int idim = 0; idim < 3; idim++)
                              for (unsigned int jdim = 0; jdim < 3; jdim++)
                                hessianRhoAtomQuads[q][idim][jdim][iSubCell] =
                                  (it2->second)[9 * q + idim * 3 + jdim];
                          }
                      }
                  }
              }
          } // subCell loop

        if (isCellOutsideRhoTail)
          continue;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            if (isXCGGA)
              {
                Tensor<1, 3, VectorizedArray<double>> temp;
                for (unsigned int i = 0; i < 3; i++)
                  {
                    temp[i] = make_vectorized_array(0.0);
                    for (unsigned int j = 0; j < 3; j++)
                      for (unsigned int k = 0; k < 3; k++)
                        temp[i] += shadowKSGradRhoMinMinusGradRhoQuads[q][j] *
                                   der2ExcWithGradRhoQuads[q][j][k] *
                                   hessianRhoAtomQuads[q][k][i];

                    for (unsigned int j = 0; j < 3; j++)
                      temp[i] += shadowKSGradRhoMinMinusGradRhoQuads[q][j] *
                                   derVxcWithGradRhoQuads[q][j] *
                                   gradRhoAtomQuads[q][i] +
                                 shadowKSRhoMinMinusRhoQuads[q] *
                                   derVxcWithGradRhoQuads[q][j] *
                                   hessianRhoAtomQuads[q][j][i];
                  }

                forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q] *
                                           gradRhoAtomQuads[q] -
                                         temp,
                                       q);

                /*
                forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q]*gradRhoAtomQuads[q]
                    -shadowKSGradRhoMinMinusGradRhoQuads[q]*(der2ExcWithGradRhoQuads[q]*hessianRhoAtomQuads[q])
                    -shadowKSGradRhoMinMinusGradRhoQuads[q]*outer_product(derVxcWithGradRhoQuads[q],gradRhoAtomQuads[q])
                    -shadowKSRhoMinMinusRhoQuads[q]*derVxcWithGradRhoQuads[q]*hessianRhoAtomQuads[q],
                    q);
                */
              }
            else
              forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q] *
                                       gradRhoAtomQuads[q],
                                     q);
          }
        Tensor<1, 3, VectorizedArray<double>> forceContributionLocalGammaiAtom =
          forceEval.integrate_value();

        if (forceContributionLocalGammaAtoms.find(atomId) ==
            forceContributionLocalGammaAtoms.end())
          forceContributionLocalGammaAtoms[atomId] =
            std::vector<double>(3, 0.0);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (unsigned int idim = 0; idim < 3; idim++)
            {
              forceContributionLocalGammaAtoms[atomId][idim] +=
                forceContributionLocalGammaiAtom[idim][iSubCell];
            }
      } // iAtom loop

  if (isNLCC)
    for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
          gradRhoCoreAtomQuads(numQuadPoints, zeroTensor1);
        dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
          hessianRhoCoreAtomQuads(numQuadPoints, zeroTensor2);

        unsigned int atomId = iAtom;
        if (iAtom >= numberGlobalAtoms)
          {
            const int imageId = iAtom - numberGlobalAtoms;
            atomId            = dftPtr->d_imageIdsTrunc[imageId];
          }

        bool isLocalDomainOutsideRhoCoreTail = false;
        if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
          isLocalDomainOutsideRhoCoreTail = true;

        if (isLocalDomainOutsideRhoCoreTail)
          continue;

        bool isCellOutsideRhoCoreTail = true;
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            // get grad rho for iAtom
            if (!isLocalDomainOutsideRhoCoreTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it = gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
                if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
                  {
                    isCellOutsideRhoCoreTail        = false;
                    const std::vector<double> &temp = it->second;
                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        gradRhoCoreAtomQuads[q][0][iSubCell] = temp[q * 3];
                        gradRhoCoreAtomQuads[q][1][iSubCell] = temp[q * 3 + 1];
                        gradRhoCoreAtomQuads[q][2][iSubCell] = temp[q * 3 + 2];
                      }
                  }

                if (isXCGGA && !isCellOutsideRhoCoreTail)
                  {
                    std::map<dealii::CellId,
                             std::vector<double>>::const_iterator it2 =
                      hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);
                    if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                      {
                        const std::vector<double> &temp = it2->second;
                        for (unsigned int q = 0; q < numQuadPoints; ++q)
                          {
                            for (unsigned int idim = 0; idim < 3; idim++)
                              for (unsigned int jdim = 0; jdim < 3; jdim++)
                                hessianRhoCoreAtomQuads
                                  [q][idim][jdim][iSubCell] =
                                    (it2->second)[9 * q + idim * 3 + jdim];
                          }
                      }
                  }
              }
          } // subCell loop

        if (isCellOutsideRhoCoreTail)
          continue;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            if (isXCGGA)
              {
                Tensor<1, 3, VectorizedArray<double>> temp;
                for (unsigned int i = 0; i < 3; i++)
                  {
                    temp[i] = make_vectorized_array(0.0);
                    for (unsigned int j = 0; j < 3; j++)
                      for (unsigned int k = 0; k < 3; k++)
                        temp[i] += shadowKSGradRhoMinMinusGradRhoQuads[q][j] *
                                   der2ExcWithGradRhoQuads[q][j][k] *
                                   hessianRhoCoreAtomQuads[q][k][i];

                    for (unsigned int j = 0; j < 3; j++)
                      temp[i] += shadowKSGradRhoMinMinusGradRhoQuads[q][j] *
                                   derVxcWithGradRhoQuads[q][j] *
                                   gradRhoCoreAtomQuads[q][i] +
                                 shadowKSRhoMinMinusRhoQuads[q] *
                                   derVxcWithGradRhoQuads[q][j] *
                                   hessianRhoCoreAtomQuads[q][j][i];
                  }

                forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q] *
                                           gradRhoCoreAtomQuads[q] -
                                         temp,
                                       q);

                /*
                forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q]*gradRhoCoreAtomQuads[q]
                    -shadowKSGradRhoMinMinusGradRhoQuads[q]*(der2ExcWithGradRhoQuads[q]*hessianRhoCoreAtomQuads[q])
                    -shadowKSGradRhoMinMinusGradRhoQuads[q]*outer_product(derVxcWithGradRhoQuads[q],gradRhoCoreAtomQuads[q])
                    -shadowKSRhoMinMinusRhoQuads[q]*derVxcWithGradRhoQuads[q]*hessianRhoCoreAtomQuads[q],
                    q);
                */
              }
            else
              forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q] *
                                       gradRhoCoreAtomQuads[q],
                                     q);
          }
        Tensor<1, 3, VectorizedArray<double>> forceContributionLocalGammaiAtom =
          forceEval.integrate_value();

        if (forceContributionLocalGammaAtoms.find(atomId) ==
            forceContributionLocalGammaAtoms.end())
          forceContributionLocalGammaAtoms[atomId] =
            std::vector<double>(3, 0.0);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (unsigned int idim = 0; idim < 3; idim++)
            {
              forceContributionLocalGammaAtoms[atomId][idim] +=
                forceContributionLocalGammaiAtom[idim][iSubCell];
            }
      } // iAtom loop
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FShadowLocalGammaAtomsElementalContributionElectrostatic(
    std::map<unsigned int, std::vector<double>>
      &                          forceContributionLocalGammaAtoms,
    FEEvaluation<3,
                 1,
                 C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
                 3> &            forceEval,
    const MatrixFree<3, double> &matrixFreeData,
    const unsigned int           cell,
    const dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
      &gradPhiRhoMinusApproxRhoElectroQuads,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &rhoAtomsQuadsSeparate)
{
  Tensor<1, 3, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);
  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  DoFHandler<3>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
    {
      dealii::AlignedVector<VectorizedArray<double>> rhoAtomQuads(
        numQuadPoints, make_vectorized_array(0.0));

      unsigned int atomId = iAtom;
      if (iAtom >= numberGlobalAtoms)
        {
          const int imageId = iAtom - numberGlobalAtoms;
          atomId            = dftPtr->d_imageIdsTrunc[imageId];
        }

      bool isLocalDomainOutsideRhoTail = false;
      if (rhoAtomsQuadsSeparate.find(iAtom) == rhoAtomsQuadsSeparate.end())
        isLocalDomainOutsideRhoTail = true;

      if (isLocalDomainOutsideRhoTail)
        continue;

      bool isCellOutsideRhoTail = true;
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();

          // get grad rho for iAtom
          if (!isLocalDomainOutsideRhoTail)
            {
              std::map<dealii::CellId, std::vector<double>>::const_iterator it =
                rhoAtomsQuadsSeparate.find(iAtom)->second.find(subCellId);
              if (it != rhoAtomsQuadsSeparate.find(iAtom)->second.end())
                {
                  isCellOutsideRhoTail            = false;
                  const std::vector<double> &temp = it->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      rhoAtomQuads[q][iSubCell] = temp[q];
                    }
                }
            }
        } // subCell loop

      if (isCellOutsideRhoTail)
        continue;

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        {
          forceEval.submit_value(gradPhiRhoMinusApproxRhoElectroQuads[q] *
                                   rhoAtomQuads[q],
                                 q);
        }
      Tensor<1, 3, VectorizedArray<double>> forceContributionLocalGammaiAtom =
        forceEval.integrate_value();

      if (forceContributionLocalGammaAtoms.find(atomId) ==
          forceContributionLocalGammaAtoms.end())
        forceContributionLocalGammaAtoms[atomId] = std::vector<double>(3, 0.0);
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            forceContributionLocalGammaAtoms[atomId][idim] +=
              forceContributionLocalGammaiAtom[idim][iSubCell];
          }
    } // iAtom loop
}
