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
// @author Sambit Das
//

// compute ESmeared contribution stress
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::addEPhiTotSmearedStressContribution(
  FEEvaluation<3, -1, 1, 3> &                               forceEval,
  const MatrixFree<3, double> &                             matrixFreeData,
  const unsigned int                                        cell,
  const std::vector<Tensor<1, 3, VectorizedArray<double>>> &gradPhiTotQuads,
  const std::vector<unsigned int> &nonTrivialAtomImageIdsMacroCell,
  const std::map<dealii::CellId, std::vector<int>> &bQuadAtomIdsAllAtomsImages,
  const std::vector<VectorizedArray<double>> &      smearedbQuads)
{
  Tensor<1, 3, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);

  Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      {
        zeroTensor2[idim][jdim] = make_vectorized_array(0.0);
      }

  const unsigned int numberGlobalAtoms  = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int numberTotalAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;

  DoFHandler<3>::active_cell_iterator subCellPtr;

  std::vector<VectorizedArray<double>> smearedbQuadsiAtom(
    numQuadPoints, make_vectorized_array(0.0));

  for (int iAtomNonTrivial = 0;
       iAtomNonTrivial < nonTrivialAtomImageIdsMacroCell.size();
       iAtomNonTrivial++)
    {
      const int        iAtom = nonTrivialAtomImageIdsMacroCell[iAtomNonTrivial];
      Point<3, double> atomLocation;
      if (iAtom < numberGlobalAtoms)
        {
          atomLocation[0] = dftPtr->atomLocations[iAtom][2];
          atomLocation[1] = dftPtr->atomLocations[iAtom][3];
          atomLocation[2] = dftPtr->atomLocations[iAtom][4];
        }
      else
        {
          const int imageId = iAtom - numberGlobalAtoms;
          atomLocation[0]   = dftPtr->d_imagePositionsTrunc[imageId][0];
          atomLocation[1]   = dftPtr->d_imagePositionsTrunc[imageId][1];
          atomLocation[2]   = dftPtr->d_imagePositionsTrunc[imageId][2];
        }

      Point<3, VectorizedArray<double>> atomLocationVect;
      atomLocationVect[0] = make_vectorized_array(atomLocation[0]);
      atomLocationVect[1] = make_vectorized_array(atomLocation[1]);
      atomLocationVect[2] = make_vectorized_array(atomLocation[2]);

      std::fill(smearedbQuadsiAtom.begin(),
                smearedbQuadsiAtom.end(),
                make_vectorized_array(0.0));

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId          subCellId = subCellPtr->id();
          const std::vector<int> &bQuadAtomIdsCell =
            bQuadAtomIdsAllAtomsImages.find(subCellId)->second;
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            if (bQuadAtomIdsCell[q] == iAtom)
              smearedbQuadsiAtom[q][iSubCell] = smearedbQuads[q][iSubCell];
        }


      Tensor<2, 3, VectorizedArray<double>> EPSPStressContribution =
        zeroTensor2;
      for (unsigned int q = 0; q < numQuadPoints; ++q)
        EPSPStressContribution -=
          outer_product(smearedbQuadsiAtom[q] * gradPhiTotQuads[q],
                        forceEval.quadrature_point(q) - atomLocationVect) *
          forceEval.JxW(q);

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            d_stress[idim][jdim] +=
              EPSPStressContribution[idim][jdim][iSubCell];
    } // iAtom loop
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::addEVselfSmearedStressContribution(
  FEEvaluation<3, -1, 1, 3> &                               forceEval,
  const MatrixFree<3, double> &                             matrixFreeData,
  const unsigned int                                        cell,
  const std::vector<Tensor<1, 3, VectorizedArray<double>>> &gradVselfQuads,
  const std::vector<unsigned int> &nonTrivialAtomImageIdsMacroCell,
  const std::map<dealii::CellId, std::vector<int>> &bQuadAtomIdsAllAtomsImages,
  const std::vector<VectorizedArray<double>> &      smearedbQuads)
{
  Tensor<1, 3, VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = make_vectorized_array(0.0);

  Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      {
        zeroTensor2[idim][jdim] = make_vectorized_array(0.0);
      }

  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;

  DoFHandler<3>::active_cell_iterator subCellPtr;

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();

  std::vector<VectorizedArray<double>> smearedbQuadsiAtom(
    numQuadPoints, make_vectorized_array(0.0));

  for (int iAtomNonTrivial = 0;
       iAtomNonTrivial < nonTrivialAtomImageIdsMacroCell.size();
       iAtomNonTrivial++)
    {
      const int atomId = nonTrivialAtomImageIdsMacroCell[iAtomNonTrivial];

      Point<3, double> atomLocation;
      if (atomId < numberGlobalAtoms)
        {
          atomLocation[0] = dftPtr->atomLocations[atomId][2];
          atomLocation[1] = dftPtr->atomLocations[atomId][3];
          atomLocation[2] = dftPtr->atomLocations[atomId][4];
        }
      else
        {
          const int imageId = atomId - numberGlobalAtoms;
          atomLocation[0]   = dftPtr->d_imagePositionsTrunc[imageId][0];
          atomLocation[1]   = dftPtr->d_imagePositionsTrunc[imageId][1];
          atomLocation[2]   = dftPtr->d_imagePositionsTrunc[imageId][2];
        }

      Point<3, VectorizedArray<double>> atomLocationVect;
      atomLocationVect[0] = make_vectorized_array(atomLocation[0]);
      atomLocationVect[1] = make_vectorized_array(atomLocation[1]);
      atomLocationVect[2] = make_vectorized_array(atomLocation[2]);

      std::fill(smearedbQuadsiAtom.begin(),
                smearedbQuadsiAtom.end(),
                make_vectorized_array(0.0));

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId          subCellId = subCellPtr->id();
          const std::vector<int> &bQuadAtomIdsCell =
            bQuadAtomIdsAllAtomsImages.find(subCellId)->second;
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            if (bQuadAtomIdsCell[q] == atomId)
              smearedbQuadsiAtom[q][iSubCell] = smearedbQuads[q][iSubCell];
        }

      Tensor<2, 3, VectorizedArray<double>> EPSPStressContribution =
        zeroTensor2;
      for (unsigned int q = 0; q < numQuadPoints; ++q)
        EPSPStressContribution +=
          outer_product(smearedbQuadsiAtom[q] * gradVselfQuads[q],
                        forceEval.quadrature_point(q) - atomLocationVect) *
          forceEval.JxW(q);

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          for (unsigned int jdim = 0; jdim < 3; jdim++)
            d_stress[idim][jdim] +=
              EPSPStressContribution[idim][jdim][iSubCell];
    } // iAtom loop
}
