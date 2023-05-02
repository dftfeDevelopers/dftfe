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
#include <force.h>
#include <dft.h>

namespace dftfe
{

//(locally used function) compute FPhiTotSmearedCharges contibution due to
// Gamma(Rj) for given set of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FPhiTotSmearedChargesGammaAtomsElementalContribution(
    std::map<unsigned int, std::vector<double>>
      &                          forceContributionSmearedChargesGammaAtoms,
    dealii::FEEvaluation<3, -1, 1, 3> &  forceEval,
    const dealii::MatrixFree<3, double> &matrixFreeData,
    const unsigned int           cell,
    const dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      &                              gradPhiTotQuads,
    const std::vector<unsigned int> &nonTrivialAtomIdsMacroCell,
    const std::map<dealii::CellId, std::vector<int>> &    bQuadAtomIdsAllAtoms,
    const dealii::AlignedVector<dealii::VectorizedArray<double>> &smearedbQuads)
{
  dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = dealii::make_vectorized_array(0.0);
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  dealii::DoFHandler<3>::active_cell_iterator subCellPtr;


  for (int iAtomNonTrivial = 0;
       iAtomNonTrivial < nonTrivialAtomIdsMacroCell.size();
       iAtomNonTrivial++)
    {
      const int iAtom = nonTrivialAtomIdsMacroCell[iAtomNonTrivial];
      dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuadsiAtom(
        numQuadPoints, dealii::make_vectorized_array(0.0));

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId          subCellId = subCellPtr->id();
          const std::vector<int> &bQuadAtomIdsCell =
            bQuadAtomIdsAllAtoms.find(subCellId)->second;
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              if (bQuadAtomIdsCell[q] == iAtom)
                smearedbQuadsiAtom[q][iSubCell] = smearedbQuads[q][iSubCell];
            }
        }

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        forceEval.submit_value(gradPhiTotQuads[q] * smearedbQuadsiAtom[q], q);


      dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        forceContributionSmearedChargesGammaiAtomCells =
          forceEval.integrate_value();

      if (forceContributionSmearedChargesGammaAtoms.find(iAtom) ==
          forceContributionSmearedChargesGammaAtoms.end())
        forceContributionSmearedChargesGammaAtoms[iAtom] =
          std::vector<double>(3, 0.0);
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            forceContributionSmearedChargesGammaAtoms[iAtom][idim] +=
              forceContributionSmearedChargesGammaiAtomCells[idim][iSubCell];
          }
    } // iAtom loop
}

//(locally used function) compute FVselfSmearedCharges contibution due to
// Gamma(Rj) for given set of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  FVselfSmearedChargesGammaAtomsElementalContribution(
    std::map<unsigned int, std::vector<double>>
      &                          forceContributionSmearedChargesGammaAtoms,
    dealii::FEEvaluation<3, -1, 1, 3> &  forceEval,
    const dealii::MatrixFree<3, double> &matrixFreeData,
    const unsigned int           cell,
    const dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      &                              gradVselfBinQuads,
    const std::vector<unsigned int> &nonTrivialAtomIdsMacroCell,
    const std::map<dealii::CellId, std::vector<int>> &    bQuadAtomIdsAllAtoms,
    const dealii::AlignedVector<dealii::VectorizedArray<double>> &smearedbQuads)
{
  dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
  for (unsigned int idim = 0; idim < 3; idim++)
    zeroTensor1[idim] = dealii::make_vectorized_array(0.0);
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEval.n_q_points;
  dealii::DoFHandler<3>::active_cell_iterator subCellPtr;


  for (int iAtomNonTrivial = 0;
       iAtomNonTrivial < nonTrivialAtomIdsMacroCell.size();
       iAtomNonTrivial++)
    {
      const int atomId = nonTrivialAtomIdsMacroCell[iAtomNonTrivial];

      dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuadsiAtom(
        numQuadPoints, dealii::make_vectorized_array(0.0));

      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId          subCellId = subCellPtr->id();
          const std::vector<int> &bQuadAtomIdsCell =
            bQuadAtomIdsAllAtoms.find(subCellId)->second;
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              if (bQuadAtomIdsCell[q] == atomId)
                smearedbQuadsiAtom[q][iSubCell] = smearedbQuads[q][iSubCell];
            }
        }

      for (unsigned int q = 0; q < numQuadPoints; ++q)
        forceEval.submit_value(-gradVselfBinQuads[q] * smearedbQuadsiAtom[q],
                               q);

      dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
        forceContributionSmearedChargesGammaiAtomCells =
          forceEval.integrate_value();

      if (forceContributionSmearedChargesGammaAtoms.find(atomId) ==
          forceContributionSmearedChargesGammaAtoms.end())
        forceContributionSmearedChargesGammaAtoms[atomId] =
          std::vector<double>(3, 0.0);
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            forceContributionSmearedChargesGammaAtoms[atomId][idim] +=
              forceContributionSmearedChargesGammaiAtomCells[idim][iSubCell];
          }
    } // iAtom loop
}
#include "../force.inst.cc"
} // namespace dftfe
