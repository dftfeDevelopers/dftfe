// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
  //(locally used function) compute Fnl contibution due to Gamma(Rj) for given
  // set
  // of cells
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::FnlGammaAtomsElementalContribution(
    std::map<dftfe::uInt, std::vector<double>> &forceContributionFnlGammaAtoms,
    const dealii::MatrixFree<3, double>        &matrixFreeData,
    FEEvaluationWrapperClass<3>                &forceEvalNLP,
    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
                                    nonLocalOp,
    dftfe::uInt                     numNonLocalAtomsCurrentProcess,
    const std::vector<dftfe::Int>  &globalChargeIdNonLocalAtoms,
    const std::vector<dftfe::uInt> &numberPseudoWaveFunctionsPerAtom,
    const dftfe::uInt               cell,
    const std::map<dealii::CellId, dftfe::uInt> &cellIdToCellNumberMap,
#ifdef USE_COMPLEX
    const std::vector<dataTypes::number>
      &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
    const std::vector<dataTypes::number> &zetaDeltaVQuadsFlattened,
    const std::vector<dataTypes::number>
      &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened)
  {
    const dftfe::uInt numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const dftfe::uInt numQuadPoints = forceEvalNLP.n_q_points;

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        zeroTensor3[idim] = dealii::make_vectorized_array(0.0);
      }

    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      FVectQuads(numQuadPoints, zeroTensor3);

    for (dftfe::Int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
      {
        //
        // get the global charge Id of the current nonlocal atom
        //

        // FIXME should use the appropriate map from oncvClassPtr
        // instead of assuming all atoms are nonlocal atoms
        const dftfe::Int globalChargeIdNonLocalAtom =
          globalChargeIdNonLocalAtoms[iAtom];

        // if map entry corresponding to current nonlocal atom id is empty,
        // initialize it to zero
        if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom) ==
            forceContributionFnlGammaAtoms.end())
          forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom] =
            std::vector<double>(3, 0.0);

        std::fill(FVectQuads.begin(), FVectQuads.end(), zeroTensor3);

        bool isPseudoWfcsAtomInMacroCell = false;
        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            bool              isPseudoWfcsAtomInCell = false;
            const dftfe::uInt elementId =
              cellIdToCellNumberMap.find(subCellPtr->id())->second;
            for (dftfe::uInt i = 0;
                 i < (nonLocalOp->getCellIdToAtomIdsLocalCompactSupportMap())
                       .find(elementId)
                       ->second.size();
                 i++)
              if ((nonLocalOp->getCellIdToAtomIdsLocalCompactSupportMap())
                    .find(elementId)
                    ->second[i] == iAtom)
                {
                  isPseudoWfcsAtomInCell      = true;
                  isPseudoWfcsAtomInMacroCell = true;
                  break;
                }

            if (isPseudoWfcsAtomInCell)
              {
                for (dftfe::uInt kPoint = 0;
                     kPoint < dftPtr->d_kPointWeights.size();
                     ++kPoint)
                  {
                    std::vector<double> kcoord(3, 0.0);
                    kcoord[0] = dftPtr->d_kPointCoordinates[kPoint * 3 + 0];
                    kcoord[1] = dftPtr->d_kPointCoordinates[kPoint * 3 + 1];
                    kcoord[2] = dftPtr->d_kPointCoordinates[kPoint * 3 + 2];

                    const dftfe::uInt startingPseudoWfcIdFlattened =
                      kPoint *
                        nonLocalOp
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPoints +
                      (nonLocalOp->getNonTrivialSphericalFnsCellStartIndex())
                          [elementId] *
                        numQuadPoints +
                      (nonLocalOp
                         ->getAtomIdToNonTrivialSphericalFnCellStartIndex())
                          .find(iAtom)
                          ->second[elementId] *
                        numQuadPoints;

                    const dftfe::uInt numberPseudoWaveFunctions =
                      numberPseudoWaveFunctionsPerAtom[iAtom];
                    std::vector<dataTypes::number> temp2(3);
                    for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                      {
                        std::vector<dataTypes::number> F(
                          3, dataTypes::number(0.0));

                        for (dftfe::uInt iPseudoWave = 0;
                             iPseudoWave < numberPseudoWaveFunctions;
                             ++iPseudoWave)
                          {
                            const dataTypes::number temp1 =
                              zetaDeltaVQuadsFlattened
                                [startingPseudoWfcIdFlattened +
                                 iPseudoWave * numQuadPoints + q];
                            temp2[0] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints + q];
                            temp2[1] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 numQuadPoints + q];
                            temp2[2] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 2 * numQuadPoints + q];
#ifdef USE_COMPLEX
                            const dataTypes::number temp3 =
                              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened +
                                 iPseudoWave * numQuadPoints + q];
                            F[0] +=
                              2.0 * (temp1 * temp2[0] +
                                     temp1 * dataTypes::number(0.0, 1.0) *
                                       temp3 * dataTypes::number(kcoord[0]));
                            F[1] +=
                              2.0 * (temp1 * temp2[1] +
                                     temp1 * dataTypes::number(0.0, 1.0) *
                                       temp3 * dataTypes::number(kcoord[1]));
                            F[2] +=
                              2.0 * (temp1 * temp2[2] +
                                     temp1 * dataTypes::number(0.0, 1.0) *
                                       temp3 * dataTypes::number(kcoord[2]));
#else
                            F[0] += 2.0 * (temp1 * temp2[0]);
                            F[1] += 2.0 * (temp1 * temp2[1]);
                            F[2] += 2.0 * (temp1 * temp2[2]);
#endif
                          } // pseudowavefunctions loop

                        FVectQuads[q][0][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[0]);
                        FVectQuads[q][1][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[1]);
                        FVectQuads[q][2][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[2]);

                      } // quad-loop
                  }     // kpoint loop
              }         // non-trivial cell check
          }             // subcell loop

        if (isPseudoWfcsAtomInMacroCell)
          {
            for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
              forceEvalNLP.submit_value(FVectQuads[q], q);

            const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
              forceContributionFnlGammaiAtomCells =
                forceEvalNLP.integrate_value();

            for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              for (dftfe::uInt idim = 0; idim < 3; idim++)
                forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]
                                              [idim] +=
                  forceContributionFnlGammaiAtomCells[idim][iSubCell];
          }
      } // iAtom loop
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::FnlGammaxElementalContribution(
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                                        &FVectQuads,
    const dealii::MatrixFree<3, double> &matrixFreeData,
    const dftfe::uInt                    numQuadPoints,
    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
                                    nonLocalOp,
    const dftfe::uInt               numNonLocalAtomsCurrentProcess,
    const std::vector<dftfe::Int>  &globalChargeIdNonLocalAtoms,
    const std::vector<dftfe::uInt> &numberPseudoWaveFunctionsPerAtom,
    const dftfe::uInt               cell,
    const std::map<dealii::CellId, dftfe::uInt> &cellIdToCellNumberMap,
    const std::vector<dataTypes::number>        &zetaDeltaVQuadsFlattened,
    const std::vector<dataTypes::number>
      &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened)
  {
    const dftfe::uInt numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        zeroTensor3[idim] = dealii::make_vectorized_array(0.0);
      }
    std::fill(FVectQuads.begin(), FVectQuads.end(), zeroTensor3);

    for (dftfe::Int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
      {
        //
        // get the global charge Id of the current nonlocal atom
        //
        // FIX ME with correct call from ONCV
        const dftfe::Int globalChargeIdNonLocalAtom =
          globalChargeIdNonLocalAtoms[iAtom];



        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            bool              isPseudoWfcsAtomInCell = false;
            const dftfe::uInt elementId =
              cellIdToCellNumberMap.find(subCellPtr->id())->second;
            for (dftfe::uInt i = 0;
                 i < (nonLocalOp->getCellIdToAtomIdsLocalCompactSupportMap())
                       .find(elementId)
                       ->second.size();
                 i++)
              if ((nonLocalOp->getCellIdToAtomIdsLocalCompactSupportMap())
                    .find(elementId)
                    ->second[i] == iAtom)
                {
                  isPseudoWfcsAtomInCell = true;
                  break;
                }

            if (isPseudoWfcsAtomInCell)
              {
                for (dftfe::uInt kPoint = 0;
                     kPoint < dftPtr->d_kPointWeights.size();
                     ++kPoint)
                  {
                    const dftfe::uInt startingPseudoWfcIdFlattened =
                      kPoint *
                        (nonLocalOp
                           ->getTotalNonTrivialSphericalFnsOverAllCells()) *
                        numQuadPoints +
                      (nonLocalOp->getNonTrivialSphericalFnsCellStartIndex())
                          [elementId] *
                        numQuadPoints +
                      (nonLocalOp
                         ->getAtomIdToNonTrivialSphericalFnCellStartIndex())
                          .find(iAtom)
                          ->second[elementId] *
                        numQuadPoints;

                    const dftfe::uInt numberPseudoWaveFunctions =
                      numberPseudoWaveFunctionsPerAtom[iAtom];
                    std::vector<dataTypes::number> temp2(3);
                    for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                      {
                        std::vector<dataTypes::number> F(
                          3, dataTypes::number(0.0));

                        for (dftfe::uInt iPseudoWave = 0;
                             iPseudoWave < numberPseudoWaveFunctions;
                             ++iPseudoWave)
                          {
                            const dataTypes::number temp1 =
                              zetaDeltaVQuadsFlattened
                                [startingPseudoWfcIdFlattened +
                                 iPseudoWave * numQuadPoints + q];
                            temp2[0] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints + q];
                            temp2[1] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 numQuadPoints + q];
                            temp2[2] =
                              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                                [startingPseudoWfcIdFlattened * 3 +
                                 iPseudoWave * 3 * numQuadPoints +
                                 2 * numQuadPoints + q];
                            F[0] -= 2.0 * (temp1 * temp2[0]);
                            F[1] -= 2.0 * (temp1 * temp2[1]);
                            F[2] -= 2.0 * (temp1 * temp2[2]);
                          } // pseudowavefunctions loop

                        FVectQuads[q][0][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[0]);
                        FVectQuads[q][1][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[1]);
                        FVectQuads[q][2][iSubCell] +=
                          dftPtr->d_kPointWeights[kPoint] * 2.0 *
                          dftfe::utils::realPart(F[2]);
                      } // quad-loop
                  }     // kpoint loop
              }         // non-trivial cell check
          }             // subcell loop
      }                 // iAtom loop
  }

  //(locally used function) accumulate and distribute Fnl contibution due to
  // Gamma(Rj)
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::distributeForceContributionFnlGammaAtoms(
    const std::map<dftfe::uInt, std::vector<double>>
      &forceContributionFnlGammaAtoms)
  {
    for (dftfe::uInt iAtom = 0; iAtom < dftPtr->atomLocations.size(); iAtom++)
      {
        bool doesAtomIdExistOnLocallyOwnedNode = false;
        if (d_atomsForceDofs.find(
              std::pair<dftfe::uInt, dftfe::uInt>(iAtom, 0)) !=
            d_atomsForceDofs.end())
          doesAtomIdExistOnLocallyOwnedNode = true;

        std::vector<double> forceContributionFnlGammaiAtomGlobal(3);
        std::vector<double> forceContributionFnlGammaiAtomLocal(3, 0.0);

        // TODO this will throw an error for hubbard
        if (forceContributionFnlGammaAtoms.find(iAtom) !=
            forceContributionFnlGammaAtoms.end())
          {
            forceContributionFnlGammaiAtomLocal =
              forceContributionFnlGammaAtoms.find(iAtom)->second;
          }
        else
          {
            std::fill(forceContributionFnlGammaiAtomLocal.begin(),
                      forceContributionFnlGammaiAtomLocal.end(),
                      0.0);
          }

        // accumulate value
        MPI_Allreduce(&(forceContributionFnlGammaiAtomLocal[0]),
                      &(forceContributionFnlGammaiAtomGlobal[0]),
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);

        if (doesAtomIdExistOnLocallyOwnedNode)
          {
            std::vector<dealii::types::global_dof_index> forceLocalDofIndices(
              3);
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              forceLocalDofIndices[idim] =
                d_atomsForceDofs[std::pair<dftfe::uInt, dftfe::uInt>(iAtom,
                                                                     idim)];
#ifdef USE_COMPLEX
            d_constraintsNoneForce.distribute_local_to_global(
              forceContributionFnlGammaiAtomGlobal,
              forceLocalDofIndices,
              d_configForceVectorLinFEKPoints);
#else
            d_constraintsNoneForce.distribute_local_to_global(
              forceContributionFnlGammaiAtomGlobal,
              forceLocalDofIndices,
              d_configForceVectorLinFE);
#endif
          }
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
