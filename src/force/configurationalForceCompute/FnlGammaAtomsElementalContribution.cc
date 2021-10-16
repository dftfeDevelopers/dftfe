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
//(locally used function) compute Fnl contibution due to Gamma(Rj) for given set
// of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::FnlGammaAtomsElementalContribution(
  std::map<unsigned int, std::vector<double>> &forceContributionFnlGammaAtoms,
  FEEvaluation<3,
               1,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               3> &                            forceEval,
  FEEvaluation<3, 1, C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(), 3>
    &                forceEvalNLP,
  const unsigned int numberMacroCells,
  const unsigned int cell,
#ifdef USE_COMPLEX
  const unsigned int kpointIndex,
  const dealii::AlignedVector<dealii::AlignedVector<
    dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>>
    &zetaDeltaVQuads,
  const dealii::AlignedVector<
    dealii::AlignedVector<Tensor<1, 3, Tensor<1, 2, VectorizedArray<double>>>>>
    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
  const dealii::AlignedVector<
    dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>
    &projectorKetTimesPsiTimesVTimesPartOccContractionPsi,
  const Tensor<1, 3, VectorizedArray<double>> kcoord,
#else
  const dealii::AlignedVector<
    dealii::AlignedVector<dealii::AlignedVector<VectorizedArray<double>>>>
    &zetaDeltaVQuads,
  const dealii::AlignedVector<
    dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>>
    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi,
#endif
  const std::vector<bool> &        isAtomInCell,
  const std::vector<unsigned int> &nonlocalPseudoWfcsAccum)
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numSubCells =
    dftPtr->matrix_free_data.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEvalNLP.n_q_points;

  const unsigned int numNonLocalAtomsCurrentProcess =
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

  for (int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
    {
      //
      // get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId =
        dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =
        dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];
      // const std::vector<double> &
      // temp2=projectorKetTimesPsiTimesVTimesPartOcc[iAtom];

      // if map entry corresponding to current nonlocal atom id is empty,
      // initialize it to zero
      if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom) ==
          forceContributionFnlGammaAtoms.end())
        forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom] =
          std::vector<double>(3, 0.0);

      if (isAtomInCell[iAtom])
        {
          const unsigned int startingPseudoWfcId =
            nonlocalPseudoWfcsAccum[iAtom];

          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
#ifdef USE_COMPLEX
              const dealii::AlignedVector<
                Tensor<1, 2, VectorizedArray<double>>> &temp1 =
                zetaDeltaVQuads[kpointIndex * numberMacroCells * numQuadPoints +
                                cell * numQuadPoints + q][iAtom];

              const Tensor<1, 3, VectorizedArray<double>> F =
                -make_vectorized_array(dftPtr->d_kPointWeights[kpointIndex]) *
                eshelbyTensor::getFnlAtom(
                  temp1,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                    [kpointIndex * numberMacroCells * numQuadPoints +
                     cell * numQuadPoints + q],
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsi
                    [kpointIndex * numberMacroCells * numQuadPoints +
                     cell * numQuadPoints + q],
                  kcoord,
                  startingPseudoWfcId);
#else
              const dealii::AlignedVector<VectorizedArray<double>> &temp1 =
                zetaDeltaVQuads[cell * numQuadPoints + q][iAtom];

              const Tensor<1, 3, VectorizedArray<double>> F =
                -eshelbyTensor::getFnlAtom(
                  temp1,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsi
                    [cell * numQuadPoints + q],
                  startingPseudoWfcId);
#endif


              forceEvalNLP.submit_value(F, q);
            }


          const Tensor<1, 3, VectorizedArray<double>>
            forceContributionFnlGammaiAtomCells =
              forceEvalNLP.integrate_value();

          for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
            for (unsigned int idim = 0; idim < 3; idim++)
              forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]
                                            [idim] +=
                forceContributionFnlGammaiAtomCells[idim][iSubCell];
        }
    } // iAtom loop
}

//(locally used function) accumulate and distribute Fnl contibution due to
// Gamma(Rj)
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::distributeForceContributionFnlGammaAtoms(
  const std::map<unsigned int, std::vector<double>>
    &forceContributionFnlGammaAtoms)
{
  for (unsigned int iAtom = 0; iAtom < dftPtr->atomLocations.size(); iAtom++)
    {
      bool doesAtomIdExistOnLocallyOwnedNode = false;
      if (d_atomsForceDofs.find(
            std::pair<unsigned int, unsigned int>(iAtom, 0)) !=
          d_atomsForceDofs.end())
        doesAtomIdExistOnLocallyOwnedNode = true;

      std::vector<double> forceContributionFnlGammaiAtomGlobal(3);
      std::vector<double> forceContributionFnlGammaiAtomLocal(3, 0.0);

      if (forceContributionFnlGammaAtoms.find(iAtom) !=
          forceContributionFnlGammaAtoms.end())
        forceContributionFnlGammaiAtomLocal =
          forceContributionFnlGammaAtoms.find(iAtom)->second;
      // accumulate value
      MPI_Allreduce(&(forceContributionFnlGammaiAtomLocal[0]),
                    &(forceContributionFnlGammaiAtomGlobal[0]),
                    3,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      if (doesAtomIdExistOnLocallyOwnedNode)
        {
          std::vector<dealii::types::global_dof_index> forceLocalDofIndices(3);
          for (unsigned int idim = 0; idim < 3; idim++)
            forceLocalDofIndices[idim] =
              d_atomsForceDofs[std::pair<unsigned int, unsigned int>(iAtom,
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
