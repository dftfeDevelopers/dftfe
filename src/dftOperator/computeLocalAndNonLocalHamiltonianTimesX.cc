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
// @author Phani Motamarri, Department of Computational and Data Sciences, IISc Bangalore
//


/** @file matrixVectorProductImplementations.cc
 *  @brief Contains linear algebra operations
 *
 */

#ifdef USE_COMPLEX
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
  computeHamiltonianTimesXInternal(
    const distributedCPUMultiVec<std::complex<double>> &src,
    std::vector<std::complex<double>> &           cellSrcWaveFunctionMatrix,
    const unsigned int                            numberWaveFunctions,
    distributedCPUMultiVec<std::complex<double>> &dst,
    std::vector<std::complex<double>> &           cellDstWaveFunctionMatrix,
    const double                                  scalar,
    const double                                  scalarA,
    const double                                  scalarB,
    bool                                          scaleFlag)
{
  AssertThrow(false, dftUtils::ExcNotImplementedYet());
}
#else
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
  computeHamiltonianTimesXInternal(
    const distributedCPUMultiVec<double> &src,
    std::vector<double> &                 cellSrcWaveFunctionMatrix,
    const unsigned int                    numberWaveFunctions,
    distributedCPUMultiVec<double> &      dst,
    std::vector<double> &                 cellDstWaveFunctionMatrix,
    const double                          scalar,
    const double                          scalarA,
    const double                          scalarB,
    bool                                  scaleFlag)

{
  const unsigned int kpointSpinIndex =
    (1 + dftPtr->d_dftParamsPtr->spinPolarized) * d_kPointIndex + d_spinIndex;
  //
  // element level matrix-vector multiplications
  //
  const char   transA = 'N', transB = 'N';
  const double scalarCoeffAlpha1 = scalar, scalarCoeffBeta = 0.0,
               scalarCoeffAlpha = 1.0;
  const unsigned int inc        = 1;

  unsigned int indexTemp1 = d_numberNodesPerElement * numberWaveFunctions;
  std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(
    d_numberNodesPerElement);


  for (unsigned int iElem = 0; iElem < d_numberCellsLocallyOwned; ++iElem)
    {
      unsigned int indexTemp2 = indexTemp1 * iElem;
      for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
        {
          if (d_nodesPerCellClassificationMap[iNode] == 1)
            {
              unsigned int indexVal = indexTemp2 + numberWaveFunctions * iNode;
              dealii::types::global_dof_index localNodeId =
                d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];

              dcopy_(&numberWaveFunctions,
                     src.data() + localNodeId,
                     &inc,
                     &cellSrcWaveFunctionMatrix[indexVal],
                     &inc);
            }
        }
    } // cell loop


  //
  // start nonlocal HX
  //
  std::map<unsigned int, std::vector<double>> projectorKetTimesVector;

  //
  // allocate memory for matrix-vector product
  //
  if (dftPtr->d_dftParamsPtr->isPseudopotential &&
      dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
    {
      for (unsigned int iAtom = 0;
           iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
           ++iAtom)
        {
          const unsigned int atomId =
            dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
          const int numberSingleAtomPseudoWaveFunctions =
            dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
          projectorKetTimesVector[atomId].resize(
            numberWaveFunctions * numberSingleAtomPseudoWaveFunctions, 0.0);
        }

      //
      // blas required settings
      //
      const double alpha = 1.0;
      const double beta  = 1.0;


      typename dealii::DoFHandler<3>::active_cell_iterator
        cell    = dftPtr->dofHandler.begin_active(),
        endc    = dftPtr->dofHandler.end();
      int iElem = -1;
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              iElem++;

              const unsigned int indexVal =
                d_numberNodesPerElement * numberWaveFunctions * iElem;
              for (unsigned int iAtom = 0;
                   iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();
                   ++iAtom)
                {
                  const unsigned int atomId =
                    dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
                  const unsigned int numberPseudoWaveFunctions =
                    dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
                  const int nonZeroElementMatrixId =
                    dftPtr->d_sparsityPattern[atomId][iElem];

                  dgemm_(&transA,
                         &transB,
                         &numberWaveFunctions,
                         &numberPseudoWaveFunctions,
                         &d_numberNodesPerElement,
                         &alpha,
                         &cellSrcWaveFunctionMatrix[indexVal],
                         &numberWaveFunctions,
                         &dftPtr->d_nonLocalProjectorElementMatricesConjugate
                            [atomId][nonZeroElementMatrixId][0],
                         &d_numberNodesPerElement,
                         &beta,
                         &projectorKetTimesVector[atomId][0],
                         &numberWaveFunctions);
                }
            }

        } // cell loop


      dftPtr->d_projectorKetTimesVectorParFlattened.setValue(0.0);

      for (unsigned int iAtom = 0;
           iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
           ++iAtom)
        {
          const unsigned int atomId =
            dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
          const unsigned int numberPseudoWaveFunctions =
            dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

          for (unsigned int iPseudoAtomicWave = 0;
               iPseudoAtomicWave < numberPseudoWaveFunctions;
               ++iPseudoAtomicWave)
            {
              const unsigned int id =
                dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(
                  atomId, iPseudoAtomicWave)];

              dcopy_(&numberWaveFunctions,
                     &projectorKetTimesVector[atomId][numberWaveFunctions *
                                                      iPseudoAtomicWave],
                     &inc,
                     dftPtr->d_projectorKetTimesVectorParFlattened.data() +
                       dftPtr->d_projectorKetTimesVectorParFlattened
                           .getMPIPatternP2P()
                           ->globalToLocal(id) *
                         numberWaveFunctions,
                     &inc);
            }
        }

      dftPtr->d_projectorKetTimesVectorParFlattened.accumulateAddLocallyOwned();
      dftPtr->d_projectorKetTimesVectorParFlattened.updateGhostValues();

      //
      // compute V*C^{T}*X
      //
      for (unsigned int iAtom = 0;
           iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
           ++iAtom)
        {
          const unsigned int atomId =
            dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
          const unsigned int numberPseudoWaveFunctions =
            dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
          for (unsigned int iPseudoAtomicWave = 0;
               iPseudoAtomicWave < numberPseudoWaveFunctions;
               ++iPseudoAtomicWave)
            {
              double nonlocalConstantV =
                dftPtr->d_nonLocalPseudoPotentialConstants[atomId]
                                                          [iPseudoAtomicWave];

              const unsigned int id =
                dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(
                  atomId, iPseudoAtomicWave)];

              dscal_(&numberWaveFunctions,
                     &nonlocalConstantV,
                     dftPtr->d_projectorKetTimesVectorParFlattened.data() +
                       dftPtr->d_projectorKetTimesVectorParFlattened
                           .getMPIPatternP2P()
                           ->globalToLocal(id) *
                         numberWaveFunctions,
                     &inc);

              dcopy_(&numberWaveFunctions,
                     dftPtr->d_projectorKetTimesVectorParFlattened.data() +
                       dftPtr->d_projectorKetTimesVectorParFlattened
                           .getMPIPatternP2P()
                           ->globalToLocal(id) *
                         numberWaveFunctions,
                     &inc,
                     &projectorKetTimesVector[atomId][numberWaveFunctions *
                                                      iPseudoAtomicWave],
                     &inc);
            }
        }
    }

  // start cell loop for assembling localHX and nonlocalHX simultaneously

  typename dealii::DoFHandler<3>::active_cell_iterator
    cell = dftPtr->dofHandler.begin_active(),
    endc = dftPtr->dofHandler.end();

  int iElem = -1;
  // blas required settings
  const char          transA1 = 'N';
  const char          transB1 = 'N';
  const double        beta1   = 1.0;
  const double        alpha2  = scalar;
  const double        alpha1  = 1.0;
  const unsigned int  inc1    = 1;
  std::vector<double> cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement *
                                                     numberWaveFunctions,
                                                   0.0);
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          iElem++;

          unsigned int indexTemp2 = indexTemp1 * iElem;

          dgemm_(&transA,
                 &transB,
                 &numberWaveFunctions,
                 &d_numberNodesPerElement,
                 &d_numberNodesPerElement,
                 &scalarCoeffAlpha1,
                 &cellSrcWaveFunctionMatrix[indexTemp2],
                 &numberWaveFunctions,
                 &d_cellHamiltonianMatrix[kpointSpinIndex][iElem][0],
                 &d_numberNodesPerElement,
                 &scalarCoeffBeta,
                 &cellHamMatrixTimesWaveMatrix[0],
                 &numberWaveFunctions);


          if (dftPtr->d_dftParamsPtr->isPseudopotential &&
              dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
            {
              for (unsigned int iAtom = 0;
                   iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();
                   ++iAtom)
                {
                  const unsigned int atomId =
                    dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
                  const unsigned int numberPseudoWaveFunctions =
                    dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
                  const int nonZeroElementMatrixId =
                    dftPtr->d_sparsityPattern[atomId][iElem];

                  dgemm_(&transA1,
                         &transB1,
                         &numberWaveFunctions,
                         &d_numberNodesPerElement,
                         &numberPseudoWaveFunctions,
                         &alpha2,
                         &projectorKetTimesVector[atomId][0],
                         &numberWaveFunctions,
                         &dftPtr->d_nonLocalProjectorElementMatricesTranspose
                            [atomId][nonZeroElementMatrixId][0],
                         &numberPseudoWaveFunctions,
                         &beta1,
                         &cellHamMatrixTimesWaveMatrix[0],
                         &numberWaveFunctions);
                }
            }

          cell->get_dof_indices(cell_dof_indicesGlobal);


          for (unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
            {
              if (d_nodesPerCellClassificationMap[iNode] == 1)
                {
                  dealii::types::global_dof_index localNodeId =
                    d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];


                  daxpy_(
                    &numberWaveFunctions,
                    &alpha1,
                    &cellHamMatrixTimesWaveMatrix[numberWaveFunctions * iNode],
                    &inc1,
                    dst.data() + localNodeId,
                    &inc1);
                }
              else
                {
                  unsigned int indexVal =
                    indexTemp2 + numberWaveFunctions * iNode;
                  if (scaleFlag)
                    {
                      dealii::types::global_dof_index localDoFId =
                        dftPtr->matrix_free_data.get_vector_partitioner()
                          ->global_to_local(cell_dof_indicesGlobal[iNode]);
                      const double scalingCoeff =
                        d_invSqrtMassVector.local_element(localDoFId);
                      const double invScalingCoeff =
                        d_sqrtMassVector.local_element(localDoFId);

                      for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                           ++iWave)
                        {
                          cellDstWaveFunctionMatrix[indexVal + iWave] =
                            scalarB *
                              cellDstWaveFunctionMatrix[indexVal + iWave] +
                            scalarA * invScalingCoeff *
                              cellSrcWaveFunctionMatrix[indexVal + iWave] +
                            scalingCoeff *
                              cellHamMatrixTimesWaveMatrix[numberWaveFunctions *
                                                             iNode +
                                                           iWave];
                        }
                    }
                  else
                    {
                      cell->get_dof_indices(cell_dof_indicesGlobal);
                      dealii::types::global_dof_index localDoFId =
                        dftPtr->matrix_free_data.get_vector_partitioner()
                          ->global_to_local(cell_dof_indicesGlobal[iNode]);
                      const double scalingCoeff =
                        d_invSqrtMassVector.local_element(localDoFId);

                      for (unsigned int iWave = 0; iWave < numberWaveFunctions;
                           ++iWave)
                        {
                          cellDstWaveFunctionMatrix[indexVal + iWave] +=
                            scalingCoeff *
                            cellHamMatrixTimesWaveMatrix[numberWaveFunctions *
                                                           iNode +
                                                         iWave];
                        }
                    }
                }
            }
        }
    } // cell loop
}
#endif
