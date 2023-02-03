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
// @author Phani Motamarri, Sambit Das
//


#ifdef USE_COMPLEX
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
  computeNonLocalProjectorKetTimesXTimesV(
    distributedCPUMultiVec<std::complex<double>> &src,
    distributedCPUMultiVec<std::complex<double>> &projectorKetTimesVectorFlattened,
    const unsigned int                       numberWaveFunctions)
{
  std::map<unsigned int, std::vector<std::complex<double>>>
    projectorKetTimesVector;

  //
  // allocate memory for matrix-vector product
  //
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


  std::vector<std::complex<double>> cellWaveFunctionMatrix(
    d_numberNodesPerElement * numberWaveFunctions, 0.0);

  //
  // blas required settings
  //
  const char                 transA = 'N';
  const char                 transB = 'N';
  const std::complex<double> alpha  = 1.0;
  const std::complex<double> beta   = 1.0;
  const unsigned int         inc    = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler
                                                        .begin_active(),
                                               endc = dftPtr->dofHandler.end();
  int iElem                                         = -1;
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          iElem++;
          if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size() > 0)
            {
              for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index localNodeId =
                    d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
                  zcopy_(&numberWaveFunctions,
                         src.data() + localNodeId,
                         &inc,
                         &cellWaveFunctionMatrix[numberWaveFunctions * iNode],
                         &inc);
                }
            }

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

              zgemm_(&transA,
                     &transB,
                     &numberWaveFunctions,
                     &numberPseudoWaveFunctions,
                     &d_numberNodesPerElement,
                     &alpha,
                     &cellWaveFunctionMatrix[0],
                     &numberWaveFunctions,
                     &dftPtr->d_nonLocalProjectorElementMatricesConjugate
                        [atomId][nonZeroElementMatrixId]
                        [d_kPointIndex * d_numberNodesPerElement *
                         numberPseudoWaveFunctions],
                     &d_numberNodesPerElement,
                     &beta,
                     &projectorKetTimesVector[atomId][0],
                     &numberWaveFunctions);
            }
        }

    } // cell loop

  projectorKetTimesVectorFlattened.setValue(std::complex<double>(0.0, 0.0));


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
          zcopy_(&numberWaveFunctions,
                 &projectorKetTimesVector[atomId][numberWaveFunctions *
                                                  iPseudoAtomicWave],
                 &inc,
                 &projectorKetTimesVectorFlattened.data()[projectorKetTimesVectorFlattened.getMPIPatternP2P()->globalToLocal(id)*numberWaveFunctions],
                 &inc);
        }
    }


  projectorKetTimesVectorFlattened.accumulateAddLocallyOwned();
  projectorKetTimesVectorFlattened.updateGhostValues();

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
          std::complex<double> nonlocalConstantV;
          nonlocalConstantV.real(
            dftPtr
              ->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave]);
          nonlocalConstantV.imag(0);

          const unsigned int id =
            dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(
              atomId, iPseudoAtomicWave)];

          zscal_(&numberWaveFunctions,
                 &nonlocalConstantV,
                 &projectorKetTimesVectorFlattened.data()[projectorKetTimesVectorFlattened.getMPIPatternP2P()->globalToLocal(id)*numberWaveFunctions],
                 &inc);
        }
    }
}
#else
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
  computeNonLocalProjectorKetTimesXTimesV(
    distributedCPUMultiVec<double> &src,
    distributedCPUMultiVec<double> &projectorKetTimesVectorFlattened,
    const unsigned int         numberWaveFunctions)
{
  std::map<unsigned int, std::vector<double>> projectorKetTimesVector;

  //
  // allocate memory for matrix-vector product
  //
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


  std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement *
                                               numberWaveFunctions,
                                             0.0);

  //
  // blas required settings
  //
  const char         transA = 'N';
  const char         transB = 'N';
  const double       alpha  = 1.0;
  const double       beta   = 1.0;
  const unsigned int inc    = 1;



  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler
                                                        .begin_active(),
                                               endc = dftPtr->dofHandler.end();
  int iElem                                         = -1;
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          iElem++;
          if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size() > 0)
            {
              for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                   ++iNode)
                {
                  dealii::types::global_dof_index localNodeId =
                    d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
                  dcopy_(&numberWaveFunctions,
                         src.data() + localNodeId,
                         &inc,
                         &cellWaveFunctionMatrix[numberWaveFunctions * iNode],
                         &inc);
                }
            }

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
                     &cellWaveFunctionMatrix[0],
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

  projectorKetTimesVectorFlattened.setValue(0.0);

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
                 &projectorKetTimesVectorFlattened.data()[projectorKetTimesVectorFlattened.getMPIPatternP2P()->globalToLocal(id)*numberWaveFunctions],
                 &inc);
        }
    }

  projectorKetTimesVectorFlattened.accumulateAddLocallyOwned();
  projectorKetTimesVectorFlattened.updateGhostValues();

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
            dftPtr
              ->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];

          const unsigned int id =
            dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(
              atomId, iPseudoAtomicWave)];

          dscal_(&numberWaveFunctions,
                 &nonlocalConstantV,
                 &projectorKetTimesVectorFlattened.data()[projectorKetTimesVectorFlattened.getMPIPatternP2P()->globalToLocal(id)*numberWaveFunctions],
                 &inc);
        }
    }
}

#endif
