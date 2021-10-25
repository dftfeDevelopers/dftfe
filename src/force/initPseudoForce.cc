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


#include "nonlocalProjectorKetTimesEigenVectors.cc"

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::computeNonLocalProjectorKetTimesPsiTimesV(
  const std::vector<distributedCPUVec<double>> &src,
  std::vector<std::vector<double>> &            projectorKetTimesPsiTimesVReal,
  std::vector<std::vector<std::complex<double>>>
    &                projectorKetTimesPsiTimesVComplex,
  const unsigned int kPointIndex)
{
  //
  // get FE data
  //
  const Quadrature<3> &quadrature_formula =
    dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);

  const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;

#ifdef USE_COMPLEX
  const unsigned int numberNodesPerElement =
    dftPtr->FEEigen.dofs_per_cell / 2; // GeometryInfo<3>::vertices_per_cell;
#else
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  //
  // compute nonlocal projector ket times x i.e C^{T}*X
  //
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef USE_COMPLEX
  std::vector<std::vector<std::complex<double>>> &projectorKetTimesVector =
    projectorKetTimesPsiTimesVComplex;
#else
  std::vector<std::vector<double>> &projectorKetTimesVector =
    projectorKetTimesPsiTimesVReal;
#endif

  const unsigned int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();
  projectorKetTimesVector.resize(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());

  //
  // allocate memory for matrix-vector product
  //
  std::map<unsigned int, unsigned int> globalToLocalMap;
  for (int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
       ++iAtom)
    {
      const int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      globalToLocalMap[atomId] = iAtom;
      int numberSingleAtomPseudoWaveFunctions =
        dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[iAtom].resize(
        numberWaveFunctions * numberSingleAtomPseudoWaveFunctions, 0.0);
    }

    //
    // some useful vectors
    //
#ifdef USE_COMPLEX
  std::vector<std::complex<double>> inputVectors(numberNodesPerElement *
                                                   numberWaveFunctions,
                                                 0.0);
#else
  std::vector<double> inputVectors(numberNodesPerElement * numberWaveFunctions,
                                   0.0);
#endif


  //
  // parallel loop over all elements to compute nonlocal projector ket times x
  // i.e C^{T}*X
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandlerEigen
                                                        .begin_active(),
                                               endc =
                                                 dftPtr->dofHandlerEigen.end();
  int iElem = -1;
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          iElem += 1;
          cell->get_dof_indices(local_dof_indices);

          unsigned int index = 0;
#ifdef USE_COMPLEX
          std::vector<double> temp(dofs_per_cell, 0.0);
          for (std::vector<distributedCPUVec<double>>::const_iterator it =
                 src.begin();
               it != src.end();
               it++)
            {
              (*it).extract_subvector_to(local_dof_indices.begin(),
                                         local_dof_indices.end(),
                                         temp.begin());
              for (int idof = 0; idof < dofs_per_cell; ++idof)
                {
                  //
                  // This is the component index 0(real) or 1(imag).
                  //
                  const unsigned int ck =
                    dftPtr->FEEigen.system_to_component_index(idof).first;
                  const unsigned int iNode =
                    dftPtr->FEEigen.system_to_component_index(idof).second;
                  if (ck == 0)
                    inputVectors[numberNodesPerElement * index + iNode].real(
                      temp[idof]);
                  else
                    inputVectors[numberNodesPerElement * index + iNode].imag(
                      temp[idof]);
                }
              index++;
            }


#else
          for (std::vector<distributedCPUVec<double>>::const_iterator it =
                 src.begin();
               it != src.end();
               it++)
            {
              (*it).extract_subvector_to(local_dof_indices.begin(),
                                         local_dof_indices.end(),
                                         inputVectors.begin() +
                                           numberNodesPerElement * index);
              index++;
            }
#endif

          for (int iAtom = 0;
               iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();
               ++iAtom)
            {
              int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
              const unsigned int numberPseudoWaveFunctions =
                dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
              int nonZeroElementMatrixId =
                dftPtr->d_sparsityPattern[atomId][iElem];
#ifdef USE_COMPLEX
              char                 transA = 'N';
              char                 transB = 'N';
              std::complex<double> alpha  = 1.0;
              std::complex<double> beta   = 1.0;
              zgemm_(&transA,
                     &transB,
                     &numberPseudoWaveFunctions,
                     &numberWaveFunctions,
                     &numberNodesPerElement,
                     &alpha,
                     &dftPtr->d_nonLocalProjectorElementMatricesConjugate
                        [atomId][nonZeroElementMatrixId]
                        [kPointIndex * numberNodesPerElement *
                         numberPseudoWaveFunctions],
                     &numberNodesPerElement,
                     &inputVectors[0],
                     &numberNodesPerElement,
                     &beta,
                     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
                     &numberPseudoWaveFunctions);
#else
              char   transA = 'T';
              char   transB = 'N';
              double alpha  = 1.0;
              double beta   = 1.0;
              dgemm_(&transA,
                     &transB,
                     &numberPseudoWaveFunctions,
                     &numberWaveFunctions,
                     &numberNodesPerElement,
                     &alpha,
                     &dftPtr->d_nonLocalProjectorElementMatricesConjugate
                        [atomId][nonZeroElementMatrixId][0],
                     &numberNodesPerElement,
                     &inputVectors[0],
                     &numberNodesPerElement,
                     &beta,
                     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
                     &numberPseudoWaveFunctions);
#endif
            }
        }

    } // element loop

    // std::cout<<"Finished Element Loop"<<std::endl;
#ifdef USE_COMPLEX
  std::vector<distributedCPUVec<std::complex<double>>>
    projectorKetTimesVectorPar(numberWaveFunctions);
#else
  std::vector<distributedCPUVec<double>> projectorKetTimesVectorPar(
    numberWaveFunctions);
#endif
#ifdef USE_COMPLEX
  distributedCPUVec<std::complex<double>> vec(
    dftPtr->d_locallyOwnedProjectorIdsCurrentProcess,
    dftPtr->d_ghostProjectorIdsCurrentProcess,
    mpi_communicator);
#else
  distributedCPUVec<double> vec(
    dftPtr->d_locallyOwnedProjectorIdsCurrentProcess,
    dftPtr->d_ghostProjectorIdsCurrentProcess,
    mpi_communicator);
#endif
  vec.update_ghost_values();
  for (unsigned int i = 0; i < numberWaveFunctions; ++i)
    {
#ifdef USE_COMPLEX
      projectorKetTimesVectorPar[i].reinit(vec);
#else
      projectorKetTimesVectorPar[i].reinit(vec);
#endif
    }

  for (unsigned int iAtom = 0;
       iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
       ++iAtom)
    {
      const int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =
        dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
        {
          for (unsigned int iPseudoAtomicWave = 0;
               iPseudoAtomicWave < numberPseudoWaveFunctions;
               ++iPseudoAtomicWave)
            {
              projectorKetTimesVectorPar
                [iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess
                          [std::make_pair(atomId, iPseudoAtomicWave)]] =
                  projectorKetTimesVector[iAtom]
                                         [numberPseudoWaveFunctions * iWave +
                                          iPseudoAtomicWave];
            }
        }
    }

  for (unsigned int i = 0; i < numberWaveFunctions; ++i)
    {
      projectorKetTimesVectorPar[i].compress(VectorOperation::add);
      projectorKetTimesVectorPar[i].update_ghost_values();
    }

  for (unsigned int iAtom = 0;
       iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
       ++iAtom)
    {
      const int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =
        dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
        {
          for (unsigned int iPseudoAtomicWave = 0;
               iPseudoAtomicWave < numberPseudoWaveFunctions;
               ++iPseudoAtomicWave)
            {
              projectorKetTimesVector[iAtom][numberPseudoWaveFunctions * iWave +
                                             iPseudoAtomicWave] =
                projectorKetTimesVectorPar
                  [iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess
                            [std::make_pair(atomId, iPseudoAtomicWave)]];
            }
        }
    }


  //
  // compute V*C^{T}*X
  //
  for (unsigned int iAtom = 0;
       iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
       ++iAtom)
    {
      const int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =
        dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for (unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
        {
          for (unsigned int iPseudoAtomicWave = 0;
               iPseudoAtomicWave < numberPseudoWaveFunctions;
               ++iPseudoAtomicWave)
            projectorKetTimesVector[iAtom][numberPseudoWaveFunctions * iWave +
                                           iPseudoAtomicWave] *=
              dftPtr
                ->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
        }
    }
}
