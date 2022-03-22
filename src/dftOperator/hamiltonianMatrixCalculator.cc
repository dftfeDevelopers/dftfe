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
// @author  Phani Motamarri
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeHamiltonianMatrix(
  const unsigned int kPointIndex,
  const unsigned int spinIndex)
{
  dealii::TimerOutput computingTimerStandard(
    mpi_communicator,
    pcout,
    dftParameters::reproducible_output || dftParameters::verbosity < 2 ?
      dealii::TimerOutput::never :
      dealii::TimerOutput::every_call,
    dealii::TimerOutput::wall_times);

  computingTimerStandard.enter_subsection(
    "Elemental Hamiltonian matrix computation on CPU");

  //
  // Get the number of locally owned cells
  //
  const unsigned int numberMacroCells =
    dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();
  if (totalLocallyOwnedCells > 0)
    {
      const unsigned int kpointSpinIndex =
        (1 + dftParameters::spinPolarized) * kPointIndex + spinIndex;

      // inputs to blas
      const char         transA = 'N', transB = 'N';
      const char         transA1 = 'N', transB1 = 'T';
      const double       alpha = 1.0;
      const double       beta  = 1.0;
      const unsigned int inc   = 1;
      const unsigned int numberNodesPerElementSquare =
        d_numberNodesPerElement * d_numberNodesPerElement;
      const unsigned int sizeNiNj =
        (d_numberNodesPerElement * (d_numberNodesPerElement + 1)) / 2;
      const unsigned int fullSizeNiNj =
        d_numberNodesPerElement * d_numberNodesPerElement;
      unsigned int numBlocks              = (FEOrder + 1);
      unsigned int numberEntriesEachBlock = sizeNiNj / numBlocks;
      unsigned int blockCount             = 0;



      if ((dftParameters::isPseudopotential ||
           dftParameters::smearedNuclearCharges) &&
          !d_isStiffnessMatrixExternalPotCorrComputed)
        {
          const unsigned int numberDofsPerElement =
            dftPtr->matrix_free_data
              .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
              .get_fe()
              .dofs_per_cell;


          FEEvaluation<3,
                       FEOrder,
                       C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                       1,
                       double>
            fe_eval(dftPtr->matrix_free_data, 0, d_externalPotCorrQuadratureId);

          const unsigned int numberQuadraturePoints = fe_eval.n_q_points;

          d_cellHamiltonianMatrixExternalPotCorr.clear();
          d_cellHamiltonianMatrixExternalPotCorr.resize(
            sizeNiNj * totalLocallyOwnedCells, 0.0);


          std::vector<double> NiNjLpspQuad_currentBlock(
            numberEntriesEachBlock * numberQuadraturePoints, 0.0);


          AssertThrow(
            dftPtr->matrix_free_data
                .get_quadrature(d_externalPotCorrQuadratureId)
                .size() == numberQuadraturePoints,
            dealii::ExcMessage(
              "DFT-FE Error: mismatch in quadrature rule usage in computeHamiltonianMatrix."));


          unsigned int iNode, jNode, tempValue, tempValue1, startIndexINode;
          while (blockCount < numBlocks)
            {
              tempValue1 = numberEntriesEachBlock * blockCount;
              for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                   ++q_point)
                {
                  iNode =
                    d_blockiNodeIndex[numberEntriesEachBlock * blockCount];

                  tempValue = (numberDofsPerElement * iNode) -
                              (iNode * (iNode + 1)) / 2 - tempValue1;


                  for (jNode =
                         d_blockjNodeIndex[numberEntriesEachBlock * blockCount];
                       jNode < numberDofsPerElement;
                       ++jNode)
                    {
                      NiNjLpspQuad_currentBlock[numberEntriesEachBlock *
                                                  q_point +
                                                tempValue + jNode] =
                        d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                      q_point +
                                                    iNode] *
                        d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                      q_point +
                                                    jNode];
                    }

                  startIndexINode = iNode + 1;

                  for (iNode = startIndexINode;
                       iNode < d_blockiNodeIndex[numberEntriesEachBlock *
                                                   (blockCount + 1) -
                                                 1];
                       ++iNode)
                    {
                      double shapeI =
                        d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                      q_point +
                                                    iNode];


                      tempValue = (numberDofsPerElement * iNode) -
                                  (iNode * (iNode + 1)) / 2 - tempValue1;

                      for (jNode = iNode; jNode < numberDofsPerElement; ++jNode)
                        {
                          double shapeJ =
                            d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                          q_point +
                                                        jNode];

                          NiNjLpspQuad_currentBlock[numberEntriesEachBlock *
                                                      q_point +
                                                    tempValue + jNode] =
                            shapeI * shapeJ;

                        } // jNode
                    }     // iNode

                  iNode = d_blockiNodeIndex[numberEntriesEachBlock *
                                              (blockCount + 1) -
                                            1];

                  tempValue = (numberDofsPerElement * iNode) -
                              (iNode * (iNode + 1)) / 2 - tempValue1;

                  for (jNode = iNode;
                       jNode <= d_blockjNodeIndex[numberEntriesEachBlock *
                                                    (blockCount + 1) -
                                                  1];
                       ++jNode)
                    {
                      NiNjLpspQuad_currentBlock[numberEntriesEachBlock *
                                                  q_point +
                                                tempValue + jNode] =
                        d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                      q_point +
                                                    iNode] *
                        d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                      q_point +
                                                    jNode];
                    }

                } // quadPoint loop


              dgemm_(
                &transA1,
                &transB1,
                &totalLocallyOwnedCells, // M
                &numberEntriesEachBlock, // N
                &numberQuadraturePoints, // K
                &alpha,
                &d_vEffExternalPotCorrJxW[0],
                &totalLocallyOwnedCells,
                &NiNjLpspQuad_currentBlock[0],
                &numberEntriesEachBlock,
                &beta,
                &d_cellHamiltonianMatrixExternalPotCorr[totalLocallyOwnedCells *
                                                        numberEntriesEachBlock *
                                                        blockCount],
                &totalLocallyOwnedCells);

              blockCount += 1;
            }

          d_isStiffnessMatrixExternalPotCorrComputed = true;
          NiNjLpspQuad_currentBlock.clear();
          std::vector<double>().swap(NiNjLpspQuad_currentBlock);
        }


      //
      // Resize the cell-level hamiltonian  matrix
      //
      d_cellHamiltonianMatrix[kpointSpinIndex].clear();
      d_cellHamiltonianMatrix[kpointSpinIndex].resize(totalLocallyOwnedCells);

      //
      // Get some FE related Data
      //
      const Quadrature<3> &quadrature =
        dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);

      const unsigned int numberDofsPerElement =
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .get_fe()
          .dofs_per_cell;

      const unsigned int numberQuadraturePoints = quadrature.size();
      const unsigned int numberQuadraturePointsTimesNine =
        9 * numberQuadraturePoints;
      const unsigned int numberQuadraturePointsTimesThree =
        3 * numberQuadraturePoints;
      //
      // create temp storage for stiffness matrix across all cells
      //
      blockCount = 0;

      std::vector<double> cellHamiltonianMatrix(totalLocallyOwnedCells *
                                                  sizeNiNj,
                                                0.0);
      std::vector<double> NiNj_currentBlock(numberEntriesEachBlock *
                                              numberQuadraturePoints,
                                            0.0);
      unsigned int        iNode, jNode, tempValue, tempValue1, startIndexINode;
      while (blockCount < numBlocks)
        {
          tempValue1 = numberEntriesEachBlock * blockCount;
          for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
               ++q_point)
            {
              iNode = d_blockiNodeIndex[numberEntriesEachBlock * blockCount];

              tempValue = (numberDofsPerElement * iNode) -
                          (iNode * (iNode + 1)) / 2 - tempValue1;

              for (jNode =
                     d_blockjNodeIndex[numberEntriesEachBlock * blockCount];
                   jNode < numberDofsPerElement;
                   ++jNode)
                {
                  NiNj_currentBlock[numberEntriesEachBlock * q_point +
                                    tempValue + jNode] =
                    d_shapeFunctionData[numberDofsPerElement * q_point +
                                        iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];
                }

              startIndexINode = iNode + 1;

              for (iNode = startIndexINode;
                   iNode <
                   d_blockiNodeIndex[numberEntriesEachBlock * (blockCount + 1) -
                                     1];
                   ++iNode)
                {
                  double shapeI =
                    d_shapeFunctionData[numberDofsPerElement * q_point + iNode];


                  tempValue = (numberDofsPerElement * iNode) -
                              (iNode * (iNode + 1)) / 2 - tempValue1;

                  for (jNode = iNode; jNode < numberDofsPerElement; ++jNode)
                    {
                      double shapeJ =
                        d_shapeFunctionData[numberDofsPerElement * q_point +
                                            jNode];
                      NiNj_currentBlock[numberEntriesEachBlock * q_point +
                                        tempValue + jNode] = shapeI * shapeJ;

                    } // jNode
                }     // iNode

              iNode =
                d_blockiNodeIndex[numberEntriesEachBlock * (blockCount + 1) -
                                  1];
              tempValue = (numberDofsPerElement * iNode) -
                          (iNode * (iNode + 1)) / 2 - tempValue1;

              for (jNode = iNode;
                   jNode <=
                   d_blockjNodeIndex[numberEntriesEachBlock * (blockCount + 1) -
                                     1];
                   ++jNode)
                {
                  NiNj_currentBlock[numberEntriesEachBlock * q_point +
                                    tempValue + jNode] =
                    d_shapeFunctionData[numberDofsPerElement * q_point +
                                        iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];
                }

            } // quadPoint loop

          dgemm_(&transA1,
                 &transB1,
                 &totalLocallyOwnedCells, // M
                 &numberEntriesEachBlock, // N
                 &numberQuadraturePoints, // K
                 &alpha,
                 &d_vEffJxW[0],
                 &totalLocallyOwnedCells,
                 &NiNj_currentBlock[0],
                 &numberEntriesEachBlock,
                 &beta,
                 &cellHamiltonianMatrix[totalLocallyOwnedCells *
                                        numberEntriesEachBlock * blockCount],
                 &totalLocallyOwnedCells);

          blockCount += 1;
        }


      NiNj_currentBlock.clear();
      std::vector<double>().swap(NiNj_currentBlock);


      if (dftParameters::xcFamilyType == "GGA")
        {
          std::vector<double> gradNiNjPlusgradNjNi_currentBlock(
            numberEntriesEachBlock * 3 * numberQuadraturePoints, 0.0);
          blockCount = 0;
          unsigned int iNode, jNode, tempValue, tempValue1, startIndexINode;
          while (blockCount < numBlocks)
            {
              tempValue1 = numberEntriesEachBlock * blockCount;
              for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                   ++q_point)
                {
                  iNode =
                    d_blockiNodeIndex[numberEntriesEachBlock * blockCount];

                  tempValue = (numberDofsPerElement * iNode) -
                              (iNode * (iNode + 1)) / 2 - tempValue1;

                  for (jNode =
                         d_blockjNodeIndex[numberEntriesEachBlock * blockCount];
                       jNode < numberDofsPerElement;
                       ++jNode)
                    {
                      gradNiNjPlusgradNjNi_currentBlock
                        [3 * numberEntriesEachBlock * q_point + tempValue +
                         jNode] =
                          d_shapeFunctionGradientValueRefX
                              [numberDofsPerElement * q_point + iNode] *
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode] +
                          d_shapeFunctionData[numberDofsPerElement * q_point +
                                              iNode] *
                            d_shapeFunctionGradientValueRefX
                              [numberDofsPerElement * q_point + jNode];



                      gradNiNjPlusgradNjNi_currentBlock
                        [3 * numberEntriesEachBlock * q_point +
                         numberEntriesEachBlock + tempValue + jNode] =
                          d_shapeFunctionGradientValueRefY
                              [numberDofsPerElement * q_point + iNode] *
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode] +
                          d_shapeFunctionData[numberDofsPerElement * q_point +
                                              iNode] *
                            d_shapeFunctionGradientValueRefY
                              [numberDofsPerElement * q_point + jNode];



                      gradNiNjPlusgradNjNi_currentBlock
                        [3 * numberEntriesEachBlock * q_point +
                         2 * numberEntriesEachBlock + tempValue + jNode] =
                          d_shapeFunctionGradientValueRefZ
                              [numberDofsPerElement * q_point + iNode] *
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode] +
                          d_shapeFunctionData[numberDofsPerElement * q_point +
                                              iNode] *
                            d_shapeFunctionGradientValueRefZ
                              [numberDofsPerElement * q_point + jNode];
                    }


                  startIndexINode = iNode + 1;


                  for (iNode = startIndexINode;
                       iNode < d_blockiNodeIndex[numberEntriesEachBlock *
                                                   (blockCount + 1) -
                                                 1];
                       ++iNode)
                    {
                      double shapeGradXRefINode =
                        d_shapeFunctionGradientValueRefX[numberDofsPerElement *
                                                           q_point +
                                                         iNode];
                      double shapeGradYRefINode =
                        d_shapeFunctionGradientValueRefY[numberDofsPerElement *
                                                           q_point +
                                                         iNode];
                      double shapeGradZRefINode =
                        d_shapeFunctionGradientValueRefZ[numberDofsPerElement *
                                                           q_point +
                                                         iNode];
                      double shapeI =
                        d_shapeFunctionData[numberDofsPerElement * q_point +
                                            iNode];

                      tempValue = (numberDofsPerElement * iNode) -
                                  (iNode * (iNode + 1)) / 2 - tempValue1;

                      for (jNode = iNode; jNode < numberDofsPerElement; ++jNode)
                        {
                          double shapeJ =
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode];
                          gradNiNjPlusgradNjNi_currentBlock
                            [3 * numberEntriesEachBlock * q_point + tempValue +
                             jNode] =
                              shapeGradXRefINode * shapeJ +
                              shapeI *
                                d_shapeFunctionGradientValueRefX
                                  [numberDofsPerElement * q_point + jNode];

                          gradNiNjPlusgradNjNi_currentBlock
                            [3 * numberEntriesEachBlock * q_point +
                             numberEntriesEachBlock + tempValue + jNode] =
                              shapeGradYRefINode * shapeJ +
                              shapeI *
                                d_shapeFunctionGradientValueRefY
                                  [numberDofsPerElement * q_point + jNode];

                          gradNiNjPlusgradNjNi_currentBlock
                            [3 * numberEntriesEachBlock * q_point +
                             2 * numberEntriesEachBlock + tempValue + jNode] =
                              shapeGradZRefINode * shapeJ +
                              shapeI *
                                d_shapeFunctionGradientValueRefZ
                                  [numberDofsPerElement * q_point + jNode];

                        } // jnode
                    }     // iNode

                  iNode = d_blockiNodeIndex[numberEntriesEachBlock *
                                              (blockCount + 1) -
                                            1];

                  tempValue = (numberDofsPerElement * iNode) -
                              (iNode * (iNode + 1)) / 2 - tempValue1;

                  for (jNode = iNode;
                       jNode <= d_blockjNodeIndex[numberEntriesEachBlock *
                                                    (blockCount + 1) -
                                                  1];
                       ++jNode)
                    {
                      gradNiNjPlusgradNjNi_currentBlock
                        [3 * numberEntriesEachBlock * q_point + tempValue +
                         jNode] =
                          d_shapeFunctionGradientValueRefX
                              [numberDofsPerElement * q_point + iNode] *
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode] +
                          d_shapeFunctionData[numberDofsPerElement * q_point +
                                              iNode] *
                            d_shapeFunctionGradientValueRefX
                              [numberDofsPerElement * q_point + jNode];


                      gradNiNjPlusgradNjNi_currentBlock
                        [3 * numberEntriesEachBlock * q_point +
                         numberEntriesEachBlock + tempValue + jNode] =
                          d_shapeFunctionGradientValueRefY
                              [numberDofsPerElement * q_point + iNode] *
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode] +
                          d_shapeFunctionData[numberDofsPerElement * q_point +
                                              iNode] *
                            d_shapeFunctionGradientValueRefY
                              [numberDofsPerElement * q_point + jNode];



                      gradNiNjPlusgradNjNi_currentBlock
                        [3 * numberEntriesEachBlock * q_point +
                         2 * numberEntriesEachBlock + tempValue + jNode] =
                          d_shapeFunctionGradientValueRefZ
                              [numberDofsPerElement * q_point + iNode] *
                            d_shapeFunctionData[numberDofsPerElement * q_point +
                                                jNode] +
                          d_shapeFunctionData[numberDofsPerElement * q_point +
                                              iNode] *
                            d_shapeFunctionGradientValueRefZ
                              [numberDofsPerElement * q_point + jNode];
                    }

                } // quadPoint loop

              dgemm_(
                &transA1,
                &transB1,
                &totalLocallyOwnedCells,           // M
                &numberEntriesEachBlock,           // N
                &numberQuadraturePointsTimesThree, // K
                &alpha,
                &d_invJacderExcWithSigmaTimesGradRhoJxW[0],
                &totalLocallyOwnedCells,
                &gradNiNjPlusgradNjNi_currentBlock[0],
                &numberEntriesEachBlock,
                &beta,
                &cellHamiltonianMatrix[totalLocallyOwnedCells *
                                       numberEntriesEachBlock * blockCount],
                &totalLocallyOwnedCells);

              blockCount += 1;
            }

          gradNiNjPlusgradNjNi_currentBlock.clear();
          std::vector<double>().swap(gradNiNjPlusgradNjNi_currentBlock);
        }

      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

      //
      // access the kPoint coordinates
      //
#ifdef USE_COMPLEX
      std::vector<double> kPointCoors(3, 0.0);
      kPointCoors[0] = dftPtr->d_kPointCoordinates[3 * kPointIndex + 0];
      kPointCoors[1] = dftPtr->d_kPointCoordinates[3 * kPointIndex + 1];
      kPointCoors[2] = dftPtr->d_kPointCoordinates[3 * kPointIndex + 2];



      double kSquareTimesHalf = 0.5 * (kPointCoors[0] * kPointCoors[0] +
                                       kPointCoors[1] * kPointCoors[1] +
                                       kPointCoors[2] * kPointCoors[2]);

      unsigned int sizeNiNjNoSym = numberDofsPerElement * numberDofsPerElement;
      unsigned int numberEntriesEachBlockNoSym = sizeNiNjNoSym / numBlocks;


      std::vector<double> elementHamiltonianMatrixImag(totalLocallyOwnedCells *
                                                         sizeNiNjNoSym,
                                                       0.0);

      blockCount = 0;
      std::vector<double> gradNiNj_currentBlock(numberEntriesEachBlockNoSym *
                                                  3 * numberQuadraturePoints,
                                                0.0);

      while (blockCount < numBlocks)
        {
          tempValue1 = numberEntriesEachBlockNoSym * blockCount;
          for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
               ++q_point)
            {
              iNode     = d_blockiNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                             blockCount];
              tempValue = (numberDofsPerElement * iNode) - tempValue1;
              for (jNode = d_blockjNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                                  blockCount];
                   jNode < numberDofsPerElement;
                   ++jNode)
                {
                  gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                          q_point +
                                        tempValue + jNode] =
                    d_shapeFunctionGradientValueRefX[numberDofsPerElement *
                                                       q_point +
                                                     iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];

                  gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                          q_point +
                                        numberEntriesEachBlockNoSym +
                                        tempValue + jNode] =
                    d_shapeFunctionGradientValueRefY[numberDofsPerElement *
                                                       q_point +
                                                     iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];

                  gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                          q_point +
                                        2 * numberEntriesEachBlockNoSym +
                                        tempValue + jNode] =
                    d_shapeFunctionGradientValueRefZ[numberDofsPerElement *
                                                       q_point +
                                                     iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];
                }

              startIndexINode = iNode + 1;

              for (iNode = startIndexINode;
                   iNode < d_blockiNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                                    (blockCount + 1) -
                                                  1];
                   ++iNode)
                {
                  double shapeGradXRefINode =
                    d_shapeFunctionGradientValueRefX[numberDofsPerElement *
                                                       q_point +
                                                     iNode];
                  double shapeGradYRefINode =
                    d_shapeFunctionGradientValueRefY[numberDofsPerElement *
                                                       q_point +
                                                     iNode];
                  double shapeGradZRefINode =
                    d_shapeFunctionGradientValueRefZ[numberDofsPerElement *
                                                       q_point +
                                                     iNode];


                  tempValue = (numberDofsPerElement * iNode) - tempValue1;
                  for (jNode = 0; jNode < numberDofsPerElement; ++jNode)
                    {
                      double shapeJ =
                        d_shapeFunctionData[numberDofsPerElement * q_point +
                                            jNode];
                      gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                              q_point +
                                            tempValue + jNode] =
                        shapeGradXRefINode * shapeJ;
                      gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                              q_point +
                                            numberEntriesEachBlockNoSym +
                                            tempValue + jNode] =
                        shapeGradYRefINode * shapeJ;
                      gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                              q_point +
                                            2 * numberEntriesEachBlockNoSym +
                                            tempValue + jNode] =
                        shapeGradZRefINode * shapeJ;
                    } // jnode
                }     // iNode

              iNode     = d_blockiNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                               (blockCount + 1) -
                                             1];
              tempValue = (numberDofsPerElement * iNode) - tempValue1;
              for (jNode = 0;
                   jNode <= d_blockjNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                                     (blockCount + 1) -
                                                   1];
                   ++jNode)
                {
                  gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                          q_point +
                                        tempValue + jNode] =
                    d_shapeFunctionGradientValueRefX[numberDofsPerElement *
                                                       q_point +
                                                     iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];

                  gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                          q_point +
                                        numberEntriesEachBlockNoSym +
                                        tempValue + jNode] =
                    d_shapeFunctionGradientValueRefY[numberDofsPerElement *
                                                       q_point +
                                                     iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];

                  gradNiNj_currentBlock[3 * numberEntriesEachBlockNoSym *
                                          q_point +
                                        2 * numberEntriesEachBlockNoSym +
                                        tempValue + jNode] =
                    d_shapeFunctionGradientValueRefZ[numberDofsPerElement *
                                                       q_point +
                                                     iNode] *
                    d_shapeFunctionData[numberDofsPerElement * q_point + jNode];
                }
            } // quadpoint loop



          dgemm_(&transA1,
                 &transB1,
                 &totalLocallyOwnedCells,           // M
                 &numberEntriesEachBlockNoSym,      // N
                 &numberQuadraturePointsTimesThree, // K
                 &alpha,
                 &d_invJacKPointTimesJxW[kPointIndex][0],
                 &totalLocallyOwnedCells,
                 &gradNiNj_currentBlock[0],
                 &numberEntriesEachBlockNoSym,
                 &beta,
                 &elementHamiltonianMatrixImag[totalLocallyOwnedCells *
                                               numberEntriesEachBlockNoSym *
                                               blockCount],
                 &totalLocallyOwnedCells);

          blockCount += 1;
        }

      gradNiNj_currentBlock.clear();
      std::vector<double>().swap(gradNiNj_currentBlock);
#endif

      //
      // compute cell-level stiffness matrix by going over dealii macrocells
      // which allows efficient integration of cell-level stiffness matrix
      // integrals using dealii vectorized arrays

      for (unsigned int iElem = 0; iElem < totalLocallyOwnedCells; ++iElem)
        {
          // FIXME: Use functions like mkl_malloc for 64 byte memory
          // alignment.
          d_cellHamiltonianMatrix[kpointSpinIndex][iElem].resize(
            numberDofsPerElement * numberDofsPerElement, 0.0);
          unsigned int count = 0;
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                   ++jNode)
                {
#ifdef USE_COMPLEX
                  d_cellHamiltonianMatrix
                    [kpointSpinIndex][iElem]
                    [numberDofsPerElement * iNode + jNode]
                      .real(
                        cellHamiltonianMatrix[totalLocallyOwnedCells * count +
                                              iElem] +
                        0.5 *
                          d_cellShapeFunctionGradientIntegral[sizeNiNj * iElem +
                                                              count] +
                        kSquareTimesHalf *
                          d_NiNjIntegral[sizeNiNj * iElem + count]);


#else
                  d_cellHamiltonianMatrix
                    [kpointSpinIndex][iElem]
                    [numberDofsPerElement * iNode + jNode] =
                      cellHamiltonianMatrix[totalLocallyOwnedCells * count +
                                            iElem] +
                      0.5 *
                        d_cellShapeFunctionGradientIntegral[sizeNiNj * iElem +
                                                            count];

#endif
                  count += 1;
                }
            }

          if (dftParameters::isPseudopotential ||
              dftParameters::smearedNuclearCharges)
            {
              count = 0;
              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                     ++jNode)
                  {
#ifdef USE_COMPLEX
                    d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                           [numberDofsPerElement * iNode +
                                            jNode] +=
                      dataTypes::number(
                        d_cellHamiltonianMatrixExternalPotCorr
                          [totalLocallyOwnedCells * count + iElem],
                        0.0);

#else
                    d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                           [numberDofsPerElement * iNode +
                                            jNode] +=
                      d_cellHamiltonianMatrixExternalPotCorr
                        [totalLocallyOwnedCells * count + iElem];


#endif
                    count += 1;
                  }
            }

#ifdef USE_COMPLEX
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for (unsigned int jNode = 0; jNode < iNode; ++jNode)
              d_cellHamiltonianMatrix
                [kpointSpinIndex][iElem][numberDofsPerElement * iNode + jNode] =
                  d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * jNode + iNode];


          count = 0;
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = 0; jNode < numberDofsPerElement;
                   ++jNode)
                {
                  d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * iNode + jNode]
                                           .imag(
                                             elementHamiltonianMatrixImag
                                               [totalLocallyOwnedCells * count +
                                                iElem]);
                  count += 1;
                }
            }
#else
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for (unsigned int jNode = 0; jNode < iNode; ++jNode)
              d_cellHamiltonianMatrix
                [kpointSpinIndex][iElem][numberDofsPerElement * iNode + jNode] =
                  d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * jNode + iNode];
#endif


        } // cell loop
    }
  computingTimerStandard.leave_subsection(
    "Elemental Hamiltonian matrix computation on CPU");
}

// This piece of code needs to be changed later when required as this is still
// in macrocell subcell mode
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeKineticMatrix()
{
  //
  // Get the number of locally owned cells
  //
  const unsigned int numberMacroCells =
    dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();

  //
  // Resize the cell-level hamiltonian  matrix
  //
  d_cellHamiltonianMatrix[0].clear();
  d_cellHamiltonianMatrix[0].resize(totalLocallyOwnedCells);

  //
  // Get some FE related Data
  //
  const Quadrature<3> &quadrature =
    dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
  FEEvaluation<3,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1,
               double>
                     fe_eval(dftPtr->matrix_free_data, 0, 0);
  FEValues<3>        fe_values(dftPtr->matrix_free_data
                          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                          .get_fe(),
                        quadrature,
                        update_gradients);
  const unsigned int numberDofsPerElement =
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
      .get_fe()
      .dofs_per_cell;


  //
  // compute cell-level stiffness matrix by going over dealii macrocells
  // which allows efficient integration of cell-level stiffness matrix integrals
  // using dealii vectorized arrays
  unsigned int iElem = 0;
  for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {
      dealii::AlignedVector<VectorizedArray<double>> elementHamiltonianMatrix;
      elementHamiltonianMatrix.resize(numberDofsPerElement *
                                      numberDofsPerElement);
      fe_eval.reinit(iMacroCell);
      const unsigned int n_sub_cells =
        dftPtr->matrix_free_data.n_components_filled(iMacroCell);

      for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
        {
          for (unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
            {
              elementHamiltonianMatrix[numberDofsPerElement * iNode + jNode] =
                d_cellShapeFunctionGradientIntegral
                  [numberDofsPerElement * numberDofsPerElement * iElem +
                   numberDofsPerElement * iNode + jNode];

            } // jNode loop

        } // iNode loop


      for (unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
        {
          // FIXME: Use functions like mkl_malloc for 64 byte memory alignment.
          d_cellHamiltonianMatrix[0][iElem].resize(numberDofsPerElement *
                                                     numberDofsPerElement,
                                                   0.0);

          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = 0; jNode < numberDofsPerElement;
                   ++jNode)
                {
                  d_cellHamiltonianMatrix[0][iElem][numberDofsPerElement *
                                                      iNode +
                                                    jNode] =
                    elementHamiltonianMatrix[numberDofsPerElement * iNode +
                                             jNode][iSubCell];
                }
            }

          iElem += 1;
        }

    } // macrocell loop
}
