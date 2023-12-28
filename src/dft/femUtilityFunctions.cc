// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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
#include <dft.h>

namespace dftfe
{
  //
  // interpolate nodal data to quadrature values using FEEvaluation
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void dftClass<FEOrder, FEOrderElectro>::
    interpolateRhoNodalDataToQuadratureDataGeneral(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalField,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureHessianValueData,
      const bool                                     isEvaluateGradData,
      const bool                                     isEvaluateHessianData)
  {
    quadratureValueData.clear();
    if (isEvaluateGradData)
      quadratureGradValueData.clear();
    if (isEvaluateHessianData)
      quadratureHessianValueData.clear();

    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                       feEvalObj(matrixFreeData, dofHandlerId, quadratureId);
    const unsigned int numQuadPoints = feEvalObj.n_q_points;

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches(); ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true,
                           isEvaluateGradData ? true : false,
                           isEvaluateHessianData ? true : false);

        for (unsigned int iSubCell = 0;
             iSubCell < matrixFreeData.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr =
              matrixFreeData.get_cell_iterator(cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId       = subCellPtr->id();
            quadratureValueData[subCellId] = std::vector<double>(numQuadPoints);

            std::vector<double> &tempVec =
              quadratureValueData.find(subCellId)->second;

            for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                quadratureGradValueData[subCellId] =
                  std::vector<double>(3 * numQuadPoints);

                std::vector<double> &tempVec2 =
                  quadratureGradValueData.find(subCellId)->second;

                for (unsigned int q_point = 0; q_point < numQuadPoints;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }

            if (isEvaluateHessianData)
              {
                quadratureHessianValueData[subCellId] =
                  std::vector<double>(9 * numQuadPoints);

                std::vector<double> &tempVec3 =
                  quadratureHessianValueData[subCellId];

                for (unsigned int q_point = 0; q_point < numQuadPoints;
                     ++q_point)
                  {
                    const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
                      &hessianVals = feEvalObj.get_hessian(q_point);
                    for (unsigned int i = 0; i < 3; i++)
                      for (unsigned int j = 0; j < 3; j++)
                        tempVec3[9 * q_point + 3 * i + j] =
                          hessianVals[i][j][iSubCell];
                  }
              }
          }
      }
  }
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::
    interpolateDensityNodalDataToQuadratureDataGeneral(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                              basisOperationsPtr,
      const unsigned int               dofHandlerId,
      const unsigned int               quadratureId,
      const distributedCPUVec<double> &nodalField,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureGradValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        quadratureHessianValueData,
      const bool isEvaluateGradData,
      const bool isEvaluateHessianData)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells        = basisOperationsPtr->nCells();
    quadratureValueData.clear();
    quadratureValueData.resize(nQuadsPerCell * nCells);
    if (isEvaluateGradData)
      {
        quadratureGradValueData.clear();
        quadratureGradValueData.resize(3 * nQuadsPerCell * nCells);
      }
    if (isEvaluateHessianData)
      {
        quadratureHessianValueData.clear();
        quadratureHessianValueData.resize(9 * nQuadsPerCell * nCells);
      }


    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
      feEvalObj(basisOperationsPtr->matrixFreeData(),
                dofHandlerId,
                quadratureId);

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      basisOperationsPtr->matrixFreeData()
          .get_quadrature(quadratureId)
          .size() == nQuadsPerCell,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0;
         cell < basisOperationsPtr->matrixFreeData().n_cell_batches();
         ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true,
                           isEvaluateGradData ? true : false,
                           isEvaluateHessianData ? true : false);

        for (unsigned int iSubCell = 0;
             iSubCell < basisOperationsPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr = basisOperationsPtr->matrixFreeData().get_cell_iterator(
              cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            unsigned int   cellIndex = basisOperationsPtr->cellIndex(subCellId);

            double *tempVec =
              quadratureValueData.data() + cellIndex * nQuadsPerCell;

            for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                double *tempVec2 = quadratureGradValueData.data() +
                                   3 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }

            if (isEvaluateHessianData)
              {
                double *tempVec3 = quadratureHessianValueData.data() +
                                   9 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
                      &hessianVals = feEvalObj.get_hessian(q_point);
                    for (unsigned int i = 0; i < 3; i++)
                      for (unsigned int j = 0; j < 3; j++)
                        tempVec3[9 * q_point + 3 * i + j] =
                          hessianVals[i][j][iSubCell];
                  }
              }
          }
      }
  }


  //
  // interpolate spin rho nodal data to quadrature values using FEEvaluation
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void dftClass<FEOrder, FEOrderElectro>::
    interpolateRhoSpinNodalDataToQuadratureDataGeneral(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalFieldSpin0,
      const distributedCPUVec<double> &              nodalFieldSpin1,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureHessianValueData,
      const bool                                     isEvaluateGradData,
      const bool                                     isEvaluateHessianData)
  {
    quadratureValueData.clear();
    if (isEvaluateGradData)
      quadratureGradValueData.clear();
    if (isEvaluateHessianData)
      quadratureHessianValueData.clear();

    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
      feEvalObjSpin0(matrixFreeData, dofHandlerId, quadratureId);
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                       feEvalObjSpin1(matrixFreeData, dofHandlerId, quadratureId);
    const unsigned int numQuadPoints = feEvalObjSpin0.n_q_points;

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches(); ++cell)
      {
        feEvalObjSpin0.reinit(cell);
        feEvalObjSpin0.read_dof_values(nodalFieldSpin0);
        feEvalObjSpin0.evaluate(true,
                                isEvaluateGradData ? true : false,
                                isEvaluateHessianData ? true : false);

        feEvalObjSpin1.reinit(cell);
        feEvalObjSpin1.read_dof_values(nodalFieldSpin1);
        feEvalObjSpin1.evaluate(true,
                                isEvaluateGradData ? true : false,
                                isEvaluateHessianData ? true : false);

        for (unsigned int iSubCell = 0;
             iSubCell < matrixFreeData.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr =
              matrixFreeData.get_cell_iterator(cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            quadratureValueData[subCellId] =
              std::vector<double>(2 * numQuadPoints);

            std::vector<double> &tempVec =
              quadratureValueData.find(subCellId)->second;

            for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
              {
                tempVec[2 * q_point + 0] =
                  feEvalObjSpin0.get_value(q_point)[iSubCell];
                tempVec[2 * q_point + 1] =
                  feEvalObjSpin1.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                quadratureGradValueData[subCellId] =
                  std::vector<double>(6 * numQuadPoints);

                std::vector<double> &tempVec2 =
                  quadratureGradValueData.find(subCellId)->second;

                for (unsigned int q_point = 0; q_point < numQuadPoints;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradValsSpin0 = feEvalObjSpin0.get_gradient(q_point);
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradValsSpin1 = feEvalObjSpin1.get_gradient(q_point);
                    tempVec2[6 * q_point + 0] = gradValsSpin0[0][iSubCell];
                    tempVec2[6 * q_point + 1] = gradValsSpin0[1][iSubCell];
                    tempVec2[6 * q_point + 2] = gradValsSpin0[2][iSubCell];
                    tempVec2[6 * q_point + 3] = gradValsSpin1[0][iSubCell];
                    tempVec2[6 * q_point + 4] = gradValsSpin1[1][iSubCell];
                    tempVec2[6 * q_point + 5] = gradValsSpin1[2][iSubCell];
                  }
              }

            if (isEvaluateHessianData)
              {
                quadratureHessianValueData[subCellId] =
                  std::vector<double>(18 * numQuadPoints);

                std::vector<double> &tempVec3 =
                  quadratureHessianValueData[subCellId];

                for (unsigned int q_point = 0; q_point < numQuadPoints;
                     ++q_point)
                  {
                    const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
                      &hessianValsSpin0 = feEvalObjSpin0.get_hessian(q_point);
                    const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
                      &hessianValsSpin1 = feEvalObjSpin1.get_hessian(q_point);
                    for (unsigned int i = 0; i < 3; i++)
                      for (unsigned int j = 0; j < 3; j++)
                        {
                          tempVec3[18 * q_point + 3 * i + j] =
                            hessianValsSpin0[i][j][iSubCell];
                          tempVec3[18 * q_point + 9 + 3 * i + j] =
                            hessianValsSpin1[i][j][iSubCell];
                        }
                  }
              }
          }
      }
  }

  //
  // interpolate nodal data to quadrature values using FEEvaluation
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void dftClass<FEOrder, FEOrderElectro>::
    interpolateElectroNodalDataToQuadratureDataGeneral(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalField,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      const bool                                     isEvaluateGradData)
  {
    quadratureValueData.clear();
    if (isEvaluateGradData)
      quadratureGradValueData.clear();

    dealii::FEEvaluation<
      3,
      FEOrderElectro,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                       feEvalObj(matrixFreeData, dofHandlerId, quadratureId);
    const unsigned int numQuadPoints = feEvalObj.n_q_points;

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches(); ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true, isEvaluateGradData ? true : false);

        for (unsigned int iSubCell = 0;
             iSubCell < matrixFreeData.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr =
              matrixFreeData.get_cell_iterator(cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId       = subCellPtr->id();
            quadratureValueData[subCellId] = std::vector<double>(numQuadPoints);

            std::vector<double> &tempVec =
              quadratureValueData.find(subCellId)->second;

            for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                quadratureGradValueData[subCellId] =
                  std::vector<double>(3 * numQuadPoints);

                std::vector<double> &tempVec2 =
                  quadratureGradValueData.find(subCellId)->second;

                for (unsigned int q_point = 0; q_point < numQuadPoints;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }
          }
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void dftClass<FEOrder, FEOrderElectro>::
    interpolateRhoNodalDataToQuadratureDataLpsp(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalField,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      const bool                                     isEvaluateGradData)
  {
    quadratureValueData.clear();
    quadratureGradValueData.clear();
    dealii::FEEvaluation<3,
                         C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
                         C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                         1,
                         double>
                       feEvalObj(matrixFreeData, dofHandlerId, quadratureId);
    const unsigned int numQuadPoints = feEvalObj.n_q_points;

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches(); ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true, true);
        for (unsigned int iSubCell = 0;
             iSubCell < matrixFreeData.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr =
              matrixFreeData.get_cell_iterator(cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId       = subCellPtr->id();
            quadratureValueData[subCellId] = std::vector<double>(numQuadPoints);
            std::vector<double> &tempVec =
              quadratureValueData.find(subCellId)->second;
            for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }
          }

        if (isEvaluateGradData)
          {
            for (unsigned int iSubCell = 0;
                 iSubCell <
                 matrixFreeData.n_active_entries_per_cell_batch(cell);
                 ++iSubCell)
              {
                subCellPtr = matrixFreeData.get_cell_iterator(cell,
                                                              iSubCell,
                                                              dofHandlerId);
                dealii::CellId subCellId = subCellPtr->id();
                quadratureGradValueData[subCellId] =
                  std::vector<double>(3 * numQuadPoints);
                std::vector<double> &tempVec =
                  quadratureGradValueData.find(subCellId)->second;
                for (unsigned int q_point = 0; q_point < numQuadPoints;
                     ++q_point)
                  {
                    tempVec[3 * q_point + 0] =
                      feEvalObj.get_gradient(q_point)[0][iSubCell];
                    tempVec[3 * q_point + 1] =
                      feEvalObj.get_gradient(q_point)[1][iSubCell];
                    tempVec[3 * q_point + 2] =
                      feEvalObj.get_gradient(q_point)[2][iSubCell];
                  }
              }
          }
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::
    interpolateDensityNodalDataToQuadratureDataLpsp(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
        &                              basisOperationsPtr,
      const unsigned int               dofHandlerId,
      const unsigned int               quadratureId,
      const distributedCPUVec<double> &nodalField,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &quadratureGradValueData,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &        quadratureHessianValueData,
      const bool isEvaluateGradData)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells        = basisOperationsPtr->nCells();
    quadratureValueData.clear();
    quadratureValueData.resize(nQuadsPerCell * nCells);
    if (isEvaluateGradData)
      {
        quadratureGradValueData.clear();
        quadratureGradValueData.resize(3 * nQuadsPerCell * nCells);
      }
 


    dealii::FEEvaluation<3,
                         C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
                         C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                         1,
                         double>
                       feEvalObj(matrixFreeData, dofHandlerId, quadratureId);

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      basisOperationsPtr->matrixFreeData()
          .get_quadrature(quadratureId)
          .size() == nQuadsPerCell,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    for (unsigned int cell = 0;
         cell < basisOperationsPtr->matrixFreeData().n_cell_batches();
         ++cell)
      {
        feEvalObj.reinit(cell);
        feEvalObj.read_dof_values(nodalField);
        feEvalObj.evaluate(true,
                           isEvaluateGradData ? true : false,
                           false);

        for (unsigned int iSubCell = 0;
             iSubCell < basisOperationsPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr = basisOperationsPtr->matrixFreeData().get_cell_iterator(
              cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            unsigned int   cellIndex = basisOperationsPtr->cellIndex(subCellId);

            double *tempVec =
              quadratureValueData.data() + cellIndex * nQuadsPerCell;

            for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point)
              {
                tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
              }

            if (isEvaluateGradData)
              {
                double *tempVec2 = quadratureGradValueData.data() +
                                   3 * cellIndex * nQuadsPerCell;

                for (unsigned int q_point = 0; q_point < nQuadsPerCell;
                     ++q_point)
                  {
                    const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                      &gradVals               = feEvalObj.get_gradient(q_point);
                    tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
                    tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
                    tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
                  }
              }

          }
      }
  }


  //
  // compute field l2 norm
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::fieldGradl2Norm(
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const distributedCPUVec<double> &    nodalField)

  {
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                       fe_evalField(matrixFreeDataObject, 0, 0);
    const unsigned int numQuadPoints = fe_evalField.n_q_points;

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeDataObject.get_vector_partitioner(0)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeDataObject.get_quadrature(0).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::VectorizedArray<double> valueVectorized =
      dealii::make_vectorized_array(0.0);
    double value = 0.0;
    for (unsigned int cell = 0; cell < matrixFreeDataObject.n_cell_batches();
         ++cell)
      {
        fe_evalField.reinit(cell);
        fe_evalField.read_dof_values(nodalField);
        fe_evalField.evaluate(false, true);
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            dealii::VectorizedArray<double> temp =
              scalar_product(fe_evalField.get_gradient(q_point),
                             fe_evalField.get_gradient(q_point));
            fe_evalField.submit_value(temp, q_point);
          }

        valueVectorized += fe_evalField.integrate_value();
        for (unsigned int iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          value += valueVectorized[iSubCell];
      }

    return dealii::Utilities::MPI::sum(value, mpi_communicator);
  }

  //
  // compute l2 projection of quad data to nodal data
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::l2ProjectionQuadToNodal(
    const dealii::MatrixFree<3, double> &                matrixFreeDataObject,
    const dealii::AffineConstraints<double> &            constraintMatrix,
    const unsigned int                                   dofHandlerId,
    const unsigned int                                   quadratureId,
    const std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
    distributedCPUVec<double> &                          nodalField)
  {
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const unsigned int                                          q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q) {
          return quadratureValueData.find(cell->id())->second[q];
        };
    dealii::VectorTools::project<3, distributedCPUVec<double>>(
      dealii::MappingQ1<3, 3>(),
      matrixFreeDataObject.get_dof_handler(dofHandlerId),
      constraintMatrix,
      matrixFreeDataObject.get_quadrature(quadratureId),
      funcRho,
      nodalField);
    constraintMatrix.set_zero(nodalField);
  }
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::l2ProjectionQuadToNodal(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                                      basisOperationsPtr,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       dofHandlerId,
    const unsigned int                       quadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                        quadratureValueData,
    distributedCPUVec<double> &nodalField)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const unsigned int                                          q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q) {
          return quadratureValueData[basisOperationsPtr->cellIndex(cell->id()) *
                                       nQuadsPerCell +
                                     q];
        };
    dealii::VectorTools::project<3, distributedCPUVec<double>>(
      dealii::MappingQ1<3, 3>(),
      basisOperationsPtr->matrixFreeData().get_dof_handler(dofHandlerId),
      constraintMatrix,
      basisOperationsPtr->matrixFreeData().get_quadrature(quadratureId),
      funcRho,
      nodalField);
    constraintMatrix.set_zero(nodalField);
  }
#include "dft.inst.cc"

} // namespace dftfe
