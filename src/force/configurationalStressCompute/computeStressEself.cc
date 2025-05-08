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
#include <dftUtils.h>
#include <eshelbyTensor.h>

namespace dftfe
{
  // compute stress contribution from nuclear self energy
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::computeStressEself(
    const dealii::DoFHandler<3>         &dofHandlerElectro,
    const vselfBinsManager              &vselfBinsManagerElectro,
    const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
    const dftfe::uInt                    smearedChargeQuadratureId)
  {
#ifdef DEBUG
    double                       dummyTest = 0;
    dealii::Tensor<1, 3, double> dummyVec;
    dealii::Tensor<2, 3, double> dummyTensor;
#endif
    const std::vector<std::vector<double>> &atomLocations =
      dftPtr->atomLocations;
    const std::vector<std::vector<double>> &imagePositions =
      dftPtr->d_imagePositionsTrunc;
    const std::vector<double> &imageCharges      = dftPtr->d_imageChargesTrunc;
    const dftfe::uInt          numberGlobalAtoms = atomLocations.size();
    //
    // First add configurational stress contribution from the volume integral
    //
    dealii::QGauss<3>   quadrature(d_dftParams.densityQuadratureRule);
    dealii::FEValues<3> feVselfValues(dofHandlerElectro.get_fe(),
                                      quadrature,
                                      dealii::update_gradients |
                                        dealii::update_JxW_values);
    const dftfe::uInt   numQuadPoints = quadrature.size();
    const dftfe::uInt   numberBins =
      vselfBinsManagerElectro.getAtomIdsBins().size();

    std::vector<dealii::Tensor<1, 3, double>> gradVselfQuad(numQuadPoints);

    // kpoint group parallelization data structures
    const dftfe::uInt numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

    const dftfe::uInt kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
    std::vector<dftfe::Int> kptGroupLowHighPlusOneIndices;

    if (numberBins > 0)
      dftUtils::createKpointParallelizationIndices(
        dftPtr->interpoolcomm, numberBins, kptGroupLowHighPlusOneIndices);



    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
            iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
          {
            const std::vector<dealii::DoFHandler<3>::active_cell_iterator>
              &cellsVselfBallDofHandler =
                d_cellsVselfBallsDofHandlerElectro[iBin];
            const distributedCPUVec<double> &iBinVselfField =
              vselfBinsManagerElectro.getVselfFieldBins()[iBin];
            std::vector<dealii::DoFHandler<3>::active_cell_iterator>::
              const_iterator iter1;
            for (iter1 = cellsVselfBallDofHandler.begin();
                 iter1 != cellsVselfBallDofHandler.end();
                 ++iter1)
              {
                dealii::DoFHandler<3>::active_cell_iterator cell = *iter1;
                feVselfValues.reinit(cell);
                feVselfValues.get_function_gradients(iBinVselfField,
                                                     gradVselfQuad);

                for (dftfe::uInt qPoint = 0; qPoint < numQuadPoints; ++qPoint)
                  {
                    d_stress += eshelbyTensor::getVselfBallEshelbyTensor(
                                  gradVselfQuad[qPoint]) *
                                feVselfValues.JxW(qPoint);
                  } // q point loop
              }     // cell loop
          }         // kpt paral loop
      }             // bin loop


    //
    // second add configurational stress contribution from the surface integral
    //
    dealii::QGauss<3 - 1>   faceQuadrature(d_dftParams.densityQuadratureRule);
    dealii::FEFaceValues<3> feVselfFaceValues(
      dofHandlerElectro.get_fe(),
      faceQuadrature,
      dealii::update_gradients | dealii::update_JxW_values |
        dealii::update_normal_vectors | dealii::update_quadrature_points);
    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt numFaceQuadPoints = faceQuadrature.size();


    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
            iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
          {
            const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                           std::vector<dftfe::uInt>>
              &cellsVselfBallSurfacesDofHandler =
                d_cellFacesVselfBallSurfacesDofHandlerElectro[iBin];
            const distributedCPUVec<double> &iBinVselfField =
              vselfBinsManagerElectro.getVselfFieldBins()[iBin];
            std::map<dealii::DoFHandler<3>::active_cell_iterator,
                     std::vector<dftfe::uInt>>::const_iterator iter1;
            for (iter1 = cellsVselfBallSurfacesDofHandler.begin();
                 iter1 != cellsVselfBallSurfacesDofHandler.end();
                 ++iter1)
              {
                dealii::DoFHandler<3>::active_cell_iterator cell = iter1->first;
                const dftfe::Int                            closestAtomId =
                  d_cellsVselfBallsClosestAtomIdDofHandlerElectro[iBin]
                                                                 [cell->id()];
                double           closestAtomCharge;
                dealii::Point<3> closestAtomLocation;
                if (closestAtomId < numberGlobalAtoms)
                  {
                    closestAtomLocation[0] = atomLocations[closestAtomId][2];
                    closestAtomLocation[1] = atomLocations[closestAtomId][3];
                    closestAtomLocation[2] = atomLocations[closestAtomId][4];
                    if (d_dftParams.isPseudopotential)
                      closestAtomCharge = atomLocations[closestAtomId][1];
                    else
                      closestAtomCharge = atomLocations[closestAtomId][0];
                  }
                else
                  {
                    const dftfe::Int imageId =
                      closestAtomId - numberGlobalAtoms;
                    closestAtomCharge      = imageCharges[imageId];
                    closestAtomLocation[0] = imagePositions[imageId][0];
                    closestAtomLocation[1] = imagePositions[imageId][1];
                    closestAtomLocation[2] = imagePositions[imageId][2];
                  }

                const std::vector<dftfe::uInt> &dirichletFaceIds =
                  iter1->second;
                for (dftfe::uInt index = 0; index < dirichletFaceIds.size();
                     index++)
                  {
                    const dftfe::uInt faceId = dirichletFaceIds[index];
                    feVselfFaceValues.reinit(cell, faceId);

                    for (dftfe::uInt qPoint = 0; qPoint < numFaceQuadPoints;
                         ++qPoint)
                      {
                        const dealii::Point<3> quadPoint =
                          feVselfFaceValues.quadrature_point(qPoint);
                        const dealii::Tensor<1, 3, double> dispClosestAtom =
                          quadPoint - closestAtomLocation;
                        const double dist = dispClosestAtom.norm();
                        const dealii::Tensor<1, 3, double>
                          gradVselfFaceQuadExact =
                            closestAtomCharge * dispClosestAtom / dist / dist /
                            dist;
                        d_stress -=
                          outer_product(
                            dispClosestAtom,
                            eshelbyTensor::getVselfBallEshelbyTensor(
                              gradVselfFaceQuadExact) *
                              feVselfFaceValues.normal_vector(qPoint)) *
                          feVselfFaceValues.JxW(qPoint);
#ifdef DEBUG
                        dummyTest +=
                          scalar_product(gradVselfFaceQuadExact,
                                         feVselfFaceValues.normal_vector(
                                           qPoint)) *
                          feVselfFaceValues.JxW(qPoint);
                        dummyVec += feVselfFaceValues.normal_vector(qPoint) *
                                    feVselfFaceValues.JxW(qPoint);
                        dummyTensor +=
                          outer_product(gradVselfFaceQuadExact,
                                        feVselfFaceValues.normal_vector(
                                          qPoint)) *
                          feVselfFaceValues.JxW(qPoint);
#endif

                      } // q point loop
                  }     // face loop
              }         // cell loop
          }             // kpt paral loop
      }                 // bin loop

    //
    // Add stress due to smeared charges
    //
    if (d_dftParams.smearedNuclearCharges)
      {
        const std::map<dftfe::Int, std::set<dftfe::Int>> &atomImageIdsBins =
          vselfBinsManagerElectro.getAtomImageIdsBins();

        FEEvaluationWrapperClass<3> forceEvalSmearedCharge(
          -1,
          1,
          matrixFreeDataElectro,
          d_forceDofHandlerIndexElectro,
          smearedChargeQuadratureId);

        dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
        const dftfe::uInt                           numQuadPointsSmearedb =
          forceEvalSmearedCharge.n_q_points;

        dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            zeroTensor[idim] = dealii::make_vectorized_array(0.0);
          }

        dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
            {
              zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
            }

        dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
          numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradVselfSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);

        for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
          {
            if (iBin < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                iBin >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                FEEvaluationWrapperClass<1> vselfEvalSmearedCharge(
                  -1,
                  1,
                  matrixFreeDataElectro,
                  dftPtr->d_binsStartDofHandlerIndexElectro + 4 * iBin,
                  smearedChargeQuadratureId);

                const std::set<dftfe::Int> &atomImageIdsInBin =
                  atomImageIdsBins.find(iBin)->second;
                for (dftfe::uInt cell = 0;
                     cell < matrixFreeDataElectro.n_cell_batches();
                     ++cell)
                  {
                    std::set<dftfe::uInt>
                      nonTrivialSmearedChargeAtomImageIdsMacroCell;
                    const dftfe::uInt numSubCells =
                      matrixFreeDataElectro.n_active_entries_per_cell_batch(
                        cell);
                    for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr =
                          matrixFreeDataElectro.get_cell_iterator(cell,
                                                                  iSubCell);
                        dealii::CellId subCellId = subCellPtr->id();
                        const std::vector<dftfe::uInt> &temp =
                          dftPtr->d_bCellNonTrivialAtomImageIdsBins[iBin]
                            .find(subCellId)
                            ->second;
                        for (dftfe::Int i = 0; i < temp.size(); i++)
                          nonTrivialSmearedChargeAtomImageIdsMacroCell.insert(
                            temp[i]);
                      }

                    if (nonTrivialSmearedChargeAtomImageIdsMacroCell.size() ==
                        0)
                      continue;

                    std::fill(smearedbQuads.begin(),
                              smearedbQuads.end(),
                              dealii::make_vectorized_array(0.0));
                    std::fill(gradVselfSmearedChargeQuads.begin(),
                              gradVselfSmearedChargeQuads.end(),
                              zeroTensor);

                    bool isCellNonTrivial = false;
                    for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                         ++iSubCell)
                      {
                        subCellPtr =
                          matrixFreeDataElectro.get_cell_iterator(cell,
                                                                  iSubCell);
                        dealii::CellId subCellId = subCellPtr->id();

                        const std::vector<dftfe::Int> &bQuadAtomIdsCell =
                          dftPtr->d_bQuadAtomIdsAllAtoms.find(subCellId)
                            ->second;
                        const std::vector<double> &bQuadValuesCell =
                          dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;

                        for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                          {
                            if (atomImageIdsInBin.find(bQuadAtomIdsCell[q]) !=
                                atomImageIdsInBin.end())
                              {
                                isCellNonTrivial           = true;
                                smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                              }
                          } // quad loop
                      }     // subcell loop

                    if (!isCellNonTrivial)
                      continue;

                    forceEvalSmearedCharge.reinit(cell);
                    vselfEvalSmearedCharge.reinit(cell);
                    vselfEvalSmearedCharge.read_dof_values_plain(
                      vselfBinsManagerElectro.getVselfFieldBins()[iBin]);
                    vselfEvalSmearedCharge.evaluate(
                      dealii::EvaluationFlags::gradients);

                    for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                      {
                        gradVselfSmearedChargeQuads[q] =
                          vselfEvalSmearedCharge.get_gradient(q);
                      }

                    addEVselfSmearedStressContribution(
                      forceEvalSmearedCharge,
                      matrixFreeDataElectro,
                      cell,
                      gradVselfSmearedChargeQuads,
                      std::vector<dftfe::uInt>(
                        nonTrivialSmearedChargeAtomImageIdsMacroCell.begin(),
                        nonTrivialSmearedChargeAtomImageIdsMacroCell.end()),
                      dftPtr->d_bQuadAtomIdsAllAtomsImages,
                      smearedbQuads);
                  } // macrocell loop
              }     // kpt paral loop
          }         // bin loop
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
