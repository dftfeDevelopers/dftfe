// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020x The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri
//

#include <dft.h>
#include <vectorUtilities.h>

namespace dftfe
{
  // compute localization lengths currently implemented for spin unpolarized
  // case
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::compute_localizationLength(
    const std::string &locLengthFileName)
  {
    dealii::QGauss<3>   quadrature_formula(C_num1DQuad<FEOrder>());
    dealii::FEValues<3> fe_values(dofHandler.get_fe(),
                                  quadrature_formula,
                                  dealii::update_values |
                                    dealii::update_JxW_values |
                                    dealii::update_quadrature_points);
    const unsigned int  dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
    const unsigned int  n_q_points    = quadrature_formula.size();
    std::vector<double> tempQuadPointValues(n_q_points);
    std::vector<double> localizationLength, secondMoment, firstMomentX,
      firstMomentY, firstMomentZ;

    localizationLength.resize(d_numEigenValues);
    secondMoment.resize(d_numEigenValues);
    firstMomentX.resize(d_numEigenValues);
    firstMomentY.resize(d_numEigenValues);
    firstMomentZ.resize(d_numEigenValues);

    std::vector<distributedCPUVec<double>> tempVec(1);
    tempVec[0].reinit(d_tempEigenVec);

    //
    // compute integral(psi_i*(x^2 + y^2 + z^2)*psi_i), integral(psi_i*x*psi_i),
    // integral(psi_i*y*psi_i), integral(psi_i*z*psi_i)
    //
    for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
      {
        vectorTools::copyFlattenedSTLVecToSingleCompVec(
          d_eigenVectorsFlattenedHost.data(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          std::make_pair(iWave, iWave + 1),
          tempVec);

        constraintsNoneEigenDataInfo.distribute(tempVec[0]);

        typename dealii::DoFHandler<3>::active_cell_iterator
          cellN = dofHandler.begin_active(),
          endcN = dofHandler.end();

        for (; cellN != endcN; ++cellN)
          {
            if (cellN->is_locally_owned())
              {
                fe_values.reinit(cellN);
                fe_values.get_function_values(tempVec[0], tempQuadPointValues);

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  {
                    dealii::Point<3> quadPointCoor =
                      fe_values.quadrature_point(q_point);
                    double distanceFromOriginSquare =
                      quadPointCoor[0] * quadPointCoor[0] +
                      quadPointCoor[1] * quadPointCoor[1] +
                      quadPointCoor[2] * quadPointCoor[2];
                    secondMoment[iWave] += tempQuadPointValues[q_point] *
                                           tempQuadPointValues[q_point] *
                                           distanceFromOriginSquare *
                                           fe_values.JxW(q_point);
                    firstMomentX[iWave] += tempQuadPointValues[q_point] *
                                           tempQuadPointValues[q_point] *
                                           quadPointCoor[0] *
                                           fe_values.JxW(q_point);
                    firstMomentY[iWave] += tempQuadPointValues[q_point] *
                                           tempQuadPointValues[q_point] *
                                           quadPointCoor[1] *
                                           fe_values.JxW(q_point);
                    firstMomentZ[iWave] += tempQuadPointValues[q_point] *
                                           tempQuadPointValues[q_point] *
                                           quadPointCoor[2] *
                                           fe_values.JxW(q_point);
                  }
              }
          }
      }

    dealii::Utilities::MPI::sum(secondMoment, mpi_communicator, secondMoment);


    dealii::Utilities::MPI::sum(firstMomentX, mpi_communicator, firstMomentX);


    dealii::Utilities::MPI::sum(firstMomentY, mpi_communicator, firstMomentY);

    dealii::Utilities::MPI::sum(firstMomentZ, mpi_communicator, firstMomentZ);

    //
    // compute localization length using above computed integrals
    //
    for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
      {
        localizationLength[iWave] =
          2.0 * std::sqrt(secondMoment[iWave] -
                          (firstMomentX[iWave] * firstMomentX[iWave] +
                           firstMomentY[iWave] * firstMomentY[iWave] +
                           firstMomentZ[iWave] * firstMomentZ[iWave]));
      }

    //
    // output the localization lengths in a file
    //
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::ofstream outFile(locLengthFileName.c_str());
        outFile.setf(std::ios_base::fixed);

        if (outFile.is_open())
          {
            for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
              {
                outFile << std::setprecision(18) << iWave << " "
                        << localizationLength[iWave] << std::endl;
              }
          }
      }
  }
#include "dft.inst.cc"
} // namespace dftfe
