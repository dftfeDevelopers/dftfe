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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh
//

// source file for all the mixing schemes
#include <dft.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  void
  calldgesv(const unsigned int dimensionMatrix,
            double *           matrix,
            double *           matrixInverse)
  {
    int              N    = dimensionMatrix;
    int              NRHS = N, lda = N, ldb = N, info;
    std::vector<int> ipiv(N);
    //

    dgesv_(
      &N, &NRHS, &matrix[0], &lda, &ipiv[0], &matrixInverse[0], &ldb, &info);
    /* Check for convergence */
    if (info != 0)
      {
        std::cout << "zgesv algorithm failed to compute inverse " << info
                  << std::endl;
        exit(1);
      }
  }

  // implement simple mixing scheme
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::mixing_simple()
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(FE, quadrature, dealii::update_JxW_values);
    const unsigned int  num_quad_points = quadrature.size();


    // create new rhoValue tables
    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;

    rhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();

    // create new gradRhoValue tables
    std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        gradRhoInValuesOld = *gradRhoInValues;
        gradRhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();

      }

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);


            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              (*gradRhoInValues)[cell->id()] =
                std::vector<double>(3 * num_quad_points);


            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                // Compute (rhoIn-rhoOut)^2
                normValue += std::pow(((rhoInValuesOld)[cell->id()][q_point]) -
                                        ((*rhoOutValues)[cell->id()][q_point]),
                                      2.0) *
                             fe_values.JxW(q_point);

                // Simple mixing scheme
                ((*rhoInValues)[cell->id()][q_point]) =
                  ((1 - d_dftParamsPtr->mixingParameter) *
                     (rhoInValuesOld)[cell->id()][q_point] +
                   d_dftParamsPtr->mixingParameter *
                     (*rhoOutValues)[cell->id()][q_point]);


                if (d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA)
                  {
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
                      ((1 - d_dftParamsPtr->mixingParameter) *
                         (gradRhoInValuesOld)[cell->id()][3 * q_point + 0] +
                       d_dftParamsPtr->mixingParameter *
                         (*gradRhoOutValues)[cell->id()][3 * q_point + 0]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
                      ((1 - d_dftParamsPtr->mixingParameter) *
                         (gradRhoInValuesOld)[cell->id()][3 * q_point + 1] +
                       d_dftParamsPtr->mixingParameter *
                         (*gradRhoOutValues)[cell->id()][3 * q_point + 1]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
                      ((1 - d_dftParamsPtr->mixingParameter) *
                         (gradRhoInValuesOld)[cell->id()][3 * q_point + 2] +
                       d_dftParamsPtr->mixingParameter *
                         (*gradRhoOutValues)[cell->id()][3 * q_point + 2]);
                  }
              }
          }
      }

    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
  }

//  // implement anderson mixing scheme
//  template <unsigned int FEOrder, unsigned int FEOrderElectro>
//  double
//  dftClass<FEOrder, FEOrderElectro>::mixing_anderson()
//  {
//    double                       normValue = 0.0;
//    const dealii::Quadrature<3> &quadrature =
//      matrix_free_data.get_quadrature(d_densityQuadratureId);
//    dealii::FEValues<3> fe_values(FE, quadrature, dealii::update_JxW_values);
//    const unsigned int  num_quad_points = quadrature.size();
//
//    // initialize data structures
//    int N = rhoOutVals.size() - 1;
//    // pcout << "\nN:" << N << "\n";
//    int                 NRHS = 1, lda = N, ldb = N, info;
//    std::vector<int>    ipiv(N);
//    std::vector<double> A(lda * N), c(ldb * NRHS);
//    for (int i = 0; i < lda * N; i++)
//      A[i] = 0.0;
//    for (int i = 0; i < ldb * NRHS; i++)
//      c[i] = 0.0;
//
//    std::vector<std::vector<double>> rhoOutTemp(
//      N + 1, std::vector<double>(num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> rhoInTemp(
//      N + 1, std::vector<double>(num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> gradRhoOutTemp(
//      N + 1, std::vector<double>(3 * num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> gradRhoInTemp(
//      N + 1, std::vector<double>(3 * num_quad_points, 0.0));
//
//
//
//    // parallel loop over all elements
//    typename dealii::DoFHandler<3>::active_cell_iterator
//      cell = dofHandler.begin_active(),
//      endc = dofHandler.end();
//    for (; cell != endc; ++cell)
//      {
//        if (cell->is_locally_owned())
//          {
//            fe_values.reinit(cell);
//
//            for (int hist = 0; hist < N + 1; hist++)
//              {
//                rhoOutTemp[hist] = (rhoOutVals[hist])[cell->id()];
//                rhoInTemp[hist]  = (rhoInVals[hist])[cell->id()];
//              }
//
//            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
//              {
//                // fill coefficient matrix, rhs
//                // double Fn=((rhoOutVals[N])[cell->id()][q_point])-
//                // ((rhoInVals[N])[cell->id()][q_point]);
//                double Fn = rhoOutTemp[N][q_point] - rhoInTemp[N][q_point];
//                for (int m = 0; m < N; m++)
//                  {
//                    // double Fnm=((rhoOutVals[N-1-m])[cell->id()][q_point])-
//                    // ((rhoInVals[N-1-m])[cell->id()][q_point]);
//                    double Fnm = rhoOutTemp[N - 1 - m][q_point] -
//                                 rhoInTemp[N - 1 - m][q_point];
//                    for (int k = 0; k < N; k++)
//                      {
//                        // double
//                        // Fnk=((rhoOutVals[N-1-k])[cell->id()][q_point])-
//                        // ((rhoInVals[N-1-k])[cell->id()][q_point]);
//                        double Fnk = rhoOutTemp[N - 1 - k][q_point] -
//                                     rhoInTemp[N - 1 - k][q_point];
//                        A[k * N + m] +=
//                          (Fn - Fnm) * (Fn - Fnk) *
//                          fe_values.JxW(q_point); // (m,k)^th entry
//                      }
//                    c[m] +=
//                      (Fn - Fnm) * (Fn)*fe_values.JxW(q_point); // (m)^th entry
//                  }
//              }
//          }
//      }
//
//
//    // accumulate over all processors
//    std::vector<double> ATotal(lda * N), cTotal(ldb * NRHS);
//    MPI_Allreduce(
//      &A[0], &ATotal[0], lda * N, MPI_DOUBLE, MPI_SUM, mpi_communicator);
//    MPI_Allreduce(
//      &c[0], &cTotal[0], ldb * NRHS, MPI_DOUBLE, MPI_SUM, mpi_communicator);
//    //
//    // pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
//    // solve for coefficients
//    dgesv_(&N, &NRHS, &ATotal[0], &lda, &ipiv[0], &cTotal[0], &ldb, &info);
//
//    if ((info > 0) && (this_mpi_process == 0))
//      {
//        printf(
//          "Anderson Mixing: The diagonal element of the triangular factor of A,\n");
//        printf(
//          "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n",
//          info,
//          info);
//        exit(1);
//      }
//    double cn = 1.0;
//    for (int i = 0; i < N; i++)
//      cn -= cTotal[i];
//    if (this_mpi_process == 0)
//      {
//        // printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
//        // for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
//        // printf("\n");
//      }
//
//    // create new rhoValue tables
//    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
//    rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
//    rhoInValues = &(rhoInVals.back());
//
//
//    // implement anderson mixing
//    cell = dofHandler.begin_active();
//    for (; cell != endc; ++cell)
//      {
//        if (cell->is_locally_owned())
//          {
//            (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
//            fe_values.reinit(cell);
//
//            for (int hist = 0; hist < N + 1; hist++)
//              {
//                rhoOutTemp[hist] = (rhoOutVals[hist])[cell->id()];
//                rhoInTemp[hist]  = (rhoInVals[hist])[cell->id()];
//              }
//
//            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
//              {
//                // Compute (rhoIn-rhoOut)^2
//                normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
//                                        (*rhoOutValues)[cell->id()][q_point],
//                                      2.0) *
//                             fe_values.JxW(q_point);
//                // Anderson mixing scheme
//                // double rhoOutBar=cn*(rhoOutVals[N])[cell->id()][q_point];
//                // double rhoInBar=cn*(rhoInVals[N])[cell->id()][q_point];
//                double rhoOutBar = cn * rhoOutTemp[N][q_point];
//                double rhoInBar  = cn * rhoInTemp[N][q_point];
//
//                for (int i = 0; i < N; i++)
//                  {
//                    // rhoOutBar+=cTotal[i]*(rhoOutVals[N-1-i])[cell->id()][q_point];
//                    // rhoInBar+=cTotal[i]*(rhoInVals[N-1-i])[cell->id()][q_point];
//                    rhoOutBar += cTotal[i] * rhoOutTemp[N - 1 - i][q_point];
//                    rhoInBar += cTotal[i] * rhoInTemp[N - 1 - i][q_point];
//                  }
//                (*rhoInValues)[cell->id()][q_point] =
//                  ((1 - d_dftParamsPtr->mixingParameter) * rhoInBar +
//                   d_dftParamsPtr->mixingParameter * rhoOutBar);
//              }
//          }
//      }
//
//    // compute gradRho for GGA using mixing constants from rho mixing
//
//
//    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
//      {
//        std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
//          *gradRhoInValues;
//        gradRhoInVals.push_back(
//          std::map<dealii::CellId, std::vector<double>>());
//        gradRhoInValues = &(gradRhoInVals.back());
//        cell            = dofHandler.begin_active();
//        for (; cell != endc; ++cell)
//          {
//            if (cell->is_locally_owned())
//              {
//                (*gradRhoInValues)[cell->id()] =
//                  std::vector<double>(3 * num_quad_points);
//                fe_values.reinit(cell);
//
//
//                for (int hist = 0; hist < N + 1; hist++)
//                  {
//                    gradRhoOutTemp[hist] = (gradRhoOutVals[hist])[cell->id()];
//                    gradRhoInTemp[hist]  = (gradRhoInVals[hist])[cell->id()];
//                  }
//
//
//                for (unsigned int q_point = 0; q_point < num_quad_points;
//                     ++q_point)
//                  {
//                    //
//                    // Anderson mixing scheme
//                    //
//                    double gradRhoXOutBar =
//                      cn *
//                      gradRhoOutTemp
//                        [N][3 * q_point +
//                            0]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
//                                // + 0];
//                    double gradRhoYOutBar =
//                      cn *
//                      gradRhoOutTemp
//                        [N][3 * q_point +
//                            1]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
//                                // + 1];
//                    double gradRhoZOutBar =
//                      cn *
//                      gradRhoOutTemp
//                        [N][3 * q_point +
//                            2]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
//                                // + 2];
//
//                    double gradRhoXInBar =
//                      cn *
//                      gradRhoInTemp
//                        [N][3 * q_point +
//                            0]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
//                                // + 0];
//                    double gradRhoYInBar =
//                      cn *
//                      gradRhoInTemp
//                        [N][3 * q_point +
//                            1]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
//                                // + 1];
//                    double gradRhoZInBar =
//                      cn *
//                      gradRhoInTemp
//                        [N][3 * q_point +
//                            2]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
//                                // + 2];
//
//                    for (int i = 0; i < N; i++)
//                      {
//                        gradRhoXOutBar +=
//                          cTotal[i] *
//                          gradRhoOutTemp
//                            [N - 1 - i]
//                            [3 * q_point +
//                             0]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
//                                 // + 0];
//                        gradRhoYOutBar +=
//                          cTotal[i] *
//                          gradRhoOutTemp
//                            [N - 1 - i]
//                            [3 * q_point +
//                             1]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
//                                 // + 1];
//                        gradRhoZOutBar +=
//                          cTotal[i] *
//                          gradRhoOutTemp
//                            [N - 1 - i]
//                            [3 * q_point +
//                             2]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
//                                 // + 2];
//
//                        gradRhoXInBar +=
//                          cTotal[i] *
//                          gradRhoInTemp
//                            [N - 1 - i]
//                            [3 * q_point +
//                             0]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
//                                 // + 0];
//                        gradRhoYInBar +=
//                          cTotal[i] *
//                          gradRhoInTemp
//                            [N - 1 - i]
//                            [3 * q_point +
//                             1]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
//                                 // + 1];
//                        gradRhoZInBar +=
//                          cTotal[i] *
//                          gradRhoInTemp
//                            [N - 1 - i]
//                            [3 * q_point +
//                             2]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
//                                 // + 2];
//                      }
//
//                    (*gradRhoInValues)[cell->id()][3 * q_point + 0] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar +
//                       d_dftParamsPtr->mixingParameter * gradRhoXOutBar);
//                    (*gradRhoInValues)[cell->id()][3 * q_point + 1] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar +
//                       d_dftParamsPtr->mixingParameter * gradRhoYOutBar);
//                    (*gradRhoInValues)[cell->id()][3 * q_point + 2] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar +
//                       d_dftParamsPtr->mixingParameter * gradRhoZOutBar);
//                  }
//              }
//          }
//      }
//
//
//
//    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
//  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::mixing_simple_spinPolarized()
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(FE, quadrature, dealii::update_JxW_values);
    const unsigned int  num_quad_points = quadrature.size();

    // create new rhoValue tables
    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
    rhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();

    std::map<dealii::CellId, std::vector<double>> rhoInValuesOldSpinPolarized =
      *rhoInValuesSpinPolarized;

    rhoInValuesSpinPolarized =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
    //

    // create new gradRhoValue tables
    std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;
    std::map<dealii::CellId, std::vector<double>>
      gradRhoInValuesOldSpinPolarized;

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        gradRhoInValuesOld = *gradRhoInValues;

        gradRhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
        //
        gradRhoInValuesOldSpinPolarized = *gradRhoInValuesSpinPolarized;

        gradRhoInValuesSpinPolarized =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
      }

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            // if (s==0) {
            (*rhoInValuesSpinPolarized)[cell->id()] =
              std::vector<double>(2 * num_quad_points);
            (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
            // }

            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              {
                (*gradRhoInValues)[cell->id()] =
                  std::vector<double>(3 * num_quad_points);
                (*gradRhoInValuesSpinPolarized)[cell->id()] =
                  std::vector<double>(6 * num_quad_points);
              }


            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                // Compute (rhoIn-rhoOut)^2
                // normValue+=std::pow(((*rhoInValuesOld)[cell->id()][2*q_point+s])-
                // ((*rhoOutValues)[cell->id()][2*q_point+s]),2.0)*fe_values.JxW(q_point);

                // Simple mixing scheme
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
                  ((1 - d_dftParamsPtr->mixingParameter) *
                     (rhoInValuesOldSpinPolarized)[cell->id()][2 * q_point] +
                   d_dftParamsPtr->mixingParameter *
                     (*rhoOutValuesSpinPolarized)[cell->id()][2 * q_point]);
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                  ((1 - d_dftParamsPtr->mixingParameter) *
                     (rhoInValuesOldSpinPolarized)[cell->id()]
                                                  [2 * q_point + 1] +
                   d_dftParamsPtr->mixingParameter *
                     (*rhoOutValuesSpinPolarized)[cell->id()][2 * q_point + 1]);

                (*rhoInValues)[cell->id()][q_point] =
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] +
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1];
                //
                normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                        (*rhoOutValues)[cell->id()][q_point],
                                      2.0) *
                             fe_values.JxW(q_point);

                if (d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA)
                  {
                    for (unsigned int i = 0; i < 6; ++i)
                      {
                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                        [6 * q_point + i]) =
                          ((1 - d_dftParamsPtr->mixingParameter) *
                             (gradRhoInValuesOldSpinPolarized)[cell->id()]
                                                              [6 * q_point +
                                                               i] +
                           d_dftParamsPtr->mixingParameter *
                             (*gradRhoOutValuesSpinPolarized)[cell->id()]
                                                             [6 * q_point + i]);
                      }

                    //
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 0]) +
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 3]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 1]) +
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 4]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 2]) +
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 5]);
                  }
              }
          }
      }

    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
  }

//  // implement anderson mixing scheme
//  template <unsigned int FEOrder, unsigned int FEOrderElectro>
//  double
//  dftClass<FEOrder, FEOrderElectro>::mixing_anderson_spinPolarized()
//  {
//    double                       normValue = 0.0;
//    const dealii::Quadrature<3> &quadrature =
//      matrix_free_data.get_quadrature(d_densityQuadratureId);
//    dealii::FEValues<3> fe_values(FE, quadrature, dealii::update_JxW_values);
//    const unsigned int  num_quad_points = quadrature.size();
//
//
//
//    // initialize data structures
//    int N = rhoOutVals.size() - 1;
//    // pcout << "\nN:" << N << "\n";
//    int                 NRHS = 1, lda = N, ldb = N, info;
//    std::vector<int>    ipiv(N);
//    std::vector<double> A(lda * N), c(ldb * NRHS);
//    for (int i = 0; i < lda * N; i++)
//      A[i] = 0.0;
//    for (int i = 0; i < ldb * NRHS; i++)
//      c[i] = 0.0;
//
//    std::vector<std::vector<double>> rhoOutTemp(
//      N + 1, std::vector<double>(num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> rhoInTemp(
//      N + 1, std::vector<double>(num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> gradRhoOutTemp(
//      N + 1, std::vector<double>(3 * num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> gradRhoInTemp(
//      N + 1, std::vector<double>(3 * num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> rhoOutSpinPolarizedTemp(
//      N + 1, std::vector<double>(2 * num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> rhoInSpinPolarizedTemp(
//      N + 1, std::vector<double>(2 * num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> gradRhoOutSpinPolarizedTemp(
//      N + 1, std::vector<double>(6 * num_quad_points, 0.0));
//
//    std::vector<std::vector<double>> gradRhoInSpinPolarizedTemp(
//      N + 1, std::vector<double>(6 * num_quad_points, 0.0));
//
//    // parallel loop over all elements
//    typename dealii::DoFHandler<3>::active_cell_iterator
//      cell = dofHandler.begin_active(),
//      endc = dofHandler.end();
//    for (; cell != endc; ++cell)
//      {
//        if (cell->is_locally_owned())
//          {
//            fe_values.reinit(cell);
//
//            for (int hist = 0; hist < N + 1; hist++)
//              {
//                rhoOutTemp[hist] = (rhoOutVals[hist])[cell->id()];
//                rhoInTemp[hist]  = (rhoInVals[hist])[cell->id()];
//              }
//
//            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
//              {
//                // fill coefficient matrix, rhs
//                double Fn = rhoOutTemp[N][q_point] - rhoInTemp[N][q_point];
//                for (int m = 0; m < N; m++)
//                  {
//                    // double Fnm=((rhoOutVals[N-1-m])[cell->id()][q_point])-
//                    // ((rhoInVals[N-1-m])[cell->id()][q_point]);
//                    double Fnm = rhoOutTemp[N - 1 - m][q_point] -
//                                 rhoInTemp[N - 1 - m][q_point];
//                    for (int k = 0; k < N; k++)
//                      {
//                        // double
//                        // Fnk=((rhoOutVals[N-1-k])[cell->id()][q_point])-
//                        // ((rhoInVals[N-1-k])[cell->id()][q_point]);
//                        double Fnk = rhoOutTemp[N - 1 - k][q_point] -
//                                     rhoInTemp[N - 1 - k][q_point];
//                        A[k * N + m] +=
//                          (Fn - Fnm) * (Fn - Fnk) *
//                          fe_values.JxW(q_point); // (m,k)^th entry
//                      }
//                    c[m] +=
//                      (Fn - Fnm) * (Fn)*fe_values.JxW(q_point); // (m)^th entry
//                  }
//              }
//          }
//      }
//    // accumulate over all processors
//    std::vector<double> ATotal(lda * N), cTotal(ldb * NRHS);
//    MPI_Allreduce(
//      &A[0], &ATotal[0], lda * N, MPI_DOUBLE, MPI_SUM, mpi_communicator);
//    MPI_Allreduce(
//      &c[0], &cTotal[0], ldb * NRHS, MPI_DOUBLE, MPI_SUM, mpi_communicator);
//    //
//    // pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
//    // solve for coefficients
//    dgesv_(&N, &NRHS, &ATotal[0], &lda, &ipiv[0], &cTotal[0], &ldb, &info);
//    if ((info > 0) && (this_mpi_process == 0))
//      {
//        printf(
//          "Anderson Mixing: The diagonal element of the triangular factor of A,\n");
//        printf(
//          "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n",
//          info,
//          info);
//        exit(1);
//      }
//    double cn = 1.0;
//    for (int i = 0; i < N; i++)
//      cn -= cTotal[i];
//    if (this_mpi_process == 0)
//      {
//        // printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
//        // for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
//        // printf("\n");
//      }
//
//    // create new rhoValue tables
//    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
//    rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
//    rhoInValues = &(rhoInVals.back());
//
//    //
//    std::map<dealii::CellId, std::vector<double>> rhoInValuesOldSpinPolarized =
//      *rhoInValuesSpinPolarized;
//    rhoInValsSpinPolarized.push_back(
//      std::map<dealii::CellId, std::vector<double>>());
//    rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
//
//    //
//    // implement anderson mixing
//    cell = dofHandler.begin_active();
//    for (; cell != endc; ++cell)
//      {
//        if (cell->is_locally_owned())
//          {
//            // if (s==0) {
//            (*rhoInValuesSpinPolarized)[cell->id()] =
//              std::vector<double>(2 * num_quad_points);
//            (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
//            //}
//            fe_values.reinit(cell);
//
//            for (int hist = 0; hist < N + 1; hist++)
//              {
//                rhoOutSpinPolarizedTemp[hist] =
//                  (rhoOutValsSpinPolarized[hist])[cell->id()];
//                rhoInSpinPolarizedTemp[hist] =
//                  (rhoInValsSpinPolarized[hist])[cell->id()];
//              }
//
//            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
//              {
//                // Compute (rhoIn-rhoOut)^2
//                // normValue+=std::pow((*rhoInValuesOld)[cell->id()][2*q_point+s]-(*rhoOutValues)[cell->id()][2*q_point+s],2.0)*fe_values.JxW(q_point);
//                // Anderson mixing scheme
//                // normValue+=std::pow((*rhoInValuesOldSpinPolarized)[cell->id()][2*q_point]-(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point],2.0)*fe_values.JxW(q_point);
//                // normValue+=std::pow((*rhoInValuesOldSpinPolarized)[cell->id()][2*q_point+1]-(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1],2.0)*fe_values.JxW(q_point);
//                normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
//                                        (*rhoOutValues)[cell->id()][q_point],
//                                      2.0) *
//                             fe_values.JxW(q_point);
//                double rhoOutBar1 =
//                  cn * rhoOutSpinPolarizedTemp[N][2 * q_point];
//                double rhoInBar1 = cn * rhoInSpinPolarizedTemp[N][2 * q_point];
//                for (int i = 0; i < N; i++)
//                  {
//                    // rhoOutBar1+=cTotal[i]*(rhoOutValsSpinPolarized[N-1-i])[cell->id()][2*q_point];
//                    // rhoInBar1+=cTotal[i]*(rhoInValsSpinPolarized[N-1-i])[cell->id()][2*q_point];
//                    rhoOutBar1 +=
//                      cTotal[i] *
//                      rhoOutSpinPolarizedTemp[N - 1 - i][2 * q_point];
//                    rhoInBar1 += cTotal[i] *
//                                 rhoInSpinPolarizedTemp[N - 1 - i][2 * q_point];
//                  }
//                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
//                  ((1 - d_dftParamsPtr->mixingParameter) * rhoInBar1 +
//                   d_dftParamsPtr->mixingParameter * rhoOutBar1);
//                //
//                double rhoOutBar2 =
//                  cn * rhoOutSpinPolarizedTemp[N][2 * q_point + 1];
//                double rhoInBar2 =
//                  cn * rhoInSpinPolarizedTemp[N][2 * q_point + 1];
//                for (int i = 0; i < N; i++)
//                  {
//                    // rhoOutBar2+=cTotal[i]*(rhoOutValsSpinPolarized[N-1-i])[cell->id()][2*q_point+1];
//                    // rhoInBar2+=cTotal[i]*(rhoInValsSpinPolarized[N-1-i])[cell->id()][2*q_point+1];
//                    rhoOutBar2 +=
//                      cTotal[i] *
//                      rhoOutSpinPolarizedTemp[N - 1 - i][2 * q_point + 1];
//                    rhoInBar2 +=
//                      cTotal[i] *
//                      rhoInSpinPolarizedTemp[N - 1 - i][2 * q_point + 1];
//                  }
//                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
//                  ((1 - d_dftParamsPtr->mixingParameter) * rhoInBar2 +
//                   d_dftParamsPtr->mixingParameter * rhoOutBar2);
//                //
//                // if (s==1)
//                //   {
//                //    (*rhoInValues)[cell->id()][q_point]+=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+s]
//                //    ;
//                //     normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
//                //   }
//                // else
//                //    (*rhoInValues)[cell->id()][q_point]=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+s]
//                //    ;
//                (*rhoInValues)[cell->id()][q_point] =
//                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] +
//                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1];
//                // normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
//              }
//          }
//      }
//
//    // compute gradRho for GGA using mixing constants from rho mixing
//
//
//    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
//      {
//        std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
//          *gradRhoInValues;
//        gradRhoInVals.push_back(
//          std::map<dealii::CellId, std::vector<double>>());
//        gradRhoInValues = &(gradRhoInVals.back());
//
//        //
//        gradRhoInValsSpinPolarized.push_back(
//          std::map<dealii::CellId, std::vector<double>>());
//        gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
//        //
//        cell = dofHandler.begin_active();
//        for (; cell != endc; ++cell)
//          {
//            if (cell->is_locally_owned())
//              {
//                (*gradRhoInValues)[cell->id()] =
//                  std::vector<double>(3 * num_quad_points);
//                (*gradRhoInValuesSpinPolarized)[cell->id()] =
//                  std::vector<double>(6 * num_quad_points);
//                //
//                fe_values.reinit(cell);
//
//                for (int hist = 0; hist < N + 1; hist++)
//                  {
//                    gradRhoOutSpinPolarizedTemp[hist] =
//                      (gradRhoOutValsSpinPolarized[hist])[cell->id()];
//                    gradRhoInSpinPolarizedTemp[hist] =
//                      (gradRhoInValsSpinPolarized[hist])[cell->id()];
//                  }
//
//                for (unsigned int q_point = 0; q_point < num_quad_points;
//                     ++q_point)
//                  {
//                    //
//                    // Anderson mixing scheme spin up
//                    //
//                    double gradRhoXOutBar1 =
//                      cn *
//                      gradRhoOutSpinPolarizedTemp
//                        [N]
//                        [6 * q_point +
//                         0]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
//                             // + 0];
//                    double gradRhoYOutBar1 =
//                      cn *
//                      gradRhoOutSpinPolarizedTemp
//                        [N]
//                        [6 * q_point +
//                         1]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
//                             // + 1];
//                    double gradRhoZOutBar1 =
//                      cn *
//                      gradRhoOutSpinPolarizedTemp
//                        [N]
//                        [6 * q_point +
//                         2]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
//                             // + 2];
//
//                    double gradRhoXInBar1 =
//                      cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 0];
//                    double gradRhoYInBar1 =
//                      cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 1];
//                    double gradRhoZInBar1 =
//                      cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 2];
//
//                    for (int i = 0; i < N; i++)
//                      {
//                        gradRhoXOutBar1 +=
//                          cTotal[i] *
//                          gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                     [6 * q_point + 0];
//                        gradRhoYOutBar1 +=
//                          cTotal[i] *
//                          gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                     [6 * q_point + 1];
//                        gradRhoZOutBar1 +=
//                          cTotal[i] *
//                          gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                     [6 * q_point + 2];
//
//                        gradRhoXInBar1 +=
//                          cTotal[i] *
//                          gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                    [6 * q_point + 0];
//                        gradRhoYInBar1 +=
//                          cTotal[i] *
//                          gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                    [6 * q_point + 1];
//                        gradRhoZInBar1 +=
//                          cTotal[i] *
//                          gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                    [6 * q_point + 2];
//                      }
//                    //
//                    // Anderson mixing scheme spin down
//                    //
//                    double gradRhoXOutBar2 =
//                      cn * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 3];
//                    double gradRhoYOutBar2 =
//                      cn * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 4];
//                    double gradRhoZOutBar2 =
//                      cn * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 5];
//
//                    double gradRhoXInBar2 =
//                      cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 3];
//                    double gradRhoYInBar2 =
//                      cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 4];
//                    double gradRhoZInBar2 =
//                      cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 5];
//
//                    for (int i = 0; i < N; i++)
//                      {
//                        gradRhoXOutBar2 +=
//                          cTotal[i] *
//                          gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                     [6 * q_point + 3];
//                        gradRhoYOutBar2 +=
//                          cTotal[i] *
//                          gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                     [6 * q_point + 4];
//                        gradRhoZOutBar2 +=
//                          cTotal[i] *
//                          gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                     [6 * q_point + 5];
//
//                        gradRhoXInBar2 +=
//                          cTotal[i] *
//                          gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                    [6 * q_point + 3];
//                        gradRhoYInBar2 +=
//                          cTotal[i] *
//                          gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                    [6 * q_point + 4];
//                        gradRhoZInBar2 +=
//                          cTotal[i] *
//                          gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                    [6 * q_point + 5];
//                      }
//                    //
//                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                0] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar1 +
//                       d_dftParamsPtr->mixingParameter * gradRhoXOutBar1);
//                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                1] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar1 +
//                       d_dftParamsPtr->mixingParameter * gradRhoYOutBar1);
//                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                2] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar1 +
//                       d_dftParamsPtr->mixingParameter * gradRhoZOutBar1);
//                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                3] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar2 +
//                       d_dftParamsPtr->mixingParameter * gradRhoXOutBar2);
//                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                4] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar2 +
//                       d_dftParamsPtr->mixingParameter * gradRhoYOutBar2);
//                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                5] =
//                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar2 +
//                       d_dftParamsPtr->mixingParameter * gradRhoZOutBar2);
//
//                    ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
//                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                      [6 * q_point + 0]) +
//                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                      [6 * q_point + 3]);
//                    ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
//                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                      [6 * q_point + 1]) +
//                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                      [6 * q_point + 4]);
//                    ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
//                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                      [6 * q_point + 2]) +
//                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                      [6 * q_point + 5]);
//                  }
//              }
//          }
//      }
//    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
//  }
#include "dft.inst.cc"
} // namespace dftfe
