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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh
//

// source file for all the mixing schemes


void
calldgesv(const unsigned int dimensionMatrix,
          double *           matrix,
          double *           matrixInverse)
{
  int              N    = dimensionMatrix;
  int              NRHS = N, lda = N, ldb = N, info;
  std::vector<int> ipiv(N);
  //

  dgesv_(&N, &NRHS, &matrix[0], &lda, &ipiv[0], &matrixInverse[0], &ldb, &info);
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
  double               normValue = 0.0;
  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(FE, quadrature, update_JxW_values);
  const unsigned int num_quad_points = quadrature.size();


  // create new rhoValue tables
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());


  // create new gradRhoValue tables
  std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;

  if (dftParameters::xcFamilyType == "GGA")
    {
      gradRhoInValuesOld = *gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
    }

  // parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);


          if (dftParameters::xcFamilyType == "GGA")
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
                std::abs((1 - dftParameters::mixingParameter) *
                           (rhoInValuesOld)[cell->id()][q_point] +
                         dftParameters::mixingParameter *
                           (*rhoOutValues)[cell->id()][q_point]);


              if (dftParameters::xcFamilyType == "GGA")
                {
                  ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
                    ((1 - dftParameters::mixingParameter) *
                       (gradRhoInValuesOld)[cell->id()][3 * q_point + 0] +
                     dftParameters::mixingParameter *
                       (*gradRhoOutValues)[cell->id()][3 * q_point + 0]);
                  ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
                    ((1 - dftParameters::mixingParameter) *
                       (gradRhoInValuesOld)[cell->id()][3 * q_point + 1] +
                     dftParameters::mixingParameter *
                       (*gradRhoOutValues)[cell->id()][3 * q_point + 1]);
                  ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
                    ((1 - dftParameters::mixingParameter) *
                       (gradRhoInValuesOld)[cell->id()][3 * q_point + 2] +
                     dftParameters::mixingParameter *
                       (*gradRhoOutValues)[cell->id()][3 * q_point + 2]);
                }
            }
        }
    }

  return Utilities::MPI::sum(normValue, mpi_communicator);
}

// implement anderson mixing scheme
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::mixing_anderson()
{
  double               normValue = 0.0;
  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(FE, quadrature, update_JxW_values);
  const unsigned int num_quad_points = quadrature.size();


  // initialize data structures
  int N = rhoOutVals.size() - 1;
  // pcout << "\nN:" << N << "\n";
  int                 NRHS = 1, lda = N, ldb = N, info;
  std::vector<int>    ipiv(N);
  std::vector<double> A(lda * N), c(ldb * NRHS);
  for (int i = 0; i < lda * N; i++)
    A[i] = 0.0;
  for (int i = 0; i < ldb * NRHS; i++)
    c[i] = 0.0;

  std::vector<std::vector<double>> rhoOutTemp(
    N + 1, std::vector<double>(num_quad_points, 0.0));

  std::vector<std::vector<double>> rhoInTemp(
    N + 1, std::vector<double>(num_quad_points, 0.0));

  std::vector<std::vector<double>> gradRhoOutTemp(
    N + 1, std::vector<double>(3 * num_quad_points, 0.0));

  std::vector<std::vector<double>> gradRhoInTemp(
    N + 1, std::vector<double>(3 * num_quad_points, 0.0));



  // parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          for (int hist = 0; hist < N + 1; hist++)
            {
              rhoOutTemp[hist] = (rhoOutVals[hist])[cell->id()];
              rhoInTemp[hist]  = (rhoInVals[hist])[cell->id()];
            }

          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              // fill coefficient matrix, rhs
              // double Fn=((rhoOutVals[N])[cell->id()][q_point])-
              // ((rhoInVals[N])[cell->id()][q_point]);
              double Fn = rhoOutTemp[N][q_point] - rhoInTemp[N][q_point];
              for (int m = 0; m < N; m++)
                {
                  // double Fnm=((rhoOutVals[N-1-m])[cell->id()][q_point])-
                  // ((rhoInVals[N-1-m])[cell->id()][q_point]);
                  double Fnm = rhoOutTemp[N - 1 - m][q_point] -
                               rhoInTemp[N - 1 - m][q_point];
                  for (int k = 0; k < N; k++)
                    {
                      // double Fnk=((rhoOutVals[N-1-k])[cell->id()][q_point])-
                      // ((rhoInVals[N-1-k])[cell->id()][q_point]);
                      double Fnk = rhoOutTemp[N - 1 - k][q_point] -
                                   rhoInTemp[N - 1 - k][q_point];
                      A[k * N + m] += (Fn - Fnm) * (Fn - Fnk) *
                                      fe_values.JxW(q_point); // (m,k)^th entry
                    }
                  c[m] +=
                    (Fn - Fnm) * (Fn)*fe_values.JxW(q_point); // (m)^th entry
                }
            }
        }
    }
  // accumulate over all processors
  std::vector<double> ATotal(lda * N), cTotal(ldb * NRHS);
  MPI_Allreduce(
    &A[0], &ATotal[0], lda * N, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  MPI_Allreduce(
    &c[0], &cTotal[0], ldb * NRHS, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  //
  // pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
  // solve for coefficients
  dgesv_(&N, &NRHS, &ATotal[0], &lda, &ipiv[0], &cTotal[0], &ldb, &info);
  if ((info > 0) && (this_mpi_process == 0))
    {
      printf(
        "Anderson Mixing: The diagonal element of the triangular factor of A,\n");
      printf(
        "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n",
        info,
        info);
      exit(1);
    }
  double cn = 1.0;
  for (int i = 0; i < N; i++)
    cn -= cTotal[i];
  if (this_mpi_process == 0)
    {
      // printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
      // for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
      // printf("\n");
    }

  // create new rhoValue tables
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());


  // implement anderson mixing
  cell = dofHandler.begin_active();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
          fe_values.reinit(cell);

          for (int hist = 0; hist < N + 1; hist++)
            {
              rhoOutTemp[hist] = (rhoOutVals[hist])[cell->id()];
              rhoInTemp[hist]  = (rhoInVals[hist])[cell->id()];
            }

          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              // Compute (rhoIn-rhoOut)^2
              normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                      (*rhoOutValues)[cell->id()][q_point],
                                    2.0) *
                           fe_values.JxW(q_point);
              // Anderson mixing scheme
              // double rhoOutBar=cn*(rhoOutVals[N])[cell->id()][q_point];
              // double rhoInBar=cn*(rhoInVals[N])[cell->id()][q_point];
              double rhoOutBar = cn * rhoOutTemp[N][q_point];
              double rhoInBar  = cn * rhoInTemp[N][q_point];

              for (int i = 0; i < N; i++)
                {
                  // rhoOutBar+=cTotal[i]*(rhoOutVals[N-1-i])[cell->id()][q_point];
                  // rhoInBar+=cTotal[i]*(rhoInVals[N-1-i])[cell->id()][q_point];
                  rhoOutBar += cTotal[i] * rhoOutTemp[N - 1 - i][q_point];
                  rhoInBar += cTotal[i] * rhoInTemp[N - 1 - i][q_point];
                }
              (*rhoInValues)[cell->id()][q_point] =
                std::abs((1 - dftParameters::mixingParameter) * rhoInBar +
                         dftParameters::mixingParameter * rhoOutBar);
            }
        }
    }

  // compute gradRho for GGA using mixing constants from rho mixing


  if (dftParameters::xcFamilyType == "GGA")
    {
      std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
        *gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
      cell            = dofHandler.begin_active();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              (*gradRhoInValues)[cell->id()] =
                std::vector<double>(3 * num_quad_points);
              fe_values.reinit(cell);


              for (int hist = 0; hist < N + 1; hist++)
                {
                  gradRhoOutTemp[hist] = (gradRhoOutVals[hist])[cell->id()];
                  gradRhoInTemp[hist]  = (gradRhoInVals[hist])[cell->id()];
                }


              for (unsigned int q_point = 0; q_point < num_quad_points;
                   ++q_point)
                {
                  //
                  // Anderson mixing scheme
                  //
                  double gradRhoXOutBar =
                    cn *
                    gradRhoOutTemp
                      [N][3 * q_point +
                          0]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
                              // + 0];
                  double gradRhoYOutBar =
                    cn *
                    gradRhoOutTemp
                      [N][3 * q_point +
                          1]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
                              // + 1];
                  double gradRhoZOutBar =
                    cn *
                    gradRhoOutTemp
                      [N][3 * q_point +
                          2]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
                              // + 2];

                  double gradRhoXInBar =
                    cn *
                    gradRhoInTemp
                      [N][3 * q_point +
                          0]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
                              // + 0];
                  double gradRhoYInBar =
                    cn *
                    gradRhoInTemp
                      [N][3 * q_point +
                          1]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
                              // + 1];
                  double gradRhoZInBar =
                    cn *
                    gradRhoInTemp
                      [N][3 * q_point +
                          2]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
                              // + 2];

                  for (int i = 0; i < N; i++)
                    {
                      gradRhoXOutBar +=
                        cTotal[i] *
                        gradRhoOutTemp
                          [N - 1 - i]
                          [3 * q_point +
                           0]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
                               // + 0];
                      gradRhoYOutBar +=
                        cTotal[i] *
                        gradRhoOutTemp
                          [N - 1 - i]
                          [3 * q_point +
                           1]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
                               // + 1];
                      gradRhoZOutBar +=
                        cTotal[i] *
                        gradRhoOutTemp
                          [N - 1 - i]
                          [3 * q_point +
                           2]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
                               // + 2];

                      gradRhoXInBar +=
                        cTotal[i] *
                        gradRhoInTemp
                          [N - 1 - i]
                          [3 * q_point +
                           0]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
                               // + 0];
                      gradRhoYInBar +=
                        cTotal[i] *
                        gradRhoInTemp
                          [N - 1 - i]
                          [3 * q_point +
                           1]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
                               // + 1];
                      gradRhoZInBar +=
                        cTotal[i] *
                        gradRhoInTemp
                          [N - 1 - i]
                          [3 * q_point +
                           2]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
                               // + 2];
                    }

                  (*gradRhoInValues)[cell->id()][3 * q_point + 0] =
                    ((1 - dftParameters::mixingParameter) * gradRhoXInBar +
                     dftParameters::mixingParameter * gradRhoXOutBar);
                  (*gradRhoInValues)[cell->id()][3 * q_point + 1] =
                    ((1 - dftParameters::mixingParameter) * gradRhoYInBar +
                     dftParameters::mixingParameter * gradRhoYOutBar);
                  (*gradRhoInValues)[cell->id()][3 * q_point + 2] =
                    ((1 - dftParameters::mixingParameter) * gradRhoZInBar +
                     dftParameters::mixingParameter * gradRhoZOutBar);
                }
            }
        }
    }



  return Utilities::MPI::sum(normValue, mpi_communicator);
}


// implement Broyden mixing scheme
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::mixing_broyden()
{
  double               normValue = 0.0;
  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(FE, quadrature, update_JxW_values);
  const unsigned int num_quad_points = quadrature.size();
  //
  int N = dFBroyden.size() + 1;

  //
  std::map<dealii::CellId, std::vector<double>> delRho, delGradRho;
  dFBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
  uBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
  if (dftParameters::xcFamilyType == "GGA")
    {
      graddFBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradUBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
    }
  //
  double              FOld;
  std::vector<double> gradFOld(3, 0.0);
  //
  // parallel loop over all elements
  double dfMag = 0.0, dfMagLoc = 0.0;
  double wtTemp = 0.0, wtTempLoc = 0.0;
  double w0Loc = 0.0;
  //
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          //
          (dFBroyden[N - 1])[cell->id()] = std::vector<double>(num_quad_points);
          delRho[cell->id()]             = std::vector<double>(num_quad_points);
          if (N == 1)
            FBroyden[cell->id()] = std::vector<double>(num_quad_points);
          //
          if (dftParameters::xcFamilyType == "GGA")
            {
              (graddFBroyden[N - 1])[cell->id()] =
                std::vector<double>(3 * num_quad_points);
              delGradRho[cell->id()] = std::vector<double>(3 * num_quad_points);
              if (N == 1)
                gradFBroyden[cell->id()] =
                  std::vector<double>(3 * num_quad_points);
            }
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              if (N == 1)
                {
                  FOld = ((rhoOutVals[0])[cell->id()][q_point]) -
                         ((rhoInVals[0])[cell->id()][q_point]);
                  w0Loc += FOld * FOld * fe_values.JxW(q_point);
                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      for (unsigned int dir = 0; dir < 3; ++dir)
                        gradFOld[dir] =
                          ((gradRhoOutVals[0])[cell->id()][3 * q_point + dir]) -
                          ((gradRhoInVals[0])[cell->id()][3 * q_point + dir]);
                    }
                }
              else
                {
                  FOld = FBroyden[cell->id()][q_point];
                  if (dftParameters::xcFamilyType == "GGA")
                    for (unsigned int dir = 0; dir < 3; ++dir)
                      gradFOld[dir] =
                        gradFBroyden[cell->id()][3 * q_point + dir];
                }
              //
              FBroyden[cell->id()][q_point] =
                (rhoOutVals[N])[cell->id()][q_point] -
                (rhoInVals[N])[cell->id()][q_point];
              delRho[cell->id()][q_point] =
                (rhoInVals[N])[cell->id()][q_point] -
                (rhoInVals[N - 1])[cell->id()][q_point];
              //
              (dFBroyden[N - 1])[cell->id()][q_point] =
                FBroyden[cell->id()][q_point] - FOld;
              if (dftParameters::xcFamilyType == "GGA")
                {
                  for (unsigned int dir = 0; dir < 3; ++dir)
                    {
                      delGradRho[cell->id()][3 * q_point + dir] =
                        (gradRhoInVals[N])[cell->id()][3 * q_point + dir] -
                        (gradRhoInVals[N - 1])[cell->id()][3 * q_point + dir];
                      gradFBroyden[cell->id()][3 * q_point + dir] =
                        (gradRhoOutVals[N])[cell->id()][3 * q_point + dir] -
                        (gradRhoInVals[N])[cell->id()][3 * q_point + dir];
                      (graddFBroyden[N - 1])[cell->id()][3 * q_point + dir] =
                        gradFBroyden[cell->id()][3 * q_point + dir] -
                        gradFOld[dir];
                    }
                }
              dfMagLoc += (dFBroyden[N - 1])[cell->id()][q_point] *
                          (dFBroyden[N - 1])[cell->id()][q_point] *
                          fe_values.JxW(q_point);
              wtTempLoc += FBroyden[cell->id()][q_point] *
                           FBroyden[cell->id()][q_point] *
                           fe_values.JxW(q_point);
            }
        }
    }
  //
  wtTemp = Utilities::MPI::sum(wtTempLoc, mpi_communicator);
  dfMag  = Utilities::MPI::sum(dfMagLoc, mpi_communicator);
  if (N == 1)
    {
      w0Broyden = Utilities::MPI::sum(w0Loc, mpi_communicator);
      w0Broyden = std::pow(w0Broyden, -0.5);
    }
  // Comment out following line, for using w0 computed from simply mixed rho
  // (not recommended)
  w0Broyden = 0.0;
  //
  wtTemp = std::pow(wtTemp, -0.5);
  //
  // Comment out push_back(1.0) and uncomment push_back(wtTemp) to include
  // history dependence in wtBroyden (not recommended)
  // wtBroyden.push_back(wtTemp) ;
  wtBroyden.push_back(1.0);
  //
  //
  double G = dftParameters::mixingParameter;
  //
  std::vector<double> c(N, 0.0), invBeta(N * N, 0.0), beta(N * N, 0.0),
    gamma(N, 0.0), cLoc(N, 0.0), invBetaLoc(N * N, 0.0);
  //
  cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        (uBroyden[N - 1])[cell->id()] = std::vector<double>(num_quad_points);
        if (dftParameters::xcFamilyType == "GGA")
          (gradUBroyden[N - 1])[cell->id()] =
            std::vector<double>(3 * num_quad_points);
        fe_values.reinit(cell);
        //
        for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
          {
            (dFBroyden[N - 1])[cell->id()][q_point] /= dfMag;
            delRho[cell->id()][q_point] /= dfMag;
            (uBroyden[N - 1])[cell->id()][q_point] =
              G * (dFBroyden[N - 1])[cell->id()][q_point] +
              delRho[cell->id()][q_point];
            //
            if (dftParameters::xcFamilyType == "GGA")
              {
                for (unsigned int dir = 0; dir < 3; ++dir)
                  {
                    (graddFBroyden[N - 1])[cell->id()][3 * q_point + dir] /=
                      dfMag;
                    delGradRho[cell->id()][3 * q_point + dir] /= dfMag;
                    (gradUBroyden[N - 1])[cell->id()][3 * q_point + dir] =
                      G *
                        (graddFBroyden[N - 1])[cell->id()][3 * q_point + dir] +
                      delGradRho[cell->id()][3 * q_point + dir];
                  }
              }
            //
            for (unsigned int k = 0; k < N; ++k)
              {
                cLoc[k] += wtBroyden[k] * (dFBroyden[k])[cell->id()][q_point] *
                           FBroyden[cell->id()][q_point] *
                           fe_values.JxW(q_point);
                for (unsigned int l = k; l < N; ++l)
                  {
                    invBetaLoc[N * k + l] +=
                      wtBroyden[k] * wtBroyden[l] *
                      (dFBroyden[k])[cell->id()][q_point] *
                      (dFBroyden[l])[cell->id()][q_point] *
                      fe_values.JxW(q_point);
                    invBetaLoc[N * l + k] = invBetaLoc[N * k + l];
                  }
              }
          }
      }
  //
  for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int l = 0; l < N; ++l)
        {
          invBeta[N * k + l] =
            Utilities::MPI::sum(invBetaLoc[N * k + l], mpi_communicator);
          if (l == k)
            {
              invBeta[N * l + l] = w0Broyden * w0Broyden + invBeta[N * l + l];
              beta[N * l + l]    = 1.0;
            }
        }
      c[k] = Utilities::MPI::sum(cLoc[k], mpi_communicator);
    }

  //
  // Invert beta
  //
  calldgesv(N, &invBeta[0], &beta[0]);


  for (unsigned int m = 0; m < N; ++m)
    for (unsigned int l = 0; l < N; ++l)
      gamma[m] += c[l] * beta[N * m + l];
  //
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());
  //
  std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;
  if (dftParameters::xcFamilyType == "GGA")
    {
      gradRhoInValuesOld = *gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
    }
  //
  cell = dofHandler.begin_active();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
          if (dftParameters::xcFamilyType == "GGA")
            (*gradRhoInValues)[cell->id()] =
              std::vector<double>(3 * num_quad_points);
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              // Compute (rhoIn-rhoOut)^2
              normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                      (*rhoOutValues)[cell->id()][q_point],
                                    2.0) *
                           fe_values.JxW(q_point);
              ;
              (*rhoInValues)[cell->id()][q_point] =
                rhoInValuesOld[cell->id()][q_point] +
                G * FBroyden[cell->id()][q_point];
              if (dftParameters::xcFamilyType == "GGA")
                for (unsigned int dir = 0; dir < 3; ++dir)
                  (*gradRhoInValues)[cell->id()][3 * q_point + dir] =
                    gradRhoInValuesOld[cell->id()][3 * q_point + dir] +
                    G * gradFBroyden[cell->id()][3 * q_point + dir];
              //
              for (int i = 0; i < N; ++i)
                {
                  (*rhoInValues)[cell->id()][q_point] -=
                    wtBroyden[i] * gamma[i] *
                    (uBroyden[i])[cell->id()][q_point];
                  if (dftParameters::xcFamilyType == "GGA")
                    for (unsigned int dir = 0; dir < 3; ++dir)
                      (*gradRhoInValues)[cell->id()][3 * q_point + dir] -=
                        wtBroyden[i] * gamma[i] *
                        (gradUBroyden[i])[cell->id()][3 * q_point + dir];
                }
            }
        }
    }
  //
  //



  return Utilities::MPI::sum(normValue, mpi_communicator);
}



// implement Broyden mixing scheme
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::mixing_broyden_spinPolarized()
{
  double               normValue = 0.0;
  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(FE, quadrature, update_JxW_values);
  const unsigned int num_quad_points = quadrature.size();
  //
  int N = dFBroyden.size() + 1;
  //
  std::map<dealii::CellId, std::vector<double>> delRho, delGradRho;
  dFBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
  uBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
  if (dftParameters::xcFamilyType == "GGA")
    {
      graddFBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradUBroyden.push_back(std::map<dealii::CellId, std::vector<double>>());
    }
  //
  double              FOld;
  std::vector<double> gradFOld(3, 0.0);
  //
  // parallel loop over all elements
  double dfMag = 0.0, dfMagLoc = 0.0;
  double wtTemp = 0.0, wtTempLoc = 0.0;
  double w0Loc = 0.0;
  //
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          //
          (dFBroyden[N - 1])[cell->id()] =
            std::vector<double>(2 * num_quad_points);
          delRho[cell->id()] = std::vector<double>(2 * num_quad_points);
          if (N == 1)
            FBroyden[cell->id()] = std::vector<double>(2 * num_quad_points);
          //
          if (dftParameters::xcFamilyType == "GGA")
            {
              (graddFBroyden[N - 1])[cell->id()] =
                std::vector<double>(6 * num_quad_points);
              delGradRho[cell->id()] = std::vector<double>(6 * num_quad_points);
              if (N == 1)
                gradFBroyden[cell->id()] =
                  std::vector<double>(6 * num_quad_points);
            }
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < 2 * num_quad_points;
               ++q_point)
            { // factor 2 due to spin splitting
              if (N == 1)
                {
                  FOld = ((rhoOutValsSpinPolarized[0])[cell->id()][q_point]) -
                         ((rhoInValsSpinPolarized[0])[cell->id()][q_point]);
                  // w0Loc += FOld * FOld * fe_values.JxW(q_point) ;
                  // F[cell->id()]=std::vector<double>(num_quad_points);
                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      // gradF[cell->id()]=std::vector<double>(6*num_quad_points);
                      for (unsigned int dir = 0; dir < 3; ++dir)
                        gradFOld[dir] =
                          ((gradRhoOutValsSpinPolarized[0])[cell->id()]
                                                           [3 * q_point +
                                                            dir]) -
                          ((gradRhoInValsSpinPolarized[0])[cell->id()]
                                                          [3 * q_point + dir]);
                    }
                }
              else
                {
                  FOld = FBroyden[cell->id()][q_point];
                  if (dftParameters::xcFamilyType == "GGA")
                    for (unsigned int dir = 0; dir < 3; ++dir)
                      gradFOld[dir] =
                        gradFBroyden[cell->id()][3 * q_point + dir];
                }

              FBroyden[cell->id()][q_point] =
                (rhoOutValsSpinPolarized[N])[cell->id()][q_point] -
                (rhoInValsSpinPolarized[N])[cell->id()][q_point];
              delRho[cell->id()][q_point] =
                (rhoInValsSpinPolarized[N])[cell->id()][q_point] -
                (rhoInValsSpinPolarized[N - 1])[cell->id()][q_point];
              //
              (dFBroyden[N - 1])[cell->id()][q_point] =
                FBroyden[cell->id()][q_point] - FOld;
              if (dftParameters::xcFamilyType == "GGA")
                {
                  for (unsigned int dir = 0; dir < 3; ++dir)
                    {
                      delGradRho[cell->id()][3 * q_point + dir] =
                        (gradRhoInValsSpinPolarized[N])[cell->id()]
                                                       [3 * q_point + dir] -
                        (gradRhoInValsSpinPolarized[N - 1])[cell->id()]
                                                           [3 * q_point + dir];
                      gradFBroyden[cell->id()][3 * q_point + dir] =
                        (gradRhoOutValsSpinPolarized[N])[cell->id()]
                                                        [3 * q_point + dir] -
                        (gradRhoInValsSpinPolarized[N])[cell->id()]
                                                       [3 * q_point + dir];
                      (graddFBroyden[N - 1])[cell->id()][3 * q_point + dir] =
                        gradFBroyden[cell->id()][3 * q_point + dir] -
                        gradFOld[dir];
                    }
                }
            }
          //
          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              dfMagLoc += (((dFBroyden[N - 1])[cell->id()][2 * q_point] +
                            (dFBroyden[N - 1])[cell->id()][2 * q_point + 1]) *
                           ((dFBroyden[N - 1])[cell->id()][2 * q_point] +
                            (dFBroyden[N - 1])[cell->id()][2 * q_point + 1])) *
                          fe_values.JxW(q_point);
              //
              wtTempLoc += ((FBroyden[cell->id()][2 * q_point] +
                             FBroyden[cell->id()][2 * q_point + 1]) *
                            ((FBroyden[cell->id()][2 * q_point] +
                              FBroyden[cell->id()][2 * q_point + 1]))) *
                           fe_values.JxW(q_point);
              if (N == 1)
                {
                  FOld = ((rhoOutVals[0])[cell->id()][q_point]) -
                         ((rhoInVals[0])[cell->id()][q_point]);
                  w0Loc += FOld * FOld * fe_values.JxW(q_point);
                }
            }
        }
    }
  //
  wtTemp = Utilities::MPI::sum(wtTempLoc, mpi_communicator);
  dfMag  = Utilities::MPI::sum(dfMagLoc, mpi_communicator);
  if (N == 1)
    {
      w0Broyden = Utilities::MPI::sum(w0Loc, mpi_communicator);
      w0Broyden = std::pow(w0Broyden, -0.5);
    }
  // Comment out following line, for using w0 computed from simply mixed rho
  // (not recommended)
  w0Broyden = 0.0;
  //
  wtTemp = std::pow(wtTemp, -0.5);
  //
  // Comment out push_back(1.0) and uncomment push_back(wtTemp) to include
  // history dependence in wtBroyden (not recommended)
  // wtBroyden.push_back(wtTemp) ;
  wtBroyden.push_back(1.0);
  //
  double G = dftParameters::mixingParameter;
  //
  std::vector<double> c(N, 0.0), invBeta(N * N, 0.0), beta(N * N, 0.0),
    gamma(N, 0.0), cLoc(N, 0.0), invBetaLoc(N * N, 0.0);
  //
  cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        (uBroyden[N - 1])[cell->id()] =
          std::vector<double>(2 * num_quad_points);
        if (dftParameters::xcFamilyType == "GGA")
          (gradUBroyden[N - 1])[cell->id()] =
            std::vector<double>(6 * num_quad_points);
        fe_values.reinit(cell);
        //
        for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
          {
            (dFBroyden[N - 1])[cell->id()][2 * q_point] /= dfMag;
            (dFBroyden[N - 1])[cell->id()][2 * q_point + 1] /= dfMag;
            delRho[cell->id()][2 * q_point] /= dfMag;
            delRho[cell->id()][2 * q_point + 1] /= dfMag;
            //
            (uBroyden[N - 1])[cell->id()][2 * q_point] =
              G * (dFBroyden[N - 1])[cell->id()][2 * q_point] +
              delRho[cell->id()][2 * q_point];
            (uBroyden[N - 1])[cell->id()][2 * q_point + 1] =
              G * (dFBroyden[N - 1])[cell->id()][2 * q_point + 1] +
              delRho[cell->id()][2 * q_point + 1];
            //
            if (dftParameters::xcFamilyType == "GGA")
              {
                for (unsigned int dir = 0; dir < 3; ++dir)
                  {
                    (graddFBroyden[N - 1])[cell->id()][6 * q_point + dir] /=
                      dfMag;
                    (graddFBroyden[N - 1])[cell->id()][6 * q_point + 3 + dir] /=
                      dfMag;
                    delGradRho[cell->id()][6 * q_point + dir] /= dfMag;
                    delGradRho[cell->id()][6 * q_point + 3 + dir] /= dfMag;
                    //
                    (gradUBroyden[N - 1])[cell->id()][6 * q_point + dir] =
                      G *
                        (graddFBroyden[N - 1])[cell->id()][6 * q_point + dir] +
                      delGradRho[cell->id()][6 * q_point + dir];
                    (gradUBroyden[N - 1])[cell->id()][6 * q_point + 3 + dir] =
                      G * (graddFBroyden[N - 1])[cell->id()]
                                                [6 * q_point + 3 + dir] +
                      delGradRho[cell->id()][6 * q_point + 3 + dir];
                  }
              }
            //
            for (unsigned int k = 0; k < N; ++k)
              {
                cLoc[k] += wtBroyden[k] *
                           ((dFBroyden[k])[cell->id()][2 * q_point] +
                            (dFBroyden[k])[cell->id()][2 * q_point + 1]) *
                           (FBroyden[cell->id()][2 * q_point] +
                            FBroyden[cell->id()][2 * q_point + 1]) *
                           fe_values.JxW(q_point);
                for (unsigned int l = k; l < N; ++l)
                  {
                    invBetaLoc[N * k + l] +=
                      wtBroyden[k] * wtBroyden[l] *
                      ((dFBroyden[k])[cell->id()][2 * q_point] +
                       (dFBroyden[k])[cell->id()][2 * q_point + 1]) *
                      ((dFBroyden[l])[cell->id()][2 * q_point] +
                       (dFBroyden[l])[cell->id()][2 * q_point + 1]) *
                      fe_values.JxW(q_point);
                    invBetaLoc[N * l + k] = invBetaLoc[N * k + l];
                  }
              }
          }
      }
  //
  for (unsigned int k = 0; k < N; ++k)
    {
      for (unsigned int l = 0; l < N; ++l)
        {
          invBeta[N * k + l] =
            Utilities::MPI::sum(invBetaLoc[N * k + l], mpi_communicator);
          //
          if (l == k)
            {
              invBeta[N * l + l] = w0Broyden * w0Broyden + invBeta[N * l + l];
              beta[N * l + l]    = 1.0;
            }
        }
      c[k] = Utilities::MPI::sum(cLoc[k], mpi_communicator);
    }

  //
  // Invert beta
  //
  calldgesv(N, &invBeta[0], &beta[0]);
  //

  for (unsigned int m = 0; m < N; ++m)
    for (unsigned int l = 0; l < N; ++l)
      gamma[m] += c[l] * beta[N * m + l];

  //
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());
  //
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOldSpinPolarized =
    *rhoInValuesSpinPolarized;
  rhoInValsSpinPolarized.push_back(
    std::map<dealii::CellId, std::vector<double>>());
  rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
  //
  std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;
  std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOldSpinPolarized;
  if (dftParameters::xcFamilyType == "GGA")
    {
      gradRhoInValuesOld = *gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
      //
      gradRhoInValuesOldSpinPolarized = *gradRhoInValuesSpinPolarized;
      gradRhoInValsSpinPolarized.push_back(
        std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
    }
  //
  cell = dofHandler.begin_active();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
          (*rhoInValuesSpinPolarized)[cell->id()] =
            std::vector<double>(2 * num_quad_points);
          if (dftParameters::xcFamilyType == "GGA")
            {
              (*gradRhoInValues)[cell->id()] =
                std::vector<double>(3 * num_quad_points);
              (*gradRhoInValuesSpinPolarized)[cell->id()] =
                std::vector<double>(6 * num_quad_points);
            }
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              // Compute (rhoIn-rhoOut)^2
              normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                      (*rhoOutValues)[cell->id()][q_point],
                                    2.0) *
                           fe_values.JxW(q_point);
              (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
                rhoInValuesOldSpinPolarized[cell->id()][2 * q_point] +
                G * FBroyden[cell->id()][2 * q_point];
              (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                rhoInValuesOldSpinPolarized[cell->id()][2 * q_point + 1] +
                G * FBroyden[cell->id()][2 * q_point + 1];
              //
              if (dftParameters::xcFamilyType == "GGA")
                for (unsigned int dir = 0; dir < 3; ++dir)
                  {
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                dir] =
                      gradRhoInValuesOldSpinPolarized[cell->id()]
                                                     [6 * q_point + dir] +
                      G * gradFBroyden[cell->id()][6 * q_point + dir];
                    (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                3 + dir] =
                      gradRhoInValuesOldSpinPolarized[cell->id()]
                                                     [6 * q_point + 3 + dir] +
                      G * gradFBroyden[cell->id()][6 * q_point + 3 + dir];
                  }
              //
              for (int i = 0; i < N; ++i)
                {
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] -=
                    wtBroyden[i] * gamma[i] *
                    (uBroyden[i])[cell->id()][2 * q_point];
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] -=
                    wtBroyden[i] * gamma[i] *
                    (uBroyden[i])[cell->id()][2 * q_point + 1];
                  if (dftParameters::xcFamilyType == "GGA")
                    for (unsigned int dir = 0; dir < 3; ++dir)
                      {
                        (*gradRhoInValuesSpinPolarized)[cell->id()]
                                                       [6 * q_point + dir] -=
                          wtBroyden[i] * gamma[i] *
                          (gradUBroyden[i])[cell->id()][6 * q_point + dir];
                        (*gradRhoInValuesSpinPolarized)[cell->id()]
                                                       [6 * q_point + 3 +
                                                        dir] -=
                          wtBroyden[i] * gamma[i] *
                          (gradUBroyden[i])[cell->id()][6 * q_point + 3 + dir];
                      }
                }
              (*rhoInValues)[cell->id()][q_point] =
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] +
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1];
              if (dftParameters::xcFamilyType == "GGA")
                for (unsigned int dir = 0; dir < 3; ++dir)
                  (*gradRhoInValues)[cell->id()][3 * q_point + dir] =
                    (*gradRhoInValuesSpinPolarized)[cell->id()]
                                                   [6 * q_point + dir] +
                    (*gradRhoInValuesSpinPolarized)[cell->id()]
                                                   [6 * q_point + 3 + dir];
            }
        }
    }
  //
  //



  return std::sqrt(Utilities::MPI::sum(normValue, mpi_communicator));
}



template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::mixing_simple_spinPolarized()
{
  double               normValue = 0.0;
  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(FE, quadrature, update_JxW_values);
  const unsigned int num_quad_points = quadrature.size();

  // create new rhoValue tables
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());

  std::map<dealii::CellId, std::vector<double>> rhoInValuesOldSpinPolarized =
    *rhoInValuesSpinPolarized;
  rhoInValsSpinPolarized.push_back(
    std::map<dealii::CellId, std::vector<double>>());
  rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());
  //

  // create new gradRhoValue tables
  std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;
  std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOldSpinPolarized;

  if (dftParameters::xcFamilyType == "GGA")
    {
      gradRhoInValuesOld = *gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());
      //
      gradRhoInValuesOldSpinPolarized = *gradRhoInValuesSpinPolarized;
      gradRhoInValsSpinPolarized.push_back(
        std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
    }

  // parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
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

          if (dftParameters::xcFamilyType == "GGA")
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
              (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] = std::abs(
                (1 - dftParameters::mixingParameter) *
                  (rhoInValuesOldSpinPolarized)[cell->id()][2 * q_point] +
                dftParameters::mixingParameter *
                  (*rhoOutValuesSpinPolarized)[cell->id()][2 * q_point]);
              (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                std::abs(
                  (1 - dftParameters::mixingParameter) *
                    (rhoInValuesOldSpinPolarized)[cell->id()][2 * q_point + 1] +
                  dftParameters::mixingParameter *
                    (*rhoOutValuesSpinPolarized)[cell->id()][2 * q_point + 1]);

              (*rhoInValues)[cell->id()][q_point] =
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] +
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1];
              //
              normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                      (*rhoOutValues)[cell->id()][q_point],
                                    2.0) *
                           fe_values.JxW(q_point);

              if (dftParameters::xcFamilyType == "GGA")
                {
                  for (unsigned int i = 0; i < 6; ++i)
                    {
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + i]) =
                        ((1 - dftParameters::mixingParameter) *
                           (gradRhoInValuesOldSpinPolarized)[cell->id()]
                                                            [6 * q_point + i] +
                         dftParameters::mixingParameter *
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

  return std::sqrt(Utilities::MPI::sum(normValue, mpi_communicator));
}

// implement anderson mixing scheme
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::mixing_anderson_spinPolarized()
{
  double               normValue = 0.0;
  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);
  FEValues<3>        fe_values(FE, quadrature, update_JxW_values);
  const unsigned int num_quad_points = quadrature.size();



  // initialize data structures
  int N = rhoOutVals.size() - 1;
  // pcout << "\nN:" << N << "\n";
  int                 NRHS = 1, lda = N, ldb = N, info;
  std::vector<int>    ipiv(N);
  std::vector<double> A(lda * N), c(ldb * NRHS);
  for (int i = 0; i < lda * N; i++)
    A[i] = 0.0;
  for (int i = 0; i < ldb * NRHS; i++)
    c[i] = 0.0;

  std::vector<std::vector<double>> rhoOutTemp(
    N + 1, std::vector<double>(num_quad_points, 0.0));

  std::vector<std::vector<double>> rhoInTemp(
    N + 1, std::vector<double>(num_quad_points, 0.0));

  std::vector<std::vector<double>> gradRhoOutTemp(
    N + 1, std::vector<double>(3 * num_quad_points, 0.0));

  std::vector<std::vector<double>> gradRhoInTemp(
    N + 1, std::vector<double>(3 * num_quad_points, 0.0));

  std::vector<std::vector<double>> rhoOutSpinPolarizedTemp(
    N + 1, std::vector<double>(2 * num_quad_points, 0.0));

  std::vector<std::vector<double>> rhoInSpinPolarizedTemp(
    N + 1, std::vector<double>(2 * num_quad_points, 0.0));

  std::vector<std::vector<double>> gradRhoOutSpinPolarizedTemp(
    N + 1, std::vector<double>(6 * num_quad_points, 0.0));

  std::vector<std::vector<double>> gradRhoInSpinPolarizedTemp(
    N + 1, std::vector<double>(6 * num_quad_points, 0.0));

  // parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),
                                               endc = dofHandler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          for (int hist = 0; hist < N + 1; hist++)
            {
              rhoOutTemp[hist] = (rhoOutVals[hist])[cell->id()];
              rhoInTemp[hist]  = (rhoInVals[hist])[cell->id()];
            }

          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              // fill coefficient matrix, rhs
              double Fn = rhoOutTemp[N][q_point] - rhoInTemp[N][q_point];
              for (int m = 0; m < N; m++)
                {
                  // double Fnm=((rhoOutVals[N-1-m])[cell->id()][q_point])-
                  // ((rhoInVals[N-1-m])[cell->id()][q_point]);
                  double Fnm = rhoOutTemp[N - 1 - m][q_point] -
                               rhoInTemp[N - 1 - m][q_point];
                  for (int k = 0; k < N; k++)
                    {
                      // double Fnk=((rhoOutVals[N-1-k])[cell->id()][q_point])-
                      // ((rhoInVals[N-1-k])[cell->id()][q_point]);
                      double Fnk = rhoOutTemp[N - 1 - k][q_point] -
                                   rhoInTemp[N - 1 - k][q_point];
                      A[k * N + m] += (Fn - Fnm) * (Fn - Fnk) *
                                      fe_values.JxW(q_point); // (m,k)^th entry
                    }
                  c[m] +=
                    (Fn - Fnm) * (Fn)*fe_values.JxW(q_point); // (m)^th entry
                }
            }
        }
    }
  // accumulate over all processors
  std::vector<double> ATotal(lda * N), cTotal(ldb * NRHS);
  MPI_Allreduce(
    &A[0], &ATotal[0], lda * N, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  MPI_Allreduce(
    &c[0], &cTotal[0], ldb * NRHS, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  //
  // pcout << "A,c:" << ATotal[0] << " " << cTotal[0] << "\n";
  // solve for coefficients
  dgesv_(&N, &NRHS, &ATotal[0], &lda, &ipiv[0], &cTotal[0], &ldb, &info);
  if ((info > 0) && (this_mpi_process == 0))
    {
      printf(
        "Anderson Mixing: The diagonal element of the triangular factor of A,\n");
      printf(
        "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n",
        info,
        info);
      exit(1);
    }
  double cn = 1.0;
  for (int i = 0; i < N; i++)
    cn -= cTotal[i];
  if (this_mpi_process == 0)
    {
      // printf("\nAnderson mixing  c%u:%12.6e, ", N+1, cn);
      // for (int i=0; i<N; i++) printf("c%u:%12.6e, ", N-i, cTotal[N-1-i]);
      // printf("\n");
    }

  // create new rhoValue tables
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
  rhoInValues = &(rhoInVals.back());

  //
  std::map<dealii::CellId, std::vector<double>> rhoInValuesOldSpinPolarized =
    *rhoInValuesSpinPolarized;
  rhoInValsSpinPolarized.push_back(
    std::map<dealii::CellId, std::vector<double>>());
  rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());

  //
  // implement anderson mixing
  cell = dofHandler.begin_active();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          // if (s==0) {
          (*rhoInValuesSpinPolarized)[cell->id()] =
            std::vector<double>(2 * num_quad_points);
          (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
          //}
          fe_values.reinit(cell);

          for (int hist = 0; hist < N + 1; hist++)
            {
              rhoOutSpinPolarizedTemp[hist] =
                (rhoOutValsSpinPolarized[hist])[cell->id()];
              rhoInSpinPolarizedTemp[hist] =
                (rhoInValsSpinPolarized[hist])[cell->id()];
            }

          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              // Compute (rhoIn-rhoOut)^2
              // normValue+=std::pow((*rhoInValuesOld)[cell->id()][2*q_point+s]-(*rhoOutValues)[cell->id()][2*q_point+s],2.0)*fe_values.JxW(q_point);
              // Anderson mixing scheme
              // normValue+=std::pow((*rhoInValuesOldSpinPolarized)[cell->id()][2*q_point]-(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point],2.0)*fe_values.JxW(q_point);
              // normValue+=std::pow((*rhoInValuesOldSpinPolarized)[cell->id()][2*q_point+1]-(*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1],2.0)*fe_values.JxW(q_point);
              normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                      (*rhoOutValues)[cell->id()][q_point],
                                    2.0) *
                           fe_values.JxW(q_point);
              double rhoOutBar1 = cn * rhoOutSpinPolarizedTemp[N][2 * q_point];
              double rhoInBar1  = cn * rhoInSpinPolarizedTemp[N][2 * q_point];
              for (int i = 0; i < N; i++)
                {
                  // rhoOutBar1+=cTotal[i]*(rhoOutValsSpinPolarized[N-1-i])[cell->id()][2*q_point];
                  // rhoInBar1+=cTotal[i]*(rhoInValsSpinPolarized[N-1-i])[cell->id()][2*q_point];
                  rhoOutBar1 +=
                    cTotal[i] * rhoOutSpinPolarizedTemp[N - 1 - i][2 * q_point];
                  rhoInBar1 +=
                    cTotal[i] * rhoInSpinPolarizedTemp[N - 1 - i][2 * q_point];
                }
              (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
                std::abs((1 - dftParameters::mixingParameter) * rhoInBar1 +
                         dftParameters::mixingParameter * rhoOutBar1);
              //
              double rhoOutBar2 =
                cn * rhoOutSpinPolarizedTemp[N][2 * q_point + 1];
              double rhoInBar2 =
                cn * rhoInSpinPolarizedTemp[N][2 * q_point + 1];
              for (int i = 0; i < N; i++)
                {
                  // rhoOutBar2+=cTotal[i]*(rhoOutValsSpinPolarized[N-1-i])[cell->id()][2*q_point+1];
                  // rhoInBar2+=cTotal[i]*(rhoInValsSpinPolarized[N-1-i])[cell->id()][2*q_point+1];
                  rhoOutBar2 +=
                    cTotal[i] *
                    rhoOutSpinPolarizedTemp[N - 1 - i][2 * q_point + 1];
                  rhoInBar2 +=
                    cTotal[i] *
                    rhoInSpinPolarizedTemp[N - 1 - i][2 * q_point + 1];
                }
              (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                std::abs((1 - dftParameters::mixingParameter) * rhoInBar2 +
                         dftParameters::mixingParameter * rhoOutBar2);
              //
              // if (s==1)
              //   {
              //    (*rhoInValues)[cell->id()][q_point]+=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+s]
              //    ;
              //     normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
              //   }
              // else
              //    (*rhoInValues)[cell->id()][q_point]=(*rhoInValuesSpinPolarized)[cell->id()][2*q_point+s]
              //    ;
              (*rhoInValues)[cell->id()][q_point] =
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] +
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1];
              // normValue+=std::pow((*rhoInValuesOld)[cell->id()][q_point]-(*rhoOutValues)[cell->id()][q_point],2.0)*fe_values.JxW(q_point);
            }
        }
    }

  // compute gradRho for GGA using mixing constants from rho mixing


  if (dftParameters::xcFamilyType == "GGA")
    {
      std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
        *gradRhoInValues;
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValues = &(gradRhoInVals.back());

      //
      gradRhoInValsSpinPolarized.push_back(
        std::map<dealii::CellId, std::vector<double>>());
      gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
      //
      cell = dofHandler.begin_active();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              (*gradRhoInValues)[cell->id()] =
                std::vector<double>(3 * num_quad_points);
              (*gradRhoInValuesSpinPolarized)[cell->id()] =
                std::vector<double>(6 * num_quad_points);
              //
              fe_values.reinit(cell);

              for (int hist = 0; hist < N + 1; hist++)
                {
                  gradRhoOutSpinPolarizedTemp[hist] =
                    (gradRhoOutValsSpinPolarized[hist])[cell->id()];
                  gradRhoInSpinPolarizedTemp[hist] =
                    (gradRhoInValsSpinPolarized[hist])[cell->id()];
                }

              for (unsigned int q_point = 0; q_point < num_quad_points;
                   ++q_point)
                {
                  //
                  // Anderson mixing scheme spin up
                  //
                  double gradRhoXOutBar1 =
                    cn *
                    gradRhoOutSpinPolarizedTemp
                      [N]
                      [6 * q_point +
                       0]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
                           // + 0];
                  double gradRhoYOutBar1 =
                    cn *
                    gradRhoOutSpinPolarizedTemp
                      [N]
                      [6 * q_point +
                       1]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
                           // + 1];
                  double gradRhoZOutBar1 =
                    cn *
                    gradRhoOutSpinPolarizedTemp
                      [N]
                      [6 * q_point +
                       2]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
                           // + 2];

                  double gradRhoXInBar1 =
                    cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 0];
                  double gradRhoYInBar1 =
                    cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 1];
                  double gradRhoZInBar1 =
                    cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 2];

                  for (int i = 0; i < N; i++)
                    {
                      gradRhoXOutBar1 +=
                        cTotal[i] *
                        gradRhoOutSpinPolarizedTemp[N - 1 - i][6 * q_point + 0];
                      gradRhoYOutBar1 +=
                        cTotal[i] *
                        gradRhoOutSpinPolarizedTemp[N - 1 - i][6 * q_point + 1];
                      gradRhoZOutBar1 +=
                        cTotal[i] *
                        gradRhoOutSpinPolarizedTemp[N - 1 - i][6 * q_point + 2];

                      gradRhoXInBar1 +=
                        cTotal[i] *
                        gradRhoInSpinPolarizedTemp[N - 1 - i][6 * q_point + 0];
                      gradRhoYInBar1 +=
                        cTotal[i] *
                        gradRhoInSpinPolarizedTemp[N - 1 - i][6 * q_point + 1];
                      gradRhoZInBar1 +=
                        cTotal[i] *
                        gradRhoInSpinPolarizedTemp[N - 1 - i][6 * q_point + 2];
                    }
                  //
                  // Anderson mixing scheme spin down
                  //
                  double gradRhoXOutBar2 =
                    cn * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 3];
                  double gradRhoYOutBar2 =
                    cn * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 4];
                  double gradRhoZOutBar2 =
                    cn * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 5];

                  double gradRhoXInBar2 =
                    cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 3];
                  double gradRhoYInBar2 =
                    cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 4];
                  double gradRhoZInBar2 =
                    cn * gradRhoInSpinPolarizedTemp[N][6 * q_point + 5];

                  for (int i = 0; i < N; i++)
                    {
                      gradRhoXOutBar2 +=
                        cTotal[i] *
                        gradRhoOutSpinPolarizedTemp[N - 1 - i][6 * q_point + 3];
                      gradRhoYOutBar2 +=
                        cTotal[i] *
                        gradRhoOutSpinPolarizedTemp[N - 1 - i][6 * q_point + 4];
                      gradRhoZOutBar2 +=
                        cTotal[i] *
                        gradRhoOutSpinPolarizedTemp[N - 1 - i][6 * q_point + 5];

                      gradRhoXInBar2 +=
                        cTotal[i] *
                        gradRhoInSpinPolarizedTemp[N - 1 - i][6 * q_point + 3];
                      gradRhoYInBar2 +=
                        cTotal[i] *
                        gradRhoInSpinPolarizedTemp[N - 1 - i][6 * q_point + 4];
                      gradRhoZInBar2 +=
                        cTotal[i] *
                        gradRhoInSpinPolarizedTemp[N - 1 - i][6 * q_point + 5];
                    }
                  //
                  (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 0] =
                    ((1 - dftParameters::mixingParameter) * gradRhoXInBar1 +
                     dftParameters::mixingParameter * gradRhoXOutBar1);
                  (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 1] =
                    ((1 - dftParameters::mixingParameter) * gradRhoYInBar1 +
                     dftParameters::mixingParameter * gradRhoYOutBar1);
                  (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 2] =
                    ((1 - dftParameters::mixingParameter) * gradRhoZInBar1 +
                     dftParameters::mixingParameter * gradRhoZOutBar1);
                  (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 3] =
                    ((1 - dftParameters::mixingParameter) * gradRhoXInBar2 +
                     dftParameters::mixingParameter * gradRhoXOutBar2);
                  (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 4] =
                    ((1 - dftParameters::mixingParameter) * gradRhoYInBar2 +
                     dftParameters::mixingParameter * gradRhoYOutBar2);
                  (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 5] =
                    ((1 - dftParameters::mixingParameter) * gradRhoZInBar2 +
                     dftParameters::mixingParameter * gradRhoZOutBar2);

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
  return std::sqrt(Utilities::MPI::sum(normValue, mpi_communicator));
}
