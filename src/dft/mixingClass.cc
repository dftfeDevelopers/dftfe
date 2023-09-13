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
// @author Vishal Subramanian
//

#include <mixingClass.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>

namespace dftfe
{
  MixingScheme::MixingScheme(const dftParameters &dftParam,
                             const MPI_Comm &mpi_comm_domain):
    d_dftParamsPtr(&dftParam),
    d_mpi_comm_domain(mpi_comm_domain)
  {

  }

  void MixingScheme::copyDensityFromInHist(std::map<dealii::CellId, std::vector<double>> *rhoInValues)
  {
    unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        numQuadPointsPerCell = 2*numQuadPointsPerCell;
      }

    unsigned int hist =d_rhoInVals.size();

    unsigned int iElem = 0;

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
              {
                (*rhoInValues)[cell->id()][iQuad] =
                  d_rhoInVals[hist-1][iElem*numQuadPointsPerCell +
                                  iQuad];
              }
            iElem++;
          }
      }

  }
  void MixingScheme::copyDensityFromOutHist(std::map<dealii::CellId, std::vector<double>> *rhoOutValues)
  {
    unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        numQuadPointsPerCell = 2*numQuadPointsPerCell;
      }

    unsigned int hist =d_rhoOutVals.size();

    unsigned int iElem = 0;

    d_dofHandler =
      &d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
              {
                (*rhoOutValues)[cell->id()][iQuad] =
                    d_rhoOutVals[hist-1][iElem*numQuadPointsPerCell +
                                 iQuad];
              }
            iElem++;
          }
      }

  }

  void MixingScheme::copyGradDensityFromInHist( std::map<dealii::CellId, std::vector<double>> *gradInput)
  {
    gradInput = &(d_gradRhoInVals.back());
  }

  void MixingScheme::copyGradDensityFromOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutput)
  {
    gradOutput = &(d_gradRhoOutVals.back());
  }

  void MixingScheme::copySpinGradDensityFromInHist( std::map<dealii::CellId, std::vector<double>> * gradInputSpin)
  {
    gradInputSpin = &(d_gradRhoInValsSpinPolarized.back());
  }

  void MixingScheme::copySpinGradDensityFromOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutputSpin)
  {
    gradOutputSpin = &(d_gradRhoOutValsSpinPolarized.back());
  }

  void MixingScheme::copyGradDensityToInHist( std::map<dealii::CellId, std::vector<double>> * gradInput)
  {
    d_gradRhoInVals.push_back(*gradInput);
  }
  void MixingScheme::copyGradDensityToOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutput)
  {
    d_gradRhoOutVals.push_back(*gradOutput);
  }

  void MixingScheme::copySpinGradDensityToInHist(std::map<dealii::CellId, std::vector<double>> * gradInputSpin)
  {
    d_gradRhoInValsSpinPolarized.push_back(*gradInputSpin);
  }
  void MixingScheme::copySpinGradDensityToOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutputSpin)
  {
    d_gradRhoOutValsSpinPolarized.push_back(*gradOutputSpin);
  }


  void MixingScheme::init(const dealii::MatrixFree<3, double> & matrixFreeData,
                     const unsigned int matrixFreeVectorComponent,
                     const unsigned int matrixFreeQuadratureComponent,
		     excManager * excManagerPtr)
  {

    d_matrixFreeDataPtr = &matrixFreeData ;
    d_matrixFreeVectorComponent = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;
    d_excManagerPtr = excManagerPtr;
        const dealii::Quadrature<3> &quadratureRhs =
          d_matrixFreeDataPtr->get_quadrature(
            d_matrixFreeQuadratureComponent);

        d_numberQuadraturePointsPerCell = quadratureRhs.size();

        d_dofHandler =
          &d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

        dealii::FEValues<3> fe_values(d_dofHandler->get_fe(),
                                      quadratureRhs,
                                      dealii::update_JxW_values);

        unsigned int iElem = 0;

	d_numCells = d_matrixFreeDataPtr->n_physical_cells();

        if(d_dftParamsPtr->spinPolarized == 0)
          {
            d_vecJxW.resize(d_numberQuadraturePointsPerCell*
                            d_numCells);
            typename dealii::DoFHandler<3>::active_cell_iterator
              cell = d_dofHandler->begin_active(),
              endc = d_dofHandler->end();
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned())
                  {
                    fe_values.reinit(cell);
                    for(unsigned int iQuad = 0 ;iQuad < d_numberQuadraturePointsPerCell; iQuad++)
                      {
                        d_vecJxW[iElem*d_numberQuadraturePointsPerCell +
                                 iQuad] =
                          fe_values.JxW(iQuad);
                      }
                    iElem++;
                  }
              }
          }
        else
          {
            d_vecJxW.resize(2*
                            d_numberQuadraturePointsPerCell*
                            d_numCells);
            typename dealii::DoFHandler<3>::active_cell_iterator
              cell = d_dofHandler->begin_active(),
              endc = d_dofHandler->end();
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned())
                  {
                    for(unsigned int iQuad = 0 ;iQuad < d_numberQuadraturePointsPerCell; iQuad++)
                      {
                        d_vecJxW[2*iElem*d_numberQuadraturePointsPerCell +
                                 2*iQuad
                                 +0] =
                          fe_values.JxW(iQuad);

                        d_vecJxW[2*iElem*d_numberQuadraturePointsPerCell +
                                 2*iQuad
                                 +1] =
                          fe_values.JxW(iQuad);
                      }
                    iElem++;
                  }
              }
          }
  }

  void MixingScheme::computeMixingMatricesDensity(const std::deque<std::vector<double>> &inHist,
                                      const std::deque<std::vector<double>> &outHist,
                                      std::vector<double> &A,
                                      std::vector<double> &c)
  {
    std::vector<double> Adensity;
    Adensity.resize(A.size());
    std::fill(Adensity.begin(), Adensity.end(),0.0);

    std::vector<double> cDensity;
    cDensity.resize(c.size());
    std::fill(cDensity.begin(),cDensity.end(),0.0);

    int N = d_rhoOutVals.size() - 1;

    std::cout<<" size of hist  = "<<N<<"\n";
    unsigned int numQuadPoints = 0;
    if( N > 0)
      numQuadPoints = inHist[0].size();

    if( numQuadPoints != d_vecJxW.size())
      {
        std::cout<<" ERROR in vec size in mixing anderson \n";
      }
    for( unsigned int iQuad = 0; iQuad < numQuadPoints; iQuad++)
      {
        double Fn = d_rhoOutVals[N][iQuad] - d_rhoInVals[N][iQuad];
        for (int m = 0; m < N; m++)
          {
            double Fnm = d_rhoOutVals[N - 1 - m][iQuad] -
                         d_rhoInVals[N - 1 - m][iQuad];
            for (int k = 0; k < N; k++)
              {
                double Fnk = d_rhoOutVals[N - 1 - k][iQuad] -
                             d_rhoInVals[N - 1 - k][iQuad];
                Adensity[k * N + m] +=
                  (Fn - Fnm) * (Fn - Fnk) *
                  d_vecJxW[iQuad]; // (m,k)^th entry
              }
            cDensity[m] +=
              (Fn - Fnm) * (Fn)*d_vecJxW[iQuad]; // (m)^th entry

          }
      }

    unsigned int aSize = Adensity.size();
    unsigned int cSize = cDensity.size();

    std::vector<double> ATotal(aSize), cTotal(cSize);
    std::fill(ATotal.begin(),ATotal.end(),0.0);
    std::fill(cTotal.begin(),cTotal.end(),0.0);
    MPI_Allreduce(
      &Adensity[0], &ATotal[0], aSize, MPI_DOUBLE, MPI_SUM, d_mpi_comm_domain);
    MPI_Allreduce(
      &cDensity[0], &cTotal[0], cSize, MPI_DOUBLE, MPI_SUM, d_mpi_comm_domain);

    for (unsigned int i = 0 ; i < aSize; i++)
      {
        A[i] += ATotal[i];
	std::cout<<"A["<<i<<"] = "<<A[i]<<"\n";
      }

    for (unsigned int i = 0 ; i < cSize; i++)
      {
        c[i] += cTotal[i];
	std::cout<<"c["<<i<<"] = "<<c[i]<<"\n";
      }
  }

  void MixingScheme::copyDensityToInHist(std::map<dealii::CellId, std::vector<double>> *rhoInValues)
  {
    std::vector<double> latestHistRhoIn;
    unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        numQuadPointsPerCell = 2*numQuadPointsPerCell;
      }

    latestHistRhoIn.resize(d_numCells*numQuadPointsPerCell);
    std::fill(latestHistRhoIn.begin(),latestHistRhoIn.end(),0.0);

    unsigned int iElem = 0;

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
              {
                latestHistRhoIn[iElem*numQuadPointsPerCell +
                                iQuad] =
                  (*rhoInValues)[cell->id()][iQuad];
              }
            iElem++;
          }
      }

    d_rhoInVals.push_back(latestHistRhoIn);
  }

  void MixingScheme::copyDensityToOutHist( std::map<dealii::CellId, std::vector<double>> *rhoOutValues)
  {
    std::vector<double> latestHistRhoOut;
    unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        numQuadPointsPerCell = 2*numQuadPointsPerCell;
      }

    latestHistRhoOut.resize(d_numCells*numQuadPointsPerCell);
    std::fill(latestHistRhoOut.begin(),latestHistRhoOut.end(),0.0);

    unsigned int iElem = 0;

    d_dofHandler =
      &d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
              {
                latestHistRhoOut[iElem*numQuadPointsPerCell +
                                iQuad] =
                  (*rhoOutValues)[cell->id()][iQuad];
              }
            iElem++;
          }
      }

    d_rhoOutVals.push_back(latestHistRhoOut);
  }

  unsigned int MixingScheme::lengthOfHistory()
  {
    return d_rhoInVals.size();
  }

  void MixingScheme::computeAndersonMixingCoeff()
  {
    // initialize data structures
    int N = d_rhoOutVals.size() - 1;
    // pcout << "\nN:" << N << "\n";
    int                 NRHS = 1, lda = N, ldb = N, info;
    std::vector<int>    ipiv(N);
    d_A.resize(lda * N);
    d_c.resize(ldb * NRHS);
    for (int i = 0; i < lda * N; i++)
      d_A[i] = 0.0;
    for (int i = 0; i < ldb * NRHS; i++)
      d_c[i] = 0.0;

    computeMixingMatricesDensity(d_rhoInVals,
                                 d_rhoOutVals,
                                 d_A,
                                 d_c);

    dgesv_(&N, &NRHS, &d_A[0], &lda, &ipiv[0], &d_c[0], &ldb, &info);
    
    for (unsigned int i = 0 ; i < ldb*NRHS; i++)
      {
        std::cout<<"d_c["<<i<<"] = "<<d_c[i]<<"\n";
      }

    d_cFinal = 1.0;
    for (int i = 0; i < N; i++)
      d_cFinal -= d_c[i];

    std::cout<<"cFinal = "<<d_cFinal<<"\n";
  }

  double MixingScheme::mixDensity(std::map<dealii::CellId, std::vector<double>> *rhoInValues,
                           std::map<dealii::CellId, std::vector<double>> *rhoOutValues,
                           std::map<dealii::CellId, std::vector<double>> *rhoInValuesSpinPolarized,
                           std::map<dealii::CellId, std::vector<double>> *rhoOutValuesSpinPolarized,
                           std::map<dealii::CellId, std::vector<double>> *gradRhoInValues,
                             std::map<dealii::CellId, std::vector<double>> *gradRhoOutValues,
                           std::map<dealii::CellId, std::vector<double>> *gradRhoInValuesSpinPolarized,
                           std::map<dealii::CellId, std::vector<double>> *gradRhoOutValuesSpinPolarized)
  {
    double normValue = 0.0;

    int N = d_rhoOutVals.size() - 1;

    std::map<dealii::CellId, std::vector<double>> rhoInputValues, rhoOutputValues;
    // create new rhoValue tables
    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;


    unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
    if(d_dftParamsPtr->spinPolarized == 0)
      {
        rhoInValuesOld = *rhoInValues;
        rhoInputValues = *rhoInValues;
        rhoOutputValues = *rhoOutValues;
      }
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        numQuadPointsPerCell = 2*numQuadPointsPerCell;

        rhoInValuesOld = *rhoInValuesSpinPolarized;
        rhoInputValues = *rhoInValuesSpinPolarized;
        rhoOutputValues = *rhoOutValuesSpinPolarized;
      }

    // implement anderson mixing
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            (rhoInputValues)[cell->id()] = std::vector<double>(numQuadPointsPerCell);

            for (unsigned int q_point = 0; q_point < numQuadPointsPerCell; ++q_point)
              {
                // Compute (rhoIn-rhoOut)^2
                normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                        (rhoOutputValues)[cell->id()][q_point],
                                      2.0) *
                             d_vecJxW[iElem*numQuadPointsPerCell + q_point];
                // Anderson mixing scheme
                // double rhoOutBar=cn*(rhoOutVals[N])[cell->id()][q_point];
                // double rhoInBar=cn*(rhoInVals[N])[cell->id()][q_point];
                double rhoOutBar = d_cFinal * d_rhoOutVals[N][iElem*numQuadPointsPerCell + q_point];
                double rhoInBar  = d_cFinal * d_rhoInVals[N][iElem*numQuadPointsPerCell + q_point];

                for (int i = 0; i < N; i++)
                  {
                    rhoOutBar += d_c[i] * d_rhoOutVals[N - 1 - i][iElem*numQuadPointsPerCell + q_point];
                    rhoInBar += d_c[i] * d_rhoInVals[N - 1 - i][iElem*numQuadPointsPerCell + q_point];
                  }
                (*rhoInValues)[cell->id()][q_point] =
                  ((1 - d_dftParamsPtr->mixingParameter) * rhoInBar +
                   d_dftParamsPtr->mixingParameter * rhoOutBar);
              }
            iElem++;
          }
      }

    // compute gradRho for GGA using mixing constants from rho mixing


    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
    {
        std::vector<std::vector<double>> gradRhoOutTemp(
          N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));

        std::vector<std::vector<double>> gradRhoInTemp(
          N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));

        std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
          *gradRhoInValues;
        d_gradRhoInVals.push_back(
          std::map<dealii::CellId, std::vector<double>>());
        gradRhoInValues = &(d_gradRhoInVals.back());
        cell            = d_dofHandler->begin_active();
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                (*gradRhoInValues)[cell->id()] =
                  std::vector<double>(3 * d_numberQuadraturePointsPerCell);


                for (int hist = 0; hist < N + 1; hist++)
                  {
                    gradRhoOutTemp[hist] = (d_gradRhoOutVals[hist])[cell->id()];
                    gradRhoInTemp[hist]  = (d_gradRhoInVals[hist])[cell->id()];
                  }


                for (unsigned int q_point = 0; q_point < d_numberQuadraturePointsPerCell;
                     ++q_point)
                  {
                    //
                    // Anderson mixing scheme
                    //
                    double gradRhoXOutBar =
                      d_cFinal *
                      gradRhoOutTemp
                        [N][3 * q_point +
                            0]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
                                // + 0];
                    double gradRhoYOutBar =
                      d_cFinal *
                      gradRhoOutTemp
                        [N][3 * q_point +
                            1]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
                                // + 1];
                    double gradRhoZOutBar =
                      d_cFinal *
                      gradRhoOutTemp
                        [N][3 * q_point +
                            2]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
                                // + 2];

                    double gradRhoXInBar =
                      d_cFinal *
                      gradRhoInTemp
                        [N][3 * q_point +
                            0]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
                                // + 0];
                    double gradRhoYInBar =
                      d_cFinal *
                      gradRhoInTemp
                        [N][3 * q_point +
                            1]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
                                // + 1];
                    double gradRhoZInBar =
                      d_cFinal *
                      gradRhoInTemp
                        [N][3 * q_point +
                            2]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
                                // + 2];

                    for (int i = 0; i < N; i++)
                      {
                        gradRhoXOutBar +=
                          d_c[i] *
                          gradRhoOutTemp
                            [N - 1 - i]
                            [3 * q_point +
                             0]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
                                 // + 0];
                        gradRhoYOutBar +=
                          d_c[i] *
                          gradRhoOutTemp
                            [N - 1 - i]
                            [3 * q_point +
                             1]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
                                 // + 1];
                        gradRhoZOutBar +=
                          d_c[i] *
                          gradRhoOutTemp
                            [N - 1 - i]
                            [3 * q_point +
                             2]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
                                 // + 2];

                        gradRhoXInBar +=
                          d_c[i] *
                          gradRhoInTemp
                            [N - 1 - i]
                            [3 * q_point +
                             0]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
                                 // + 0];
                        gradRhoYInBar +=
                          d_c[i] *
                          gradRhoInTemp
                            [N - 1 - i]
                            [3 * q_point +
                             1]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
                                 // + 1];
                        gradRhoZInBar +=
                          d_c[i] *
                          gradRhoInTemp
                            [N - 1 - i]
                            [3 * q_point +
                             2]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
                                 // + 2];
                      }

                    (*gradRhoInValues)[cell->id()][3 * q_point + 0] =
                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar +
                       d_dftParamsPtr->mixingParameter * gradRhoXOutBar);
                    (*gradRhoInValues)[cell->id()][3 * q_point + 1] =
                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar +
                       d_dftParamsPtr->mixingParameter * gradRhoYOutBar);
                    (*gradRhoInValues)[cell->id()][3 * q_point + 2] =
                      ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar +
                       d_dftParamsPtr->mixingParameter * gradRhoZOutBar);
                  }
              }
          }
    if(d_dftParamsPtr->spinPolarized == 1)
      {
        std::vector<std::vector<double>> gradRhoOutTemp(
          N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));

        std::vector<std::vector<double>> gradRhoInTemp(
          N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));

        std::vector<std::vector<double>> gradRhoOutSpinPolarizedTemp(
          N + 1, std::vector<double>(6 * d_numberQuadraturePointsPerCell, 0.0));

        std::vector<std::vector<double>> gradRhoInSpinPolarizedTemp(
          N + 1, std::vector<double>(6 * d_numberQuadraturePointsPerCell, 0.0));

          std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
            *gradRhoInValues;
          d_gradRhoInVals.push_back(
            std::map<dealii::CellId, std::vector<double>>());
          gradRhoInValues = &(d_gradRhoInVals.back());

          //
          d_gradRhoInValsSpinPolarized.push_back(
            std::map<dealii::CellId, std::vector<double>>());
          gradRhoInValuesSpinPolarized = &(d_gradRhoInValsSpinPolarized.back());
          //
          cell = d_dofHandler->begin_active();
          for (; cell != endc; ++cell)
            {
              if (cell->is_locally_owned())
                {
                  (*gradRhoInValues)[cell->id()] =
                    std::vector<double>(3 * d_numberQuadraturePointsPerCell);
                  (*gradRhoInValuesSpinPolarized)[cell->id()] =
                    std::vector<double>(6 * d_numberQuadraturePointsPerCell);
                  //

                  for (int hist = 0; hist < N + 1; hist++)
                    {
                      gradRhoOutSpinPolarizedTemp[hist] =
                        (d_gradRhoOutValsSpinPolarized[hist])[cell->id()];
                      gradRhoInSpinPolarizedTemp[hist] =
                        (d_gradRhoInValsSpinPolarized[hist])[cell->id()];
                    }

                  for (unsigned int q_point = 0; q_point < d_numberQuadraturePointsPerCell;
                       ++q_point)
                    {
                      //
                      // Anderson mixing scheme spin up
                      //
                      double gradRhoXOutBar1 =
                        d_cFinal *
                        gradRhoOutSpinPolarizedTemp
                          [N]
                          [6 * q_point +
                           0]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
                               // + 0];
                      double gradRhoYOutBar1 =
                        d_cFinal *
                        gradRhoOutSpinPolarizedTemp
                          [N]
                          [6 * q_point +
                           1]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
                               // + 1];
                      double gradRhoZOutBar1 =
                        d_cFinal *
                        gradRhoOutSpinPolarizedTemp
                          [N]
                          [6 * q_point +
                           2]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
                               // + 2];

                      double gradRhoXInBar1 =
                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 0];
                      double gradRhoYInBar1 =
                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 1];
                      double gradRhoZInBar1 =
                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 2];

                      for (int i = 0; i < N; i++)
                        {
                          gradRhoXOutBar1 +=
                            d_c[i] *
                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
                                                       [6 * q_point + 0];
                          gradRhoYOutBar1 +=
                            d_c[i] *
                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
                                                       [6 * q_point + 1];
                          gradRhoZOutBar1 +=
                            d_c[i] *
                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
                                                       [6 * q_point + 2];

                          gradRhoXInBar1 +=
                            d_c[i] *
                            gradRhoInSpinPolarizedTemp[N - 1 - i]
                                                      [6 * q_point + 0];
                          gradRhoYInBar1 +=
                            d_c[i] *
                            gradRhoInSpinPolarizedTemp[N - 1 - i]
                                                      [6 * q_point + 1];
                          gradRhoZInBar1 +=
                            d_c[i] *
                            gradRhoInSpinPolarizedTemp[N - 1 - i]
                                                      [6 * q_point + 2];
                        }
                      //
                      // Anderson mixing scheme spin down
                      //
                      double gradRhoXOutBar2 =
                        d_cFinal * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 3];
                      double gradRhoYOutBar2 =
                        d_cFinal * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 4];
                      double gradRhoZOutBar2 =
                        d_cFinal * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 5];

                      double gradRhoXInBar2 =
                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 3];
                      double gradRhoYInBar2 =
                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 4];
                      double gradRhoZInBar2 =
                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 5];

                      for (int i = 0; i < N; i++)
                        {
                          gradRhoXOutBar2 +=
                            d_c[i] *
                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
                                                       [6 * q_point + 3];
                          gradRhoYOutBar2 +=
                            d_c[i] *
                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
                                                       [6 * q_point + 4];
                          gradRhoZOutBar2 +=
                            d_c[i] *
                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
                                                       [6 * q_point + 5];

                          gradRhoXInBar2 +=
                            d_c[i] *
                            gradRhoInSpinPolarizedTemp[N - 1 - i]
                                                      [6 * q_point + 3];
                          gradRhoYInBar2 +=
                            d_c[i] *
                            gradRhoInSpinPolarizedTemp[N - 1 - i]
                                                      [6 * q_point + 4];
                          gradRhoZInBar2 +=
                            d_c[i] *
                            gradRhoInSpinPolarizedTemp[N - 1 - i]
                                                      [6 * q_point + 5];
                        }
                      //
                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                  0] =
                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar1 +
                         d_dftParamsPtr->mixingParameter * gradRhoXOutBar1);
                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                  1] =
                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar1 +
                         d_dftParamsPtr->mixingParameter * gradRhoYOutBar1);
                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                  2] =
                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar1 +
                         d_dftParamsPtr->mixingParameter * gradRhoZOutBar1);
                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                  3] =
                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar2 +
                         d_dftParamsPtr->mixingParameter * gradRhoXOutBar2);
                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                  4] =
                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar2 +
                         d_dftParamsPtr->mixingParameter * gradRhoYOutBar2);
                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
                                                                  5] =
                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar2 +
                         d_dftParamsPtr->mixingParameter * gradRhoZOutBar2);

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
      }
    //copyDensityToInHist(rhoInValues);
    return std::sqrt(dealii::Utilities::MPI::sum(normValue, d_mpi_comm_domain));
  }

  void MixingScheme::clearHistory()
  {
	  d_rhoInVals.clear();
	  d_rhoOutVals.clear();
	  d_gradRhoInVals.clear();
	  d_gradRhoOutVals.clear();
	  d_gradRhoInValsSpinPolarized.clear();
	  d_gradRhoOutValsSpinPolarized.clear();

  }
  void MixingScheme::popOldHistory()
  {
    if (d_rhoInVals.size() == d_dftParamsPtr->mixingHistory)
      {
        d_rhoInVals.pop_front();
        d_rhoOutVals.pop_front();

        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA) // GGA
          {
            d_gradRhoInVals.pop_front();
            d_gradRhoOutVals.pop_front();
          }

        if (d_dftParamsPtr->spinPolarized == 1 &&
            d_excManagerPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA)
          {
            d_gradRhoInValsSpinPolarized.pop_front();
            d_gradRhoOutValsSpinPolarized.pop_front();
          }

      }
  }

  void MixingScheme::popRhoInHist()
  {
    d_rhoInVals.pop_front();
  }

  void MixingScheme::popRhoOutHist()
  {
    d_rhoOutVals.pop_front();
  }

  void MixingScheme::popGradRhoInHist()
  {
    d_gradRhoInVals.pop_front();
  }

  void MixingScheme::popGradRhoOutHist()
  {
    d_gradRhoOutVals.pop_front();
  }

  void MixingScheme::popGradRhoSpinInHist()
  {
    d_gradRhoInValsSpinPolarized.pop_front();
  }

  void MixingScheme::popGradRhoSpinOutHist()
  {
    d_gradRhoOutValsSpinPolarized.pop_front();
  }

} // end of namespace
