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
  MixingScheme::MixingScheme(const MPI_Comm &mpi_comm_domain):
    d_mpi_comm_domain(mpi_comm_domain)
  {

  }

  void MixingScheme::addMixingVariable(mixingVariable &mixingVariableList,
                         std::vector<double> &weightDotProducts,
                         bool performMPIReduce,
                                  double mixingValue)
  {
    d_variableHistoryIn.insert({mixingVariableList, std::deque<std::vector<double>>()});

    d_variableHistoryOut.insert({mixingVariableList, std::deque<std::vector<double>>()});

    d_vectorDotProductWeights.insert({mixingVariableList,weightDotProducts});

    d_performMPIReduce.insert({mixingVariableList,performMPIReduce});

    d_mixingParameter.insert({mixingVariableList,mixingValue});
  }

  void MixingScheme::computeMixingMatrices(const std::deque<std::vector<double>> &inHist,
                                      const std::deque<std::vector<double>> &outHist,
                                      const std::vector<double> &weightDotProducts,
                                      const bool isMPIAllReduce,
                                      std::vector<double> &A,
                                      std::vector<double> &c)
  {
    std::vector<double> Adensity;
    Adensity.resize(A.size());
    std::fill(Adensity.begin(), Adensity.end(),0.0);

    std::vector<double> cDensity;
    cDensity.resize(c.size());
    std::fill(cDensity.begin(),cDensity.end(),0.0);

    int N = inHist.size() - 1;

//    std::cout<<" size of hist  = "<<N<<"\n";
    unsigned int numQuadPoints = 0;
    if( N > 0)
      numQuadPoints = inHist[0].size();

    if( numQuadPoints != weightDotProducts.size())
      {
        std::cout<<" ERROR in vec size in mixing anderson \n";
      }
    for( unsigned int iQuad = 0; iQuad < numQuadPoints; iQuad++)
      {
        double Fn = outHist[N][iQuad] - inHist[N][iQuad];
        for (int m = 0; m < N; m++)
          {
            double Fnm = outHist[N - 1 - m][iQuad] -
                         inHist[N - 1 - m][iQuad];
            for (int k = 0; k < N; k++)
              {
                double Fnk = outHist[N - 1 - k][iQuad] -
                             inHist[N - 1 - k][iQuad];
                Adensity[k * N + m] +=
                  (Fn - Fnm) * (Fn - Fnk) *
                  weightDotProducts[iQuad]; // (m,k)^th entry
              }
            cDensity[m] +=
              (Fn - Fnm) * (Fn)*weightDotProducts[iQuad]; // (m)^th entry

          }
      }

    unsigned int aSize = Adensity.size();
    unsigned int cSize = cDensity.size();

    std::vector<double> ATotal(aSize), cTotal(cSize);
    std::fill(ATotal.begin(),ATotal.end(),0.0);
    std::fill(cTotal.begin(),cTotal.end(),0.0);
    if (isMPIAllReduce)
      {
        MPI_Allreduce(
          &Adensity[0], &ATotal[0], aSize, MPI_DOUBLE, MPI_SUM, d_mpi_comm_domain);
        MPI_Allreduce(
          &cDensity[0], &cTotal[0], cSize, MPI_DOUBLE, MPI_SUM, d_mpi_comm_domain);
      }
    else
      {
        ATotal = Adensity;
        cTotal = cDensity;
      }



    for (unsigned int i = 0 ; i < aSize; i++)
      {
        A[i] += ATotal[i];
//	std::cout<<"A["<<i<<"] = "<<A[i]<<"\n";
      }

    for (unsigned int i = 0 ; i < cSize; i++)
      {
        c[i] += cTotal[i];
//	std::cout<<"c["<<i<<"] = "<<c[i]<<"\n";
      }
  }



  unsigned int MixingScheme::lengthOfHistory()
  {
    return variableHistoryIn[mixingVariable::rho].size();
  }

  void MixingScheme::computeAndersonMixingCoeff()
  {
    // initialize data structures
    int N = variableHistoryIn[mixingVariable::rho].size() - 1;
    // pcout << "\nN:" << N << "\n";
    int                 NRHS = 1, lda = N, ldb = N, info;
    std::vector<int>    ipiv(N);
    d_A.resize(lda * N);
    d_c.resize(ldb * NRHS);
    for (int i = 0; i < lda * N; i++)
      d_A[i] = 0.0;
    for (int i = 0; i < ldb * NRHS; i++)
      d_c[i] = 0.0;

    for (const auto& [key, value] : variableHistoryIn)
      {
        computeMixingMatrices(variableHistoryIn[key],
                              variableHistoryOut[key],
                              vectorDotProductWeights[key],
                              performMPIReduce[key],
                              d_A,
                              d_c);
      }

    dgesv_(&N, &NRHS, &d_A[0], &lda, &ipiv[0], &d_c[0], &ldb, &info);
    
//    for (unsigned int i = 0 ; i < ldb*NRHS; i++)
//      {
//        std::cout<<"d_c["<<i<<"] = "<<d_c[i]<<"\n";
//      }

    d_cFinal = 1.0;
    for (int i = 0; i < N; i++)
      d_cFinal -= d_c[i];

//    std::cout<<"cFinal = "<<d_cFinal<<"\n";
  }

  void MixingScheme::addVariableToInHist(mixingVariable &mixingVariableName,
                             std::vector<double> &inputVariableToInHist)
  {
    variableHistoryIn[mixingVariableName].push_back(inputVariableToInHist);
  }

  void MixingScheme::addVariableToOutHist(mixingVariable &mixingVariableName,
                       std::vector<double> &inputVariableToOutHist)
  {
    variableHistoryOut[mixingVariableName].push_back(inputVariableToOutHist);
  }

  double MixingScheme::mixVariable(mixingVariable &mixingVariableName,
                     std::vector<double> &outputVariable)
  {
    double normValue = 0.0;
    int N = variableHistoryIn[mixingVariableName].size() - 1;
    unsigned int lenVar = variableHistoryIn[mixingVariableName][0].size();
    outputVariable.resize(lenVar);
    std::fill(outputVariable.begin(),outputVariable.end(),0.0);

    for( unsigned int iQuad = 0; iQuad < lenVar; iQuad++)
      {
        normValue += std::pow(variableHistoryOut[mixingVariableName][N][iQuad] -
                                variableHistoryIn[mixingVariableName][N][iQuad],
                              2.0) *
                     vectorDotProductWeights[mixingVariableName][iQuad];

        double varOutBar = d_cFinal * variableHistoryOut[mixingVariableName][N][iQuad];
        double varInBar  = d_cFinal * variableHistoryIn[mixingVariableName][N][iQuad];

        for (int i = 0; i < N; i++)
          {
            varOutBar += d_c[i] * variableHistoryOut[mixingVariableName][N - 1 - i][iQuad];
            varInBar += d_c[i] * variableHistoryIn[mixingVariableName][N - 1 - i][iQuad];
          }
        outputVariable[iQuad] = ((1 - d_mixingParameter[mixingVariableName]) * varInBar +
                                 d_mixingParameter[mixingVariableName] * varOutBar);

      }

  }

  void MixingScheme::clearHistory()
  {
    for (const auto& [key, value] : variableHistoryIn)
      {
        variableHistoryIn[key].clear();
        variableHistoryOut[key].clear();
      }
  }
  void MixingScheme::popOldHistory()
  {
    if (variableHistoryIn[mixingVariable::rho].size() >=  d_dftParamsPtr->mixingHistory)
      {

        for (const auto& [key, value] : variableHistoryIn)
          {
            variableHistoryIn[key].pop_front();
            variableHistoryOut[key].pop_front();
          }
      }
  }

} // end of namespace


//  void MixingScheme::popRhoInHist()
//  {
//    d_rhoInVals.pop_front();
//  }
//
//  void MixingScheme::popRhoOutHist()
//  {
//    d_rhoOutVals.pop_front();
//  }
//
//  void MixingScheme::popGradRhoInHist()
//  {
//    d_gradRhoInVals.pop_front();
//  }
//
//  void MixingScheme::popGradRhoOutHist()
//  {
//    d_gradRhoOutVals.pop_front();
//  }
//
//  void MixingScheme::popGradRhoSpinInHist()
//  {
//    d_gradRhoInValsSpinPolarized.pop_front();
//  }
//
//  void MixingScheme::popGradRhoSpinOutHist()
//  {
//    d_gradRhoOutValsSpinPolarized.pop_front();
//  }

//double MixingScheme::mixDensity(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValues,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoOutValues,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValuesSpinPolarized,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoOutValuesSpinPolarized,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoInValues,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoOutValues,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoInValuesSpinPolarized,
//                         std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoOutValuesSpinPolarized)
//{
//  double normValue = 0.0;
//
//  //    std::cout<<" size of rhoInHist = "<<d_rhoInVals.size()<<"\n";
//  //    std::cout<<" size of rhoOutHist = "<<d_rhoOutVals.size()<<"\n";
//  //    std::cout<<" size of gradRhoInHist = "<<d_gradRhoInVals.size()<<"\n";
//  //    std::cout<<" size of gradRhoOutHist = "<<d_gradRhoOutVals.size()<<"\n";
//  //    std::cout<<" size of gradSpinRhoInHist = "<<d_gradRhoInValsSpinPolarized.size()<<"\n";
//  //    std::cout<<" size of gradSpinRhoOutHist = "<<d_gradRhoOutValsSpinPolarized.size()<<"\n";
//
//  int N = d_rhoOutVals.size() - 1;
//
//  std::vector<double> rhoInHistNorm(N+1,0.0), rhoOutHistNorm(N+1,0.0), gradRhoInHistNorm(N+1,0.0), gradRhoOutHistNorm(N+1,0.0), gradSpinRhoInHistNorm(N+1,0.0), gradSpinRhoOutHistNorm(N+1,0.0);
//  double rhoOutputNorm = 0.0, gradRhoOutputNorm = 0.0;
//
//  std::fill(rhoInHistNorm.begin(),rhoInHistNorm.end(),0.0);
//  std::fill(rhoOutHistNorm.begin(),rhoOutHistNorm.end(),0.0);
//  std::fill(gradRhoInHistNorm.begin(),gradRhoInHistNorm.end(),0.0);
//  std::fill(gradRhoOutHistNorm.begin(),gradRhoOutHistNorm.end(),0.0);
//  std::fill(gradSpinRhoInHistNorm.begin(),gradSpinRhoInHistNorm.end(),0.0);
//  std::fill(gradSpinRhoOutHistNorm.begin(),gradSpinRhoOutHistNorm.end(),0.0);
//
//  std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> rhoInputValues, rhoOutputValues;
//  // create new rhoValue tables
//  std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
//
//
//  unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
//  if(d_dftParamsPtr->spinPolarized == 0)
//    {
//      rhoInValuesOld = *rhoInValues;
//      rhoInputValues = rhoInValues;
//      rhoOutputValues = rhoOutValues;
//    }
//  if (d_dftParamsPtr->spinPolarized == 1)
//    {
//      numQuadPointsPerCell = 2*numQuadPointsPerCell;
//
//      rhoInValuesOld = *rhoInValuesSpinPolarized;
//      rhoInputValues = rhoInValuesSpinPolarized;
//      rhoOutputValues = rhoOutValuesSpinPolarized;
//    }
//
//  // implement anderson mixing
//  typename dealii::DoFHandler<3>::active_cell_iterator
//    cell = d_dofHandler->begin_active(),
//    endc = d_dofHandler->end();
//  unsigned int iElem = 0;
//  for (; cell != endc; ++cell)
//    {
//      if (cell->is_locally_owned())
//        {
//          (*rhoInputValues)[cell->id()].resize(numQuadPointsPerCell);
//
//          for (unsigned int q_point = 0; q_point < numQuadPointsPerCell; ++q_point)
//            {
//
//              for(unsigned int iHist = 0;  iHist < N+1; iHist++)
//                {
//                  rhoInHistNorm[iHist] += std::abs(d_rhoInVals[iHist][iElem*numQuadPointsPerCell + q_point]);
//                  rhoOutHistNorm[iHist] += std::abs(d_rhoOutVals[iHist][iElem*numQuadPointsPerCell + q_point]);
//                }
//              // Compute (rhoIn-rhoOut)^2
//              normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
//                                      (*rhoOutputValues)[cell->id()][q_point],
//                                    2.0) *
//                           d_vecJxW[iElem*numQuadPointsPerCell + q_point];
//              // Anderson mixing scheme
//              // double rhoOutBar=cn*(rhoOutVals[N])[cell->id()][q_point];
//              // double rhoInBar=cn*(rhoInVals[N])[cell->id()][q_point];
//              double rhoOutBar = d_cFinal * d_rhoOutVals[N][iElem*numQuadPointsPerCell + q_point];
//              double rhoInBar  = d_cFinal * d_rhoInVals[N][iElem*numQuadPointsPerCell + q_point];
//
//              for (int i = 0; i < N; i++)
//                {
//                  rhoOutBar += d_c[i] * d_rhoOutVals[N - 1 - i][iElem*numQuadPointsPerCell + q_point];
//                  rhoInBar += d_c[i] * d_rhoInVals[N - 1 - i][iElem*numQuadPointsPerCell + q_point];
//                }
//              (*rhoInputValues)[cell->id()][q_point] =
//                ((1 - d_dftParamsPtr->mixingParameter) * rhoInBar +
//                 d_dftParamsPtr->mixingParameter * rhoOutBar);
//              rhoOutputNorm += (*rhoInputValues)[cell->id()][q_point];
//            }
//
//          if(d_dftParamsPtr->spinPolarized == 1)
//            {
//              for (unsigned int q_point = 0; q_point < d_numberQuadraturePointsPerCell; ++q_point)
//                {
//                  (*rhoInValues)[cell->id()][q_point] = (*rhoInputValues)[cell->id()][2*q_point + 0] +
//                                                        (*rhoInputValues)[cell->id()][2*q_point + 1];
//                }
//            }
//          iElem++;
//        }
//    }
//
//  MPI_Allreduce(MPI_IN_PLACE,
//                &rhoOutputNorm,
//                1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//
//  MPI_Allreduce(MPI_IN_PLACE,
//                &rhoInHistNorm[0],
//                N+1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//  MPI_Allreduce(MPI_IN_PLACE,
//                &rhoOutHistNorm[0],
//                N+1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//
//  //        std::cout<<" norm of output rho = "<<rhoOutputNorm<<"\n";
//  //
//  //        for(unsigned int iHist = 0; iHist < N+1; iHist++)
//  //          {
//  //            std::cout<<" norm of rhoIn["<<iHist<<"] = "<<rhoInHistNorm[iHist]<<" norm of rhoOut["<<iHist<<"] = "<<rhoOutHistNorm[iHist]<<"\n";
//  //          }
//
//
//
//
//  // compute gradRho for GGA using mixing constants from rho mixing
//
//
//  if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
//    {
//      if(d_dftParamsPtr->spinPolarized == 0)
//        {
//          std::vector<std::vector<double>> gradRhoOutTemp(
//            N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));
//
//          std::vector<std::vector<double>> gradRhoInTemp(
//            N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));
//
//          std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
//            *gradRhoInValues;
//          gradRhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
//          cell            = d_dofHandler->begin_active();
//          for (; cell != endc; ++cell)
//            {
//              if (cell->is_locally_owned())
//                {
//                  (*gradRhoInValues)[cell->id()] =
//                    std::vector<double>(3 * d_numberQuadraturePointsPerCell);
//
//
//                  for (int hist = 0; hist < N + 1; hist++)
//                    {
//                      gradRhoOutTemp[hist] = (*(d_gradRhoOutVals[hist]))[cell->id()];
//                      gradRhoInTemp[hist]  = (*(d_gradRhoInVals[hist]))[cell->id()];
//                    }
//
//
//                  for (unsigned int q_point = 0; q_point < d_numberQuadraturePointsPerCell;
//                       ++q_point)
//                    {
//                      for( unsigned int iHist = 0 ;iHist < N+1;iHist++)
//                        {
//                          gradRhoInHistNorm[iHist] += std::abs(gradRhoInTemp[iHist][3*q_point+0]);
//                          gradRhoInHistNorm[iHist] += std::abs(gradRhoInTemp[iHist][3*q_point+1]);
//                          gradRhoInHistNorm[iHist] += std::abs(gradRhoInTemp[iHist][3*q_point+2]);
//                          gradRhoOutHistNorm[iHist] += std::abs(gradRhoOutTemp[iHist][3*q_point+0]);
//                          gradRhoOutHistNorm[iHist] += std::abs(gradRhoOutTemp[iHist][3*q_point+1]);
//                          gradRhoOutHistNorm[iHist] += std::abs(gradRhoOutTemp[iHist][3*q_point+2]);
//
//                        }
//                      //
//                      // Anderson mixing scheme
//                      //
//                      double gradRhoXOutBar =
//                        d_cFinal *
//                        gradRhoOutTemp
//                          [N][3 * q_point +
//                              0]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
//                                  // + 0];
//                      double gradRhoYOutBar =
//                        d_cFinal *
//                        gradRhoOutTemp
//                          [N][3 * q_point +
//                              1]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
//                                  // + 1];
//                      double gradRhoZOutBar =
//                        d_cFinal *
//                        gradRhoOutTemp
//                          [N][3 * q_point +
//                              2]; // cn*(gradRhoOutVals[N])[cell->id()][3*q_point
//                                  // + 2];
//
//                      double gradRhoXInBar =
//                        d_cFinal *
//                        gradRhoInTemp
//                          [N][3 * q_point +
//                              0]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
//                                  // + 0];
//                      double gradRhoYInBar =
//                        d_cFinal *
//                        gradRhoInTemp
//                          [N][3 * q_point +
//                              1]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
//                                  // + 1];
//                      double gradRhoZInBar =
//                        d_cFinal *
//                        gradRhoInTemp
//                          [N][3 * q_point +
//                              2]; // cn*(gradRhoInVals[N])[cell->id()][3*q_point
//                                  // + 2];
//
//                      for (int i = 0; i < N; i++)
//                        {
//                          gradRhoXOutBar +=
//                            d_c[i] *
//                            gradRhoOutTemp
//                              [N - 1 - i]
//                              [3 * q_point +
//                               0]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
//                                   // + 0];
//                          gradRhoYOutBar +=
//                            d_c[i] *
//                            gradRhoOutTemp
//                              [N - 1 - i]
//                              [3 * q_point +
//                               1]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
//                                   // + 1];
//                          gradRhoZOutBar +=
//                            d_c[i] *
//                            gradRhoOutTemp
//                              [N - 1 - i]
//                              [3 * q_point +
//                               2]; // cTotal[i]*(gradRhoOutVals[N-1-i])[cell->id()][3*q_point
//                                   // + 2];
//
//                          gradRhoXInBar +=
//                            d_c[i] *
//                            gradRhoInTemp
//                              [N - 1 - i]
//                              [3 * q_point +
//                               0]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
//                                   // + 0];
//                          gradRhoYInBar +=
//                            d_c[i] *
//                            gradRhoInTemp
//                              [N - 1 - i]
//                              [3 * q_point +
//                               1]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
//                                   // + 1];
//                          gradRhoZInBar +=
//                            d_c[i] *
//                            gradRhoInTemp
//                              [N - 1 - i]
//                              [3 * q_point +
//                               2]; // cTotal[i]*(gradRhoInVals[N-1-i])[cell->id()][3*q_point
//                                   // + 2];
//                        }
//
//                      (*gradRhoInValues)[cell->id()][3 * q_point + 0] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar +
//                         d_dftParamsPtr->mixingParameter * gradRhoXOutBar);
//                      (*gradRhoInValues)[cell->id()][3 * q_point + 1] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar +
//                         d_dftParamsPtr->mixingParameter * gradRhoYOutBar);
//                      (*gradRhoInValues)[cell->id()][3 * q_point + 2] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar +
//                         d_dftParamsPtr->mixingParameter * gradRhoZOutBar);
//
//                      gradRhoOutputNorm += std::abs((*gradRhoInValues)[cell->id()][3 * q_point + 0]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValues)[cell->id()][3 * q_point + 1]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValues)[cell->id()][3 * q_point + 2]);
//                    }
//                }
//            }
//        }
//      else
//        {
//          std::vector<std::vector<double>> gradRhoOutTemp(
//            N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));
//
//          std::vector<std::vector<double>> gradRhoInTemp(
//            N + 1, std::vector<double>(3 * d_numberQuadraturePointsPerCell, 0.0));
//
//          std::vector<std::vector<double>> gradRhoOutSpinPolarizedTemp(
//            N + 1, std::vector<double>(6 * d_numberQuadraturePointsPerCell, 0.0));
//
//          std::vector<std::vector<double>> gradRhoInSpinPolarizedTemp(
//            N + 1, std::vector<double>(6 * d_numberQuadraturePointsPerCell, 0.0));
//
//          std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld =
//            *gradRhoInValues;
//
//          gradRhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
//
//          gradRhoInValuesSpinPolarized =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
//          cell = d_dofHandler->begin_active();
//          for (; cell != endc; ++cell)
//            {
//              if (cell->is_locally_owned())
//                {
//                  (*gradRhoInValues)[cell->id()] =
//                    std::vector<double>(3 * d_numberQuadraturePointsPerCell);
//                  (*gradRhoInValuesSpinPolarized)[cell->id()] =
//                    std::vector<double>(6 * d_numberQuadraturePointsPerCell);
//                  //
//
//                  for (int hist = 0; hist < N + 1; hist++)
//                    {
//                      gradRhoOutSpinPolarizedTemp[hist] =
//                        (*(d_gradRhoOutValsSpinPolarized[hist]))[cell->id()];
//                      gradRhoInSpinPolarizedTemp[hist] =
//                        (*(d_gradRhoInValsSpinPolarized[hist]))[cell->id()];
//                    }
//
//                  for (unsigned int q_point = 0; q_point < d_numberQuadraturePointsPerCell;
//                       ++q_point)
//                    {
//                      for( unsigned int iHist = 0 ;iHist < N+1;iHist++)
//                        {
//                          //                            gradRhoInHistNorm[iHist] += std::abs(gradRhoInTemp[iHist][3*q_point+0]);
//                          //                            gradRhoInHistNorm[iHist] += std::abs(gradRhoInTemp[iHist][3*q_point+1]);
//                          //                            gradRhoInHistNorm[iHist] += std::abs(gradRhoInTemp[iHist][3*q_point+2]);
//                          //                            gradRhoOutHistNorm[iHist] += std::abs(gradRhoOutTemp[iHist][3*q_point+0]);
//                          //                            gradRhoOutHistNorm[iHist] += std::abs(gradRhoOutTemp[iHist][3*q_point+1]);
//                          //                            gradRhoOutHistNorm[iHist] += std::abs(gradRhoOutTemp[iHist][3*q_point+2]);
//
//                          gradSpinRhoInHistNorm[iHist] += std::abs(gradRhoInSpinPolarizedTemp[iHist][6*q_point+0]);
//                          gradSpinRhoInHistNorm[iHist] += std::abs(gradRhoInSpinPolarizedTemp[iHist][6*q_point+1]);
//                          gradSpinRhoInHistNorm[iHist] += std::abs(gradRhoInSpinPolarizedTemp[iHist][6*q_point+2]);
//                          gradSpinRhoInHistNorm[iHist] += std::abs(gradRhoInSpinPolarizedTemp[iHist][6*q_point+3]);
//                          gradSpinRhoInHistNorm[iHist] += std::abs(gradRhoInSpinPolarizedTemp[iHist][6*q_point+4]);
//                          gradSpinRhoInHistNorm[iHist] += std::abs(gradRhoInSpinPolarizedTemp[iHist][6*q_point+5]);
//
//                          gradSpinRhoOutHistNorm[iHist] += std::abs(gradRhoOutSpinPolarizedTemp[iHist][6*q_point+0]);
//                          gradSpinRhoOutHistNorm[iHist] += std::abs(gradRhoOutSpinPolarizedTemp[iHist][6*q_point+1]);
//                          gradSpinRhoOutHistNorm[iHist] += std::abs(gradRhoOutSpinPolarizedTemp[iHist][6*q_point+2]);
//                          gradSpinRhoOutHistNorm[iHist] += std::abs(gradRhoOutSpinPolarizedTemp[iHist][6*q_point+3]);
//                          gradSpinRhoOutHistNorm[iHist] += std::abs(gradRhoOutSpinPolarizedTemp[iHist][6*q_point+4]);
//                          gradSpinRhoOutHistNorm[iHist] += std::abs(gradRhoOutSpinPolarizedTemp[iHist][6*q_point+5]);
//
//
//                        }
//                      //
//                      // Anderson mixing scheme spin up
//                      //
//                      double gradRhoXOutBar1 =
//                        d_cFinal *
//                        gradRhoOutSpinPolarizedTemp
//                          [N]
//                          [6 * q_point +
//                           0]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
//                               // + 0];
//                      double gradRhoYOutBar1 =
//                        d_cFinal *
//                        gradRhoOutSpinPolarizedTemp
//                          [N]
//                          [6 * q_point +
//                           1]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
//                               // + 1];
//                      double gradRhoZOutBar1 =
//                        d_cFinal *
//                        gradRhoOutSpinPolarizedTemp
//                          [N]
//                          [6 * q_point +
//                           2]; // cn*(gradRhoOutValsSpinPolarized[N])[cell->id()][6*q_point
//                               // + 2];
//
//                      double gradRhoXInBar1 =
//                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 0];
//                      double gradRhoYInBar1 =
//                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 1];
//                      double gradRhoZInBar1 =
//                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 2];
//
//                      for (int i = 0; i < N; i++)
//                        {
//                          gradRhoXOutBar1 +=
//                            d_c[i] *
//                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                       [6 * q_point + 0];
//                          gradRhoYOutBar1 +=
//                            d_c[i] *
//                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                       [6 * q_point + 1];
//                          gradRhoZOutBar1 +=
//                            d_c[i] *
//                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                       [6 * q_point + 2];
//
//                          gradRhoXInBar1 +=
//                            d_c[i] *
//                            gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                      [6 * q_point + 0];
//                          gradRhoYInBar1 +=
//                            d_c[i] *
//                            gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                      [6 * q_point + 1];
//                          gradRhoZInBar1 +=
//                            d_c[i] *
//                            gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                      [6 * q_point + 2];
//                        }
//                      //
//                      // Anderson mixing scheme spin down
//                      //
//                      double gradRhoXOutBar2 =
//                        d_cFinal * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 3];
//                      double gradRhoYOutBar2 =
//                        d_cFinal * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 4];
//                      double gradRhoZOutBar2 =
//                        d_cFinal * gradRhoOutSpinPolarizedTemp[N][6 * q_point + 5];
//
//                      double gradRhoXInBar2 =
//                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 3];
//                      double gradRhoYInBar2 =
//                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 4];
//                      double gradRhoZInBar2 =
//                        d_cFinal * gradRhoInSpinPolarizedTemp[N][6 * q_point + 5];
//
//                      for (int i = 0; i < N; i++)
//                        {
//                          gradRhoXOutBar2 +=
//                            d_c[i] *
//                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                       [6 * q_point + 3];
//                          gradRhoYOutBar2 +=
//                            d_c[i] *
//                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                       [6 * q_point + 4];
//                          gradRhoZOutBar2 +=
//                            d_c[i] *
//                            gradRhoOutSpinPolarizedTemp[N - 1 - i]
//                                                       [6 * q_point + 5];
//
//                          gradRhoXInBar2 +=
//                            d_c[i] *
//                            gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                      [6 * q_point + 3];
//                          gradRhoYInBar2 +=
//                            d_c[i] *
//                            gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                      [6 * q_point + 4];
//                          gradRhoZInBar2 +=
//                            d_c[i] *
//                            gradRhoInSpinPolarizedTemp[N - 1 - i]
//                                                      [6 * q_point + 5];
//                        }
//                      //
//                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                  0] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar1 +
//                         d_dftParamsPtr->mixingParameter * gradRhoXOutBar1);
//                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                  1] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar1 +
//                         d_dftParamsPtr->mixingParameter * gradRhoYOutBar1);
//                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                  2] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar1 +
//                         d_dftParamsPtr->mixingParameter * gradRhoZOutBar1);
//                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                  3] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoXInBar2 +
//                         d_dftParamsPtr->mixingParameter * gradRhoXOutBar2);
//                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                  4] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoYInBar2 +
//                         d_dftParamsPtr->mixingParameter * gradRhoYOutBar2);
//                      (*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point +
//                                                                  5] =
//                        ((1 - d_dftParamsPtr->mixingParameter) * gradRhoZInBar2 +
//                         d_dftParamsPtr->mixingParameter * gradRhoZOutBar2);
//
//                      ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
//                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                        [6 * q_point + 0]) +
//                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                        [6 * q_point + 3]);
//                      ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
//                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                        [6 * q_point + 1]) +
//                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                        [6 * q_point + 4]);
//                      ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
//                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                        [6 * q_point + 2]) +
//                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
//                                                        [6 * q_point + 5]);
//
//                      gradRhoOutputNorm += std::abs((*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 0]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 1]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 2]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 3]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 4]);
//                      gradRhoOutputNorm += std::abs((*gradRhoInValuesSpinPolarized)[cell->id()][6 * q_point + 5]);
//                    }
//                }
//            }
//        }
//    }
//
//  MPI_Allreduce(MPI_IN_PLACE,
//                &gradRhoOutputNorm,
//                1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//
//  MPI_Allreduce(MPI_IN_PLACE,
//                &gradRhoInHistNorm[0],
//                N+1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//  MPI_Allreduce(MPI_IN_PLACE,
//                &gradRhoOutHistNorm[0],
//                N+1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//
//  MPI_Allreduce(MPI_IN_PLACE,
//                &gradSpinRhoInHistNorm[0],
//                N+1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//  MPI_Allreduce(MPI_IN_PLACE,
//                &gradSpinRhoOutHistNorm[0],
//                N+1,
//                MPI_DOUBLE,
//                MPI_SUM,
//                d_mpi_comm_domain);
//
//  //    std::cout<<" norm of output  grad rho = "<<gradRhoOutputNorm<<"\n";
//  //
//  //    for(unsigned int iHist = 0; iHist < N+1; iHist++)
//  //      {
//  //        std::cout<<" norm of grad rhoIn["<<iHist<<"] = "<<gradRhoInHistNorm[iHist]<<" norm of grad rhoOut["<<iHist<<"] = "<<gradRhoOutHistNorm[iHist]<<"\n";
//  //        std::cout<<" norm of spin grad rhoIn["<<iHist<<"] = "<<gradSpinRhoInHistNorm[iHist]<<" norm of grad rhoOut["<<iHist<<"] = "<<gradSpinRhoOutHistNorm[iHist]<<"\n";
//  //      }
//
//
//  //copyDensityToInHist(rhoInValues);
//  return std::sqrt(dealii::Utilities::MPI::sum(normValue, d_mpi_comm_domain));
//}

//void MixingScheme::init(const dealii::MatrixFree<3, double> & matrixFreeData,
//                   const unsigned int matrixFreeVectorComponent,
//                   const unsigned int matrixFreeQuadratureComponent,
//                   excManager * excManagerPtr)
//{
//
//  d_matrixFreeDataPtr = &matrixFreeData ;
//  d_matrixFreeVectorComponent = matrixFreeVectorComponent;
//  d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;
//  d_excManagerPtr = excManagerPtr;
//  const dealii::Quadrature<3> &quadratureRhs =
//    d_matrixFreeDataPtr->get_quadrature(
//      d_matrixFreeQuadratureComponent);
//
//  d_numberQuadraturePointsPerCell = quadratureRhs.size();
//
//  d_dofHandler =
//    &d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);
//
//  dealii::FEValues<3> fe_values(d_dofHandler->get_fe(),
//                                quadratureRhs,
//                                dealii::update_JxW_values);
//
//  unsigned int iElem = 0;
//
//  d_numCells = d_matrixFreeDataPtr->n_physical_cells();
//
//  if(d_dftParamsPtr->spinPolarized == 0)
//    {
//      d_vecJxW.resize(d_numberQuadraturePointsPerCell*
//                      d_numCells);
//      typename dealii::DoFHandler<3>::active_cell_iterator
//        cell = d_dofHandler->begin_active(),
//        endc = d_dofHandler->end();
//      for (; cell != endc; ++cell)
//        {
//          if (cell->is_locally_owned())
//            {
//              fe_values.reinit(cell);
//              for(unsigned int iQuad = 0 ;iQuad < d_numberQuadraturePointsPerCell; iQuad++)
//                {
//                  d_vecJxW[iElem*d_numberQuadraturePointsPerCell +
//                           iQuad] =
//                    fe_values.JxW(iQuad);
//                }
//              iElem++;
//            }
//        }
//    }
//  else
//    {
//      d_vecJxW.resize(2*
//                        d_numberQuadraturePointsPerCell*
//                        d_numCells,0.0);
//      typename dealii::DoFHandler<3>::active_cell_iterator
//        cell = d_dofHandler->begin_active(),
//        endc = d_dofHandler->end();
//      for (; cell != endc; ++cell)
//        {
//          if (cell->is_locally_owned())
//            {
//              fe_values.reinit(cell);
//              for(unsigned int iQuad = 0 ;iQuad < d_numberQuadraturePointsPerCell; iQuad++)
//                {
//                  d_vecJxW[2*iElem*d_numberQuadraturePointsPerCell +
//                           2*iQuad
//                           +0] =
//                    fe_values.JxW(iQuad);
//
//                  d_vecJxW[2*iElem*d_numberQuadraturePointsPerCell +
//                           2*iQuad
//                           +1] =
//                    fe_values.JxW(iQuad);
//                }
//              iElem++;
//            }
//        }
//    }
//  //        std::cout<<" size of JxW = "<<d_vecJxW.size()<<"\n";
//}

//void MixingScheme::copyGradDensityFromInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradInput)
//{
//  gradInput = d_gradRhoInVals.back();
//}
//
//void MixingScheme::copyGradDensityFromOutHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradOutput)
//{
//  gradOutput = d_gradRhoOutVals.back();
//}
//
//void MixingScheme::copySpinGradDensityFromInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradInputSpin)
//{
//  gradInputSpin = d_gradRhoInValsSpinPolarized.back();
//}
//
//void MixingScheme::copySpinGradDensityFromOutHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradOutputSpin)
//{
//  gradOutputSpin = d_gradRhoOutValsSpinPolarized.back();
//}
//
//void MixingScheme::copyGradDensityToInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradInput)
//{
//  d_gradRhoInVals.push_back(gradInput);
//}
//void MixingScheme::copyGradDensityToOutHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradOutput)
//{
//  d_gradRhoOutVals.push_back(gradOutput);
//}
//
//void MixingScheme::copySpinGradDensityToInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradInputSpin)
//{
//  d_gradRhoInValsSpinPolarized.push_back(gradInputSpin);
//}
//void MixingScheme::copySpinGradDensityToOutHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradOutputSpin)
//{
//  d_gradRhoOutValsSpinPolarized.push_back(gradOutputSpin);
//}

//void MixingScheme::copyDensityFromInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValues)
//{
//  unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
//  if (d_dftParamsPtr->spinPolarized == 1)
//    {
//      numQuadPointsPerCell = 2*numQuadPointsPerCell;
//    }
//
//  unsigned int hist =d_rhoInVals.size();
//
//  unsigned int iElem = 0;
//
//  typename dealii::DoFHandler<3>::active_cell_iterator
//    cell = d_dofHandler->begin_active(),
//    endc = d_dofHandler->end();
//  for (; cell != endc; ++cell)
//    {
//      if (cell->is_locally_owned())
//        {
//          for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
//            {
//              (*rhoInValues)[cell->id()][iQuad] =
//                d_rhoInVals[hist-1][iElem*numQuadPointsPerCell +
//                                      iQuad];
//            }
//          iElem++;
//        }
//    }
//
//}
//void MixingScheme::copyDensityFromOutHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoOutValues)
//{
//  unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
//  if (d_dftParamsPtr->spinPolarized == 1)
//    {
//      numQuadPointsPerCell = 2*numQuadPointsPerCell;
//    }
//
//  unsigned int hist =d_rhoOutVals.size();
//
//  unsigned int iElem = 0;
//
//  d_dofHandler =
//    &d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);
//
//  typename dealii::DoFHandler<3>::active_cell_iterator
//    cell = d_dofHandler->begin_active(),
//    endc = d_dofHandler->end();
//  for (; cell != endc; ++cell)
//    {
//      if (cell->is_locally_owned())
//        {
//          for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
//            {
//              (*rhoOutValues)[cell->id()][iQuad] =
//                d_rhoOutVals[hist-1][iElem*numQuadPointsPerCell +
//                                       iQuad];
//            }
//          iElem++;
//        }
//    }
//
//}
//
//void MixingScheme::copyDensityToInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValues)
//{
//  std::vector<double> latestHistRhoIn;
//  unsigned int numQuadPointsPerCell = d_numberQuadraturePointsPerCell;
//  if (d_dftParamsPtr->spinPolarized == 1)
//    {
//      numQuadPointsPerCell = 2*numQuadPointsPerCell;
//    }
//
//  latestHistRhoIn.resize(d_numCells*numQuadPointsPerCell);
//  std::fill(latestHistRhoIn.begin(),latestHistRhoIn.end(),0.0);
//
//  unsigned int iElem = 0;
//
//  typename dealii::DoFHandler<3>::active_cell_iterator
//    cell = d_dofHandler->begin_active(),
//    endc = d_dofHandler->end();
//  for (; cell != endc; ++cell)
//    {
//      if (cell->is_locally_owned())
//        {
//          for(unsigned int iQuad = 0 ;iQuad < numQuadPointsPerCell; iQuad++)
//            {
//              latestHistRhoIn[iElem*numQuadPointsPerCell +
//                              iQuad] =
//                (*rhoInValues)[cell->id()][iQuad];
//            }
//          iElem++;
//        }
//    }
//
//  d_rhoInVals.push_back(latestHistRhoIn);
//}
