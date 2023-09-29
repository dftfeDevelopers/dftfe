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
  MixingScheme::MixingScheme(const MPI_Comm &mpi_comm_domain)
    : d_mpi_comm_domain(mpi_comm_domain)
  {}

  void
  MixingScheme::addMixingVariable(const mixingVariable       mixingVariableList,
                                  const std::vector<double> &weightDotProducts,
                                  const bool                 performMPIReduce,
                                  const double               mixingValue)
  {
    d_variableHistoryIn[mixingVariableList] = std::deque<std::vector<double>>();
    d_variableHistoryOut[mixingVariableList] =
      std::deque<std::vector<double>>();
    d_vectorDotProductWeights[mixingVariableList] = weightDotProducts;

    d_performMPIReduce[mixingVariableList] = performMPIReduce;
    d_mixingParameter[mixingVariableList]  = mixingValue;
  }

  void
  MixingScheme::computeMixingMatrices(
    const std::deque<std::vector<double>> &inHist,
    const std::deque<std::vector<double>> &outHist,
    const std::vector<double> &            weightDotProducts,
    const bool                             isMPIAllReduce,
    std::vector<double> &                  A,
    std::vector<double> &                  c)
  {
    std::vector<double> Adensity;
    Adensity.resize(A.size());
    std::fill(Adensity.begin(), Adensity.end(), 0.0);

    std::vector<double> cDensity;
    cDensity.resize(c.size());
    std::fill(cDensity.begin(), cDensity.end(), 0.0);

    int          N             = inHist.size() - 1;
    unsigned int numQuadPoints = 0;
    if (N > 0)
      numQuadPoints = inHist[0].size();

    if (weightDotProducts.size() > 0)
      {
        AssertThrow(numQuadPoints == weightDotProducts.size(),
                    dealii::ExcMessage(
                      "DFT-FE Error: The size of the weight dot products vec "
                      "does not match the size of the vectors in history."
                      "Please resize the vectors appropriately."));
        for (unsigned int iQuad = 0; iQuad < numQuadPoints; iQuad++)
          {
            double Fn = outHist[N][iQuad] - inHist[N][iQuad];
            for (int m = 0; m < N; m++)
              {
                double Fnm =
                  outHist[N - 1 - m][iQuad] - inHist[N - 1 - m][iQuad];
                for (int k = 0; k < N; k++)
                  {
                    double Fnk =
                      outHist[N - 1 - k][iQuad] - inHist[N - 1 - k][iQuad];
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
        std::fill(ATotal.begin(), ATotal.end(), 0.0);
        std::fill(cTotal.begin(), cTotal.end(), 0.0);
        if (isMPIAllReduce)
          {
            MPI_Allreduce(&Adensity[0],
                          &ATotal[0],
                          aSize,
                          MPI_DOUBLE,
                          MPI_SUM,
                          d_mpi_comm_domain);
            MPI_Allreduce(&cDensity[0],
                          &cTotal[0],
                          cSize,
                          MPI_DOUBLE,
                          MPI_SUM,
                          d_mpi_comm_domain);
          }
        else
          {
            ATotal = Adensity;
            cTotal = cDensity;
          }
        for (unsigned int i = 0; i < aSize; i++)
          {
            A[i] += ATotal[i];
          }

        for (unsigned int i = 0; i < cSize; i++)
          {
            c[i] += cTotal[i];
          }
      }
  }

  unsigned int
  MixingScheme::lengthOfHistory()
  {
    return d_variableHistoryIn[mixingVariable::rho].size();
  }

  // Fucntion to compute the mixing coefficients based on anderson scheme
  void
  MixingScheme::computeAndersonMixingCoeff()
  {
    // initialize data structures
    // assumes rho is a mixing variable
    int              N    = d_variableHistoryIn[mixingVariable::rho].size() - 1;
    int              NRHS = 1, lda = N, ldb = N, info;
    std::vector<int> ipiv(N);
    d_A.resize(lda * N);
    d_c.resize(ldb * NRHS);
    for (int i = 0; i < lda * N; i++)
      d_A[i] = 0.0;
    for (int i = 0; i < ldb * NRHS; i++)
      d_c[i] = 0.0;

    for (const auto &[key, value] : d_variableHistoryIn)
      {
        computeMixingMatrices(d_variableHistoryIn[key],
                              d_variableHistoryOut[key],
                              d_vectorDotProductWeights[key],
                              d_performMPIReduce[key],
                              d_A,
                              d_c);
      }

    dgesv_(&N, &NRHS, &d_A[0], &lda, &ipiv[0], &d_c[0], &ldb, &info);

    d_cFinal = 1.0;
    for (int i = 0; i < N; i++)
      d_cFinal -= d_c[i];
  }

  // Fucntions to add to the history
  void
  MixingScheme::addVariableToInHist(
    const mixingVariable       mixingVariableName,
    const std::vector<double> &inputVariableToInHist)
  {
    d_variableHistoryIn[mixingVariableName].push_back(inputVariableToInHist);
  }

  void
  MixingScheme::addVariableToOutHist(
    const mixingVariable       mixingVariableName,
    const std::vector<double> &inputVariableToOutHist)
  {
    d_variableHistoryOut[mixingVariableName].push_back(inputVariableToOutHist);
  }

  // Computes the new variable after mixing. It returns the L2 norm of
  // the difference between the old and new input variables
  double
  MixingScheme::mixVariable(mixingVariable       mixingVariableName,
                            std::vector<double> &outputVariable)
  {
    double       normValue = 0.0;
    unsigned int N         = d_variableHistoryIn[mixingVariableName].size() - 1;
    unsigned int lenVar = d_vectorDotProductWeights[mixingVariableName].size();

    if (lenVar > 0)
      {
        for (unsigned int iQuad = 0; iQuad < lenVar; iQuad++)
          {
            double rhodiff =
              std::abs(d_variableHistoryIn[mixingVariableName][N][iQuad] -
                       d_variableHistoryOut[mixingVariableName][N][iQuad]);

            normValue += rhodiff * rhodiff *
                         d_vectorDotProductWeights[mixingVariableName][iQuad];
          }
      }
    else
      {
        // Assumes the variable is present otherwise will lead to a seg fault
        lenVar = d_variableHistoryIn[mixingVariableName][0].size();
      }

    outputVariable.resize(lenVar);
    std::fill(outputVariable.begin(), outputVariable.end(), 0.0);

    for (unsigned int iQuad = 0; iQuad < lenVar; iQuad++)
      {
        double varOutBar =
          d_cFinal * d_variableHistoryOut[mixingVariableName][N][iQuad];
        double varInBar =
          d_cFinal * d_variableHistoryIn[mixingVariableName][N][iQuad];

        for (int i = 0; i < N; i++)
          {
            varOutBar +=
              d_c[i] *
              d_variableHistoryOut[mixingVariableName][N - 1 - i][iQuad];
            varInBar +=
              d_c[i] *
              d_variableHistoryIn[mixingVariableName][N - 1 - i][iQuad];
          }
        outputVariable[iQuad] =
          ((1 - d_mixingParameter[mixingVariableName]) * varInBar +
           d_mixingParameter[mixingVariableName] * varOutBar);
      }

    double totalNormValue = 0.0;
    if (d_performMPIReduce[mixingVariableName])
      {
        MPI_Allreduce(&normValue,
                      &totalNormValue,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpi_comm_domain);
      }
    else
      {
        totalNormValue = normValue;
      }

    return std::sqrt(totalNormValue);
  }

  // Clears the history
  // But it does not clear the list of variables
  // and its corresponding JxW values
  void
  MixingScheme::clearHistory()
  {
    for (const auto &[key, value] : d_variableHistoryIn)
      {
        d_variableHistoryIn[key].clear();
        d_variableHistoryOut[key].clear();
      }
  }


  // Deletes old history.
  // This is not recursively
  // If the length is greater or equal to mixingHistory then the
  // oldest history is deleted
  void
  MixingScheme::popOldHistory(unsigned int mixingHistory)
  {
    if (d_variableHistoryIn[mixingVariable::rho].size() >= mixingHistory)
      {
        for (const auto &[key, value] : d_variableHistoryIn)
          {
            d_variableHistoryIn[key].pop_front();
            d_variableHistoryOut[key].pop_front();
          }
      }
  }

} // namespace dftfe
