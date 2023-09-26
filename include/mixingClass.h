// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_MIXINGCLASS_H
#define DFTFE_MIXINGCLASS_H

#include <deque>
#include <headers.h>
#include <dftParameters.h>
#include <excManager.h>

namespace dftfe
{

  enum class mixingVariable
  {
    rho,
    gradRho
  };
  class MixingScheme
  {
   public:

    MixingScheme(const MPI_Comm &mpi_comm_domain);

    unsigned int lengthOfHistory();

    void computeAndersonMixingCoeff();

    void popOldHistory(unsigned int mixingHistory);

    void clearHistory();

    void addMixingVariable(const mixingVariable mixingVariableList,
                      std::vector<double> &weightDotProducts,
                      const bool performMPIReduce,
                      const double mixingValue);

    void addVariableToInHist(const mixingVariable mixingVariableName,
                      const std::vector<double> &inputVariableToInHist);

    void addVariableToOutHist(const mixingVariable mixingVariableName,
                        const std::vector<double> &inputVariableToOutHist);

    double mixVariable(const mixingVariable mixingVariableName,
                std::vector<double> &outputVariable);


  private:

    void computeMixingMatrices(const std::deque<std::vector<double>> &inHist,
                          const std::deque<std::vector<double>> &outHist,
                          const std::vector<double> &weightDotProducts,
                          const bool isMPIAllReduce,
                          std::vector<double> &A,
                          std::vector<double> &c);

    std::vector<double> d_A, d_c;
    double d_cFinal;

    std::map<mixingVariable, std::deque<std::vector<double>>> d_variableHistoryIn, d_variableHistoryOut ;
    std::map<mixingVariable, std::vector<double>> d_vectorDotProductWeights;
    std::map<mixingVariable, bool> d_performMPIReduce;

    const MPI_Comm d_mpi_comm_domain;

    std::map<mixingVariable, double> d_mixingParameter;
    unsigned int d_mixingHistory;

  };
} //  end of namespace dftfe


#endif // DFTFE_MIXINGCLASS_H

//
//double mixDensity(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValues,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoOutValues,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValuesSpinPolarized,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoOutValuesSpinPolarized,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoInValues,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoOutValues,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoInValuesSpinPolarized,
//           std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &gradRhoOutValuesSpinPolarized);

//    void copyDensityToInHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoInValues);
//
//    void copyDensityToOutHist(std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> &rhoOutValues);