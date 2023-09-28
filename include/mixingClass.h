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

namespace dftfe
{
  /**
   * @brief Enum class that stores he list of variables that will can be
   * used in the mixing scheme
   *
   */
  enum class mixingVariable
  {
    rho,
    gradRho
  };

  /**
   * @brief This class performs the anderson mixing in a variable agnostic way
   * This class takes can take different input variables as input in a
   * std::vector format and computes the mixing coefficients These coefficients
   * can then be used to compute the new variable at the start of the SCF.
   * @author Vishal Subramanian
   */
  class MixingScheme
  {
  public:
    MixingScheme(const MPI_Comm &mpi_comm_domain);

    unsigned int
    lengthOfHistory();

    /**
     * @brief Computes the mixing coefficients.
     *
     */
    void
    computeAndersonMixingCoeff();

    /**
     * @brief Deletes the old history if the length exceeds max length of history
     *
     */
    void
    popOldHistory(unsigned int mixingHistory);

    /**
     * @brief Clears all the the history.
     *
     */
    void
    clearHistory();

    /**
     * @brief This function is used to add the mixing variables and its corresponding
     * JxW values
     *
     */
    void
    addMixingVariable(const mixingVariable       mixingVariableList,
                      const std::vector<double> &weightDotProducts,
                      const bool                 performMPIReduce,
                      const double               mixingValue);

    /**
     * @brief Adds to the input history
     *
     */
    void
    addVariableToInHist(const mixingVariable       mixingVariableName,
                        const std::vector<double> &inputVariableToInHist);

    /**
     * @brief Adds to the output history
     *
     */
    void
    addVariableToOutHist(const mixingVariable       mixingVariableName,
                         const std::vector<double> &inputVariableToOutHist);

    /**
     * @brief Computes the input for the next iteration based on the anderson coefficients
     *
     */
    double
    mixVariable(const mixingVariable mixingVariableName,
                std::vector<double> &outputVariable);


  private:
    /**
     * @brief Computes the matrix A and c vector that will be needed for anderson mixing.
     * This function computes the A and c values for each variable which will be
     * then added up in computeAndersonMixingCoeff()
     */
    void
    computeMixingMatrices(const std::deque<std::vector<double>> &inHist,
                          const std::deque<std::vector<double>> &outHist,
                          const std::vector<double> &weightDotProducts,
                          const bool                 isMPIAllReduce,
                          std::vector<double> &      A,
                          std::vector<double> &      c);

    std::vector<double> d_A, d_c;
    double              d_cFinal;

    std::map<mixingVariable, std::deque<std::vector<double>>>
                                                  d_variableHistoryIn, d_variableHistoryOut;
    std::map<mixingVariable, std::vector<double>> d_vectorDotProductWeights;
    std::map<mixingVariable, bool>                d_performMPIReduce;

    const MPI_Comm d_mpi_comm_domain;

    std::map<mixingVariable, double> d_mixingParameter;
    unsigned int                     d_mixingHistory;
  };
} //  end of namespace dftfe
#endif // DFTFE_MIXINGCLASS_H
