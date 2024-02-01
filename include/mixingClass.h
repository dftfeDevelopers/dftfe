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
    gradRho,
    magZ,
    gradMagZ
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
    MixingScheme(const MPI_Comm &   mpi_comm_parent,
                 const MPI_Comm &   mpi_comm_domain,
                 const unsigned int verbosity);

    unsigned int
    lengthOfHistory();

    /**
     * @brief Computes the mixing coefficients.
     *
     */
    void
    computeAndersonMixingCoeff(
      const std::vector<mixingVariable> mixingVariablesList);

    /**
     * @brief Computes the adaptive mixing parameter.
     *
     */
    void
    computeAdaptiveAndersonMixingParameter();

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
     * For dependent variables which are not used in mixing, the
     * weightDotProducts is set to a vector of size zero. Later the dependent
     * variables can be mixed with the same mixing coefficients.
     *
     */
    void
    addMixingVariable(
      const mixingVariable mixingVariableList,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &          weightDotProducts,
      const bool   performMPIReduce,
      const double mixingValue,
      const bool   adaptMixingValue);

    /**
     * @brief Adds to the input history
     *
     */
    void
    addVariableToInHist(const mixingVariable mixingVariableName,
                        const double *       inputVariableToInHist,
                        const unsigned int   length);

    /**
     * @brief Adds to the residual history
     *
     */
    void
    addVariableToResidualHist(const mixingVariable mixingVariableName,
                              const double *       inputVariableToResidualHist,
                              const unsigned int   length);

    /**
     * @brief Computes the input for the next iteration based on the anderson coefficients
     *
     */
    void
    mixVariable(const mixingVariable mixingVariableName,
                double *             outputVariable,
                const unsigned int   lenVar);


  private:
    /**
     * @brief Computes the matrix A and c vector that will be needed for anderson mixing.
     * This function computes the A and c values for each variable which will be
     * then added up in computeAndersonMixingCoeff()
     */
    void
    computeMixingMatrices(
      const std::deque<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &inHist,
      const std::deque<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &outHist,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                  weightDotProducts,
      const bool           isPerformMixing,
      const bool           isMPIAllReduce,
      std::vector<double> &A,
      std::vector<double> &c);

    std::vector<double> d_A, d_c;
    double              d_cFinal;

    std::map<
      mixingVariable,
      std::deque<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>>
      d_variableHistoryIn, d_variableHistoryResidual;
    std::map<
      mixingVariable,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                                   d_vectorDotProductWeights;
    std::map<mixingVariable, bool> d_performMPIReduce;

    const MPI_Comm d_mpi_comm_domain, d_mpi_comm_parent;

    std::map<mixingVariable, double> d_mixingParameter;
    std::map<mixingVariable, bool>   d_adaptMixingParameter;
    bool                             d_anyMixingParameterAdaptive;
    bool                             d_adaptiveMixingParameterDecLastIteration;
    bool                             d_adaptiveMixingParameterDecAllIterations;
    bool                             d_adaptiveMixingParameterIncAllIterations;
    unsigned int                     d_mixingHistory;
    std::map<mixingVariable, bool>   d_performMixing;
    const int               d_verbosity;


    /// conditional stream object
    dealii::ConditionalOStream pcout;
  };
} //  end of namespace dftfe
#endif // DFTFE_MIXINGCLASS_H
