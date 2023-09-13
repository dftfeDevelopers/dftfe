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
  class MixingScheme
  {
   public:

    MixingScheme(const dftParameters &dftParam,
                 const MPI_Comm &mpi_comm_domain);

    void init(const dealii::MatrixFree<3, double> & matrixFreeData,
         const unsigned int matrixFreeVectorComponent,
                 const unsigned int matrixFreeQuadratureComponent,
		 excManager * excManagerPtr);

    void dotProduct(const std::deque<std::vector<double>> &inHist,
               const std::deque<std::vector<double>> &outHist,
               std::vector<double> &A,
               std::vector<double> &c,
               std::string fieldName);

    void copyDensityToInHist(std::map<dealii::CellId, std::vector<double>> *rhoInValues);

    void copyDensityToOutHist(std::map<dealii::CellId, std::vector<double>> *rhoOutValues);

    unsigned int lengthOfHistory();

    void computeAndersonMixingCoeff();


    void popRhoInHist();

    void popRhoOutHist();

    void copyDensityFromInHist(std::map<dealii::CellId, std::vector<double>> *rhoInValues);
    void copyDensityFromOutHist(std::map<dealii::CellId, std::vector<double>> *rhoOutValues);

    void copyGradDensityFromInHist(std::map<dealii::CellId, std::vector<double>>* gradInput);
    void copyGradDensityFromOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutput);

    void copySpinGradDensityFromInHist(std::map<dealii::CellId, std::vector<double>> * gradInputSpin);
    void copySpinGradDensityFromOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutputSpin);

    void copyGradDensityToInHist(std::map<dealii::CellId, std::vector<double>>  *gradInput);
    void copyGradDensityToOutHist(std::map<dealii::CellId, std::vector<double>> * gradOutput);

    void copySpinGradDensityToInHist(std::map<dealii::CellId, std::vector<double>> * gradInputSpin);
    void copySpinGradDensityToOutHist(std::map<dealii::CellId, std::vector<double>>* gradOutputSpin);

    void computeMixingMatricesDensity(const std::deque<std::vector<double>> &inHist,
                                 const std::deque<std::vector<double>> &outHist,
                                 std::vector<double> &A,
                                 std::vector<double> &c);

    double mixDensity(std::map<dealii::CellId, std::vector<double>> *rhoInValues,
                             std::map<dealii::CellId, std::vector<double>> *rhoOutValues,
                             std::map<dealii::CellId, std::vector<double>> *rhoInValuesSpinPolarized,
                             std::map<dealii::CellId, std::vector<double>> *rhoOutValuesSpinPolarized,
                                 std::map<dealii::CellId, std::vector<double>> *gradRhoInValues,
                                 std::map<dealii::CellId, std::vector<double>> *gradRhoOutValues,
                             std::map<dealii::CellId, std::vector<double>> *gradRhoInValuesSpinPolarized,
                             std::map<dealii::CellId, std::vector<double>> *gradRhoOutValuesSpinPolarized);

    void popOldHistory();

    void clearHistory();

    void popGradRhoInHist();
    void popGradRhoOutHist();
    void popGradRhoSpinInHist();
    void popGradRhoSpinOutHist();

  private:

    std::vector<double> d_A, d_c;
    double d_cFinal;
    std::deque<std::vector<double>> d_rhoInVals,
      d_rhoOutVals, d_rhoInValsSpinPolarized, d_rhoOutValsSpinPolarized;

    std::deque<std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>> d_gradRhoInVals,
      d_gradRhoInValsSpinPolarized, d_gradRhoOutVals, d_gradRhoOutValsSpinPolarized;

    const dealii::MatrixFree<3,double>  *d_matrixFreeDataPtr;
    unsigned int d_matrixFreeVectorComponent,d_matrixFreeQuadratureComponent;

    unsigned int d_numberQuadraturePointsPerCell;
    unsigned int d_numCells;
    const dealii::DoFHandler<3> * d_dofHandler;

    std::vector<double> d_vecJxW;

    const excManager *         d_excManagerPtr;

    const MPI_Comm d_mpi_comm_domain;

    const dftParameters *d_dftParamsPtr;

  };
} //  end of namespace dftfe


#endif // DFTFE_MIXINGCLASS_H
