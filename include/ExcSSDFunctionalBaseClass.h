// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXCSSDFUNCTIONALBASECLASS_H
#define DFTFE_EXCSSDFUNCTIONALBASECLASS_H

#include "AuxDensityMatrix.h"
#include <vector>
#include <fstream>
#include <iostream>
namespace dftfe
{
  enum class ExcFamilyType
  {
    LDA,
    GGA,
    LLMGGA,
    HYBRID,
    DFTPlusU,
    MGGA
  };

  enum class densityFamilyType
  {
    LDA,
    GGA,
    LLMGGA,
  };


  /*
   * XC attributes for the derivatives for the remainder functional
   *
   */

  enum class xcRemainderOutputDataAttributes
  {
    e,       // energy density per unit volume for the remainder functional
    vSpinUp, // the local multiplicative potential for spin up arising from
    // remainder functional
    vSpinDown, // the local multiplicative potential for spin down arising from
               // remainder functional
    pdeDensitySpinUp,
    pdeDensitySpinDown,
    pdeSigma,
    pdeLaplacianSpinUp,
    pdeLaplacianSpinDown,
    pdeTauSpinUp,
    pdeTauSpinDown
  };



  /**
   * @brief This class provides the structure for all
   * Exc functionals that can be written as a combination of
   * functional of Single Slater determinant that results in a
   * non-multiplicative potential plus a remainder functional
   * dependent on density and Tau.
   *
   * Exc = S{\phi} + R [\rho, \tau]
   * @author Vishal Subramanian, Sambit Das
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class ExcSSDFunctionalBaseClass
  {
  public:
    ExcSSDFunctionalBaseClass(const ExcFamilyType     excFamType,
                              const densityFamilyType densityFamType,
                              const std::vector<DensityDescriptorDataAttributes>
                                &densityDescriptorAttributesList);

    virtual ~ExcSSDFunctionalBaseClass();

    const std::vector<DensityDescriptorDataAttributes> &
    getDensityDescriptorAttributesList() const;

    densityFamilyType
    getDensityBasedFamilyType() const;


    /*
     * @brief The apply function that will be called in HX().
     * The distribute() and updateGhostValues() for src
     * has to be called before this function.
     * Similarly for dst, accumulateLocallyOwned() should be called in HX()
     * after this function is called. param[in] src The input vector param[out]
     * dst The output vector param[in] inputVecSize The size of the input vector
     * param[in] kPointIndex the k point for which the HX() is called
     * param[in] spinIndex the spin index for which the HX() is called
     */
    virtual void
    applyWaveFunctionDependentFuncDerWrtPsi(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        &                                                                src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const unsigned int inputVecSize,
      const unsigned int kPointIndex,
      const unsigned int spinIndex) = 0;

    /*
     * @brief The function that updates the Wave function dependent part
     * of the Exc functional and its derivative wrt \psi
     */
    virtual void
    updateWaveFunctionDependentFuncDerWrtPsi(
      const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
      const std::vector<double> &                           kPointWeights) = 0;


    /*
     * @brief The function that computes the Wave function dependent part
     * of the Exc functional energy
     */
    virtual void
    computeWaveFunctionDependentExcEnergy(
      const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
      const std::vector<double> &                           kPointWeights) = 0;

    /*
     * @brief Returns the Wavefunction dependent part of the Exc energy.
     */
    virtual double
    getWaveFunctionDependentExcEnergy() = 0;

    /*
     * @brief Returns the Expectation value of the WaveFunctionDependentExcFuncDerWrtPsi
     * While using band energy approach to compute the total free energy
     * the expectation of the WaveFunctionDependentExcFuncDerWrtPsi is included
     * in the band energy. Hence it has to be subtracted and the correct energy
     * has to be added to the free energy.
     */
    virtual double
    getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi() = 0;

    /**
     * x and c denotes exchange and correlation respectively.
     * This function computes the rho and tau dependent part of
     * the Exc functional energy density and its partial derivatives
     */
    virtual void
    computeRhoTauDependentXCData(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &xDataOut,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &cDataout) const = 0;

    ExcFamilyType
    getExcFamilyType() const;

    virtual void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const = 0;

    virtual void
    reinitKPointDependentVariables(unsigned int kPointIndex) = 0;

  protected:
    const std::vector<DensityDescriptorDataAttributes>
      d_densityDescriptorAttributesList;

    ExcFamilyType     d_ExcFamilyType;
    densityFamilyType d_densityFamilyType;
  };
} // namespace dftfe

#include "ExcSSDFunctionalBaseClass.t.cc"
#endif // DFTFE_EXCSSDFUNCTIONALBASECLASS_H
