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
     * @brief The apply function that will be called in HX()
     * param[in] src The input vector
     * param[out] dst The output vector
     * param[in] inputVecSize The size of the input vector
     * param[in] factor the factor with which the output is scaled in HX()
     * param[in] kPointIndex the k point for which the HX() is called
     * param[in] spinIndex the spin index for which the HX() is called
     */
    virtual void
    applyWaveFunctionDependentFuncDer(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        &                                                                src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const unsigned int inputVecSize,
      const double       factor,
      const unsigned int kPointIndex,
      const unsigned int spinIndex) = 0;
    virtual void
    updateWaveFunctionDependentFuncDer(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) = 0;
    virtual double
    computeWaveFunctionDependentExcEnergy(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) = 0;

    /**
     * x and c denotes exchange and correlation respectively.
     * This function computes the rho and tau dependent parts
     * of the Exc and Vxc functionals
     */
    virtual void
    computeOutputXCData(
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

  protected:
    const std::vector<DensityDescriptorDataAttributes>
      d_densityDescriptorAttributesList;

    ExcFamilyType     d_ExcFamilyType;
    densityFamilyType d_densityFamilyType;
  };
} // namespace dftfe

#include "ExcSSDFunctionalBaseClass.t.cc"
#endif // DFTFE_EXCSSDFUNCTIONALBASECLASS_H
