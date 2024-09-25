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

#ifndef DFTFE_EXCMANAGER_H
#define DFTFE_EXCMANAGER_H

#include <xc.h>
#include <ExcSSDFunctionalBaseClass.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class excManager
  {
  public:
    /**
     * @brief Constructor
     *
     */
    excManager();

    /**
     * @brief  destructor
     */
    ~excManager();

    void
    clear();


    void
    init(std::string XCType,
         bool        isSpinPolarized,
         std::string modelXCInputFile);

    ExcSSDFunctionalBaseClass<memorySpace> *
    getExcSSDFunctionalObj();


    const ExcSSDFunctionalBaseClass<memorySpace> *
    getExcSSDFunctionalObj() const;

    const std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> &
    getSSDSharedObj() const;

    std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> &
    getSSDSharedObj();


  private:
    /// objects for various exchange-correlations (from libxc package)
    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;

    std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> d_excObj;
  };
} // namespace dftfe

#endif // DFTFE_EXCMANAGER_H
