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

#ifndef DFTFE_EXPCONFININGPOTENTIAL_H
#define DFTFE_EXPCONFININGPOTENTIAL_H

#include "headers.h"
#include "FEBasisOperations.h"
#include "dftParameters.h"
#include "MemoryStorage.h"

namespace dftfe
{
  class expConfiningPotential
  {
  public:
    expConfiningPotential();
    void
    init(const std::shared_ptr<
           dftfe::basis::FEBasisOperations<dataTypes::number,
                                           double,
                                           dftfe::utils::MemorySpace::HOST>>
                                                &feBasisOp,
         const dftParameters                    &dftParams,
         const std::vector<std::vector<double>> &atomLocations);


    void
    addConfiningPotential(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &externalPotential) const;


  private:
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_confiningPotential;
  };

} // namespace dftfe


#endif // DFTFE_EXPCONFININGPOTENTIAL_H
