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

/** @file dftUtils.cc
 *  @brief Contains repeatedly used functions in the KSDFT calculations
 *
 *  @author Sambit Das
 */

#include "../include/dftUtils.h"
#include "../include/headers.h"

namespace dftUtils
{
   double getPartialOccupancy(const double eigenValue,const double fermiEnergy,const double kb,const double T)
   {
      const double factor=(eigenValue-fermiEnergy)/(kb*T);       
      return (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
   }
}
