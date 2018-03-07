//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------

/** @file dftUtils.h
 *  @brief Contains repeatedly used functions in the KSDFT calculations
 *
 *  @author Sambit Das
 */

#ifndef dftUtils_H_
#define dftUtils_H_
namespace dftUtils
{
/** @brief Calculates partial occupancy of the atomic orbital using 
 *  Fermi-Dirac smearing.
 *
 *  @param  eigenValue
 *  @param  fermiEnergy 
 *  @param  kb Boltzmann constant
 *  @param  T smearing temperature 
 *  @return double The partial occupancy of the orbital
 */    
double getPartialOccupancy(const double eigenValue,const double fermiEnergy,const double kb,const double T);

}
#endif
