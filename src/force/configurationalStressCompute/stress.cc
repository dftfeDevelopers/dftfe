// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das(2017)
//


template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStress()
{
  //configurational stress contribution from all terms except those from nuclear self energy
  if (dftParameters::spinPolarized)
     std::cout<<"TO BE IMPLEMENTED"<<std::endl;  
  else
     computeStressEEshelbyEPSPEnlEk(); 
  //configurational stress contribution from nuclear self energy. This is handled separately as it involves
  // a surface integral over the vself ball surface
  computeStressEself();
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::printStress()
{
}
//
