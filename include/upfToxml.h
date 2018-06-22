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
// @author Phani Motamarri (2018)
//

#ifndef upfToxml_h
#define upfToxml_h
#include <string>
#include "string.h"
#include <headers.h>
#include <dftParameters.h>

namespace dftfe
{
 namespace pseudoUtils
 {
   int upfToxml(std::string &inputFile,
		std::string &outputFile);
 }
}
#endif 
