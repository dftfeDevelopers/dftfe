// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri
//

#ifndef upfToxml_h
#define upfToxml_h
#include <string>
#include "string.h"
#include <headers.h>
#include <dftParameters.h>

namespace dftfe
{
  //
  //Declare pseudoUtils function
  //

  /** @file upfxml.h
   *  @brief converts pseudopotential file from upf to xml format
   *
   *  The functionality reads the upfile and identifies appropriate tags and converts
   *  into xml file format
   *
   *  @author Phani Motamarri
   */
 namespace pseudoUtils
 {
   /**
    * @brief read a given upf pseudopotential file name in upf format  and convert to xml format
    * 
    * @param inputFile filePath location of the upf file
    * @param outputFile filePath location of xml file
    *
    * @return int errorCode indicating success or failure of conversion
    */
   int upfToxml(const std::string &inputFile,
		const std::string &outputFile);
 }
}
#endif 
