/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Bikash Kanungo
 */
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "StringOperations.h"
namespace dftfe
{
  namespace utils
  {
    namespace stringOps
    {
      bool
      strToInt(const std::string s, int &i)
      {
        try
          {
            i = boost::lexical_cast<int>(s);
          }
        catch (const boost::bad_lexical_cast &e)
          {
            return false;
          }
        return true;
      }

      bool
      strToDouble(const std::string s, double &x)
      {
        try
          {
            x = boost::lexical_cast<double>(s);
          }
        catch (const boost::bad_lexical_cast &e)
          {
            return false;
          }
        return true;
      }

      void
      trim(std::string &s)
      {
        boost::algorithm::trim(s);
      }

      std::string
      trimCopy(const std::string &s)
      {
        return boost::algorithm::trim_copy(s);
      }

      std::vector<std::string>
      split(const std::string &inpStr,
            std::string        delimiter /* = " " */,
            bool               skipAdjacentDelimiters /* = true */,
            bool               skipLeadTrailWhiteSpace /* = true */)
      {
        std::string inpStrCopy = inpStr;
        if (skipLeadTrailWhiteSpace)
          trim(inpStrCopy);

        std::vector<std::string> outStrs(0);
        if (skipAdjacentDelimiters)
          {
            // NOTE: The thrid argument (token_compress_mode_type eCompress) is
            // set
            // to token_compress_on to treat consecutive delimiters to be one,
            // otherwise boost assumes an empty string in between the adjacent
            // delimiters
            boost::algorithm::split(outStrs,
                                    inpStrCopy,
                                    boost::is_any_of(delimiter),
                                    boost::token_compress_on);
          }
        else
          {
            boost::algorithm::split(outStrs,
                                    inpStrCopy,
                                    boost::is_any_of(delimiter));
          }
        return outStrs;
      }
    } // end of namespace stringOps
  }   // end of namespace utils
} // end of namespace dftfe
