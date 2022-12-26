// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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


/*
 * @author Bikash Kanungo
 */
#include "Exceptions.h"
#include <exception>
namespace dftfe
{
  namespace utils
  {
    namespace
    {
      /**
       * @brief A class that derives from the std::exception to throw a custom message
       */
      class ExceptionWithMsg : public std::exception
      {
      public:
        ExceptionWithMsg(std::string const &msg)
          : d_msg(msg)
        {}
        virtual char const *
        what() const noexcept override
        {
          return d_msg.c_str();
        }

      private:
        std::string d_msg;
      };

    } // end of unnamed namespace


    void
    throwException(bool condition, std::string msg)
    {
      if (!condition)
        {
          throw ExceptionWithMsg(msg);
        }
    }



  } // end of namespace utils

} // end of namespace dftfe
