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
#ifndef dftfeExceptions_h
#define dftfeExceptions_h

#include <string>
#include <stdexcept>
/**
@brief provides an interface for exception handling.
It two overrides on the assert(expr) function in C/C++ and two wrappers on
std::exception.

The overrides on assert(expr) are useful for debug mode testing. That is,
you want the assert to be executed *only in debug mode* and not in release
mode (i.e., if NDEBUG is defined). The two assert overrides are
1. DFTFE_Assert(expr): same as std::assert(expr). Throws an assert
if expr is false
2. DFTFE_AssertWithMsg(expr,msg): same as above but takes an
additional message in the form of string to display if expr is false.

It also provides two preprocessor flags, DFTFE_DISABLE_ASSERT and
DFTFE_ENABLE_ASSERT, that you can set to override the NDEBUG flag in a
particular source file. This is provided to allow selective enabling or
disabling of Assert and AssertWithMsg without any relation to whether NDEBUG
is defined or not (NDEBUG is typically defined globally for all files
through compiler options).
For example, if in a file you have
#define DFTFE_DISABLE_ASSERT
#include "Exceptions.h"
then it would disable all calls to Assert or AssertWithMsg in that file,
regardless of whether NDEBUG is defined. Also, it has no bearing on
std::assert (i.e., any calls to std::assert in that file will still be
governed by NDEBUG). Similarly, if in a file you have
#define
DFTFE_ENABLE_ASSERT
#include "Exceptions.h"
then it would enable all calls to Assert or AssertWithMsg in that file,
regardless of whether NDEBUG is defined.
Also, it has no bearning on std::assert (i.e., any calls
to std::assert in that file will still be governed by NDEBUG)

It also provides two wrappers on std::exception and its derived classes
(e.g., std::runtime_error, std::domain_error, etc.) The two wrappers are:
1. dftfe::utils::throwException(expr,msg): a generic exception handler
which throws an optional message (msg) if expr evaluates to false. It
combines std::exception with an  additional messgae. (Note: the
std::exception has no easy way of taking in a message).
2. dftfe::utils::throwException<T>(expr, msg): similar to the above, but
takes a specific derived class of std::exception handler as a template
parameter. The derived std::exception must have a constructor that takes in
a string. For the ease of the user, we have typedef-ed some commonly used
derived classes of std::exception. A user can use the typedefs as the
template parameter instead. Available typedefs LogicError - std::logic_error
   InvalidArgument - std::invalid_argument
   DomainError - std::domain_error
   LengthError - std::length_error
   OutOfRangeError - std::out_of_range
   FutureError - std::future_error
   RuntimeError	- std::runtime_error
   OverflowError - std::overflow_error
   UnderflowError - std::underflow_error
*/

#undef DFTFE_Assert
#undef DFTFE_AssertWithMsg

#if defined(DFTFE_DISABLE_ASSERT) || \
  (!defined(DFTFE_ENABLE_ASSERT) && defined(NDEBUG))
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define DFTFE_Assert(expr) ((void)0)
#  define DFTFE_AssertWithMsg(expr, msg) ((void)0)

#elif defined(DFTFE_ENABLE_ASSERT) && defined(NDEBUG)
#  undef NDEBUG // disabling NDEBUG to forcibly enable assert for sources that
                // set DFTFE_ENABLE_ASSERT even when in release mode (with
                // NDEBUG)
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define DFTFE_Assert(expr) assert(expr)
#  define DFTFE_AssertWithMsg(expr, msg) assert((expr) && (msg))

#else
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define DFTFE_Assert(expr) assert(expr)
#  define DFTFE_AssertWithMsg(expr, msg) assert((expr) && (msg))

#endif

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
#  include <DeviceExceptions.cu.h>
#elif DFTFE_WITH_DEVICE_LANG_HIP
#  include <DeviceExceptions.hip.h>
#endif

namespace dftfe
{
  namespace utils
  {
    typedef std::logic_error      LogicError;
    typedef std::invalid_argument InvalidArgument;
    typedef std::domain_error     DomainError;
    typedef std::length_error     LengthError;
    typedef std::out_of_range     OutOfRangeError;
    typedef std::runtime_error    RuntimeError;
    typedef std::overflow_error   OverflowError;
    typedef std::underflow_error  UnderflowError;

    void
    throwException(bool condition, std::string msg = "");

    template <class T>
    void
    throwException(bool condition, std::string msg = "");

  } // namespace utils
} // namespace dftfe
#include "../utils/Exceptions.t.cc"
#endif // dftfeExceptions_h
