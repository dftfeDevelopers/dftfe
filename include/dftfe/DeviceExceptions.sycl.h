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
//
#ifndef dftfeDeviceExceptions_syclh
#define dftfeDeviceExceptions_syclh

#define DEVICE_API_CHECK(event)                                               \
  {                                                                           \
    try                                                                       \
      {                                                                       \
        event.wait();                                                         \
      }                                                                       \
    catch (const sycl::exception &e)                                          \
      {                                                                       \
        std::cerr << "SYCL error on or before line number" << __LINE__        \
                  << " in file: " << __FILE__ << ". Error code: " << e.what() \
                  << ".\n";                                                   \
      }                                                                       \
  }

#define DEVICEBLAS_API_CHECK(expr)                           \
  do                                                         \
    {                                                        \
      try                                                    \
        {                                                    \
          (void)(expr);                                      \
        }                                                    \
      catch (sycl::exception const &__sycl_err)              \
        {                                                    \
          std::printf("oneMKL enqueue error at %s:%d: %s\n", \
                      __FILE__,                              \
                      __LINE__,                              \
                      __sycl_err.what());                    \
        }                                                    \
  } while (0)

#endif // dftfeDeviceExceptions_syclh
