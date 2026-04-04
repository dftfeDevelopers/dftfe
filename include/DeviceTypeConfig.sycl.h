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
#ifndef dftfeDeviceTypeConfig_syclh
#define dftfeDeviceTypeConfig_syclh

#include <oneapi/mkl/types.hpp>
#include <oneapi/mkl/blas/types.hpp>
#include <complex>
#include <set>
#include <map>
#include <TypeConfig.h>
#include <sycl/sycl.hpp>

namespace dftfe
{
  namespace utils
  {
    typedef std::error_code      deviceError_t;
    typedef dftfe::uInt          deviceStream_t;
    typedef sycl::event          deviceEvent_t;
    typedef std::complex<double> deviceDoubleComplex;
    typedef std::complex<float>  deviceFloatComplex;

    // static consts
    static std::error_code deviceSuccess = make_error_code(sycl::errc::success);
    // static deviceError_t deviceSuccess(success_code);

    // vendor blas related typedef and static consts
    typedef sycl::queue                      deviceBlasHandle_t;
    typedef oneapi::mkl::transpose           deviceBlasOperation_t;
    typedef oneapi::mkl::blas::compute_mode  deviceBlasComputeType_t;
    typedef sycl::info::event_command_status deviceBlasStatus_t;

    static const sycl::info::event_command_status deviceBlasSuccess =
      sycl::info::event_command_status::complete;

    static const oneapi::mkl::transpose DEVICEBLAS_OP_N =
      oneapi::mkl::transpose::nontrans;
    static const oneapi::mkl::transpose DEVICEBLAS_OP_T =
      oneapi::mkl::transpose::trans;
    static const oneapi::mkl::transpose DEVICEBLAS_OP_C =
      oneapi::mkl::transpose::conjtrans;
    static const oneapi::mkl::blas::compute_mode DEVICEBLAS_COMPUTE_32F =
      oneapi::mkl::blas::compute_mode::standard;
    static const oneapi::mkl::blas::compute_mode
      DEVICEBLAS_COMPUTE_32F_FAST_TF32 =
        oneapi::mkl::blas::compute_mode::float_to_tf32;
    static const oneapi::mkl::blas::compute_mode
      DEVICEBLAS_COMPUTE_32F_FAST_16BF =
        oneapi::mkl::blas::compute_mode::float_to_bf16;
    static const oneapi::mkl::blas::compute_mode
      DEVICEBLAS_COMPUTE_32F_FAST_16F =
        oneapi::mkl::blas::compute_mode::float_to_bf16x2;
    static deviceStream_t            defaultStream = 0;
    inline std::vector<sycl::device> allSyclGPUDevices =
      sycl::device::get_devices(sycl::info::device_type::gpu);
    inline dftfe::uInt   syclDeviceId = 0;
    inline sycl::device  syclDevice   = allSyclGPUDevices[syclDeviceId];
    inline sycl::context syclContext{syclDevice};
    inline std::map<dftfe::uInt, sycl::queue> queueRegistry{
      {defaultStream,
       sycl::queue(syclContext,
                   syclDevice,
                   sycl::property::queue::in_order{})}};
    inline std::set<dftfe::uInt> usedStreamIds{defaultStream};
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfig_syclh
