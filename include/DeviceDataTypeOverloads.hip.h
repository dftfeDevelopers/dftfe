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


#ifndef dftfeDeviceDataTypeOverloads_hiph
#define dftfeDeviceDataTypeOverloads_hiph

#include <complex>
#include <hip/hip_complex.h>
#include <TypeConfig.h>
namespace dftfe
{
  namespace utils
  {
    __forceinline__ __device__ hipDoubleComplex
                               makeComplex(double realPart, double imagPart)
    {
      return make_hipDoubleComplex(realPart, imagPart);
    }

    __forceinline__ __device__ hipFloatComplex
                               makeComplex(float realPart, float imagPart)
    {
      return make_hipFloatComplex(realPart, imagPart);
    }

    //
    // copyValue for homogeneous types
    //
    __forceinline__ __device__ void
                    copyValue(double *a, const double b)
    {
      *a = b;
    }

    __forceinline__ __device__ void
                    copyValue(float *a, const float b)
    {
      *a = b;
    }

    __forceinline__ __device__ void
                    copyValue(hipDoubleComplex *a, const hipDoubleComplex b)
    {
      *a = b;
    }

    __forceinline__ __device__ void
                    copyValue(hipFloatComplex *a, const hipFloatComplex b)
    {
      *a = b;
    }

    //
    // copyValue for heteregenous types
    //
    __forceinline__ __device__ void
                    copyValue(float *a, const double b)
    {
      *a = b;
    }

    __forceinline__ __device__ void
                    copyValue(double *a, const float b)
    {
      *a = b;
    }

    __forceinline__ __device__ void
                    copyValue(hipDoubleComplex *a, const hipFloatComplex b)
    {
      *a = make_hipDoubleComplex(b.x, b.y);
    }

    __forceinline__ __device__ void
                    copyValue(hipFloatComplex *a, const hipDoubleComplex b)
    {
      *a = make_hipFloatComplex(b.x, b.y);
    }

    __forceinline__ __device__ void
                    copyValue(hipDoubleComplex *a, const double b)
    {
      *a = make_hipDoubleComplex(b, 0);
    }

    __forceinline__ __device__ void
                    copyValue(hipFloatComplex *a, const float b)
    {
      *a = make_hipFloatComplex(b, 0);
    }

    __forceinline__ __device__ void
                    copyValue(hipDoubleComplex *a, const float b)
    {
      *a = make_hipDoubleComplex(b, 0);
    }

    __forceinline__ __device__ void
                    copyValue(hipFloatComplex *a, const double b)
    {
      *a = make_hipFloatComplex(b, 0);
    }

    // real part obverloads

    __forceinline__ __device__ double
                    realPartDevice(double a)
    {
      return a;
    }

    __forceinline__ __device__ float
                    realPartDevice(float a)
    {
      return a;
    }

    __forceinline__ __device__ double
                    realPartDevice(hipDoubleComplex a)
    {
      return a.x;
    }

    __forceinline__ __device__ float
                    realPartDevice(hipFloatComplex a)
    {
      return a.x;
    }

    // imag part obverloads

    __forceinline__ __device__ double
                    imagPartDevice(double a)
    {
      return 0;
    }

    __forceinline__ __device__ float
                    imagPartDevice(float a)
    {
      return 0;
    }

    __forceinline__ __device__ double
                    imagPartDevice(hipDoubleComplex a)
    {
      return a.y;
    }

    __forceinline__ __device__ float
                    imagPartDevice(hipFloatComplex a)
    {
      return a.y;
    }

    // abs obverloads

    __forceinline__ __device__ double
                    abs(double a)
    {
      return fabs(a);
    }

    __forceinline__ __device__ float
                    abs(float a)
    {
      return fabs(a);
    }

    __forceinline__ __device__ double
                    abs(hipDoubleComplex a)
    {
      return hipCabs(a);
    }

    __forceinline__ __device__ float
                    abs(hipFloatComplex a)
    {
      return hipCabsf(a);
    }

    //
    // conjugate overloads
    //

    __forceinline__ __device__ size_type
                               conj(size_type a)
    {
      return a;
    }

    __forceinline__ __device__ global_size_type
                               conj(global_size_type a)
    {
      return a;
    }

    __forceinline__ __device__ int
                    conj(int a)
    {
      return a;
    }

    __forceinline__ __device__ float
                    conj(float a)
    {
      return a;
    }
    __forceinline__ __device__ double
                    conj(double a)
    {
      return a;
    }

    __forceinline__ __device__ hipDoubleComplex
                               conj(hipDoubleComplex a)
    {
      return hipConj(a);
    }

    __forceinline__ __device__ hipFloatComplex
                               conj(hipFloatComplex a)
    {
      return hipConjf(a);
    }


    //
    // mult for real homogeneous types e.g. (double, double)
    //
    __forceinline__ __device__ size_type
                               mult(size_type a, size_type b)
    {
      return a * b;
    }

    __forceinline__ __device__ global_size_type
                               mult(global_size_type a, global_size_type b)
    {
      return a * b;
    }

    __forceinline__ __device__ int
                    mult(int a, int b)
    {
      return a * b;
    }

    __forceinline__ __device__ double
                    mult(double a, double b)
    {
      return a * b;
    }

    __forceinline__ __device__ float
                    mult(float a, float b)
    {
      return a * b;
    }

    //
    // mult for complex homogenous types
    // (e.g., hipDoubleComplex and hipDoubleComplex)
    //
    __forceinline__ __device__ hipDoubleComplex
                               mult(hipDoubleComplex a, hipDoubleComplex b)
    {
      return hipCmul(a, b);
    }

    __forceinline__ __device__ hipFloatComplex
                               mult(hipFloatComplex a, hipFloatComplex b)
    {
      return hipCmulf(a, b);
    }


    //
    // mult for complex heterogeneous types e.g. (hipDoubleComplex,
    // hipFloatComplex)
    //
    __forceinline__ __device__ hipDoubleComplex
                               mult(hipFloatComplex a, hipDoubleComplex b)
    {
      return hipCmul(make_hipDoubleComplex(a.x, a.y), b);
    }

    __forceinline__ __device__ hipDoubleComplex
                               mult(hipDoubleComplex a, hipFloatComplex b)
    {
      return hipCmul(a, make_hipDoubleComplex(b.x, b.y));
    }


    //
    // mult for real-complex heterogeneous types e.g. (double, hipFloatComplex)
    //
    __forceinline__ __device__ hipDoubleComplex
                               mult(double a, hipDoubleComplex b)
    {
      return make_hipDoubleComplex(a * b.x, a * b.y);
    }

    __forceinline__ __device__ hipDoubleComplex
                               mult(hipDoubleComplex a, double b)
    {
      return make_hipDoubleComplex(b * a.x, b * a.y);
    }

    __forceinline__ __device__ hipFloatComplex
                               mult(float a, hipFloatComplex b)
    {
      return make_hipFloatComplex(a * b.x, a * b.y);
    }

    __forceinline__ __device__ hipFloatComplex
                               mult(hipFloatComplex a, float b)
    {
      return make_hipFloatComplex(b * a.x, b * a.y);
    }

    __forceinline__ __device__ hipDoubleComplex
                               mult(double a, hipFloatComplex b)
    {
      return make_hipDoubleComplex(a * b.x, a * b.y);
    }

    __forceinline__ __device__ hipDoubleComplex
                               mult(hipFloatComplex a, double b)
    {
      return make_hipDoubleComplex(b * a.x, b * a.y);
    }


    __forceinline__ __device__ size_type
                               add(size_type a, size_type b)
    {
      return a + b;
    }

    __forceinline__ __device__ global_size_type
                               add(global_size_type a, global_size_type b)
    {
      return a + b;
    }

    __forceinline__ __device__ int
                    add(int a, int b)
    {
      return a + b;
    }

    __forceinline__ __device__ double
                    add(double a, double b)
    {
      return a + b;
    }

    __forceinline__ __device__ float
                    add(float a, float b)
    {
      return a + b;
    }

    __forceinline__ __device__ hipDoubleComplex
                               add(hipDoubleComplex a, hipDoubleComplex b)
    {
      return hipCadd(a, b);
    }


    __forceinline__ __device__ hipFloatComplex
                               add(hipFloatComplex a, hipFloatComplex b)
    {
      return hipCaddf(a, b);
    }

    __forceinline__ __device__ double
                    add(double a, float b)
    {
      return a + b;
    }

    __forceinline__ __device__ double
                    add(float a, double b)
    {
      return a + b;
    }

    __forceinline__ __device__ cuDoubleComplex
                               add(hipDoubleComplex a, hipFloatComplex b)
    {
      return hipCadd(a, make_hipDoubleComplex(b.x, b.y));
    }


    __forceinline__ __device__ cuDoubleComplex
                               add(hipFloatComplex a, hipDoubleComplex b)
    {
      return hipCadd(make_hipDoubleComplex(a.x, a.y), b);
    }


    __forceinline__ __device__ size_type
                               sub(size_type a, size_type b)
    {
      return a - b;
    }

    __forceinline__ __device__ global_size_type
                               sub(global_size_type a, global_size_type b)
    {
      return a - b;
    }

    __forceinline__ __device__ int
                    sub(int a, int b)
    {
      return a - b;
    }

    __forceinline__ __device__ double
                    sub(double a, double b)
    {
      return a - b;
    }

    __forceinline__ __device__ float
                    sub(float a, float b)
    {
      return a - b;
    }

    __forceinline__ __device__ hipDoubleComplex
                               sub(hipDoubleComplex a, hipDoubleComplex b)
    {
      return hipCsub(a, b);
    }

    __forceinline__ __device__ hipFloatComplex
                               sub(hipFloatComplex a, hipFloatComplex b)
    {
      return hipCsubf(a, b);
    }

    __forceinline__ __device__ size_type
                               div(size_type a, size_type b)
    {
      return a / b;
    }

    __forceinline__ __device__ global_size_type
                               div(global_size_type a, global_size_type b)
    {
      return a / b;
    }

    __forceinline__ __device__ int
                    div(int a, int b)
    {
      return a / b;
    }

    __forceinline__ __device__ double
                    div(double a, double b)
    {
      return a / b;
    }

    __forceinline__ __device__ float
                    div(float a, float b)
    {
      return a / b;
    }

    __forceinline__ __device__ hipDoubleComplex
                               div(hipDoubleComplex a, hipDoubleComplex b)
    {
      return hipCdiv(a, b);
    }

    __forceinline__ __device__ hipFloatComplex
                               div(hipFloatComplex a, hipFloatComplex b)
    {
      return hipCdivf(a, b);
    }

    //
    // div for complex heterogeneous types e.g. (hipDoubleComplex,
    // hipFloatComplex)
    //
    __forceinline__ __device__ hipDoubleComplex
                               div(hipFloatComplex a, hipDoubleComplex b)
    {
      return hipCdiv(make_hipDoubleComplex(a.x, a.y), b);
    }

    __forceinline__ __device__ hipDoubleComplex
                               div(hipDoubleComplex a, hipFloatComplex b)
    {
      return hipCdiv(a, make_hipDoubleComplex(b.x, b.y));
    }


    //
    // div for real-complex heterogeneous types e.g. (double, hipFloatComplex)
    //
    __forceinline__ __device__ hipDoubleComplex
                               div(double a, hipDoubleComplex b)
    {
      return make_hipDoubleComplex(a / b.x, a / b.y);
    }

    __forceinline__ __device__ hipDoubleComplex
                               div(hipDoubleComplex a, double b)
    {
      return make_hipDoubleComplex(b / a.x, b / a.y);
    }

    __forceinline__ __device__ hipFloatComplex
                               div(float a, hipFloatComplex b)
    {
      return make_hipFloatComplex(a / b.x, a / b.y);
    }

    __forceinline__ __device__ hipFloatComplex
                               div(hipFloatComplex a, float b)
    {
      return make_hipFloatComplex(b / a.x, b / a.y);
    }

    __forceinline__ __device__ hipDoubleComplex
                               div(double a, hipFloatComplex b)
    {
      return make_hipDoubleComplex(a / b.x, a / b.y);
    }

    __forceinline__ __device__ hipDoubleComplex
                               div(hipFloatComplex a, double b)
    {
      return make_hipDoubleComplex(b / a.x, b / a.y);
    }

    ////


    inline int *
    makeDataTypeDeviceCompatible(int *a)
    {
      return a;
    }

    inline const int *
    makeDataTypeDeviceCompatible(const int *a)
    {
      return a;
    }

    inline size_type *
    makeDataTypeDeviceCompatible(size_type *a)
    {
      return a;
    }

    inline const size_type *
    makeDataTypeDeviceCompatible(const size_type *a)
    {
      return a;
    }

    inline global_size_type *
    makeDataTypeDeviceCompatible(global_size_type *a)
    {
      return a;
    }

    inline const global_size_type *
    makeDataTypeDeviceCompatible(const global_size_type *a)
    {
      return a;
    }

    inline double *
    makeDataTypeDeviceCompatible(double *a)
    {
      return a;
    }

    inline const double *
    makeDataTypeDeviceCompatible(const double *a)
    {
      return a;
    }

    inline float *
    makeDataTypeDeviceCompatible(float *a)
    {
      return a;
    }

    inline const float *
    makeDataTypeDeviceCompatible(const float *a)
    {
      return a;
    }

    inline hipDoubleComplex *
    makeDataTypeDeviceCompatible(std::complex<double> *a)
    {
      return reinterpret_cast<hipDoubleComplex *>(a);
    }

    inline const hipDoubleComplex *
    makeDataTypeDeviceCompatible(const std::complex<double> *a)
    {
      return reinterpret_cast<const hipDoubleComplex *>(a);
    }

    inline hipFloatComplex *
    makeDataTypeDeviceCompatible(std::complex<float> *a)
    {
      return reinterpret_cast<hipFloatComplex *>(a);
    }

    inline const hipFloatComplex *
    makeDataTypeDeviceCompatible(const std::complex<float> *a)
    {
      return reinterpret_cast<const hipFloatComplex *>(a);
    }

    inline int
    makeDataTypeDeviceCompatible(int a)
    {
      return a;
    }

    inline size_type
    makeDataTypeDeviceCompatible(size_type a)
    {
      return a;
    }

    inline global_size_type
    makeDataTypeDeviceCompatible(global_size_type a)
    {
      return a;
    }

    inline double
    makeDataTypeDeviceCompatible(double a)
    {
      return a;
    }

    inline float
    makeDataTypeDeviceCompatible(float a)
    {
      return a;
    }

    inline hipDoubleComplex
    makeDataTypeDeviceCompatible(std::complex<double> a)
    {
      return make_hipDoubleComplex(a.real(), a.imag());
    }

    inline hipFloatComplex
    makeDataTypeDeviceCompatible(std::complex<float> a)
    {
      return make_hipFloatComplex(a.real(), a.imag());
    }


  } // namespace utils

} // namespace dftfe

#endif
