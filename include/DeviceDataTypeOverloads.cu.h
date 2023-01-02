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


#ifndef dftfeDeviceDataTypeOverloads_cuh
#define dftfeDeviceDataTypeOverloads_cuh

#include <complex>
#include <cuComplex.h>
#include <TypeConfig.h>
namespace dftfe
{
  namespace utils
  {
    __forceinline__ __device__ cuDoubleComplex
                               makeComplex(double realPart, double imagPart)
    {
      return make_cuDoubleComplex(realPart, imagPart);
    }

    __forceinline__ __device__ cuFloatComplex
                               makeComplex(float realPart, float imagPart)
    {
      return make_cuFloatComplex(realPart, imagPart);
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
                    copyValue(cuDoubleComplex *a, const cuDoubleComplex b)
    {
      *a = b;
    }

    __forceinline__ __device__ void
                    copyValue(cuFloatComplex *a, const cuFloatComplex b)
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
                    copyValue(cuDoubleComplex *a, const cuFloatComplex b)
    {
      *a = make_cuDoubleComplex(b.x, b.y);
    }

    __forceinline__ __device__ void
                    copyValue(cuFloatComplex *a, const cuDoubleComplex b)
    {
      *a = make_cuFloatComplex(b.x, b.y);
    }

    __forceinline__ __device__ void
                    copyValue(cuDoubleComplex *a, const double b)
    {
      *a = make_cuDoubleComplex(b, 0);
    }

    __forceinline__ __device__ void
                    copyValue(cuFloatComplex *a, const float b)
    {
      *a = make_cuFloatComplex(b, 0);
    }

    __forceinline__ __device__ void
                    copyValue(cuDoubleComplex *a, const float b)
    {
      *a = make_cuDoubleComplex(b, 0);
    }

    __forceinline__ __device__ void
                    copyValue(cuFloatComplex *a, const double b)
    {
      *a = make_cuFloatComplex(b, 0);
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
                    realPartDevice(cuDoubleComplex a)
    {
      return a.x;
    }

    __forceinline__ __device__ float
                    realPartDevice(cuFloatComplex a)
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
                    imagPartDevice(cuDoubleComplex a)
    {
      return a.y;
    }

    __forceinline__ __device__ float
                    imagPartDevice(cuFloatComplex a)
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
                    abs(cuDoubleComplex a)
    {
      return cuCabs(a);
    }

    __forceinline__ __device__ float
                    abs(cuFloatComplex a)
    {
      return cuCabsf(a);
    }

    //
    // conjugate overloads
    //

    __forceinline__ __device__ unsigned int
                    conj(unsigned int a)
    {
      return a;
    }

    __forceinline__ __device__ unsigned long int
                    conj(unsigned long int a)
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

    __forceinline__ __device__ cuDoubleComplex
                               conj(cuDoubleComplex a)
    {
      return cuConj(a);
    }

    __forceinline__ __device__ cuFloatComplex
                               conj(cuFloatComplex a)
    {
      return cuConjf(a);
    }


    //
    // mult for real homogeneous types e.g. (double, double)
    //
    __forceinline__ __device__ unsigned int
                    mult(unsigned int a, unsigned int b)
    {
      return a * b;
    }

    __forceinline__ __device__ unsigned long int
                    mult(unsigned long int a, unsigned long int b)
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

    __forceinline__ __device__ double
                    mult(float a, double b)
    {
      return a * b;
    }

    __forceinline__ __device__ double
                    mult(double a, float b)
    {
      return a * b;
    }


    //
    // mult for complex homogenous types
    // (e.g., cuDoubleComplex and cuDoubleComplex)
    //
    __forceinline__ __device__ cuDoubleComplex
                               mult(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCmul(a, b);
    }

    __forceinline__ __device__ cuFloatComplex
                               mult(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCmulf(a, b);
    }


    //
    // mult for complex heterogeneous types e.g. (cuDoubleComplex,
    // cuFloatComplex)
    //
    __forceinline__ __device__ cuDoubleComplex
                               mult(cuFloatComplex a, cuDoubleComplex b)
    {
      return cuCmul(make_cuDoubleComplex(a.x, a.y), b);
    }

    __forceinline__ __device__ cuDoubleComplex
                               mult(cuDoubleComplex a, cuFloatComplex b)
    {
      return cuCmul(a, make_cuDoubleComplex(b.x, b.y));
    }


    //
    // mult for real-complex heterogeneous types e.g. (double, cuFloatComplex)
    //
    __forceinline__ __device__ cuDoubleComplex
                               mult(double a, cuDoubleComplex b)
    {
      return make_cuDoubleComplex(a * b.x, a * b.y);
    }

    __forceinline__ __device__ cuDoubleComplex
                               mult(cuDoubleComplex a, double b)
    {
      return make_cuDoubleComplex(b * a.x, b * a.y);
    }

    __forceinline__ __device__ cuFloatComplex
                               mult(float a, cuFloatComplex b)
    {
      return make_cuFloatComplex(a * b.x, a * b.y);
    }

    __forceinline__ __device__ cuFloatComplex
                               mult(cuFloatComplex a, float b)
    {
      return make_cuFloatComplex(b * a.x, b * a.y);
    }

    __forceinline__ __device__ cuDoubleComplex
                               mult(double a, cuFloatComplex b)
    {
      return make_cuDoubleComplex(a * b.x, a * b.y);
    }

    __forceinline__ __device__ cuDoubleComplex
                               mult(cuFloatComplex a, double b)
    {
      return make_cuDoubleComplex(b * a.x, b * a.y);
    }


    __forceinline__ __device__ unsigned int
                    add(unsigned int a, unsigned int b)
    {
      return a + b;
    }

    __forceinline__ __device__ unsigned long int
                    add(unsigned long int a, unsigned long int b)
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

    __forceinline__ __device__ cuDoubleComplex
                               add(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCadd(a, b);
    }


    __forceinline__ __device__ cuFloatComplex
                               add(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCaddf(a, b);
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
                               add(cuDoubleComplex a, cuFloatComplex b)
    {
      return cuCadd(a, make_cuDoubleComplex(b.x, b.y));
    }


    __forceinline__ __device__ cuDoubleComplex
                               add(cuFloatComplex a, cuDoubleComplex b)
    {
      return cuCadd(make_cuDoubleComplex(a.x, a.y), b);
    }


    __forceinline__ __device__ unsigned int
                    sub(unsigned int a, unsigned int b)
    {
      return a - b;
    }

    __forceinline__ __device__ unsigned long int
                    sub(unsigned long int a, unsigned long int b)
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

    __forceinline__ __device__ cuDoubleComplex
                               sub(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCsub(a, b);
    }

    __forceinline__ __device__ cuFloatComplex
                               sub(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCsubf(a, b);
    }

    __forceinline__ __device__ unsigned int
                    div(unsigned int a, unsigned int b)
    {
      return a / b;
    }

    __forceinline__ __device__ unsigned long int
                    div(unsigned long int a, unsigned long int b)
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

    __forceinline__ __device__ cuDoubleComplex
                               div(cuDoubleComplex a, cuDoubleComplex b)
    {
      return cuCdiv(a, b);
    }

    __forceinline__ __device__ cuFloatComplex
                               div(cuFloatComplex a, cuFloatComplex b)
    {
      return cuCdivf(a, b);
    }

    //
    // div for complex heterogeneous types e.g. (cuDoubleComplex,
    // cuFloatComplex)
    //
    __forceinline__ __device__ cuDoubleComplex
                               div(cuFloatComplex a, cuDoubleComplex b)
    {
      return cuCdiv(make_cuDoubleComplex(a.x, a.y), b);
    }

    __forceinline__ __device__ cuDoubleComplex
                               div(cuDoubleComplex a, cuFloatComplex b)
    {
      return cuCdiv(a, make_cuDoubleComplex(b.x, b.y));
    }


    //
    // div for real-complex heterogeneous types e.g. (double, cuFloatComplex)
    //
    __forceinline__ __device__ cuDoubleComplex
                               div(double a, cuDoubleComplex b)
    {
      return make_cuDoubleComplex(a / b.x, a / b.y);
    }

    __forceinline__ __device__ cuDoubleComplex
                               div(cuDoubleComplex a, double b)
    {
      return make_cuDoubleComplex(b / a.x, b / a.y);
    }

    __forceinline__ __device__ cuFloatComplex
                               div(float a, cuFloatComplex b)
    {
      return make_cuFloatComplex(a / b.x, a / b.y);
    }

    __forceinline__ __device__ cuFloatComplex
                               div(cuFloatComplex a, float b)
    {
      return make_cuFloatComplex(b / a.x, b / a.y);
    }

    __forceinline__ __device__ cuDoubleComplex
                               div(double a, cuFloatComplex b)
    {
      return make_cuDoubleComplex(a / b.x, a / b.y);
    }

    __forceinline__ __device__ cuDoubleComplex
                               div(cuFloatComplex a, double b)
    {
      return make_cuDoubleComplex(b / a.x, b / a.y);
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

    inline long int *
    makeDataTypeDeviceCompatible(long int *a)
    {
      return a;
    }

    inline const long int *
    makeDataTypeDeviceCompatible(const long int *a)
    {
      return a;
    }


    inline unsigned int *
    makeDataTypeDeviceCompatible(unsigned int *a)
    {
      return a;
    }

    inline const unsigned int *
    makeDataTypeDeviceCompatible(const unsigned int *a)
    {
      return a;
    }

    inline unsigned long int *
    makeDataTypeDeviceCompatible(unsigned long int *a)
    {
      return a;
    }

    inline const unsigned long int *
    makeDataTypeDeviceCompatible(const unsigned long int *a)
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

    inline cuDoubleComplex *
    makeDataTypeDeviceCompatible(std::complex<double> *a)
    {
      return reinterpret_cast<cuDoubleComplex *>(a);
    }

    inline const cuDoubleComplex *
    makeDataTypeDeviceCompatible(const std::complex<double> *a)
    {
      return reinterpret_cast<const cuDoubleComplex *>(a);
    }

    inline cuFloatComplex *
    makeDataTypeDeviceCompatible(std::complex<float> *a)
    {
      return reinterpret_cast<cuFloatComplex *>(a);
    }

    inline const cuFloatComplex *
    makeDataTypeDeviceCompatible(const std::complex<float> *a)
    {
      return reinterpret_cast<const cuFloatComplex *>(a);
    }

    inline int
    makeDataTypeDeviceCompatible(int a)
    {
      return a;
    }

    inline long int
    makeDataTypeDeviceCompatible(long int a)
    {
      return a;
    }


    inline unsigned int
    makeDataTypeDeviceCompatible(unsigned int a)
    {
      return a;
    }

    inline unsigned long int
    makeDataTypeDeviceCompatible(unsigned long int a)
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

    inline cuDoubleComplex
    makeDataTypeDeviceCompatible(std::complex<double> a)
    {
      return make_cuDoubleComplex(a.real(), a.imag());
    }

    inline cuFloatComplex
    makeDataTypeDeviceCompatible(std::complex<float> a)
    {
      return make_cuFloatComplex(a.real(), a.imag());
    }


  } // namespace utils

} // namespace dftfe

#endif
