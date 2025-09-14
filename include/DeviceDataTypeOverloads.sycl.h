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

#include <sycl/sycl.hpp>
#include <complex>
#include <TypeConfig.h>

namespace dftfe
{
  namespace utils
  {

    template <typename T1, typename T2>
    inline void
    atomicAddWrapper(T1 *addr, T2 value)
    {
      auto atomic_add =
        sycl::atomic_ref<T1,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(addr[0]);
      atomic_add += value;
    }

    inline std::complex<double>
    makeComplex(double realPart, double imagPart)
    {
      return std::complex<double>(realPart, imagPart);
    }

    inline std::complex<float>
    makeComplex(float realPart, float imagPart)
    {
      return std::complex<float>(realPart, imagPart);
    }

    //
    // copyValue for homogeneous types
    //
    inline void
    copyValue(double *a, const double b)
    {
      *a = b;
    }

    inline void
    copyValue(float *a, const float b)
    {
      *a = b;
    }

    inline void
    copyValue(std::complex<double> *a, const std::complex<double> b)
    {
      *a = b;
    }

    inline void
    copyValue(std::complex<float> *a, const std::complex<float> b)
    {
      *a = b;
    }

    //
    // copyValue for heteregenous types
    //
    inline void
    copyValue(float *a, const double b)
    {
      *a = b;
    }

    inline void
    copyValue(double *a, const float b)
    {
      *a = b;
    }

    inline void
    copyValue(std::complex<double> *a, const std::complex<float> b)
    {
      *a = std::complex<double>(b.real(), b.imag());
    }

    inline void
    copyValue(std::complex<float> *a, const std::complex<double> b)
    {
      *a = std::complex<float>(b.real(), b.imag());
    }

    inline void
    copyValue(std::complex<double> *a, const double b)
    {
      *a = std::complex<double>(b, 0);
    }

    inline void
    copyValue(std::complex<float> *a, const float b)
    {
      *a = std::complex<float>(b, 0);
    }

    inline void
    copyValue(std::complex<double> *a, const float b)
    {
      *a = std::complex<double>(b, 0);
    }

    inline void
    copyValue(std::complex<float> *a, const double b)
    {
      *a = std::complex<float>(b, 0);
    }

    // real part obverloads

    inline double
    realPartDevice(double a)
    {
      return a;
    }

    inline float
    realPartDevice(float a)
    {
      return a;
    }

    inline double
    realPartDevice(std::complex<double> a)
    {
      return a.real();
    }

    inline float
    realPartDevice(std::complex<float> a)
    {
      return a.real();
    }

    // imag part obverloads

    inline double
    imagPartDevice(double a)
    {
      return 0;
    }

    inline float
    imagPartDevice(float a)
    {
      return 0;
    }

    inline double
    imagPartDevice(std::complex<double> a)
    {
      return a.imag();
    }

    inline float
    imagPartDevice(std::complex<float> a)
    {
      return a.imag();
    }

    // abs obverloads

    inline double
    abs(double a)
    {
      return fabs(a);
    }

    inline float
    abs(float a)
    {
      return fabs(a);
    }

    inline double
    abs(std::complex<double> a)
    {
      return std::abs(a);
    }

    inline float
    abs(std::complex<float> a)
    {
      return std::abs(a);
    }

    //
    // conjugate overloads
    //

    inline unsigned int
    conj(unsigned int a)
    {
      return a;
    }

    inline unsigned long int
    conj(unsigned long int a)
    {
      return a;
    }

    inline int
    conj(int a)
    {
      return a;
    }

    inline float
    conj(float a)
    {
      return a;
    }

    inline double
    conj(double a)
    {
      return a;
    }

    inline std::complex<double>
    conj(std::complex<double> a)
    {
      return std::conj(a);
    }

    inline std::complex<float>
    conj(std::complex<float> a)
    {
      return std::conj(a);
    }


    //
    // mult for real homogeneous types e.g. (double, double)
    //
    inline unsigned int
    mult(unsigned int a, unsigned int b)
    {
      return a * b;
    }

    inline unsigned long int
    mult(unsigned long int a, unsigned long int b)
    {
      return a * b;
    }

    inline int
    mult(int a, int b)
    {
      return a * b;
    }

    inline double
    mult(double a, double b)
    {
      return a * b;
    }

    inline float
    mult(float a, float b)
    {
      return a * b;
    }

    inline double
    mult(float a, double b)
    {
      return a * b;
    }

    inline double
    mult(double a, float b)
    {
      return a * b;
    }


    //
    // mult for complex homogenous types
    // (e.g., std::complex<double> and std::complex<double>)
    //
    inline std::complex<double>
    mult(std::complex<double> a, std::complex<double> b)
    {
      return a * b;
    }

    inline std::complex<float>
    mult(std::complex<float> a, std::complex<float> b)
    {
      return a * b;
    }


    //
    // mult for complex heterogeneous types e.g. (std::complex<double>,
    // std::complex<float>)
    //
    inline std::complex<double>
    mult(std::complex<float> a, std::complex<double> b)
    {
      return std::complex<double>(a.real(), a.imag()) * b;
    }

    inline std::complex<double>
    mult(std::complex<double> a, std::complex<float> b)
    {
      return a * std::complex<double>(b.real(), b.imag());
    }


    //
    // mult for real-complex heterogeneous types e.g. (double,
    // std::complex<float>)
    //
    inline std::complex<double>
    mult(double a, std::complex<double> b)
    {
      return std::complex<double>(a * b.real(), a * b.imag());
    }

    inline std::complex<double>
    mult(std::complex<double> a, double b)
    {
      return std::complex<double>(b * a.real(), b * a.imag());
    }

    inline std::complex<float>
    mult(float a, std::complex<float> b)
    {
      return std::complex<float>(a * b.real(), a * b.imag());
    }

    inline std::complex<float>
    mult(std::complex<float> a, float b)
    {
      return std::complex<float>(b * a.real(), b * a.imag());
    }

    inline std::complex<double>
    mult(double a, std::complex<float> b)
    {
      return std::complex<double>(a * b.real(), a * b.imag());
    }

    inline std::complex<double>
    mult(std::complex<float> a, double b)
    {
      return std::complex<double>(b * a.real(), b * a.imag());
    }


    inline unsigned int
    add(unsigned int a, unsigned int b)
    {
      return a + b;
    }

    inline unsigned long int
    add(unsigned long int a, unsigned long int b)
    {
      return a + b;
    }

    inline int
    add(int a, int b)
    {
      return a + b;
    }

    inline double
    add(double a, double b)
    {
      return a + b;
    }

    inline float
    add(float a, float b)
    {
      return a + b;
    }

    inline std::complex<double>
    add(std::complex<double> a, std::complex<double> b)
    {
      return a + b;
    }


    inline std::complex<float>
    add(std::complex<float> a, std::complex<float> b)
    {
      return a + b;
    }

    inline double
    add(double a, float b)
    {
      return a + b;
    }

    inline double
    add(float a, double b)
    {
      return a + b;
    }

    inline std::complex<double>
    add(std::complex<double> a, std::complex<float> b)
    {
      return a + std::complex<double>(b.real(), b.imag());
    }


    inline std::complex<double>
    add(std::complex<float> a, std::complex<double> b)
    {
      return std::complex<double>(a.real(), a.imag()) + b;
    }


    inline unsigned int
    sub(unsigned int a, unsigned int b)
    {
      return a - b;
    }

    inline unsigned long int
    sub(unsigned long int a, unsigned long int b)
    {
      return a - b;
    }

    inline int
    sub(int a, int b)
    {
      return a - b;
    }

    inline double
    sub(double a, double b)
    {
      return a - b;
    }

    inline float
    sub(float a, float b)
    {
      return a - b;
    }

    inline std::complex<double>
    sub(std::complex<double> a, std::complex<double> b)
    {
      return a - b;
    }

    inline std::complex<float>
    sub(std::complex<float> a, std::complex<float> b)
    {
      return a - b;
    }

    inline unsigned int
    div(unsigned int a, unsigned int b)
    {
      return a / b;
    }

    inline unsigned long int
    div(unsigned long int a, unsigned long int b)
    {
      return a / b;
    }

    inline int
    div(int a, int b)
    {
      return a / b;
    }

    inline double
    div(double a, double b)
    {
      return a / b;
    }

    inline float
    div(float a, float b)
    {
      return a / b;
    }

    inline std::complex<double>
    div(std::complex<double> a, std::complex<double> b)
    {
      return a / b;
    }

    inline std::complex<float>
    div(std::complex<float> a, std::complex<float> b)
    {
      return a / b;
    }

    //
    // div for complex heterogeneous types e.g. (std::complex<double>,
    // std::complex<float>)
    //
    inline std::complex<double>
    div(std::complex<float> a, std::complex<double> b)
    {
      return std::complex<double>(a.real(), a.imag()) / b;
    }

    inline std::complex<double>
    div(std::complex<double> a, std::complex<float> b)
    {
      return a / std::complex<double>(b.real(), b.imag());
    }


    //
    // div for real-complex heterogeneous types e.g. (double,
    // std::complex<float>)
    //
    inline std::complex<double>
    div(double a, std::complex<double> b)
    {
      return std::complex<double>(a / b.real(), a / b.imag());
    }

    inline std::complex<double>
    div(std::complex<double> a, double b)
    {
      return std::complex<double>(b / a.real(), b / a.imag());
    }

    inline std::complex<float>
    div(float a, std::complex<float> b)
    {
      return std::complex<float>(a / b.real(), a / b.imag());
    }

    inline std::complex<float>
    div(std::complex<float> a, float b)
    {
      return std::complex<float>(b / a.real(), b / a.imag());
    }

    inline std::complex<double>
    div(double a, std::complex<float> b)
    {
      return std::complex<double>(a / b.real(), a / b.imag());
    }

    inline std::complex<double>
    div(std::complex<float> a, double b)
    {
      return std::complex<double>(b / a.real(), b / a.imag());
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

    inline long long int *
    makeDataTypeDeviceCompatible(long long int *a)
    {
      return a;
    }

    inline const long long int *
    makeDataTypeDeviceCompatible(const long long int *a)
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

    inline unsigned long long int *
    makeDataTypeDeviceCompatible(unsigned long long int *a)
    {
      return a;
    }

    inline const unsigned long long int *
    makeDataTypeDeviceCompatible(const unsigned long long int *a)
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

    inline std::complex<double> *
    makeDataTypeDeviceCompatible(std::complex<double> *a)
    {
      return a;
    }

    inline const std::complex<double> *
    makeDataTypeDeviceCompatible(const std::complex<double> *a)
    {
      return a;
    }

    inline std::complex<float> *
    makeDataTypeDeviceCompatible(std::complex<float> *a)
    {
      return a;
    }

    inline const std::complex<float> *
    makeDataTypeDeviceCompatible(const std::complex<float> *a)
    {
      return a;
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

    inline std::complex<double>
    makeDataTypeDeviceCompatible(std::complex<double> a)
    {
      return a;
    }

    inline std::complex<float>
    makeDataTypeDeviceCompatible(std::complex<float> a)
    {
      return a;
    }

    inline bool
    makeDataTypeDeviceCompatible(bool a)
    {
      return a;
    }

    inline bool *
    makeDataTypeDeviceCompatible(bool *a)
    {
      return a;
    }

  } // namespace utils

} // namespace dftfe

#endif
