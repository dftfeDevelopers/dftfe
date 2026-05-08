// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
//  GPU Block Floating Point (BFP) compression: encode/decode helpers,
//  GPU kernels, and the public dftfe::compressionWrapper entry points.
//  Specialised for bits_per_value in {8, 10, 12, 16} with a 1-thread-per-
//  4-value-block design (no atomics on compress, no cross-block packing).
//

#ifdef DFTFE_WITH_DEVICE

#  include <DeviceKernelLauncherHelpers.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceTypeConfig.h>
#  include <TypeConfig.h>
#  include <compressionWrapper.h>
#  include <cassert>
#  include <climits>
#  include <cmath>
#  include <complex>
#  include <cstddef>
#  include <cstdint>

namespace dftfe
{
  namespace compressionWrapper
  {
    namespace
    {
      using uint64 = std::uint64_t;

#  if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTFE_WITH_DEVICE_LANG_HIP)
#    define DFTFE_COMP_DEVICE_INLINE __device__ __forceinline__
#  else
#    define DFTFE_COMP_DEVICE_INLINE inline
#  endif

      /* ---- traits: IEEE 754 constants used by BFP quantisation ---- */
      template <typename ValueType>
      struct traits;

      template <>
      struct traits<float>
      {
        static constexpr int EBIAS  = 127;
        static constexpr int EBITS  = 8;
        static constexpr int PREC   = 32;
        static constexpr int MINEXP = -149;
      };

      template <>
      struct traits<double>
      {
        static constexpr int EBIAS  = 1023;
        static constexpr int EBITS  = 11;
        static constexpr int PREC   = 64;
        static constexpr int MINEXP = -1074;
      };

      /* ---- partial-block padding (< 4 values) ---- */
      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE void
      pad_block(ValueType *q, unsigned int n)
      {
        if (n == 0)
          q[0] = 0;
        if (n <= 1)
          q[1] = q[0];
        if (n <= 2)
          q[2] = q[1];
        if (n <= 3)
          q[3] = q[n <= 0 ? 0 : n - 1];
      }

      /* ---- exponent of largest absolute value in a block ---- */
      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE int
      block_exponent(ValueType x)
      {
        int e = -traits<ValueType>::EBIAS;
        if (x > 0)
          {
            dftfe::utils::frexp(x, &e);
            e = dftfe::utils::max(e, 1 - traits<ValueType>::EBIAS);
          }
        return e;
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE int
      max_exponent(const ValueType *p)
      {
        ValueType mx = 0;
        for (int i = 0; i < 4; i++)
          {
            ValueType f = dftfe::utils::abs(p[i]);
            mx       = dftfe::utils::max(mx, f);
          }
        return block_exponent<ValueType>(mx);
      }

      DFTFE_COMP_DEVICE_INLINE int
      calc_precision(int maxexp, int maxprec, int minexp)
      {
        return dftfe::utils::min(maxprec,
                                 dftfe::utils::max(0, maxexp - minexp + 8));
      }

      /* =====================================================================
         BFP encode / decode helpers (constexpr vbits per rate)

         Each block of 4 values packs into a fixed-width word:
           bpv == 8:   uint32_t       (32 bits)
           bpv == 10:  uint64 lo 40   (5 x uint8_t per block on the wire)
           bpv == 12:  uint64 lo 48   (3 x uint16_t per block on the wire)
           bpv == 16:  uint64         (64 bits)

         Layout per block (LSB first):
           [0 .. EBITS-1]    biased exponent (0 = zero block)
           [EBITS .. end]    4 x vbits-bit signed coefficients
         ==================================================================== */

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE unsigned int
      encode_block_32(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (32u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;

        int emax    = max_exponent<ValueType>(fblock);
        int maxprec = calc_precision(emax,
                                     traits<ValueType>::PREC,
                                     traits<ValueType>::MINEXP);
        unsigned int e =
          maxprec ? (unsigned int)(emax + traits<ValueType>::EBIAS) : 0u;
        if (!e)
          return 0u;

        unsigned int packed = e;
        ValueType       s =
          dftfe::utils::ldexp((ValueType)1.0, (int)vbits - 1 - emax);
        const int qmax = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int q_raw = (int)rint(s * fblock[i]);
            const int q     = q_raw > qmax ? qmax : q_raw;
            packed |= ((unsigned int)q & vmask)
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE void
      decode_block_32(unsigned int packed, ValueType *fblock)
      {
        constexpr unsigned int ebits          = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits          = (32u - ebits) / 4u;
        constexpr unsigned int vmask          = (1u << vbits) - 1u;
        constexpr int          sign_threshold = 1 << (vbits - 1);
        constexpr unsigned int emask          = (1u << ebits) - 1u;

        unsigned int e_raw = packed & emask;
        if (!e_raw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int    emax  = (int)e_raw - traits<ValueType>::EBIAS;
        ValueType scale = dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw =
              (packed >> (ebits + (unsigned int)i * vbits)) & vmask;
            int q = (int)raw;
            if (q >= sign_threshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE uint64
      encode_block_40(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (40u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;

        int emax    = max_exponent<ValueType>(fblock);
        int maxprec = calc_precision(emax,
                                     traits<ValueType>::PREC,
                                     traits<ValueType>::MINEXP);
        unsigned int e =
          maxprec ? (unsigned int)(emax + traits<ValueType>::EBIAS) : 0u;
        if (!e)
          return (uint64)0;

        uint64    packed = (uint64)e;
        ValueType    s      = dftfe::utils::ldexp((ValueType)1.0,
                                          (int)vbits - 1 - emax);
        const int qmax   = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int q_raw = (int)rint(s * fblock[i]);
            const int q     = q_raw > qmax ? qmax : q_raw;
            packed |= ((uint64)((unsigned int)q & vmask))
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE void
      decode_block_40(uint64 packed, ValueType *fblock)
      {
        constexpr unsigned int ebits          = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits          = (40u - ebits) / 4u;
        constexpr unsigned int vmask          = (1u << vbits) - 1u;
        constexpr int          sign_threshold = 1 << (vbits - 1);
        constexpr unsigned int emask          = (1u << ebits) - 1u;

        unsigned int e_raw = (unsigned int)(packed & emask);
        if (!e_raw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int    emax  = (int)e_raw - traits<ValueType>::EBIAS;
        ValueType scale = dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw = (unsigned int)(
              (packed >> (ebits + (unsigned int)i * vbits)) & vmask);
            int q = (int)raw;
            if (q >= sign_threshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE uint64
      encode_block_48(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (48u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;

        int emax    = max_exponent<ValueType>(fblock);
        int maxprec = calc_precision(emax,
                                     traits<ValueType>::PREC,
                                     traits<ValueType>::MINEXP);
        unsigned int e =
          maxprec ? (unsigned int)(emax + traits<ValueType>::EBIAS) : 0u;
        if (!e)
          return (uint64)0;

        uint64    packed = (uint64)e;
        ValueType    s      = dftfe::utils::ldexp((ValueType)1.0,
                                          (int)vbits - 1 - emax);
        const int qmax   = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int q_raw = (int)rint(s * fblock[i]);
            const int q     = q_raw > qmax ? qmax : q_raw;
            packed |= ((uint64)((unsigned int)q & vmask))
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE void
      decode_block_48(uint64 packed, ValueType *fblock)
      {
        constexpr unsigned int ebits          = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits          = (48u - ebits) / 4u;
        constexpr unsigned int vmask          = (1u << vbits) - 1u;
        constexpr int          sign_threshold = 1 << (vbits - 1);
        constexpr unsigned int emask          = (1u << ebits) - 1u;

        unsigned int e_raw = (unsigned int)(packed & emask);
        if (!e_raw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int    emax  = (int)e_raw - traits<ValueType>::EBIAS;
        ValueType scale = dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw = (unsigned int)(
              (packed >> (ebits + (unsigned int)i * vbits)) & vmask);
            int q = (int)raw;
            if (q >= sign_threshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE uint64
      encode_block_64(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (64u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;

        int emax    = max_exponent<ValueType>(fblock);
        int maxprec = calc_precision(emax,
                                     traits<ValueType>::PREC,
                                     traits<ValueType>::MINEXP);
        unsigned int e =
          maxprec ? (unsigned int)(emax + traits<ValueType>::EBIAS) : 0u;
        if (!e)
          return (uint64)0;

        uint64    packed = (uint64)e;
        ValueType    s      = dftfe::utils::ldexp((ValueType)1.0,
                                          (int)vbits - 1 - emax);
        const int qmax   = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int q_raw = (int)rint(s * fblock[i]);
            const int q     = q_raw > qmax ? qmax : q_raw;
            packed |= ((uint64)((unsigned int)q & vmask))
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_COMP_DEVICE_INLINE void
      decode_block_64(uint64 packed, ValueType *fblock)
      {
        constexpr unsigned int ebits          = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits          = (64u - ebits) / 4u;
        constexpr unsigned int vmask          = (1u << vbits) - 1u;
        constexpr int          sign_threshold = 1 << (vbits - 1);
        constexpr unsigned int emask          = (1u << ebits) - 1u;

        unsigned int e_raw = (unsigned int)(packed & emask);
        if (!e_raw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int    emax  = (int)e_raw - traits<ValueType>::EBIAS;
        ValueType scale = dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw = (unsigned int)(
              (packed >> (ebits + (unsigned int)i * vbits)) & vmask);
            int q = (int)raw;
            if (q >= sign_threshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      /* =====================================================================
         GPU kernels — 1 thread per 4-value block.
         Use DFTFE_CREATE_KERNEL so the same source covers CUDA / HIP / SYCL.
         ==================================================================== */

      /* ---- 8 bpv: 1 x uint32_t per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_8_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          unsigned int block_start = block_idx * 4u;
          ValueType       fblock[4];
          if (block_start + 4u <= dim)
            {
              fblock[0] = data[block_start];
              fblock[1] = data[block_start + 1];
              fblock[2] = data[block_start + 2];
              fblock[3] = data[block_start + 3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[block_start + j];
              pad_block(fblock, nx);
            }
          stream[block_idx] = encode_block_32(fblock);
        },
        const ValueType *data,
        unsigned int *stream,
        unsigned int  dim,
        unsigned int  tot_blocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_8_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          ValueType fblock[4];
          decode_block_32(stream[block_idx], fblock);
          unsigned int block_start = block_idx * 4u;
          if (block_start + 4u <= dim)
            {
              data[block_start]     = fblock[0];
              data[block_start + 1] = fblock[1];
              data[block_start + 2] = fblock[2];
              data[block_start + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                data[block_start + j] = fblock[j];
            }
        },
        const unsigned int *stream,
        ValueType             *data,
        unsigned int        dim,
        unsigned int        tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_gather_8_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0]         = dataArray[base];
          fblock[1]         = dataArray[base + 1];
          fblock[2]         = dataArray[base + 2];
          fblock[3]         = dataArray[base + 3];
          stream[block_idx] = encode_block_32(fblock);
        },
        const ValueType    *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        unsigned int    *stream,
        unsigned int     tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_scatter_add_8_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          ValueType fblock[4];
          decode_block_32(stream[block_idx], fblock);

          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const unsigned int *stream,
        const IndexType    *indices,
        unsigned int        gatherBlockSize,
        ValueType             *dataArray,
        unsigned int        tot_blocks);

      /* ---- 10 bpv: 5 x uint8_t per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_10_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          unsigned int block_start = block_idx * 4u;
          ValueType       fblock[4];
          if (block_start + 4u <= dim)
            {
              fblock[0] = data[block_start];
              fblock[1] = data[block_start + 1];
              fblock[2] = data[block_start + 2];
              fblock[3] = data[block_start + 3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[block_start + j];
              pad_block(fblock, nx);
            }
          uint64 packed   = encode_block_40(fblock);
          size_t out      = (size_t)block_idx * 5u;
          stream[out]     = (uint8_t)(packed);
          stream[out + 1] = (uint8_t)(packed >> 8);
          stream[out + 2] = (uint8_t)(packed >> 16);
          stream[out + 3] = (uint8_t)(packed >> 24);
          stream[out + 4] = (uint8_t)(packed >> 32);
        },
        const ValueType *data,
        uint8_t      *stream,
        unsigned int  dim,
        unsigned int  tot_blocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_10_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          size_t out    = (size_t)block_idx * 5u;
          uint64 packed = (uint64)stream[out] |
                          ((uint64)stream[out + 1] << 8) |
                          ((uint64)stream[out + 2] << 16) |
                          ((uint64)stream[out + 3] << 24) |
                          ((uint64)stream[out + 4] << 32);

          ValueType fblock[4];
          decode_block_40(packed, fblock);

          unsigned int block_start = block_idx * 4u;
          if (block_start + 4u <= dim)
            {
              data[block_start]     = fblock[0];
              data[block_start + 1] = fblock[1];
              data[block_start + 2] = fblock[2];
              data[block_start + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                data[block_start + j] = fblock[j];
            }
        },
        const uint8_t *stream,
        ValueType        *data,
        unsigned int   dim,
        unsigned int   tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_gather_10_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0] = dataArray[base];
          fblock[1] = dataArray[base + 1];
          fblock[2] = dataArray[base + 2];
          fblock[3] = dataArray[base + 3];

          uint64 packed   = encode_block_40(fblock);
          size_t out      = (size_t)block_idx * 5u;
          stream[out]     = (uint8_t)(packed);
          stream[out + 1] = (uint8_t)(packed >> 8);
          stream[out + 2] = (uint8_t)(packed >> 16);
          stream[out + 3] = (uint8_t)(packed >> 24);
          stream[out + 4] = (uint8_t)(packed >> 32);
        },
        const ValueType    *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        uint8_t         *stream,
        unsigned int     tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_scatter_add_10_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          size_t out    = (size_t)block_idx * 5u;
          uint64 packed = (uint64)stream[out] |
                          ((uint64)stream[out + 1] << 8) |
                          ((uint64)stream[out + 2] << 16) |
                          ((uint64)stream[out + 3] << 24) |
                          ((uint64)stream[out + 4] << 32);

          ValueType fblock[4];
          decode_block_40(packed, fblock);

          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const uint8_t   *stream,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        ValueType          *dataArray,
        unsigned int     tot_blocks);

      /* ---- 12 bpv: 3 x uint16_t per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_12_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          unsigned int block_start = block_idx * 4u;
          ValueType       fblock[4];
          if (block_start + 4u <= dim)
            {
              fblock[0] = data[block_start];
              fblock[1] = data[block_start + 1];
              fblock[2] = data[block_start + 2];
              fblock[3] = data[block_start + 3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[block_start + j];
              pad_block(fblock, nx);
            }
          uint64 packed   = encode_block_48(fblock);
          size_t out      = (size_t)block_idx * 3u;
          stream[out]     = (uint16_t)(packed);
          stream[out + 1] = (uint16_t)(packed >> 16);
          stream[out + 2] = (uint16_t)(packed >> 32);
        },
        const ValueType *data,
        uint16_t     *stream,
        unsigned int  dim,
        unsigned int  tot_blocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_12_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          size_t out    = (size_t)block_idx * 3u;
          uint64 packed = (uint64)stream[out] |
                          ((uint64)stream[out + 1] << 16) |
                          ((uint64)stream[out + 2] << 32);

          ValueType fblock[4];
          decode_block_48(packed, fblock);

          unsigned int block_start = block_idx * 4u;
          if (block_start + 4u <= dim)
            {
              data[block_start]     = fblock[0];
              data[block_start + 1] = fblock[1];
              data[block_start + 2] = fblock[2];
              data[block_start + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                data[block_start + j] = fblock[j];
            }
        },
        const uint16_t *stream,
        ValueType         *data,
        unsigned int    dim,
        unsigned int    tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_gather_12_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0] = dataArray[base];
          fblock[1] = dataArray[base + 1];
          fblock[2] = dataArray[base + 2];
          fblock[3] = dataArray[base + 3];

          uint64 packed   = encode_block_48(fblock);
          size_t out      = (size_t)block_idx * 3u;
          stream[out]     = (uint16_t)(packed);
          stream[out + 1] = (uint16_t)(packed >> 16);
          stream[out + 2] = (uint16_t)(packed >> 32);
        },
        const ValueType    *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        uint16_t        *stream,
        unsigned int     tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_scatter_add_12_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          size_t out    = (size_t)block_idx * 3u;
          uint64 packed = (uint64)stream[out] |
                          ((uint64)stream[out + 1] << 16) |
                          ((uint64)stream[out + 2] << 32);

          ValueType fblock[4];
          decode_block_48(packed, fblock);

          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const uint16_t  *stream,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        ValueType          *dataArray,
        unsigned int     tot_blocks);

      /* ---- 16 bpv: 1 x uint64 per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_16_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          unsigned int block_start = block_idx * 4u;
          ValueType       fblock[4];
          if (block_start + 4u <= dim)
            {
              fblock[0] = data[block_start];
              fblock[1] = data[block_start + 1];
              fblock[2] = data[block_start + 2];
              fblock[3] = data[block_start + 3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[block_start + j];
              pad_block(fblock, nx);
            }
          stream[block_idx] = encode_block_64(fblock);
        },
        const ValueType *data,
        uint64       *stream,
        unsigned int  dim,
        unsigned int  tot_blocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_16_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          ValueType fblock[4];
          decode_block_64(stream[block_idx], fblock);
          unsigned int block_start = block_idx * 4u;
          if (block_start + 4u <= dim)
            {
              data[block_start]     = fblock[0];
              data[block_start + 1] = fblock[1];
              data[block_start + 2] = fblock[2];
              data[block_start + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - block_start;
              for (unsigned int j = 0; j < nx; j++)
                data[block_start + j] = fblock[j];
            }
        },
        const uint64 *stream,
        ValueType       *data,
        unsigned int  dim,
        unsigned int  tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compress_gather_16_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0]         = dataArray[base];
          fblock[1]         = dataArray[base + 1];
          fblock[2]         = dataArray[base + 2];
          fblock[3]         = dataArray[base + 3];
          stream[block_idx] = encode_block_64(fblock);
        },
        const ValueType    *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        uint64          *stream,
        unsigned int     tot_blocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress_scatter_add_16_kernel,
        {
          const unsigned int block_idx = (unsigned int)globalThreadId;
          if (block_idx >= tot_blocks)
            return;
          ValueType fblock[4];
          decode_block_64(stream[block_idx], fblock);

          const unsigned int blocks_per_entry = gatherBlockSize >> 2;
          unsigned int       gatherIdx        = block_idx / blocks_per_entry;
          unsigned int       localBlock       = block_idx - gatherIdx * blocks_per_entry;
          unsigned int       intraIdx         = localBlock * 4u;
          size_t             base =
            (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const uint64    *stream,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        ValueType          *dataArray,
        unsigned int     tot_blocks);

#  undef DFTFE_COMP_DEVICE_INLINE

      /* =====================================================================
         Internal dispatch: switches on bits_per_value, launches the matching
         specialised kernel via DFTFE_LAUNCH_KERNEL.
         ==================================================================== */

#  define DFTFE_COMP_ASSERT_BPV(bpv)                              \
    assert(((bpv) == 8 || (bpv) == 10 || (bpv) == 12 ||           \
            (bpv) == 16) &&                                       \
           "bits_per_value must be 8, 10, 12, or 16")

      template <typename ValueType>
      void
      compress_impl(const ValueType                *d_data,
                    void                        *d_stream,
                    size_t                       num_values,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
      {
        if (num_values == 0)
          return;
        assert(num_values <= (size_t)UINT_MAX &&
               "num_values exceeds 32-bit limit");
        DFTFE_COMP_ASSERT_BPV(bits_per_value);

        const unsigned int dim        = (unsigned int)num_values;
        const unsigned int num_blocks = (dim + 3u) / 4u;
        const unsigned int grid       = (num_blocks +
                                   dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bits_per_value)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL((compress_8_kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  d_data,
                                  reinterpret_cast<unsigned int *>(d_stream),
                                  dim,
                                  num_blocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL((compress_10_kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  d_data,
                                  reinterpret_cast<uint8_t *>(d_stream),
                                  dim,
                                  num_blocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL((compress_12_kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  d_data,
                                  reinterpret_cast<uint16_t *>(d_stream),
                                  dim,
                                  num_blocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL((compress_16_kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  d_data,
                                  reinterpret_cast<uint64 *>(d_stream),
                                  dim,
                                  num_blocks);
              break;
          }
      }

      template <typename ValueType>
      void
      decompress_impl(const void                  *d_stream,
                      ValueType                      *d_data,
                      size_t                       num_values,
                      int                          bits_per_value,
                      dftfe::utils::deviceStream_t stream)
      {
        if (num_values == 0)
          return;
        assert(num_values <= (size_t)UINT_MAX &&
               "num_values exceeds 32-bit limit");
        DFTFE_COMP_ASSERT_BPV(bits_per_value);

        const unsigned int dim        = (unsigned int)num_values;
        const unsigned int num_blocks = (dim + 3u) / 4u;
        const unsigned int grid       = (num_blocks +
                                   dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bits_per_value)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL(
                (decompress_8_kernel<ValueType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const unsigned int *>(d_stream),
                d_data,
                dim,
                num_blocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL((decompress_10_kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  reinterpret_cast<const uint8_t *>(d_stream),
                                  d_data,
                                  dim,
                                  num_blocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL(
                (decompress_12_kernel<ValueType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint16_t *>(d_stream),
                d_data,
                dim,
                num_blocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL((decompress_16_kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  reinterpret_cast<const uint64 *>(d_stream),
                                  d_data,
                                  dim,
                                  num_blocks);
              break;
          }
      }

      template <typename ValueType, typename IndexType>
      void
      compress_gather_impl(const ValueType                *dataArray,
                           const IndexType             *indices,
                           size_t                       num_indices,
                           unsigned int                 gather_block_size,
                           void                        *d_stream,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
      {
        const size_t num_values = num_indices * gather_block_size;
        if (num_values == 0)
          return;
        assert(num_values <= (size_t)UINT_MAX &&
               "num_values exceeds 32-bit limit");
        DFTFE_COMP_ASSERT_BPV(bits_per_value);

        const unsigned int dim        = (unsigned int)num_values;
        const unsigned int num_blocks = (dim + 3u) / 4u;
        const unsigned int grid       = (num_blocks +
                                   dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bits_per_value)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL(
                (compress_gather_8_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gather_block_size,
                reinterpret_cast<unsigned int *>(d_stream),
                num_blocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL(
                (compress_gather_10_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gather_block_size,
                reinterpret_cast<uint8_t *>(d_stream),
                num_blocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL(
                (compress_gather_12_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gather_block_size,
                reinterpret_cast<uint16_t *>(d_stream),
                num_blocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL(
                (compress_gather_16_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gather_block_size,
                reinterpret_cast<uint64 *>(d_stream),
                num_blocks);
              break;
          }
      }

      template <typename ValueType, typename IndexType>
      void
      decompress_scatter_add_impl(const void      *d_stream,
                                  const IndexType *indices,
                                  size_t           num_indices,
                                  unsigned int     gather_block_size,
                                  ValueType          *dataArray,
                                  int              bits_per_value,
                                  dftfe::utils::deviceStream_t stream)
      {
        const size_t num_values = num_indices * gather_block_size;
        if (num_values == 0)
          return;
        assert(num_values <= (size_t)UINT_MAX &&
               "num_values exceeds 32-bit limit");
        DFTFE_COMP_ASSERT_BPV(bits_per_value);

        const unsigned int dim        = (unsigned int)num_values;
        const unsigned int num_blocks = (dim + 3u) / 4u;
        const unsigned int grid       = (num_blocks +
                                   dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bits_per_value)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL(
                (decompress_scatter_add_8_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const unsigned int *>(d_stream),
                indices,
                gather_block_size,
                dataArray,
                num_blocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL(
                (decompress_scatter_add_10_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint8_t *>(d_stream),
                indices,
                gather_block_size,
                dataArray,
                num_blocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL(
                (decompress_scatter_add_12_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint16_t *>(d_stream),
                indices,
                gather_block_size,
                dataArray,
                num_blocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL(
                (decompress_scatter_add_16_kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint64 *>(d_stream),
                indices,
                gather_block_size,
                dataArray,
                num_blocks);
              break;
          }
      }

#  undef DFTFE_COMP_ASSERT_BPV

    } // anonymous namespace

    /* =======================================================================
       Public API — non-templated overloads forward to the *_impl templates.
       Complex<T> overloads are inline in compressionWrapper.h and forward
       via reinterpret_cast to the real-T overloads here.

       The double overloads exist only because MPICommunicatorP2P is
       explicitly instantiated for ValueType = double / complex<double> and
       its compress branch is a runtime if (d_commPrecision == compress);
       those calls must resolve at instantiation time. At runtime the
       COMPRESSED branch is only set on FP32 multivectors, so the double
       overloads here link but are never executed.
       ====================================================================== */

    void
    compress(const double                *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream)
    {
      compress_impl<double>(
        d_data, d_compressed, num_values, bits_per_value, stream);
    }

    void
    compress(const float                 *d_data,
             void                        *d_compressed,
             size_t                       num_values,
             int                          bits_per_value,
             dftfe::utils::deviceStream_t stream)
    {
      compress_impl<float>(
        d_data, d_compressed, num_values, bits_per_value, stream);
    }

    void
    decompress(const void                  *d_compressed,
               double                      *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream)
    {
      decompress_impl<double>(
        d_compressed, d_data, num_values, bits_per_value, stream);
    }

    void
    decompress(const void                  *d_compressed,
               float                       *d_data,
               size_t                       num_values,
               int                          bits_per_value,
               dftfe::utils::deviceStream_t stream)
    {
      decompress_impl<float>(
        d_compressed, d_data, num_values, bits_per_value, stream);
    }

    void
    compress_gather(const double                *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
    {
      compress_gather_impl<double, dftfe::uInt>(dataArray,
                                                indices,
                                                num_indices,
                                                (unsigned int)gather_block_size,
                                                d_compressed,
                                                bits_per_value,
                                                stream);
    }

    void
    compress_gather(const float                 *dataArray,
                    const dftfe::uInt           *indices,
                    size_t                       num_indices,
                    dftfe::uInt                  gather_block_size,
                    void                        *d_compressed,
                    int                          bits_per_value,
                    dftfe::utils::deviceStream_t stream)
    {
      compress_gather_impl<float, dftfe::uInt>(dataArray,
                                               indices,
                                               num_indices,
                                               (unsigned int)gather_block_size,
                                               d_compressed,
                                               bits_per_value,
                                               stream);
    }

    void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           double                      *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
    {
      decompress_scatter_add_impl<double, dftfe::uInt>(
        d_compressed,
        indices,
        num_indices,
        (unsigned int)gather_block_size,
        dataArray,
        bits_per_value,
        stream);
    }

    void
    decompress_scatter_add(const void                  *d_compressed,
                           const dftfe::uInt           *indices,
                           size_t                       num_indices,
                           dftfe::uInt                  gather_block_size,
                           float                       *dataArray,
                           int                          bits_per_value,
                           dftfe::utils::deviceStream_t stream)
    {
      decompress_scatter_add_impl<float, dftfe::uInt>(
        d_compressed,
        indices,
        num_indices,
        (unsigned int)gather_block_size,
        dataArray,
        bits_per_value,
        stream);
    }

  } // namespace compressionWrapper
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
