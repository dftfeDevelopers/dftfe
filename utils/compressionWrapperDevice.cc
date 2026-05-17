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
//  4-value-block design (no atomics, no cross-block packing).

#ifdef DFTFE_WITH_DEVICE

#  include <DeviceKernelLauncherHelpers.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceTypeConfig.h>
#  include <TypeConfig.h>
#  include <compressionWrapper.h>
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
#    define DFTFE_DEVICE_INLINE __device__ __forceinline__
#  elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
#    define DFTFE_DEVICE_INLINE inline
#  endif

      template <typename ValueType>
      struct traits;

      template <>
      struct traits<float>
      {
        static constexpr int EBIAS = 127;
        static constexpr int EBITS = 8;
      };

      template <>
      struct traits<double>
      {
        static constexpr int EBIAS = 1023;
        static constexpr int EBITS = 11;
      };

      /* ---- partial-block padding (< 4 values) ---- */
      template <typename ValueType>
      DFTFE_DEVICE_INLINE void
      padBlock(ValueType *q, unsigned int n)
      {
        for (unsigned int i = n; i < 4; ++i)
          q[i] = ValueType(0);
      }

      /* ---- exponent of largest absolute value in a block ---- */
      template <typename ValueType>
      DFTFE_DEVICE_INLINE int
      blockExponent(ValueType x)
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
      DFTFE_DEVICE_INLINE int
      maxExponent(const ValueType *p)
      {
        ValueType mx = 0;
        for (int i = 0; i < 4; i++)
          {
            ValueType f = dftfe::utils::abs(p[i]);
            mx          = dftfe::utils::max(mx, f);
          }
        return blockExponent<ValueType>(mx);
      }

      /* =====================================================================
         BFP encode / decode helpers (constexpr vbits per rate)

         Each block of 4 values packs into a fixed-width word:
           bpv == 8:   uint32_t       (32 bits)
           bpv == 10:  uint64 lo 40   (5 x uint8_t = 40 bits)
           bpv == 12:  uint64 lo 48   (3 x uint16_t = 48 bits)
           bpv == 16:  uint64         (64 bits)

         Layout per block (LSB first):
           [0 .. EBITS-1]    biased exponent (0 = zero block)
           [EBITS .. end]    4 x vbits-bit signed coefficients
         ==================================================================== */

      template <typename ValueType>
      DFTFE_DEVICE_INLINE unsigned int
      encodeBlock32(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (32u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int minEmax = (int)vbits - 1 - traits<ValueType>::EBIAS;

        int emax = maxExponent<ValueType>(fblock);
        if (emax < minEmax)
          return 0u;

        unsigned int packed = (unsigned int)(emax + traits<ValueType>::EBIAS);
        ValueType    s =
          dftfe::utils::ldexp((ValueType)1.0, (int)vbits - 1 - emax);
        const int qmax = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int qRaw = (int)dftfe::utils::rint(s * fblock[i]);
            const int q    = qRaw > qmax ? qmax : qRaw;
            packed |= ((unsigned int)q & vmask)
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE void
      decodeBlock32(unsigned int packed, ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (32u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int          signThreshold = 1 << (vbits - 1);
        constexpr unsigned int emask         = (1u << ebits) - 1u;

        unsigned int eRaw = packed & emask;
        if (!eRaw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int       emax = (int)eRaw - traits<ValueType>::EBIAS;
        ValueType scale =
          dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw =
              (packed >> (ebits + (unsigned int)i * vbits)) & vmask;
            int q = (int)raw;
            if (q >= signThreshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE uint64
      encodeBlock40(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (40u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int minEmax = (int)vbits - 1 - traits<ValueType>::EBIAS;

        int emax = maxExponent<ValueType>(fblock);
        if (emax < minEmax)
          return (uint64)0;

        uint64    packed = (uint64)(emax + traits<ValueType>::EBIAS);
        ValueType s =
          dftfe::utils::ldexp((ValueType)1.0, (int)vbits - 1 - emax);
        const int qmax = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int qRaw = (int)dftfe::utils::rint(s * fblock[i]);
            const int q    = qRaw > qmax ? qmax : qRaw;
            packed |= ((uint64)((unsigned int)q & vmask))
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE void
      decodeBlock40(uint64 packed, ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (40u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int          signThreshold = 1 << (vbits - 1);
        constexpr unsigned int emask         = (1u << ebits) - 1u;

        unsigned int eRaw = (unsigned int)(packed & emask);
        if (!eRaw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int       emax = (int)eRaw - traits<ValueType>::EBIAS;
        ValueType scale =
          dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw =
              (unsigned int)((packed >> (ebits + (unsigned int)i * vbits)) &
                             vmask);
            int q = (int)raw;
            if (q >= signThreshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE uint64
      encodeBlock48(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (48u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int minEmax = (int)vbits - 1 - traits<ValueType>::EBIAS;

        int emax = maxExponent<ValueType>(fblock);
        if (emax < minEmax)
          return (uint64)0;

        uint64    packed = (uint64)(emax + traits<ValueType>::EBIAS);
        ValueType s =
          dftfe::utils::ldexp((ValueType)1.0, (int)vbits - 1 - emax);
        const int qmax = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int qRaw = (int)dftfe::utils::rint(s * fblock[i]);
            const int q    = qRaw > qmax ? qmax : qRaw;
            packed |= ((uint64)((unsigned int)q & vmask))
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE void
      decodeBlock48(uint64 packed, ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (48u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int          signThreshold = 1 << (vbits - 1);
        constexpr unsigned int emask         = (1u << ebits) - 1u;

        unsigned int eRaw = (unsigned int)(packed & emask);
        if (!eRaw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int       emax = (int)eRaw - traits<ValueType>::EBIAS;
        ValueType scale =
          dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw =
              (unsigned int)((packed >> (ebits + (unsigned int)i * vbits)) &
                             vmask);
            int q = (int)raw;
            if (q >= signThreshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE uint64
      encodeBlock64(ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (64u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int minEmax = (int)vbits - 1 - traits<ValueType>::EBIAS;

        int emax = maxExponent<ValueType>(fblock);
        if (emax < minEmax)
          return (uint64)0;

        uint64    packed = (uint64)(emax + traits<ValueType>::EBIAS);
        ValueType s =
          dftfe::utils::ldexp((ValueType)1.0, (int)vbits - 1 - emax);
        const int qmax = (int)(vmask >> 1u);
        for (int i = 0; i < 4; i++)
          {
            const int qRaw = (int)dftfe::utils::rint(s * fblock[i]);
            const int q    = qRaw > qmax ? qmax : qRaw;
            packed |= ((uint64)((unsigned int)q & vmask))
                      << (ebits + (unsigned int)i * vbits);
          }
        return packed;
      }

      template <typename ValueType>
      DFTFE_DEVICE_INLINE void
      decodeBlock64(uint64 packed, ValueType *fblock)
      {
        constexpr unsigned int ebits = (unsigned int)traits<ValueType>::EBITS;
        constexpr unsigned int vbits = (64u - ebits) / 4u;
        constexpr unsigned int vmask = (1u << vbits) - 1u;
        constexpr int          signThreshold = 1 << (vbits - 1);
        constexpr unsigned int emask         = (1u << ebits) - 1u;

        unsigned int eRaw = (unsigned int)(packed & emask);
        if (!eRaw)
          {
            fblock[0] = fblock[1] = fblock[2] = fblock[3] = (ValueType)0;
            return;
          }

        int       emax = (int)eRaw - traits<ValueType>::EBIAS;
        ValueType scale =
          dftfe::utils::ldexp((ValueType)1.0, emax - (int)vbits + 1);
        for (int i = 0; i < 4; i++)
          {
            unsigned int raw =
              (unsigned int)((packed >> (ebits + (unsigned int)i * vbits)) &
                             vmask);
            int q = (int)raw;
            if (q >= signThreshold)
              q -= (1 << vbits);
            fblock[i] = scale * (ValueType)q;
          }
      }

      /* GPU kernels — 1 thread per 4-value block.
         Use DFTFE_CREATE_KERNEL so the same source covers CUDA / HIP / SYCL */

      /* ---- 8 bpv: 1 x uint32_t per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress8Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          unsigned int blockStart = blockIdx * 4u;
          ValueType    fblock[4];
          if (blockStart + 4u <= dim)
            {
              fblock[0] = data[blockStart];
              fblock[1] = data[blockStart + 1];
              fblock[2] = data[blockStart + 2];
              fblock[3] = data[blockStart + 3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[blockStart + j];
              padBlock(fblock, nx);
            }
          stream[blockIdx] = encodeBlock32(fblock);
        },
        const ValueType *data,
        unsigned int    *stream,
        unsigned int     dim,
        unsigned int     totBlocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress8Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          ValueType fblock[4];
          decodeBlock32(stream[blockIdx], fblock);
          unsigned int blockStart = blockIdx * 4u;
          if (blockStart + 4u <= dim)
            {
              data[blockStart]     = fblock[0];
              data[blockStart + 1] = fblock[1];
              data[blockStart + 2] = fblock[2];
              data[blockStart + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                data[blockStart + j] = fblock[j];
            }
        },
        const unsigned int *stream,
        ValueType          *data,
        unsigned int        dim,
        unsigned int        totBlocks);

      /* gatherBlockSize must be a multiple of 4 (blocksPerEntry = size>>2, no
         partial-block path) */
      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compressGather8Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0]        = dataArray[base];
          fblock[1]        = dataArray[base + 1];
          fblock[2]        = dataArray[base + 2];
          fblock[3]        = dataArray[base + 3];
          stream[blockIdx] = encodeBlock32(fblock);
        },
        const ValueType *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        unsigned int    *stream,
        unsigned int     totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompressScatterAdd8Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          ValueType fblock[4];
          decodeBlock32(stream[blockIdx], fblock);

          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const unsigned int *stream,
        const IndexType    *indices,
        unsigned int        gatherBlockSize,
        ValueType          *dataArray,
        unsigned int        totBlocks);

      /* ---- 10 bpv: 5 x uint8_t per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress10Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          unsigned int blockStart = blockIdx * 4u;
          ValueType    fblock[4];
          if (blockStart + 4u <= dim)
            {
              fblock[0] = data[blockStart];
              fblock[1] = data[blockStart + 1];
              fblock[2] = data[blockStart + 2];
              fblock[3] = data[blockStart + 3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[blockStart + j];
              padBlock(fblock, nx);
            }
          uint64 packed   = encodeBlock40(fblock);
          size_t out      = (size_t)blockIdx * 5u;
          stream[out]     = (uint8_t)(packed);
          stream[out + 1] = (uint8_t)(packed >> 8);
          stream[out + 2] = (uint8_t)(packed >> 16);
          stream[out + 3] = (uint8_t)(packed >> 24);
          stream[out + 4] = (uint8_t)(packed >> 32);
        },
        const ValueType *data,
        uint8_t         *stream,
        unsigned int     dim,
        unsigned int     totBlocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress10Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          size_t out    = (size_t)blockIdx * 5u;
          uint64 packed = (uint64)stream[out] | ((uint64)stream[out + 1] << 8) |
                          ((uint64)stream[out + 2] << 16) |
                          ((uint64)stream[out + 3] << 24) |
                          ((uint64)stream[out + 4] << 32);

          ValueType fblock[4];
          decodeBlock40(packed, fblock);
          unsigned int blockStart = blockIdx * 4u;
          if (blockStart + 4u <= dim)
            {
              data[blockStart]     = fblock[0];
              data[blockStart + 1] = fblock[1];
              data[blockStart + 2] = fblock[2];
              data[blockStart + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                data[blockStart + j] = fblock[j];
            }
        },
        const uint8_t *stream,
        ValueType     *data,
        unsigned int   dim,
        unsigned int   totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compressGather10Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0] = dataArray[base];
          fblock[1] = dataArray[base + 1];
          fblock[2] = dataArray[base + 2];
          fblock[3] = dataArray[base + 3];

          uint64 packed   = encodeBlock40(fblock);
          size_t out      = (size_t)blockIdx * 5u;
          stream[out]     = (uint8_t)(packed);
          stream[out + 1] = (uint8_t)(packed >> 8);
          stream[out + 2] = (uint8_t)(packed >> 16);
          stream[out + 3] = (uint8_t)(packed >> 24);
          stream[out + 4] = (uint8_t)(packed >> 32);
        },
        const ValueType *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        uint8_t         *stream,
        unsigned int     totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompressScatterAdd10Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          size_t out    = (size_t)blockIdx * 5u;
          uint64 packed = (uint64)stream[out] | ((uint64)stream[out + 1] << 8) |
                          ((uint64)stream[out + 2] << 16) |
                          ((uint64)stream[out + 3] << 24) |
                          ((uint64)stream[out + 4] << 32);

          ValueType fblock[4];
          decodeBlock40(packed, fblock);

          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const uint8_t   *stream,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        ValueType       *dataArray,
        unsigned int     totBlocks);

      /* ---- 12 bpv: 3 x uint16_t per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress12Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          unsigned int blockStart = blockIdx * 4u;
          ValueType    fblock[4];
          if (blockStart + 4u <= dim)
            {
              fblock[0] = data[blockStart];
              fblock[1] = data[blockStart + 1];
              fblock[2] = data[blockStart + 2];
              fblock[3] = data[blockStart + 3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[blockStart + j];
              padBlock(fblock, nx);
            }
          uint64 packed   = encodeBlock48(fblock);
          size_t out      = (size_t)blockIdx * 3u;
          stream[out]     = (uint16_t)(packed);
          stream[out + 1] = (uint16_t)(packed >> 16);
          stream[out + 2] = (uint16_t)(packed >> 32);
        },
        const ValueType *data,
        uint16_t        *stream,
        unsigned int     dim,
        unsigned int     totBlocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress12Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          size_t out    = (size_t)blockIdx * 3u;
          uint64 packed = (uint64)stream[out] |
                          ((uint64)stream[out + 1] << 16) |
                          ((uint64)stream[out + 2] << 32);

          ValueType fblock[4];
          decodeBlock48(packed, fblock);

          unsigned int blockStart = blockIdx * 4u;
          if (blockStart + 4u <= dim)
            {
              data[blockStart]     = fblock[0];
              data[blockStart + 1] = fblock[1];
              data[blockStart + 2] = fblock[2];
              data[blockStart + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                data[blockStart + j] = fblock[j];
            }
        },
        const uint16_t *stream,
        ValueType      *data,
        unsigned int    dim,
        unsigned int    totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compressGather12Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0] = dataArray[base];
          fblock[1] = dataArray[base + 1];
          fblock[2] = dataArray[base + 2];
          fblock[3] = dataArray[base + 3];

          uint64 packed   = encodeBlock48(fblock);
          size_t out      = (size_t)blockIdx * 3u;
          stream[out]     = (uint16_t)(packed);
          stream[out + 1] = (uint16_t)(packed >> 16);
          stream[out + 2] = (uint16_t)(packed >> 32);
        },
        const ValueType *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        uint16_t        *stream,
        unsigned int     totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompressScatterAdd12Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          size_t out    = (size_t)blockIdx * 3u;
          uint64 packed = (uint64)stream[out] |
                          ((uint64)stream[out + 1] << 16) |
                          ((uint64)stream[out + 2] << 32);

          ValueType fblock[4];
          decodeBlock48(packed, fblock);

          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const uint16_t  *stream,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        ValueType       *dataArray,
        unsigned int     totBlocks);

      /* ---- 16 bpv: 1 x uint64 per block ---- */
      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        compress16Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          unsigned int blockStart = blockIdx * 4u;
          ValueType    fblock[4];
          if (blockStart + 4u <= dim)
            {
              fblock[0] = data[blockStart];
              fblock[1] = data[blockStart + 1];
              fblock[2] = data[blockStart + 2];
              fblock[3] = data[blockStart + 3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                fblock[j] = data[blockStart + j];
              padBlock(fblock, nx);
            }
          stream[blockIdx] = encodeBlock64(fblock);
        },
        const ValueType *data,
        uint64          *stream,
        unsigned int     dim,
        unsigned int     totBlocks);

      template <typename ValueType>
      DFTFE_CREATE_KERNEL(
        void,
        decompress16Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          ValueType fblock[4];
          decodeBlock64(stream[blockIdx], fblock);
          unsigned int blockStart = blockIdx * 4u;
          if (blockStart + 4u <= dim)
            {
              data[blockStart]     = fblock[0];
              data[blockStart + 1] = fblock[1];
              data[blockStart + 2] = fblock[2];
              data[blockStart + 3] = fblock[3];
            }
          else
            {
              unsigned int nx = dim - blockStart;
              for (unsigned int j = 0; j < nx; j++)
                data[blockStart + j] = fblock[j];
            }
        },
        const uint64 *stream,
        ValueType    *data,
        unsigned int  dim,
        unsigned int  totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        compressGather16Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          ValueType fblock[4];
          fblock[0]        = dataArray[base];
          fblock[1]        = dataArray[base + 1];
          fblock[2]        = dataArray[base + 2];
          fblock[3]        = dataArray[base + 3];
          stream[blockIdx] = encodeBlock64(fblock);
        },
        const ValueType *dataArray,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        uint64          *stream,
        unsigned int     totBlocks);

      template <typename ValueType, typename IndexType>
      DFTFE_CREATE_KERNEL(
        void,
        decompressScatterAdd16Kernel,
        {
          const unsigned int blockIdx = (unsigned int)globalThreadId;
          if (blockIdx >= totBlocks)
            return;
          ValueType fblock[4];
          decodeBlock64(stream[blockIdx], fblock);

          const unsigned int blocksPerEntry = gatherBlockSize >> 2;
          unsigned int       gatherIdx      = blockIdx / blocksPerEntry;
          unsigned int       localBlock = blockIdx - gatherIdx * blocksPerEntry;
          unsigned int       intraIdx   = localBlock * 4u;
          size_t base = (size_t)indices[gatherIdx] * gatherBlockSize + intraIdx;

          dftfe::utils::atomicAddWrapper(&dataArray[base], fblock[0]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 1], fblock[1]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 2], fblock[2]);
          dftfe::utils::atomicAddWrapper(&dataArray[base + 3], fblock[3]);
        },
        const uint64    *stream,
        const IndexType *indices,
        unsigned int     gatherBlockSize,
        ValueType       *dataArray,
        unsigned int     totBlocks);

#  undef DFTFE_DEVICE_INLINE

      /* Internal dispatch: switch with bpv, launches the matching kernel */

      template <typename ValueType>
      void
      compressWrapper(const ValueType             *dData,
                      void                        *dStream,
                      size_t                       numValues,
                      int                          bitsPerValue,
                      dftfe::utils::deviceStream_t stream)
      {
        if (numValues == 0)
          return;

        const unsigned int dim       = (unsigned int)numValues;
        const unsigned int numBlocks = (dim + 3u) / 4u;
        const unsigned int grid =
          (numBlocks + dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
          dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bitsPerValue)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL((compress8Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  dData,
                                  reinterpret_cast<unsigned int *>(dStream),
                                  dim,
                                  numBlocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL((compress10Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  dData,
                                  reinterpret_cast<uint8_t *>(dStream),
                                  dim,
                                  numBlocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL((compress12Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  dData,
                                  reinterpret_cast<uint16_t *>(dStream),
                                  dim,
                                  numBlocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL((compress16Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  dData,
                                  reinterpret_cast<uint64 *>(dStream),
                                  dim,
                                  numBlocks);
              break;
          }
      }

      template <typename ValueType>
      void
      decompressWrapper(const void                  *dStream,
                        ValueType                   *dData,
                        size_t                       numValues,
                        int                          bitsPerValue,
                        dftfe::utils::deviceStream_t stream)
      {
        if (numValues == 0)
          return;

        const unsigned int dim       = (unsigned int)numValues;
        const unsigned int numBlocks = (dim + 3u) / 4u;
        const unsigned int grid =
          (numBlocks + dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
          dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bitsPerValue)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL((decompress8Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  reinterpret_cast<const unsigned int *>(
                                    dStream),
                                  dData,
                                  dim,
                                  numBlocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL((decompress10Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  reinterpret_cast<const uint8_t *>(dStream),
                                  dData,
                                  dim,
                                  numBlocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL((decompress12Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  reinterpret_cast<const uint16_t *>(dStream),
                                  dData,
                                  dim,
                                  numBlocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL((decompress16Kernel<ValueType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  reinterpret_cast<const uint64 *>(dStream),
                                  dData,
                                  dim,
                                  numBlocks);
              break;
          }
      }

      template <typename ValueType, typename IndexType>
      void
      compressGatherWrapper(const ValueType             *dataArray,
                            const IndexType             *indices,
                            size_t                       numIndices,
                            unsigned int                 gatherBlockSize,
                            void                        *dStream,
                            int                          bitsPerValue,
                            dftfe::utils::deviceStream_t stream)
      {
        const size_t numValues = numIndices * gatherBlockSize;
        if (numValues == 0)
          return;

        const unsigned int dim       = (unsigned int)numValues;
        const unsigned int numBlocks = (dim + 3u) / 4u;
        const unsigned int grid =
          (numBlocks + dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
          dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bitsPerValue)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL((compressGather8Kernel<ValueType, IndexType>),
                                  grid,
                                  dftfe::utils::DEVICE_BLOCK_SIZE,
                                  stream,
                                  dataArray,
                                  indices,
                                  gatherBlockSize,
                                  reinterpret_cast<unsigned int *>(dStream),
                                  numBlocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL(
                (compressGather10Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gatherBlockSize,
                reinterpret_cast<uint8_t *>(dStream),
                numBlocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL(
                (compressGather12Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gatherBlockSize,
                reinterpret_cast<uint16_t *>(dStream),
                numBlocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL(
                (compressGather16Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                dataArray,
                indices,
                gatherBlockSize,
                reinterpret_cast<uint64 *>(dStream),
                numBlocks);
              break;
          }
      }

      template <typename ValueType, typename IndexType>
      void
      decompressScatterAddWrapper(const void                  *dStream,
                                  const IndexType             *indices,
                                  size_t                       numIndices,
                                  unsigned int                 gatherBlockSize,
                                  ValueType                   *dataArray,
                                  int                          bitsPerValue,
                                  dftfe::utils::deviceStream_t stream)
      {
        const size_t numValues = numIndices * gatherBlockSize;
        if (numValues == 0)
          return;

        const unsigned int dim       = (unsigned int)numValues;
        const unsigned int numBlocks = (dim + 3u) / 4u;
        const unsigned int grid =
          (numBlocks + dftfe::utils::DEVICE_BLOCK_SIZE - 1) /
          dftfe::utils::DEVICE_BLOCK_SIZE;

        switch (bitsPerValue)
          {
            case 8:
              DFTFE_LAUNCH_KERNEL(
                (decompressScatterAdd8Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const unsigned int *>(dStream),
                indices,
                gatherBlockSize,
                dataArray,
                numBlocks);
              break;
            case 10:
              DFTFE_LAUNCH_KERNEL(
                (decompressScatterAdd10Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint8_t *>(dStream),
                indices,
                gatherBlockSize,
                dataArray,
                numBlocks);
              break;
            case 12:
              DFTFE_LAUNCH_KERNEL(
                (decompressScatterAdd12Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint16_t *>(dStream),
                indices,
                gatherBlockSize,
                dataArray,
                numBlocks);
              break;
            case 16:
              DFTFE_LAUNCH_KERNEL(
                (decompressScatterAdd16Kernel<ValueType, IndexType>),
                grid,
                dftfe::utils::DEVICE_BLOCK_SIZE,
                stream,
                reinterpret_cast<const uint64 *>(dStream),
                indices,
                gatherBlockSize,
                dataArray,
                numBlocks);
              break;
          }
      }
    } // namespace

    void
    compress(const double                *dData,
             void                        *dCompressed,
             size_t                       numValues,
             int                          bitsPerValue,
             dftfe::utils::deviceStream_t stream)
    {
      compressWrapper<double>(
        dData, dCompressed, numValues, bitsPerValue, stream);
    }

    void
    compress(const float                 *dData,
             void                        *dCompressed,
             size_t                       numValues,
             int                          bitsPerValue,
             dftfe::utils::deviceStream_t stream)
    {
      compressWrapper<float>(
        dData, dCompressed, numValues, bitsPerValue, stream);
    }

    void
    decompress(const void                  *dCompressed,
               double                      *dData,
               size_t                       numValues,
               int                          bitsPerValue,
               dftfe::utils::deviceStream_t stream)
    {
      decompressWrapper<double>(
        dCompressed, dData, numValues, bitsPerValue, stream);
    }

    void
    decompress(const void                  *dCompressed,
               float                       *dData,
               size_t                       numValues,
               int                          bitsPerValue,
               dftfe::utils::deviceStream_t stream)
    {
      decompressWrapper<float>(
        dCompressed, dData, numValues, bitsPerValue, stream);
    }

    void
    compressGather(const double                *dataArray,
                   const dftfe::uInt           *indices,
                   size_t                       numIndices,
                   dftfe::uInt                  gatherBlockSize,
                   void                        *dCompressed,
                   int                          bitsPerValue,
                   dftfe::utils::deviceStream_t stream)
    {
      compressGatherWrapper<double, dftfe::uInt>(dataArray,
                                                 indices,
                                                 numIndices,
                                                 (unsigned int)gatherBlockSize,
                                                 dCompressed,
                                                 bitsPerValue,
                                                 stream);
    }

    void
    compressGather(const float                 *dataArray,
                   const dftfe::uInt           *indices,
                   size_t                       numIndices,
                   dftfe::uInt                  gatherBlockSize,
                   void                        *dCompressed,
                   int                          bitsPerValue,
                   dftfe::utils::deviceStream_t stream)
    {
      compressGatherWrapper<float, dftfe::uInt>(dataArray,
                                                indices,
                                                numIndices,
                                                (unsigned int)gatherBlockSize,
                                                dCompressed,
                                                bitsPerValue,
                                                stream);
    }

    void
    decompressScatterAdd(const void                  *dCompressed,
                         const dftfe::uInt           *indices,
                         size_t                       numIndices,
                         dftfe::uInt                  gatherBlockSize,
                         double                      *dataArray,
                         int                          bitsPerValue,
                         dftfe::utils::deviceStream_t stream)
    {
      decompressScatterAddWrapper<double, dftfe::uInt>(dCompressed,
                                                       indices,
                                                       numIndices,
                                                       (unsigned int)
                                                         gatherBlockSize,
                                                       dataArray,
                                                       bitsPerValue,
                                                       stream);
    }

    void
    decompressScatterAdd(const void                  *dCompressed,
                         const dftfe::uInt           *indices,
                         size_t                       numIndices,
                         dftfe::uInt                  gatherBlockSize,
                         float                       *dataArray,
                         int                          bitsPerValue,
                         dftfe::utils::deviceStream_t stream)
    {
      decompressScatterAddWrapper<float, dftfe::uInt>(dCompressed,
                                                      indices,
                                                      numIndices,
                                                      (unsigned int)
                                                        gatherBlockSize,
                                                      dataArray,
                                                      bitsPerValue,
                                                      stream);
    }
  } // namespace compressionWrapper
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
