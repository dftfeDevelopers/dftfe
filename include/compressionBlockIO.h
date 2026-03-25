/*
** compressionBlockIO.h - GPU compressor: bit-level block I/O helper structs.
**
**   LocalBlockWriter  – write bits to a thread-local Word array (super-block)
**   LocalBlockReader  – read bits from a thread-local Word array (super-block)
**   InlineBlockWriter – accumulate bits into a single uint64 (bpt=1 fast path)
**   InlineBlockReader – read bits from a single uint64 (bpt=1 fast path)
**   BlockReader       – read bits from the global compressed stream (__ldg)
**
** Included by compressionZFP.h and compressionBFP.h.
** Never included directly by application code.
*/

#ifndef COMPRESSION_BLOCK_IO_H
#define COMPRESSION_BLOCK_IO_H

#include <compressionTypes.h>

namespace compression
{

  /* =========================================================================
     Device: LocalBlockWriter -- write bits to a thread-local Word array

     Accumulates bits using |= (no atomics). The local buffer must be
     zero-initialized by the caller. This replaces the old AtomicBlockWriter
     that used atomicAdd to global memory.
     =========================================================================
   */

  struct LocalBlockWriter
  {
    unsigned int m_word_index;
    unsigned int m_start_bit;
    unsigned int m_current_bit;
    Word        *m_local_words;

    COMPRESSION_DEVICE_INLINE
    LocalBlockWriter(Word        *local_words,
                     unsigned int maxbits,
                     unsigned int local_block_idx)
      : m_current_bit(0)
      , m_local_words(local_words)
    {
      size_t bit_offset = (size_t)local_block_idx * maxbits;
      m_word_index      = (unsigned int)(bit_offset / WSIZE);
      m_start_bit       = (unsigned int)(bit_offset % WSIZE);
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    write_bits(uint64 bits, unsigned int n_bits)
    {
      if (n_bits == 0)
        return bits;

      unsigned int seg_start = (m_start_bit + m_current_bit) % WSIZE;
      unsigned int write_index =
        m_word_index + (m_start_bit + m_current_bit) / WSIZE;
      unsigned int seg_end = seg_start + n_bits - 1;

      /* mask to lower n_bits - avoids UB shift-by-64 when n_bits == WSIZE */
      Word b = n_bits < WSIZE ? (bits & (((Word)1 << n_bits) - 1u)) : bits;
      m_local_words[write_index] |= (b << seg_start);

      if (seg_start < WSIZE && seg_end >= WSIZE)
        m_local_words[write_index + 1] |= (b >> (WSIZE - seg_start));

      m_current_bit += n_bits;
      return bits >> (Word)n_bits;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    write_bit(unsigned int bit)
    {
      unsigned int seg_start = (m_start_bit + m_current_bit) % WSIZE;
      unsigned int write_index =
        m_word_index + (m_start_bit + m_current_bit) / WSIZE;
      m_local_words[write_index] |= ((Word)bit << seg_start);
      m_current_bit += 1;
      return bit;
    }
  };

  /* =========================================================================
     Device: LocalBlockReader -- read bits from a thread-local Word array

     Used by the super-block decompress kernel: the caller loads wpt Words
     from global memory into a local array (L1-cached), then decodes all bpt
     blocks from that array using plain (non-__ldg) loads.  Carries the same
     advance-first invariant as BlockReader so decode_ints/decode_block work
     with either reader type via template dispatch.
     =========================================================================
   */

  struct LocalBlockReader
  {
    int         m_current_bit;
    const Word *m_words;
    Word        m_buffer;

    COMPRESSION_DEVICE_INLINE
    LocalBlockReader(const Word  *local_words,
                     unsigned int maxbits,
                     unsigned int local_block_idx)
    {
      size_t bit_offset = (size_t)local_block_idx * maxbits;
      m_words           = local_words + bit_offset / WSIZE;
      m_buffer          = m_words[0];
      m_current_bit     = (int)(bit_offset % WSIZE);
      m_buffer >>= m_current_bit;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    read_bit()
    {
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = m_words[0];
        }
      unsigned int bit = m_buffer & 1;
      ++m_current_bit;
      m_buffer >>= 1;
      return bit;
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    read_bits(unsigned int n_bits)
    {
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = m_words[0];
        }
      int    rem_bits   = (int)WSIZE - m_current_bit;
      int    first_read = portable_min(rem_bits, (int)n_bits);
      Word   mask       = ((Word)1 << first_read) - 1;
      uint64 bits       = m_buffer & mask;
      m_buffer >>= first_read;
      m_current_bit += first_read;

      int next_read = 0;
      if ((int)n_bits > rem_bits)
        {
          ++m_words;
          m_buffer      = m_words[0];
          m_current_bit = 0;
          next_read     = (int)n_bits - first_read;
        }
      mask = ((Word)1 << next_read) - 1;
      bits += (m_buffer & mask) << first_read;
      m_buffer >>= next_read;
      m_current_bit += next_read;
      return bits;
    }
  };

  /* =========================================================================
     Device: InlineBlockWriter -- accumulate bits into a single uint64

     For bpt=1 specialized kernels: no Word array, no cross-word writes.
     The caller encodes one block, then stores the packed uint64 as
     uint32_t (8 bpv) or 3×uint16_t (12 bpv).
     =========================================================================
   */

  struct InlineBlockWriter
  {
    unsigned int m_current_bit;
    uint64       m_packed;

    COMPRESSION_DEVICE_INLINE
    InlineBlockWriter()
      : m_current_bit(0)
      , m_packed(0)
    {}

    COMPRESSION_DEVICE_INLINE
    uint64
    write_bits(uint64 bits, unsigned int n_bits)
    {
      if (n_bits == 0)
        return bits;
      uint64 b = n_bits < 64u ? (bits & (((uint64)1 << n_bits) - 1u)) : bits;
      m_packed |= (b << m_current_bit);
      m_current_bit += n_bits;
      return bits >> (uint64)n_bits;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    write_bit(unsigned int bit)
    {
      m_packed |= ((uint64)bit << m_current_bit);
      m_current_bit += 1;
      return bit;
    }
  };

  /* =========================================================================
     Device: InlineBlockReader -- read bits from a single uint64

     For bpt=1 specialized decompress kernels. No word-boundary crossing.
     =========================================================================
   */

  struct InlineBlockReader
  {
    uint64 m_buffer;

    COMPRESSION_DEVICE_INLINE
    InlineBlockReader(uint64 packed)
      : m_buffer(packed)
    {}

    COMPRESSION_DEVICE_INLINE
    unsigned int
    read_bit()
    {
      unsigned int bit = (unsigned int)(m_buffer & 1u);
      m_buffer >>= 1;
      return bit;
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    read_bits(unsigned int n_bits)
    {
      uint64 mask = n_bits < 64u ? (((uint64)1 << n_bits) - 1u) : ~(uint64)0;
      uint64 bits = m_buffer & mask;
      m_buffer >>= n_bits;
      return bits;
    }
  };

  /* =========================================================================
     Device: BlockReader -- read bits from compressed stream (fixed-rate)

     Uses read-only load where available (CUDA __ldg).
     =========================================================================
   */

  struct BlockReader
  {
    int         m_current_bit;
    const Word *m_words;
    Word        m_buffer;

    COMPRESSION_DEVICE_INLINE
    BlockReader(const Word  *blocks,
                unsigned int maxbits,
                unsigned int block_idx)
    {
      size_t bit_offset = (size_t)block_idx * maxbits;
      size_t word_index = bit_offset / WSIZE;
      m_words           = blocks + word_index;
      m_buffer          = portable_ldg(m_words);
      m_current_bit     = (int)(bit_offset % WSIZE);
      m_buffer >>= m_current_bit;
    }

    COMPRESSION_DEVICE_INLINE
    unsigned int
    read_bit()
    {
      /* Advance-first: normalise if previous call left m_current_bit == WSIZE
       */
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = portable_ldg(m_words);
        }
      unsigned int bit = m_buffer & 1;
      ++m_current_bit;
      m_buffer >>= 1;
      return bit;
    }

    COMPRESSION_DEVICE_INLINE
    uint64
    read_bits(unsigned int n_bits)
    {
      /* Advance-first: normalise if previous call left m_current_bit == WSIZE.
         This also guarantees rem_bits >= 1 so first_read <= 63 (no UB shift).
       */
      if (m_current_bit >= (int)WSIZE)
        {
          m_current_bit = 0;
          ++m_words;
          m_buffer = portable_ldg(m_words);
        }

      int    rem_bits   = (int)WSIZE - m_current_bit;
      int    first_read = portable_min(rem_bits, (int)n_bits);
      Word   mask       = ((Word)1 << first_read) - 1;
      uint64 bits       = m_buffer & mask;
      m_buffer >>= first_read;
      m_current_bit += first_read;

      int next_read = 0;
      /* Strict >: only advance to next word when bits actually spill over.
         Using >= would load word[num_words] (OOB) whenever n_bits == rem_bits.
       */
      if ((int)n_bits > rem_bits)
        {
          ++m_words;
          m_buffer      = portable_ldg(m_words);
          m_current_bit = 0;
          next_read     = (int)n_bits - first_read;
        }

      /* next_read <= n_bits - 1 <= 62, so shift is safe */
      mask = ((Word)1 << next_read) - 1;
      bits += (m_buffer & mask) << first_read;
      m_buffer >>= next_read;
      m_current_bit += next_read;
      return bits;
    }
  };

} /* namespace compression */

#endif /* COMPRESSION_BLOCK_IO_H */
