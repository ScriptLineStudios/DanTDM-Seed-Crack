
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include <list>
#include <vector>
#include <algorithm>

///=============================================================================
///                    C implementation of Java Random
///=============================================================================

__device__ __host__ static inline void setSeed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66d) & ((1ULL << 48) - 1);
}

__device__ __host__ static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

__device__ __host__ static inline int nextInt(uint64_t *seed, const int n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (uint64_t)next(seed, 31);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = next(seed, 31);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}

__device__ __host__ static inline uint64_t nextLong(uint64_t *seed)
{
    return ((uint64_t) next(seed, 32) << 32) + next(seed, 32);
}

static inline float nextFloat(uint64_t *seed)
{
    return next(seed, 24) / (float) (1 << 24);
}

static inline double nextDouble(uint64_t *seed)
{
    uint64_t x = (uint64_t)next(seed, 26);
    x <<= 27;
    x += next(seed, 27);
    return (int64_t) x / (double) (1ULL << 53);
}

/* A macro to generate the ideal assembly for X = nextInt(*S, 24)
 * This is a macro and not an inline function, as many compilers can make use
 * of the additional optimisation passes for the surrounding code.
 */
#define JAVA_NEXT_INT24(S,X)                \
    do {                                    \
        uint64_t a = (1ULL << 48) - 1;      \
        uint64_t c = 0x5deece66dULL * (S);  \
        c += 11; a &= c;                    \
        (S) = a;                            \
        a = (uint64_t) ((int64_t)a >> 17);  \
        c = 0xaaaaaaab * a;                 \
        c = (uint64_t) ((int64_t)c >> 36);  \
        (X) = (int)a - (int)(c << 3) * 3;   \
    } while (0)


/* Jumps forwards in the random number sequence by simulating 'n' calls to next.
 */
static inline void skipNextN(uint64_t *seed, uint64_t n)
{
    uint64_t m = 1;
    uint64_t a = 0;
    uint64_t im = 0x5deece66dULL;
    uint64_t ia = 0xb;
    uint64_t k;

    for (k = n; k; k >>= 1)
    {
        if (k & 1)
        {
            m *= im;
            a = im * a + ia;
        }
        ia = (im + 1) * ia;
        im *= im;
    }

    *seed = *seed * m + a;
    *seed &= 0xffffffffffffULL;
}

__device__ __host__ static inline int nextIntBounded(uint64_t *seed, int min, int max) {
    if (min >= max) {
        return min;
    }
    return nextInt(seed, max - min + 1) + min;
    //return random.nextInt(max - min + 1) + min;
}

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

__constant__ int precomputedRPLootTable[398];
__managed__ unsigned long long int seeds_checked = 0;

typedef enum {
    //clock,
    flint,
    flint_and_steel,
    none
} Item;

typedef struct {
    Item item;
    int amount;
} ItemStack;

__global__ void kernel(uint64_t s) {
    uint64_t input_seed = blockDim.x * blockIdx.x + threadIdx.x + s;
    atomicAdd(&seeds_checked, 1);

    uint64_t internal;
    setSeed(&internal, input_seed);

    int rolls = nextIntBounded(&internal, 4, 8);
    if (rolls != 4) {
      return;
    }

    //If we are on a 4 roll seed, we can continue.
    // We can only accept 1 (flint) 3 (flint and steel) x2 20 (clock)
    int mask = 0;

    int flint_index = 0;

    for (int i = 0; i < 4; i++) {
      int item = precomputedRPLootTable[nextInt(&internal, 398)];
      switch (item) {
        case 1: //flint
          if (nextIntBounded(&internal, 1, 4) != 3) {
            return;
          }
          flint_index = i;
          mask |= 0b0001;
          break;
        case 3: //flint and steel
          if (!(mask >> 1 & 1)) {
            mask |= 0b0010;
          }
          else {
            mask |= 0b0100;
          }
          break;
        case 20: //clock
          mask |= 0b1000;
          break;
        default:
          return;
      }
    }

    if (mask != 0b1111) {
        return;
    }

    int container[27] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    };

    for (int i = 27; i > 1; i--) {
        // printf("swap i = %d\n", i);
        int j = nextInt(&internal, i);
        int tmp = container[i - 1];
        container[i - 1] = container[j];
        container[j] = tmp;
    }

    int mask2 = 0;
    for (int i = 27 - 6; i < 27; i++) { // We're interested in the last 6 elements.
        switch (container[i]) {
            case 11:
                mask2 |= 0b000001;
                break;
            case 15:
                mask2 |= 0b000010;
                break;
            case 18:
                mask2 |= 0b000100;
                break;
            case 21:
                mask2 |= 0b001000;
                break;
            case 22:
                mask2 |= 0b010000;
                break;
            case 24:
                mask2 |= 0b100000;
                break;
            default:
                return;
        }
    }
    if (mask2 != 0b111111) {
        return;
    }

    printf("%" PRIu64 "\n", input_seed);

    return;
}

int main(void) {
  //There are 2^48 possible states for the loot seed. We need the one that yields

  int rpLootTable[25] = {
      40, 40, 40, 40, 40,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      5, 5, 5, 5, 5, 5,
      1, 1, 1
  };
  int precomp_H[398];

  int ix = 0;
  for (int i = 0; i < 25; i++)
      for (int j = 0; j < rpLootTable[i]; j++)
          precomp_H[ix++] = i;

  cudaMemcpyToSymbol(precomputedRPLootTable, precomp_H, 398 * sizeof(int));

  int threads_per_block = 512;
  int num_blocks = 32768;

  const int max = 16777216;
  for (uint64_t s = 0; s < max; s++) {
      kernel<<<num_blocks, threads_per_block>>>(threads_per_block * num_blocks * s);
  }
  cudaDeviceSynchronize();
  printf("seeds checked: %llu\n", seeds_checked);

  return 0;
}