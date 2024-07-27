#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef RNG_H_
#define RNG_H_

#define __STDC_FORMAT_MACROS 1

#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>


///=============================================================================
///                      Compiler and Platform Features
///=============================================================================

typedef int8_t      i8;
typedef uint8_t     u8;
typedef int16_t     i16;
typedef uint16_t    u16;
typedef int32_t     i32;
typedef uint32_t    u32;
typedef int64_t     i64;
typedef uint64_t    u64;
typedef float       f32;
typedef double      f64;

#define XRSR_MIX1          0xbf58476d1ce4e5b9
#define XRSR_MIX2          0x94d049bb133111eb
#define XRSR_MIX1_INVERSE  0x96de1b173f119089
#define XRSR_MIX2_INVERSE  0x319642b2d24d8ec3
#define XRSR_SILVER_RATIO  0x6a09e667f3bcc909
#define XRSR_GOLDEN_RATIO  0x9e3779b97f4a7c15

__device__ __host__ uint64_t mix64(uint64_t a) {
	a = (a ^ a >> 30) * XRSR_MIX1;
	a = (a ^ a >> 27) * XRSR_MIX2;
	return a ^ a >> 31;
}

#define STRUCT(S) typedef struct S S; struct S

#if __GNUC__

#define IABS(X)                 __builtin_abs(X)
#define PREFETCH(PTR,RW,LOC)    __builtin_prefetch(PTR,RW,LOC)
#define likely(COND)            (__builtin_expect(!!(COND),1))
#define unlikely(COND)          (__builtin_expect((COND),0))
#define ATTR(...)               __attribute__((__VA_ARGS__))
#define BSWAP32(X)              __builtin_bswap32(X)
#define UNREACHABLE()           __builtin_unreachable()

#else

#define IABS(X)                 ((int)abs(X))
#define PREFETCH(PTR,RW,LOC)
#define likely(COND)            (COND)
#define unlikely(COND)          (COND)
#define ATTR(...)
static inline uint32_t BSWAP32(uint32_t x) {
    x = ((x & 0x000000ff) << 24) | ((x & 0x0000ff00) <<  8) |
        ((x & 0x00ff0000) >>  8) | ((x & 0xff000000) >> 24);
    return x;
}
#if _MSC_VER
#define UNREACHABLE()           __assume(0)
#else
#define UNREACHABLE()           exit(1) // [[noreturn]]
#endif

#endif

/// imitate amd64/x64 rotate instructions

static inline ATTR(const, always_inline, artificial)
__device__ __host__ uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

static inline ATTR(const, always_inline, artificial)
__device__ __host__ uint32_t rotr32(uint32_t a, uint8_t b)
{
    return (a >> b) | (a << (32-b));
}

/// integer floor divide
static inline ATTR(const, always_inline)
int32_t floordiv(int32_t a, int32_t b)
{
    int32_t q = a / b;
    int32_t r = a % b;
    return q - ((a ^ b) < 0 && !!r);
}

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

__device__ __host__ static inline int nextInt(uint64_t *seed, const int n) {
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

__device__ __host__ static inline int nextIntBounded(uint64_t *seed, const int min, const int max) {
    if (min >= max) {
        return min;
    }
    return nextInt(seed, max - min + 1) + min;
}

__device__ __host__ static inline uint64_t nextLong(uint64_t *seed)
{
    return ((uint64_t) next(seed, 32) << 32) + next(seed, 32);
}

__device__ __host__ static inline float nextFloat(uint64_t *seed)
{
    return next(seed, 24) / (float) (1 << 24);
}

__device__ __host__ static inline double nextDouble(uint64_t *seed)
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


///=============================================================================
///                               Xoroshiro 128
///=============================================================================

STRUCT(Xoroshiro)
{
    uint64_t lo, hi;
};

__device__ __host__ static inline void xSetSeed(Xoroshiro *xr, uint64_t value)
{
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

__device__ __host__ static inline void xSetFeatureSeed(Xoroshiro *xr, uint64_t p_190065_, int p_190066_, int p_190067_) {
    uint64_t i = p_190065_ + (long)p_190066_ + (long)(10000 * p_190067_);
    xSetSeed(xr, i);
}

__device__ __host__ static inline uint64_t xNextLong(Xoroshiro *xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

__device__ __host__ static inline uint64_t xSetDecorationSeed(Xoroshiro *xr, uint64_t p_64691_, int p_64692_, int p_64693_) {
    // this.setSeed(p_64691_);
    xSetSeed(xr, p_64691_);
    uint64_t i = xNextLong(xr) | 1L;
    uint64_t j = xNextLong(xr) | 1L;
    uint64_t k = (uint64_t)p_64692_ * i + (uint64_t)p_64693_ * j ^ p_64691_;
    // this.setSeed(k);
    xSetSeed(xr, k);
    return k;
}

__device__ __host__ static inline int xNextInt(Xoroshiro *xr, uint32_t n)
{
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
        }
    }
    return r >> 32;
}

__device__ __host__ static inline double xNextDouble(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-53)) * 1.1102230246251565E-16;
}

__device__ __host__ static inline float xNextFloat(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-24)) * 5.9604645E-8F;
}

__device__ __host__ static inline void xSkipN(Xoroshiro *xr, int count)
{
    while (count --> 0)
        xNextLong(xr);
}

__device__ __host__ static inline uint64_t xNextLongJ(Xoroshiro *xr)
{
    int32_t a = xNextLong(xr) >> 32;
    int32_t b = xNextLong(xr) >> 32;
    return ((uint64_t)a << 32) + b;
}

__device__ __host__ static inline int xNextIntJ(Xoroshiro *xr, uint32_t n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (xNextLong(xr) >> 33);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = (xNextLong(xr) >> 33);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}


//==============================================================================
//                              MC Seed Helpers
//==============================================================================

/**
 * The seed pipeline:
 *
 * getLayerSalt(n)                -> layerSalt (ls)
 * layerSalt (ls), worldSeed (ws) -> startSalt (st), startSeed (ss)
 * startSeed (ss), coords (x,z)   -> chunkSeed (cs)
 *
 * The chunkSeed alone is enough to generate the first PRNG integer with:
 *   mcFirstInt(cs, mod)
 * subsequent PRNG integers are generated by stepping the chunkSeed forwards,
 * salted with startSalt:
 *   cs_next = mcStepSeed(cs, st)
 */

static inline uint64_t mcStepSeed(uint64_t s, uint64_t salt)
{
    return s * (s * 6364136223846793005ULL + 1442695040888963407ULL) + salt;
}

static inline int mcFirstInt(uint64_t s, int mod)
{
    int ret = (int)(((int64_t)s >> 24) % mod);
    if (ret < 0)
        ret += mod;
    return ret;
}

static inline int mcFirstIsZero(uint64_t s, int mod)
{
    return (int)(((int64_t)s >> 24) % mod) == 0;
}

static inline uint64_t getChunkSeed(uint64_t ss, int x, int z)
{
    uint64_t cs = ss + x;
    cs = mcStepSeed(cs, z);
    cs = mcStepSeed(cs, x);
    cs = mcStepSeed(cs, z);
    return cs;
}

static inline uint64_t getLayerSalt(uint64_t salt)
{
    uint64_t ls = mcStepSeed(salt, salt);
    ls = mcStepSeed(ls, salt);
    ls = mcStepSeed(ls, salt);
    return ls;
}

static inline uint64_t getStartSalt(uint64_t ws, uint64_t ls)
{
    uint64_t st = ws;
    st = mcStepSeed(st, ls);
    st = mcStepSeed(st, ls);
    st = mcStepSeed(st, ls);
    return st;
}

static inline uint64_t getStartSeed(uint64_t ws, uint64_t ls)
{
    uint64_t ss = ws;
    ss = getStartSalt(ss, ls);
    ss = mcStepSeed(ss, 0);
    return ss;
}


///============================================================================
///                               Arithmatic
///============================================================================


/* Linear interpolations
 */
__device__ __host__ static inline double lerp(double part, double from, double to)
{
    return from + part * (to - from);
}

__device__ __host__ static inline double lerp2(
        double dx, double dy, double v00, double v10, double v01, double v11)
{
    return lerp(dy, lerp(dx, v00, v10), lerp(dx, v01, v11));
}

__device__ __host__ static inline double lerp3(
        double dx, double dy, double dz,
        double v000, double v100, double v010, double v110,
        double v001, double v101, double v011, double v111)
{
    v000 = lerp2(dx, dy, v000, v100, v010, v110);
    v001 = lerp2(dx, dy, v001, v101, v011, v111);
    return lerp(dz, v000, v001);
}

__device__ __host__ static inline double clampedLerp(double part, double from, double to)
{
    if (part <= 0) return from;
    if (part >= 1) return to;
    return lerp(part, from, to);
}

/* Find the modular inverse: (1/x) | mod m.
 * Assumes x and m are positive (less than 2^63), co-prime.
 */
static inline ATTR(const)
__device__ __host__ uint64_t mulInv(uint64_t x, uint64_t m)
{
    uint64_t t, q, a, b, n;
    if ((int64_t)m <= 1)
        return 0; // no solution

    n = m;
    a = 0; b = 1;

    while ((int64_t)x > 1)
    {
        if (m == 0)
            return 0; // x and m are co-prime
        q = x / m;
        t = m; m = x % m;     x = t;
        t = a; a = b - q * a; b = t;
    }

    if ((int64_t)b < 0)
        b += n;
    return b;
}


typedef struct {
    Xoroshiro internal;
} RNG; // Bruh I really didn't want to have to do this.

__device__ __host__ RNG rng_new() {
    return (RNG){.internal=(Xoroshiro){0}};
}

__device__ __host__ static inline void rng_set_seed(RNG *rng, uint64_t seed) {
    seed ^= XRSR_SILVER_RATIO;
    rng->internal.lo = mix64(seed);
    rng->internal.hi = mix64(seed + XRSR_GOLDEN_RATIO);
}

__device__ __host__ static inline void rng_set_internal(RNG *rng, uint64_t lo, uint64_t hi) {
    rng->internal.lo = lo;
    rng->internal.hi = hi;
}

__device__ __host__ static inline uint32_t rng_next(RNG *rng, int32_t bits) {
    return xNextLong(&rng->internal) >> (64 - bits);
}

__device__ __host__ static inline int32_t rng_next_int(RNG *rng, uint32_t bound) {
    uint32_t r = rng_next(rng, 31);
    uint32_t m = bound - 1;
    if ((bound & m) == 0) {
        // (int)((long)p_188504_ * (long)this.next(31) >> 31);
        r = (uint32_t)((uint64_t)bound * (uint64_t)r >> 31);
    }
    else {
        for (uint32_t u = r; (int32_t)(u - (r = u % bound) + m) < 0; u = rng_next(rng, 31));
    }
    return r;
}

__device__ __host__ static inline uint64_t rng_next_long(RNG *rng) {
    int32_t i = rng_next(rng, 32);
    int32_t j = rng_next(rng, 32);
    uint64_t k = (uint64_t)i << 32;
    return k + (uint64_t)j;
}

__device__ __host__ static inline uint64_t rng_set_decoration_seed(RNG *rng, uint64_t world_seed, int32_t x, int32_t z) {
    rng_set_seed(rng, world_seed);

    uint64_t a = rng_next_long(rng) | 1L;
    uint64_t b = rng_next_long(rng) | 1L;

    uint64_t k = (a * (uint64_t)x + b * (uint64_t)z) ^ world_seed;
    rng_set_seed(rng, k);
    return k;
}

__device__ __host__ static inline void rng_set_feature_seed(RNG *rng, uint64_t p_190065_, int32_t p_190066_, int32_t p_190067_) {
    uint64_t i = p_190065_ + (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_);
    //printf("Salt = %" PRIu64 "\n", (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_));
    rng_set_seed(rng, i);
}

#endif /* RNG_H_ */

#define ll  long long int

#define printu64(val) printf("%" PRIu64 "\n", (val))
#define printi64(val) printf("%" PRIi64 "\n", (val))
#define printi(val) printf("%d\n", (val))
#define print_seed(val) printi64(val)

__device__ const static uint64_t valid_loot_seeds[68] = {
    4835198300983L,
    7332035751452L,
    14125018631592L,
    16843522132883L,
    22558606346032L,
    25603415556092L,
    27917408974186L,
    28780665201524L,
    36763672506642L,
    49898174216347L,
    54851118410986L,
    67429533035729L,
    68921484724337L,
    79541903247155L,
    83760782148839L,
    83841116665372L,
    84176774411871L,
    90797196476473L,
    91357893484262L,
    92571097990103L,
    94798470879408L,
    96960812053490L,
    101164170990904L,
    101235091411158L,
    106565953754466L,
    106921894927966L,
    107365059663880L,
    111512993735855L,
    115233097849165L,
    116230392124471L,
    118051218661506L,
    118109354894248L,
    122663339622580L,
    123764680956139L,
    126157375814142L,
    128872024651236L,
    128896829384277L,
    130556325114174L,
    140455946282600L,
    143000543024508L,
    152240665975058L,
    154690659645272L,
    158158777792049L,
    164936178522953L,
    178065525828803L,
    183823501087772L,
    186275823599766L,
    187078375095561L,
    188030536518365L,
    196793879718854L,
    201845975496614L,
    204573765959703L,
    208270839479119L,
    209156717123810L,
    211393967678402L,
    215988698279009L,
    225715937633625L,
    226717405268749L,
    234029363196676L,
    237623813325737L,
    237888192141868L,
    253892783093514L,
    255547815557168L,
    258960756868937L,
    261560426974177L,
    269655404299125L,
    272605054575375L,
    280486279432499L
};

#define MASK48 0xFFFFFFFFFFFFULL
__managed__ unsigned long long seedsChecked = 0;

__global__ void kernel(uint64_t o)
{
    uint64_t input_seed = blockDim.x * blockIdx.x + threadIdx.x + o;

    uint64_t seed;
    setSeed(&seed, input_seed);
    uint64_t world_seed = nextLong(&seed);

    RNG rng = rng_new();

    uint64_t i = rng_set_decoration_seed(&rng, world_seed, 192, 0);
    rng_set_feature_seed(&rng, i, 10, 4);

    uint64_t loot_seed = rng_next_long(&rng);

    for (int i = 0; i < 1; i++) {
        if ((loot_seed & MASK48) == valid_loot_seeds[i]) {
            print_seed(world_seed);
        }
    }
    atomicAdd(&seedsChecked, 1);
}

int main(int argc,char **argv)
{
    uint64_t threads_per_block = 512L;
    uint64_t num_blocks = 32768L;

    printf("CPU: Hello!\n");
    const uint64_t max = threads_per_block * num_blocks * 100000L; //(1ll << 48);
	  for (ll o = 0; o < max; o += threads_per_block * num_blocks) {
      //printf("%lld %lld\n", o, max);
		  kernel<<<num_blocks, threads_per_block>>>(o);
    }
    cudaDeviceSynchronize();
    printf("%lld\n", seedsChecked);
    return 0;
}