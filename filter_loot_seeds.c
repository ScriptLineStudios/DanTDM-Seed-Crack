#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include <vector>
#include <algorithm>
///=============================================================================
///                    C implementation of Java Random
///=============================================================================

static inline void setSeed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66d) & ((1ULL << 48) - 1);
}

static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

static inline int nextInt(uint64_t *seed, const int n)
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

static inline uint64_t nextLong(uint64_t *seed)
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

static inline int nextIntBounded(uint64_t *seed, int min, int max) {
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

int precomp_H[398];

typedef enum {
    item_clock = 0,
    flint, 
    flint_and_steel,
    none
} Item;

char *item_names[4] = {
    // "item_clock", "flint", "flint_and_steel", "none"
};

typedef struct {
    Item item;
    int amount;
} ItemStack;

int get_index(std::vector<int> v, int K) { 
    auto it = std::find(v.begin(), v.end(), K); 
  
    if (it != v.end()) { 
        int index = it - v.begin(); 
        return index;
    } 
    return -1;
} 

void kernel(uint64_t s) {
    uint64_t input_seed = s;
    // atomicAdd(&seeds_checked, 1);

    uint64_t internal;
    setSeed(&internal, input_seed);

    int rolls = nextIntBounded(&internal, 4, 8);
    if (rolls != 4) {
      return;
    }

    //If we are on a 4 roll seed, we can continue.
    // We can only accept 1 (flint) 3 (flint and steel) x2 20 (item_clock)
    int mask = 0;

    std::vector<ItemStack> items;
    int flint_index = 0;

    for (int i = 0; i < 4; i++) {
      int item = precomp_H[nextInt(&internal, 398)];
      switch (item) {
        case 1: //flint
          if (nextIntBounded(&internal, 1, 4) != 3) {
            return;
          }
          items.push_back((ItemStack){.item=flint, .amount=3});
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
          items.push_back((ItemStack){.item=flint_and_steel, .amount=1});
          break;
        case 20: //item_clock
          mask |= 0b1000;
          items.push_back((ItemStack){.item=item_clock, .amount=1});
          break;
        default:
          return;
      }
    }

    if (mask != 0b1111) {
        return;
    }

    std::vector<int> container{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    };

    for (int i = container.size(); i > 1; i--) {
        // printf("swap i = %d\n", i);
        int j = nextInt(&internal, i);
		int tmp = container.at(i - 1);
		container[i - 1] = container.at(j);
		container[j] = tmp;
    }

    // for (int i = 0; i < container.size(); i++) {
    //     printf("%d, ", container.at(i));
    // }
    // puts("");
    
    int mask2 = 0;
    for (int i = container.size() - 6; i < container.size(); i++) { // We're interested in the last 6 elements.
        switch (container.at(i)) {
            case 11:
                mask |= 0b000001;
                break;
            case 15:
                mask |= 0b000010;
                break;
            case 18:
                mask |= 0b000100;
                break;
            case 21:
                mask |= 0b001000;
                break;
            case 22:
                mask |= 0b010000;
                break;
            case 24:
                mask |= 0b100000;
                break;
            default:
                return;
        }
    }
    if (mask != 0b111111) {
        return;
    }

    // return; 
    // exit(1);

    std::vector<ItemStack> list;

    items.erase(items.begin() + flint_index); //remove the flint.
    list.push_back((ItemStack){.item=flint, .amount=3}); //put the flint in the list

    while (list.size() > 0) {
        //get and remove the item from the list
        int k = nextIntBounded(&internal, 0, list.size() - 1);
        ItemStack itemstack2 = list.at(k);
        list.erase(list.begin() + k); 

        int half = itemstack2.amount / 2;
        int i = nextIntBounded(&internal, 1, half);
        // printf("i = %d\n", i);

        ItemStack itemstack1 = (ItemStack){.item=flint, .amount=i};
        itemstack2.amount -= i;

        if (itemstack2.amount > 1 && (next(&internal, 1) == 1)) {
            list.push_back(itemstack2);
        }
        else {
            items.push_back(itemstack2);
        }

        if (itemstack1.amount > 1 && (next(&internal, 1) == 1)) {
            list.push_back(itemstack1);
        }
        else {
            items.push_back(itemstack1);
        }
    }

    if (items.size() != 6) {
        return;
    }

    // for (auto stack : list) {
    //     items.push_back(stack);
    // }
    
    // for (auto stack : items) {
    //     printf("%s %d\n", item_names[stack.item], stack.amount);
    // }
    //the final shit is now stored in ITEMS.

    //everyday i'm shuffling...
    for (int i = items.size(); i > 1; i--) {
        // printf("swap i = %d\n", i);
        int j = nextInt(&internal, i);
		ItemStack tmp = items.at(i - 1);
		items[i - 1] = items.at(j);
		items[j] = tmp;
    }

    int index = 0;
    int mask3 = 0;
    for (auto stack : items) {
        // printf("%d -> %s %d\n", container.at(26 - index), item_names[stack.item], stack.amount);
        switch (container.at(26 - index)) {
            case 11:
                if (!(stack.item == flint && stack.amount == 1)) {
                    return;
                }
                break;
            case 15:
                if (!(stack.item == flint && stack.amount == 1)) {
                    return;
                }
                break;
            case 18:
                if (!(stack.item == flint_and_steel && stack.amount == 1)) {
                    return;
                }
                break;
            case 21:
                if (!(stack.item == item_clock && stack.amount == 1)) {
                    return;
                }
                break;
            case 22:
                if (!(stack.item == flint && stack.amount == 1)) {
                    return;
                }
                break;
            case 24:
                if (!(stack.item == flint_and_steel && stack.amount == 1)) {
                    return;
                }
                break;
            default: {
                //if this happens something has gone very wrong
                printf("FATAL ERROR!\n");
                exit(1);
            }
        }
        index++;
    }
    printf("%" PRIu64 "\n", input_seed);

    return;
}

#include <iostream>
#include <fstream>

int main(void) {
    //There are 2^48 possible states for the loot seed. We need the one that yields 

    int rpLootTable[25] = {
        40, 40, 40, 40, 40,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        5, 5, 5, 5, 5, 5,
        1, 1, 1
    };


    int ix = 0;
    for (int i = 0; i < 25; i++)
        for (int j = 0; j < rpLootTable[i]; j++)
            precomp_H[ix++] = i;
        
    
    std::ifstream file("seeds.txt"); 
  
    // String to store each line of the file. 
    std::string line; 
  
    if (file.is_open()) { 
        // Read each line from the file and store it in the 
        // 'line' variable. 
        while (std::getline(file, line)) { 
            uint64_t seed = strtoull(line.c_str(), NULL, 10);
            kernel(seed);
        } 
    }
    exit(1);

    for (int i = 0; i < 10000; i++) {
    }

    return 0;
}