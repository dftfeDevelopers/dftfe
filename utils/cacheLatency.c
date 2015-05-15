//obtained this file from : http://stackoverflow.com/questions/21369381/measuring-cache-latencies
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>

int i386_cpuid_caches (size_t * data_caches) {
    int i;
    int num_data_caches = 0;
    for (i = 0; i < 32; i++) {

        // Variables to hold the contents of the 4 i386 legacy registers
        uint32_t eax, ebx, ecx, edx; 

        eax = 4; // get cache info
        ecx = i; // cache id

        asm (
            "cpuid" // call i386 cpuid instruction
            : "+a" (eax) // contains the cpuid command code, 4 for cache query
            , "=b" (ebx)
            , "+c" (ecx) // contains the cache id
            , "=d" (edx)
        ); // generates output in 4 registers eax, ebx, ecx and edx 

        // taken from http://download.intel.com/products/processor/manual/325462.pdf Vol. 2A 3-149
        int cache_type = eax & 0x1F; 

        if (cache_type == 0) // end of valid cache identifiers
            break;

        char * cache_type_string;
        switch (cache_type) {
            case 1: cache_type_string = "Data Cache"; break;
            case 2: cache_type_string = "Instruction Cache"; break;
            case 3: cache_type_string = "Unified Cache"; break;
            default: cache_type_string = "Unknown Type Cache"; break;
        }

        int cache_level = (eax >>= 5) & 0x7;

        int cache_is_self_initializing = (eax >>= 3) & 0x1; // does not need SW initialization
        int cache_is_fully_associative = (eax >>= 1) & 0x1;


        // taken from http://download.intel.com/products/processor/manual/325462.pdf 3-166 Vol. 2A
        // ebx contains 3 integers of 10, 10 and 12 bits respectively
        unsigned int cache_sets = ecx + 1;
        unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
        unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
        unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;

        // Total cache size is the product
        size_t cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

        if (cache_type == 1 || cache_type == 3) {
            data_caches[num_data_caches++] = cache_total_size;
        }

        printf(
            "Cache ID %d:\n"
            "- Level: %d\n"
            "- Type: %s\n"
            "- Sets: %d\n"
            "- System Coherency Line Size: %d bytes\n"
            "- Physical Line partitions: %d\n"
            "- Ways of associativity: %d\n"
            "- Total Size: %zu bytes (%zu kb)\n"
            "- Is fully associative: %s\n"
            "- Is Self Initializing: %s\n"
            "\n"
            , i
            , cache_level
            , cache_type_string
            , cache_sets
            , cache_coherency_line_size
            , cache_physical_line_partitions
            , cache_ways_of_associativity
            , cache_total_size, cache_total_size >> 10
            , cache_is_fully_associative ? "true" : "false"
            , cache_is_self_initializing ? "true" : "false"
        );
    }

    return num_data_caches;
}

int test_cache(size_t attempts, size_t lower_cache_size, int * latencies, size_t max_latency) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        perror("open");
        abort();
    }
    char * random_data = mmap(
          NULL
        , lower_cache_size
        , PROT_READ | PROT_WRITE
        , MAP_PRIVATE | MAP_ANON // | MAP_POPULATE
        , -1
        , 0
        ); // get some random data
    if (random_data == MAP_FAILED) {
        perror("mmap");
        abort();
    }

    size_t i;
    for (i = 0; i < lower_cache_size; i += sysconf(_SC_PAGESIZE)) {
        random_data[i] = 1;
    }


    int64_t random_offset = 0;
    while (attempts--) {
        // use processor clock timer for exact measurement
        random_offset += rand();
        random_offset %= lower_cache_size;
        int32_t cycles_used, edx, temp1, temp2;
        asm (
            "mfence\n\t"        // memory fence
            "rdtsc\n\t"         // get cpu cycle count
            "mov %%edx, %2\n\t"
            "mov %%eax, %3\n\t"
            "mfence\n\t"        // memory fence
            "mov %4, %%al\n\t"  // load data
            "mfence\n\t"
            "rdtsc\n\t"
            "sub %2, %%edx\n\t" // substract cycle count
            "sbb %3, %%eax"     // substract cycle count
            : "=a" (cycles_used)
            , "=d" (edx)
            , "=r" (temp1)
            , "=r" (temp2)
            : "m" (random_data[random_offset])
            );
        // printf("%d\n", cycles_used);
        if (cycles_used < max_latency)
            latencies[cycles_used]++;
        else 
            latencies[max_latency - 1]++;
    }

    munmap(random_data, lower_cache_size);

    return 0;
} 

int main() {
    size_t cache_sizes[32];
    int num_data_caches = i386_cpuid_caches(cache_sizes);

    int latencies[0x400];
    memset(latencies, 0, sizeof(latencies));

    int empty_cycles = 0;

    int i;
    int attempts = 1000000;
    for (i = 0; i < attempts; i++) { // measure how much overhead we have for counting cyscles
        int32_t cycles_used, edx, temp1, temp2;
        asm (
            "mfence\n\t"        // memory fence
            "rdtsc\n\t"         // get cpu cycle count
            "mov %%edx, %2\n\t"
            "mov %%eax, %3\n\t"
            "mfence\n\t"        // memory fence
            "mfence\n\t"
            "rdtsc\n\t"
            "sub %2, %%edx\n\t" // substract cycle count
            "sbb %3, %%eax"     // substract cycle count
            : "=a" (cycles_used)
            , "=d" (edx)
            , "=r" (temp1)
            , "=r" (temp2)
            :
            );
        if (cycles_used < sizeof(latencies) / sizeof(*latencies))
            latencies[cycles_used]++;
        else 
            latencies[sizeof(latencies) / sizeof(*latencies) - 1]++;

    }

    {
        int j;
        size_t sum = 0;
        for (j = 0; j < sizeof(latencies) / sizeof(*latencies); j++) {
            sum += latencies[j];
        }
        size_t sum2 = 0;
        for (j = 0; j < sizeof(latencies) / sizeof(*latencies); j++) {
            sum2 += latencies[j];
            if (sum2 >= sum * .75) {
                empty_cycles = j;
                fprintf(stderr, "Empty counting takes %d cycles\n", empty_cycles);
                break;
            }
        }
    }

    for (i = 0; i < num_data_caches; i++) {
        test_cache(attempts, cache_sizes[i] * 4, latencies, sizeof(latencies) / sizeof(*latencies));

        int j;
        size_t sum = 0;
        for (j = 0; j < sizeof(latencies) / sizeof(*latencies); j++) {
            sum += latencies[j];
        }
        size_t sum2 = 0;
        for (j = 0; j < sizeof(latencies) / sizeof(*latencies); j++) {
            sum2 += latencies[j];
            if (sum2 >= sum * .75) {
                fprintf(stderr, "Cache ID %i has latency %d cycles\n", i, j - empty_cycles);
                break;
            }
        }

    }

    return 0;

}
