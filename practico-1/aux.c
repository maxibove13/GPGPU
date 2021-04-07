#include "bench.h"

// *(a+i) = a[i] WHEN a is an ARRAY
void random_vector(VALT *a, size_t n) {
    for (unsigned int i = 0; i < n; i++)
        a[i] = 1;
}
void array_zeros(int *rec, size_t n) {
    for (unsigned int i = 0; i < n; i++)
        *(rec+i) = 0;
}