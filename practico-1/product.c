#include "aux.h"

// __restrict__ keyword hints to the compiler that only this pointer can point to the obcol_Aect that is pointing to.
int mult_simple   (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n) {
    for (int row_C = 0; row_C < n; ++row_C) {
        for (int row_A = 0; row_A < n; ++row_A) {
            for (int col_A = 0; col_A < n; ++col_A) {
                C[row_C*n+row_A] += A[row_A*n+col_A]*B[row_A+col_A*n];
            }
        }
    }
}

int mult_fila     (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n) {
    for (int row_A = 0; row_A < n; ++row_A) {
        for (int row_B = 0; row_B < n; ++row_B) {
            for (int col_B = 0; col_B < n; ++col_B) {
                C[row_A*n+col_B] += A[row_A*n+row_B]*B[row_B*n+col_B];
            }
        }
    }
}


    // inner loop me voy moviendo en filas de B y columnas de A. filas de C corresponden a filas de A, y columnas de C corresponden a columnas de B