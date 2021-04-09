#include "aux.h"

// __restrict__ keyword hints to the compiler that only this pointer can point to the obcol_Aect that is pointing to.
int mult_simple   (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n) {
    for (int row_C = 0; row_C < n; ++row_C) {
        for (int col_B = 0; col_B < n; ++col_B) {
            for (int col_A = 0; col_A < n; ++col_A) {
                C[row_C*n+col_B] += A[row_C*n+col_A]*B[col_A*n+col_B];
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

int mult_bl_simple(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t nb) {

    int index_A, index_B, index_C, bl = n/nb;
    
    for (int Ib = 0; Ib < bl; Ib++) {
        for (int Jb = 0; Jb < bl; Jb++) {
                for (int i = Ib*nb; i < nb*(Ib+1) ; i++) {
                    for (int j = Jb*nb; j < nb*(Jb+1); j++) {
                        for (int k = 0; k < n; k++) {
                            index_A = n*i + k;
                            index_B = n*j + k;
                            index_C = n*i + k;
                            *(C+index_C)+= ( *(A+index_A) * *(B+index_B) );
                        }
                    }
                }
            }
        }
    }