#include "aux.h"

void suma_est_fil(double  A[N][N]) {
    double volatile suma_fil = 0.0;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            suma_fil += A[i][j];
}

void suma_est_col(double  A[N][N]) {
    double volatile suma_col = 0.0;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            suma_col += A[j][i];
}

void suma_din_fil (VALT * A, size_t n) {
    double volatile suma_din_fil = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            suma_din_fil += A[j+i*n];
    }
}
void suma_din_fil2 (VALT * A, size_t n) {
    double volatile suma_din_fil = 0.0;

    for (int i = 0; i < n*n; ++i) {
        suma_din_fil += A[i];
    }
}

void suma_din_col (VALT * A, size_t n) {
    double volatile suma_din_col = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            suma_din_col += A[i+j*n];
    }
}

void suma_din_rand (VALT * A, size_t n) {
    // int *rec, posrand, i ;
    int rec[n*n], posrand; 
    int j = 0;
    double volatile  suma_din_rand = 0.0;
    array_zeros(rec,n*n);
    for (int i = 0;;) {
        // j++;
        posrand = rand() * n * n / RAND_MAX;
        if (rec[posrand] == 0) {
            suma_din_rand += A[posrand];
            rec[posrand] = 1;
            i++;
        }
        if (i == n*n) {
            // printf("loops: %d\n",j);
            break;
        }
    }
}