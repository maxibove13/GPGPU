// Directives - libraries
#include <stdio.h>
#include "bench.h"

// Functions declarations

double suma_est_fil(VALT  A[N][N]);
double suma_est_col(VALT  A[N][N]);

// size_t is a return type, generally used with the sizeof operator.
double suma_din_fil (VALT * A, size_t n);
double suma_din_col (VALT * A, size_t n);
// int suma_rand(VALT * A, size_t n);

// __restrict__ keyword: it tells the compiler that that pointer is the only way to access the object pointed by it.
// int mult_simple   (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n); 
// int mult_fila     (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n); 
// int mult_bl_simple(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t bl_sz); 
// int mult_bl_fila  (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t bl_sz); 

// *(a+i) = a[i]
void random_vector(VALT *a, size_t n) {
    for (unsigned int i = 0; i < n; i++)
        a[i] = i;
}

// main program
// argc (argument count) is the number of strings pointed to by argv (argument vector) (1 + number of arguments)
int main(char argc, char * argv[]){

    // const char * fname;

    if (argc < 2) {
        printf("El programa recibe n, y nb: n es la dimensión de las matrices y nb es el tamaño de bloque\n");
        exit(1);
    }

    // Converts argv (strings) to int
    int n = atoi(argv[1]);
    int nb = atoi(argv[2]);

    // srand(0); // Inicializa la semilla aleatoria

    VALT * A = (VALT *) aligned_alloc( 64, n*n*sizeof(VALT) );
    // VALT * B = (VALT *) aligned_alloc( 64, n*n*sizeof(VALT) );
    // VALT * C = (VALT *) aligned_alloc( 64, n*n*sizeof(VALT) );

    random_vector(A, n*n);
    // random_vector(B, n*n);

    struct timeb t_ini;
    struct timeb t_fin;

    // declared the array
    VALT A_est[N][N];

    // assign random doubles to each array's element
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_est[i][j]=1;

    // 1.a)
    suma_est_fil (A_est);
    // 1.b)
    suma_est_col (A_est);
    // 2.a)
    suma_din_fil (A, n);
    // 2.b)
    suma_din_col (A, n);
        
    BENCH_RUN(suma_est_fil (A_est), t_suma_est_fil, t_suma_est_fil_runs );
    BENCH_RUN(suma_est_col (A_est), t_suma_est_col, t_suma_est_col_runs );
    // BENCH_RUN(suma_din_fil (A,n)  , t_suma_din_fil, t_suma_din_fil_runs );
    // BENCH_RUN(suma_din_col (A,n)  , t_suma_din_col, t_suma_din_col_runs );
    // BENCH_RUN( suma_rand    (A,n)  , t_suma_rand   ,  t_suma_rand_runs    );

    printf("suma_est_fil: %.2f ms\truns:%d\n", t_suma_est_fil, t_suma_est_fil_runs );
    printf("suma_est_col: %.2f ms\truns:%d\n", t_suma_est_col, t_suma_est_col_runs );
    // printf("suma_din_fil: %.2f ms\truns:%d\n", t_suma_din_fil, t_suma_din_fil_runs );
    // printf("suma_din_col: %.2f ms\truns:%d\n", t_suma_din_col, t_suma_din_col_runs );
    // printf("suma_rand: %.2f ms\truns:%d\n"    , t_suma_rand   ,t_suma_rand_runs    );

    // BENCH_RUN( mult_simple   (A,B,C,n)   , t_mm_simple   , t_mm_simple_runs    )
    // BENCH_RUN( mult_fila     (A,B,C,n)   , t_mm_fila     , t_mm_fila_runs      )
    // BENCH_RUN( mult_bl_simple(A,B,C,n,nb), t_mm_bl_simple, t_mm_bl_simple_runs )
    // BENCH_RUN( mult_bl_fila  (A,B,C,n,nb), t_mm_bl_fila  , t_mm_bl_fila_runs   )

    // printf("mult_simple: %.2f ms, %.2f GFlops runs: %d\n" , t_mm_simple , ( ((double)n/t_mm_simple )*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_simple_runs  );
    // printf("mult_fila: %.2f ms, %.2f GFlops runs: %d\n"   , t_mm_fila   , ( ((double)n/t_mm_fila   )*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_fila_runs    );
    // printf("mult_bl_simple: %.2f ms, %.2f GFlops runs: %d\n", t_mm_bl_simple, ( ((double)n/t_mm_bl_simple)*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_bl_simple_runs );
    // printf("mult_bl_fila: %.2f ms, %.2f GFlops runs: %d\n", t_mm_bl_fila, ( ((double)n/t_mm_bl_fila)*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_bl_fila_runs );

	return 0;
}

double suma_est_fil(double  A[N][N]) {
    double volatile suma_fil = 0.0;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            suma_fil += A[i][j];
}

double suma_est_col(double  A[N][N]) {
    double volatile suma_col = 0.0;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            suma_col += A[j][i];
}

double suma_din_fil (VALT * A, size_t n) {
    double volatile suma_din_fil = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            suma_din_fil += A[j+i*n];
    }

    printf("%f\n", suma_din_fil);
}

double suma_din_col (VALT * A, size_t n) {
    double volatile suma_din_col = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            suma_din_col += A[i+j*n];
    }

    printf("%f\n", suma_din_col);
}