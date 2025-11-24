// MPI Matrix Multiplication using Point-to-Point Communication
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void initialize_matrices(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n * n; i++) {
        A[i] = i % 100;
        B[i] = (i % 100) + 1;
        C[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    int rank, size, n = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A, *B, *C;
    A = (double*)malloc(n * n * sizeof(double));
    B = (double*)malloc(n * n * sizeof(double));
    C = (double*)malloc(n * n * sizeof(double));

    if (rank == 0) {
        initialize_matrices(n, A, B, C);
    }

    double* local_A = (double*)malloc((n * n / size) * sizeof(double));
    double* local_C = (double*)malloc((n * n / size) * sizeof(double));

    /*
     * Timing variables. We continue to collect times only on rank 0
     * (as requested), but use MPI_Barrier to synchronize all processes
     * so the measured intervals represent the real, global durations.
     */
    double T_total_start = 0.0, T_total_end = 0.0;
    double T_comm = 0.0;
    double t_start, t_end;
    double T_comp = 0.0;

    /* Ensure all processes finished setup before starting measurements */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) T_total_start = MPI_Wtime();

    /* --- SCATTER (measure as global operation) --- */
    MPI_Barrier(MPI_COMM_WORLD);               // sync before measuring
    if (rank == 0) t_start = MPI_Wtime();

    MPI_Scatter(A, n * n / size, MPI_DOUBLE,
                local_A, n * n / size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);               // ensure all finished scatter
    if (rank == 0) {
        t_end = MPI_Wtime();
        T_comm += t_end - t_start;
    }

    /* --- BCAST (measure as global operation) --- */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_start = MPI_Wtime();

    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_end = MPI_Wtime();
        T_comm += t_end - t_start;
    }

    /* --- COMPUTATION (measure as the global compute time) --- */
    MPI_Barrier(MPI_COMM_WORLD);               // sync so all start compute together
    if (rank == 0) t_start = MPI_Wtime();

    for (int i = 0; i < n / size; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);               // wait until all finish compute
    if (rank == 0) {
        t_end = MPI_Wtime();
        T_comp = t_end - t_start;
    }

    /* --- GATHER (measure as global operation) --- */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_start = MPI_Wtime();

    MPI_Gather(local_C, n * n / size, MPI_DOUBLE,
               C, n * n / size, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_end = MPI_Wtime();
        T_comm += t_end - t_start;
    }

    /* Final total time (global) */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        T_total_end = MPI_Wtime();
        printf("Tempo total: %.6f\n", T_total_end - T_total_start);
        printf("Tempo de comunicacao: %.6f\n", T_comm);
        printf("Tempo de computacao: %.6f\n", T_comp);
    }

/*    if (rank == 0) {
        printf("Result Matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", C[i * n + j]);
            }
            printf("\n");
        }
    }
*/
    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
