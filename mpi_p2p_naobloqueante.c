// MPI Matrix Multiplication using Non-Blocking Communication
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

    MPI_Request request;

    double t_total_start, t_total_end;
    double t_comm_start,  t_comm_end;
    double t_comp_start,  t_comp_end;
    double t_comp=0.00,t_comm=0.00;

    t_total_start = MPI_Wtime();

    t_comm_start = MPI_Wtime();
        
    if (rank == 0) {
        /* CORREÇÃO: usar um array de requests ao postar múltiplos Isend
           em vez de sobrescrever o mesmo 'request' várias vezes. */
        MPI_Request *reqs = NULL;
        if (size > 1) {
            reqs = (MPI_Request*)malloc((size - 1) * sizeof(MPI_Request));
            for (int i = 1; i < size; i++) {
                MPI_Isend(A + i * (n * n / size),
                          n * n / size, MPI_DOUBLE,
                          i, 0, MPI_COMM_WORLD,
                          &reqs[i-1]);
            }
        }
        t_comp_start = MPI_Wtime();
        for (int i = 0; i < n * n / size; i++) {
            local_A[i] = A[i];
        }
        t_comp_end = MPI_Wtime();
        t_comp+=t_comp_end-t_comp_start;

        /* esperar todos os Isend postados */
        if (size > 1) {
            MPI_Waitall(size - 1, reqs, MPI_STATUSES_IGNORE);
            free(reqs);
        }
        
    } else {
        MPI_Irecv(local_A, n * n / size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    MPI_Ibcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    t_comm_end = MPI_Wtime();
    t_comm+=t_comm_end-t_comm_start;

    t_comp_start = MPI_Wtime();

    for (int i = 0; i < n / size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local_C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }
    t_comp_end = MPI_Wtime();
    t_comp+=t_comp_end-t_comp_start;

    if (rank == 0) {
        t_comp_start = MPI_Wtime();
        for (int i = 0; i < n * n / size; i++) {
            C[i] = local_C[i];
        }
        t_comp_end = MPI_Wtime();
        t_comp+=t_comp_end-t_comp_start;
        t_comm_start = MPI_Wtime();
        for (int i = 1; i < size; i++) {
            MPI_Irecv(C + i * (n * n / size), n * n / size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        t_comm_end = MPI_Wtime();
        t_comm+=t_comm_end-t_comm_start;
    } else {
        MPI_Isend(local_C, n * n / size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_total_end = MPI_Wtime();
        printf("Tempo total: %.6f\n", t_total_end - t_total_start);
        printf("Tempo de comunicacao: %.6f\n", t_comm);
        printf("Tempo de computacao: %.6f\n", t_comp);
    }

/*
    if (rank == 0) {
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
