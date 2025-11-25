```
Compilar com:
mpicc mpi_p2p_naobloqueante.c -o mpi_p2p_naobloqueante
mpicc mpi_p2p_bloqueante.c -o mpi_p2p_bloqueante
mpicc mpi_coletiva.c -o mpi_coletiva

Executar com:
mpirun -np NUM_PROCESSOS ./mpi_coletiva DIMENSAO_DA_MATRIZ
mpirun -np NUM_PROCESSOS ./mpi_p2p_naobloqueante DIMENSAO_DA_MATRIZ
mpirun -np NUM_PROCESSOS ./mpi_p2p_bloqueante DIMENSAO_DA_MATRIZ

Exemplo:
mpirun -np 3 ./mpi_p2p_bloqueante 100
```
