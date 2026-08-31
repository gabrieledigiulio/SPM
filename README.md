# Iterative Sparse Matrix-Vector Product

Questo progetto confronta diverse implementazioni di una iterazione SpMV su una matrice sparsa che evolve nel tempo tramite shift circolari delle righe. Le varianti confrontate sono:

- sequenziale
- thread pool in C++
- OpenMP con task
- OpenMP con work-sharing (`omp for`)
- MPI + OpenMP con task
- MPI + OpenMP con work-sharing (`omp for`)

L'obiettivo e' misurare tempi, scalabilita' e comportamento al variare della granularita' del lavoro, del numero di thread e del numero di nodi MPI.

## Cosa fa il programma

Ogni eseguibile:

1. genera una matrice sparsa in formato CSR;
2. inizializza un vettore con un PRNG deterministico;
3. normalizza il vettore;
4. esegue `NUM_ITERS = 500` iterazioni di SpMV;
5. ogni `EPOCH_LEN = 25` iterazioni aggiorna l'evoluzione della matrice con uno shift di righe;
6. alla fine calcola checksum e Rayleigh quotient;
7. stampa il breakdown temporale e, se richiesto, salva il vettore finale.

La matrice puo' essere generata in due modalita':

- `regular`: non-zero distribuiti in modo quasi uniforme tra le righe;
- `irregular`: non-zero concentrati in alcune fasce di righe, utile per stressare il load balancing.

## Struttura dei file

### Sorgenti principali

- [iterative_SpMV.cpp](iterative_SpMV.cpp): riferimento sequenziale del progetto.
- [threadpool_SpMV.cpp](threadpool_SpMV.cpp): versione parallela con thread pool C++.
- [omp_tasks_SpMV.cpp](omp_tasks_SpMV.cpp): versione OpenMP basata su task.
- [omp_worksharing_SpMV.cpp](omp_worksharing_SpMV.cpp): versione OpenMP basata su work-sharing (`omp for`).
- [mpi_omp_tasks_SpMV.cpp](mpi_omp_tasks_SpMV.cpp): versione ibrida MPI + OpenMP con task.
- [mpi_omp_worksharing_SpMV.cpp](mpi_omp_worksharing_SpMV.cpp): versione ibrida MPI + OpenMP con work-sharing.

### Header e supporto

- [matrix_generation.hpp](matrix_generation.hpp): generazione della matrice CSR e dei pattern `regular`/`irregular`.
- [utils.hpp](utils.hpp): parsing degli argomenti, PRNG, checksum, dump del vettore e timer.
- [threadpool.hpp](threadpool.hpp): implementazione del thread pool usato dalla variante C++.

### Analisi e grafici

- [plot.py](plot.py): genera i grafici a partire dai CSV prodotti dagli esperimenti.

### Esperimenti

La cartella [exp](exp) contiene gli script di benchmark. Ogni script lancia uno o piu' eseguibili con parametri fissati, raccoglie i tempi medi/mediani e salva i risultati in [exp/results](exp/results).

## Requisiti

- compilatore C++20
- supporto OpenMP
- MPI con `mpicxx`/`mpic++`
- Slurm, perche' gli script usano `srun`
- Python 3 con `pandas`, `numpy`, `matplotlib` per i grafici

Per installare le dipendenze Python:

```bash
python -m pip install pandas numpy matplotlib
```

## Compilazione

I comandi seguenti assumono di essere nella root del progetto: `project/`.

```bash
# Sequenziale
g++ -O3 -std=c++20 -I . -Wall -Wextra iterative_SpMV.cpp -o iterative_SpMV

# Thread pool
g++ -O3 -std=c++20 -I . -Wall -Wextra -pthread threadpool_SpMV.cpp -o threadpool_SpMV

# OpenMP task
g++ -O3 -std=c++20 -I . -Wall -Wextra -fopenmp omp_tasks_SpMV.cpp -o omp_tasks_SpMV

# OpenMP work-sharing
g++ -O3 -std=c++20 -I . -Wall -Wextra -fopenmp omp_worksharing_SpMV.cpp -o omp_worksharing_SpMV

# MPI + OpenMP task
mpicxx -O3 -std=c++20 -I . -Wall -Wextra -fopenmp mpi_omp_tasks_SpMV.cpp -o mpi_omp_tasks_SpMV

# MPI + OpenMP work-sharing
mpicxx -O3 -std=c++20 -I . -Wall -Wextra -fopenmp mpi_omp_worksharing_SpMV.cpp -o mpi_omp_worksharing_SpMV
```

Se il tuo ambiente usa un nome diverso per il compilatore MPI, sostituisci `mpicxx` con `mpiCC` o con il wrapper disponibile nel sistema.

## Come lanciare i singoli programmi

Formato base degli argomenti:

```bash
-n N              dimensione della matrice N x N
-nz K             numero totale di non-zero
-m regular|irregular
-s SEED           seme opzionale, default 111
-t THREADS        numero di thread (solo versioni parallele)
-c CHUNK_SIZE     granularita' per SpMV
-nc NORM_CHUNK    granularita' per la normalizzazione
--dump-vector FILE  salva il vettore finale normalizzato
```

Esempi:

```bash
./iterative_SpMV -n 500000 -nz 20000000 -m irregular
./threadpool_SpMV -n 500000 -nz 20000000 -m irregular -t 16 -c 2048 -nc 2048
./omp_worksharing_SpMV -n 500000 -nz 20000000 -m irregular -t 16 -c 2048 -nc 2048
```

Per le varianti MPI + OpenMP, l'esecuzione va fatta tramite `srun`:

```bash
OMP_NUM_THREADS=16 srun --mpi=pmix -N 8 -n 8 --cpus-per-task=16 \
	./mpi_omp_worksharing_SpMV \
	-n 1000000 -nz 250000000 -m irregular -s 111 \
	-t 16 -c 2048 -nc 2048
```

Gli script in [exp](exp) impostano anche:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

## Esperimenti

Gli script vanno lanciati dalla cartella [exp](exp), perche' usano binari referenziati come `../nome_binario`.

### Baseline sequenziale

- Script: [exp/run_seq.sh](exp/run_seq.sh)
- Scopo: misura il tempo della versione sequenziale come riferimento.
- Input tipico: `N=1000000`, `NZ=250000000`, `MODE=irregular`, `SEED=111`, `REPEATS=3`.
- Output: [exp/results/sequential_results.csv](exp/results/sequential_results.csv)

Esecuzione:

```bash
cd exp
bash run_seq.sh
```

### Cross-validation corretta'

- Script: [exp/run_correctness_test.sh](exp/run_correctness_test.sh)
- Scopo: confronta checksum, Rayleigh e, se abilitato, dump del vettore finale tra piu' implementazioni.
- Confronta la versione sequenziale con le implementazioni parallele selezionate nello script.
- Output: CSV di validazione in [exp/results](exp/results) e dump opzionali dei vettori.

Esecuzione:

```bash
cd exp
bash run_correctness_test.sh
```

### OpenMP task vs work-sharing

- Script: [exp/run_omp_task_vs_work.sh](exp/run_omp_task_vs_work.sh)
- Scopo: confronta OpenMP task e OpenMP `omp for` variando il numero di thread.
- Thread testati: 4, 8, 16, 32.
- Output: [exp/results/omp_task_vs_work.csv](exp/results/omp_task_vs_work.csv)

### MPI + OpenMP task vs work-sharing

- Script: [exp/run_mpi_task_vs_work.sh](exp/run_mpi_task_vs_work.sh)
- Scopo: confronta le due varianti ibride al crescere dei nodi MPI, con 1 rank per nodo.
- Nodi testati: 1, 2, 4, 8.
- Output: [exp/results/mpi_task_vs_work.csv](exp/results/mpi_task_vs_work.csv)

### Sweep MPI

- Script: [exp/run_mpi_sweep.sh](exp/run_mpi_sweep.sh)
- Scopo: studia l'effetto del numero di thread per rank nella versione MPI + OpenMP.
- Varia `MPI_THREADS` su 1, 2, 4, 8, 16, 32 e adatta il numero di rank per nodo.
- Output: [exp/results/mpi_sweep_results.csv](exp/results/mpi_sweep_results.csv)

### Granularita' della SpMV

- Script: [exp/run_granularity.sh](exp/run_granularity.sh)
- Scopo: studia come cambia il tempo al variare del chunk size della SpMV.
- Confronta `CPP_THREADS`, `OMP_TASKS` e `MPI_OMP`.
- Chunk testati: 256, 512, 1024, 2048, 4096, 8192, 16384.
- Output: [exp/results/granularity_results_16threads_medians.csv](exp/results/granularity_results_16threads_medians.csv)

### Granularita' della normalizzazione

- Script: [exp/run_granularity_norm.sh](exp/run_granularity_norm.sh)
- Scopo: studia il chunk size della normalizzazione mantenendo fisso il chunk della SpMV.
- Confronta `CPP_THREADS`, `OMP_TASKS` e `MPI_OMP`.
- Chunk testati: 256, 512, 1024, 2048, 4096, 8192, 16384.
- Output: [exp/results/norm_granularity_results_16threads.csv](exp/results/norm_granularity_results_16threads.csv)

Nota: il nome del file di output conserva la storicizzazione dello script, anche se i parametri correnti usano 32 thread.

### Regular vs irregular

- Script: [exp/run_regular_irregular.sh](exp/run_regular_irregular.sh)
- Scopo: confronta il comportamento su matrice `regular` e `irregular`.
- Confronta `CPP_THREADS`, `OMP_TASKS` e `MPI_OMP`.
- Output: [exp/results/regular_vs_irregular.csv](exp/results/regular_vs_irregular.csv)

### Strong scaling

- Script: [exp/run_strong_scalability.sh](exp/run_strong_scalability.sh)
- Scopo: misura la scalabilita' forte della versione MPI + OpenMP.
- Nodi testati: 1, 2, 4, 8.
- Output: [exp/results/strong_scaling_results.csv](exp/results/strong_scaling_results.csv)

### Weak scaling

- Script: [exp/run_weak_scalability.sh](exp/run_weak_scalability.sh)
- Scopo: weak scaling con dimensione del problema proporzionale al numero di nodi.
- Output: [exp/results/weak_scaling_results.csv](exp/results/weak_scaling_results.csv)

### Weak scaling a N costante

- Script: [exp/run_weak_scalability_constant.sh](exp/run_weak_scalability_constant.sh)
- Scopo: weak scaling con `N` fisso e `NZ` che cresce con i nodi.
- Output: [exp/results/weak_scaling_constant_results.csv](exp/results/weak_scaling_constant_results.csv)

## Generazione dei grafici

Lo script [plot.py](plot.py) legge i CSV prodotti dagli esperimenti e salva i grafici nella cartella [img](img).

Esempi:

```bash
python plot.py
python plot.py --all
python plot.py --strong-scaling
python plot.py --weak-scaling
python plot.py --breakdown
python plot.py --regular-vs-irregular
```

Se non passi flag, lo script genera tutto.

### Grafici prodotti

- breakdown dei tempi per strong scaling
- breakdown dei tempi per weak scaling
- breakdown dei tempi per weak scaling con `N` costante
- confronto regular vs irregular
- grafici di strong scaling
- grafici di weak scaling

## Output dei programmi

Ogni eseguibile stampa:

- i parametri della run;
- il checksum del vettore finale;
- il Rayleigh quotient;
- il breakdown temporale delle sezioni misurate.

Le cartelle rilevanti sono:

- [exp/results](exp/results): CSV generati dagli esperimenti;
- [img](img): grafici generati da `plot.py`.

## Note pratiche

- Gli script sono pensati per ambiente Slurm.
- Per confronti affidabili usa gli stessi seed e gli stessi parametri del problema.
- La matrice `irregular` e' la piu' interessante per studiare il load balancing.
- Per i test di correttezza, il checksum e il Rayleigh devono restare coerenti tra le varianti.
