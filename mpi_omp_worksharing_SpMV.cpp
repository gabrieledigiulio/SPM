#include "matrix_generation.hpp"
#include "utils.hpp"
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;


// the matrix rows are statically partitioned among the ranks
// each rank holds a contiguous physical subset of them, which is never recalculated
struct BlockRange { //[row_begin, row_end)
    std::size_t row_begin = 0;
    std::size_t row_end = 0;
};

// the submatrix that each rank physically holds in memory—only a portion of the global CSR matrix.
struct LocalMatrix {
    std::size_t row_begin = 0; //offset, in global numbering
    std::size_t num_rows  = 0; // n of physical rows this rank has
    std::vector<std::uint64_t> row_ptr; // index into the rank's small local values/col_idx
    std::vector<std::uint32_t> col_idx; // which column of the vec x (0..n-1) to read
    std::vector<double> values; // non zero values belonging to the rows of this rank
};

// how to distribute the matrix among the ranks so that each has
// the same workload not the same number of rows  but the same number of non-zero elements

static std::vector<BlockRange> partition_rows_by_nnz(const CSRMatrix& A,
                                                      std::size_t num_blocks) {

    const std::size_t n = A.n; // n rows
    const std::uint64_t total_nnz = A.row_ptr[n]; // n of non zeros elements
    const std::uint64_t target_per_block = (total_nnz + num_blocks - 1) / num_blocks; // how many non zero elements we would like to assign to each block

    // create blocks
    std::vector<BlockRange> blocks;
    blocks.reserve(num_blocks);

    // init start
    std::size_t row_start = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) { // for each block
        // security check rows run out
        if (row_start >= n) {
            blocks.push_back({row_start, row_start}); // empty block is created
            continue; // next block
        }
        // how many non zeros are there before row_start
        const std::uint64_t nnz_floor = A.row_ptr[row_start];
        //the point at which this block has accumulated approximately target_per_block non-zero values
        const std::uint64_t target_prefix = nnz_floor + target_per_block;

        // end block
        std::size_t row_end;
        // if is the last one
        if (b == num_blocks - 1) {
            row_end = n; // take everything that remains
        } else {
            // find the first row where the cumulative count
            // of non-zero elements exceeds or equals our target
            row_end = std::distance(
                A.row_ptr.begin(),
                std::lower_bound(A.row_ptr.begin() + row_start, A.row_ptr.end(), target_prefix));
            row_end = std::clamp(row_end, row_start + 1, n); // places the result within a valid range
        }

        // the newly calculated block is recorded
        // and row_start is advanced to the end of this block
        blocks.push_back({row_start, row_end});
        row_start = row_end;
    }

    return blocks;
}

// computes the local matrix for each block
// matrix A
// range of rows to extract
static LocalMatrix extract_local_matrix(const CSRMatrix& A, const BlockRange& range) {
    LocalMatrix L;
    L.row_begin = range.row_begin; // global offset
    L.num_rows  = range.row_end - range.row_begin; // n rows of this block

    const std::uint64_t begin_p = A.row_ptr[range.row_begin]; // where they begin
    const std::uint64_t end_p   = A.row_ptr[range.row_end]; // where they end

    L.row_ptr.resize(L.num_rows + 1); // at least num_rows + 1
    for (std::size_t i = 0; i <= L.num_rows; ++i) { // for every rows
        L.row_ptr[i] = A.row_ptr[range.row_begin + i] - begin_p; // rebasing
    }

    // copy of col_idx and values
    L.col_idx.assign(A.col_idx.begin() + begin_p, A.col_idx.begin() + end_p);
    L.values.assign(A.values.begin() + begin_p, A.values.begin() + end_p);

    return L;
}

// MPI process memory is separate—each rank runs in its own address space
// must serialize the data and explicitly send it across the network or communication system
// local matrix L
// numeric identifier of the target rank
static void send_local_matrix(const LocalMatrix& L, int dest_rank) {
    // send dim
    // number of local rows
    // number of non-zero
    // global offset
    std::uint64_t sizes[3] = {L.num_rows, L.col_idx.size(), L.row_begin};
    MPI_Send(sizes, 3, MPI_UINT64_T, dest_rank, 0, MPI_COMM_WORLD);

    // three separate calls, one for each of the arrays that make up the LocalMatrix
    MPI_Send(L.row_ptr.data(), static_cast<int>(L.row_ptr.size()), // row_ptr
              MPI_UINT64_T, dest_rank, 1, MPI_COMM_WORLD);
    MPI_Send(L.col_idx.data(), static_cast<int>(L.col_idx.size()), // col_idx
              MPI_UINT32_T, dest_rank, 2, MPI_COMM_WORLD);
    MPI_Send(L.values.data(), static_cast<int>(L.values.size()), // values
              MPI_DOUBLE, dest_rank, 3, MPI_COMM_WORLD);
}

// each rank (other than 0) executes to receive its own slice of the matrix
// source_rank > from which rank expect data
// local matrix L
static LocalMatrix recv_local_matrix(int source_rank) {
    LocalMatrix L;
    // recive the dimensions
    std::uint64_t sizes[3];
    MPI_Recv(sizes, 3, MPI_UINT64_T, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // MPI_Recv is blocking
    // extract the three values
    L.num_rows  = sizes[0];
    const std::uint64_t nnz_local = sizes[1];
    L.row_begin = sizes[2];

    // pre alloc
    L.row_ptr.resize(L.num_rows + 1);
    L.col_idx.resize(nnz_local);
    L.values.resize(nnz_local);
    // recive three data arrays
    MPI_Recv(L.row_ptr.data(), static_cast<int>(L.row_ptr.size()), // row_ptr
             MPI_UINT64_T, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(L.col_idx.data(), static_cast<int>(L.col_idx.size()), // col_idx
             MPI_UINT32_T, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(L.values.data(), static_cast<int>(L.values.size()), // values
             MPI_DOUBLE, source_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return L;
}

// coordinator
// n, nz, seed, mode > generating matrix params
// rank, num_ranks > identity of this process and the total number of processes
// all_blocks > each rank will have this list populated with all the blocks from all the ranks
// generation_sec, distribution_sec > timer
static LocalMatrix setup_and_distribute(std::size_t n, std::uint64_t nz,
                                        std::uint64_t seed, const std::string& mode,
                                        int rank, int num_ranks,
                                        std::vector<BlockRange>& all_blocks,
                                        double& generation_sec,
                                        double& distribution_sec) {
    // inti timer
    generation_sec = 0.0;
    distribution_sec = 0.0;

    // rank 0 generates and distributes
    if (rank == 0) {

        // matrix generation
        int ok = 1;
        std::string error_msg;
        GeneratedMatrix G;

        // timer
        const double tg0 = MPI_Wtime();
        try {
            G = generate_matrix(n, nz, seed, mode);
        } catch (const std::exception& e) {
            ok = 0;
            error_msg = e.what();
        }
        generation_sec = MPI_Wtime() - tg0;

        // collective communication: rank 0 sends the same data to all other ranks
        // in the communicator in a single synchronized operation
        // all ranks must call MPI_Bcast it is a collective synchronization point
        // no rank proceeds until all have participated in this call
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) {
            throw std::runtime_error("matrix generation failed: " + error_msg);
        }
        // only rank 0 prints the stats
        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n";

        // calculate balanced partitioning for nnz
        all_blocks = partition_rows_by_nnz(G.A, static_cast<std::size_t>(num_ranks));

        // spread partitioning to all ranks
        // rank 0 broadcasts the entire all_blocks vector to all other ranks
        MPI_Bcast(all_blocks.data(), num_ranks * 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        // extraction and distribution of the pieces
        const double td0 = MPI_Wtime();
        LocalMatrix local;
        // rank 0 iterates over all ranks
        for (int r = 0; r < num_ranks; ++r) {
            LocalMatrix piece = extract_local_matrix(G.A, all_blocks[r]); // extracts the corresponding matrix segment
            if (r == 0) { // already in its own memory space
                local = std::move(piece);
            } else { // sends it to the other rank
                send_local_matrix(piece, r);
            }
        }

        distribution_sec = MPI_Wtime() - td0;

        return local;
    } else { // the branch of the other ranks
        // any rank other than 0
        int ok = 1;
        // participates in the same MPI_Bcast
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) {
            throw std::runtime_error("matrix generation failed on rank 0");
        }
        // pre alloc
        all_blocks.resize(num_ranks);
        // participate in the second MPI_Bcast
        MPI_Bcast(all_blocks.data(), num_ranks * 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        // receive one's own slice of the matrix from rank 0
        return recv_local_matrix(0);
    }
}

// compute shift

static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// dot product over the local slice [begin, begin + count)
// a, b > input vectors
// begin > start index inside a/b
// count > number of elements to process
// chunk size
static double local_dot_omp_for(const std::vector<double>& a,
                                const std::vector<double>& b,
                                std::size_t begin, std::size_t count,
                                std::size_t chunk_size) {

    static double sum = 0.0;

    #pragma omp single
    {
        sum = 0.0;
    }

    #pragma omp for schedule(static, chunk_size) reduction(+:sum)
    for (std::size_t i = 0; i < count; ++i) {
        sum += a[begin + i] * b[begin + i]; // sum element-wise products of this slice
    }
    return sum;
}

// scales v in place by factor over the local slice [begin, begin + count)
// v > input/output vector
// begin, count > local range
// factor > scalar multiplier computed by the caller
// chunk size
static void local_scale_omp_for(std::vector<double>& v,
                                std::size_t begin, std::size_t count,
                                double factor, std::size_t chunk_size) {
    // omp for > distributes the loop iterations across the active thread team
    #pragma omp for schedule(static, chunk_size)
    for (std::size_t i = 0; i < count; ++i) {
        v[begin + i] *= factor;
    }
}

// normalizes v in place (L2 norm) over the local slice [begin, begin + count)
// This is only valid when the whole range is already local to the rank.
// When the norm spans multiple ranks, the caller must combine the partial
// dot products with MPI_Allreduce and then pass the global factor here.
static void normalize_local_omp_for(std::vector<double>& v,
                                    std::size_t begin, std::size_t count,
                                    std::size_t chunk_size) {
    const double sumsq = local_dot_omp_for(v, v, begin, count, chunk_size); // identical on every thread, thanks to the reduction

    // every thread recomputes inv locally instead of broadcasting it
    const double inv = 1.0 / std::sqrt(sumsq);

    local_scale_omp_for(v, begin, count, inv, chunk_size);
}

// multiplication of this rank's local matrix slice by the full vector x
// L > local matrix of this rank
// x_full > global input vector
// y_phys > local output vector
// chunk size
static void spmv_local_omp_for(const LocalMatrix& L,
                               const std::vector<double>& x_full,
                               std::vector<double>& y_phys,
                               std::size_t chunk_size) {
    // schedule(dynamic) for the same reason as in the shared-memory version:
    // rows with more nonzeros cost more, so light chunks finish earlier and
    // threads can keep pulling new work from the queue
    #pragma omp for schedule(dynamic, chunk_size)
    for (std::size_t i = 0; i < L.num_rows; ++i) {
        double sum = 0.0;
        for (std::uint64_t p = L.row_ptr[i]; p < L.row_ptr[i + 1]; ++p) { // iterate only the nonzero values
            sum += L.values[p] * x_full[L.col_idx[p]]; // multiply by the matching x entry
        }
        y_phys[i] = sum; // save the row result in the local output vector
    }
}

// MPI_Allgatherv lets every rank receive all local pieces in a single vector
// assembled in the correct global order

// AllgatherPlan stores the row counts and displacements needed by MPI_Allgatherv
struct AllgatherPlan {
    std::vector<int> counts; // number of physical rows of rank r
    std::vector<int> displs; // physical offset (== row_begin) of the rank r
};

// the construction function
static AllgatherPlan build_allgather_plan(const std::vector<BlockRange>& all_blocks) {
    AllgatherPlan plan;
    // pre alloc
    plan.counts.reserve(all_blocks.size());
    plan.displs.reserve(all_blocks.size());

    for (const BlockRange& b : all_blocks) { // for every block
        plan.counts.push_back(static_cast<int>(b.row_end - b.row_begin)); // n rows has that rank
        plan.displs.push_back(static_cast<int>(b.row_begin)); // corresponds to the correct position in the final vec
    }

    return plan;
}

// the point at which the local pieces from each rank are actually
// assembled into a single complete vector, visible across all ranks

// y_phys_local > the local component of this rank
// y_phys_full > the output vector, of dimension n
// plan > the calculated plan
static void gather_full_vector(const std::vector<double>& y_phys_local,
                               std::vector<double>& y_phys_full,
                               const AllgatherPlan& plan) {
    // collective communication called by one thread per rank
    // y_phys_local.data() > the buffer that this rank sends its own local piece
    // static_cast<int>(y_phys_local.size()) > n elements is this rank sending
    // MPI_DOUBLE > type of the sent data
    // y_phys_full.data() > the destination buffer
    // plan.counts.data() > an array: n elements to receive from each rank
    // plan.displs.data() > an array: the position in the destination buffer where the contribution is to be written
    // MPI_DOUBLE > type of received data
    // MPI_COMM_WORLD > the group of processes involved
    MPI_Allgatherv(y_phys_local.data(), static_cast<int>(y_phys_local.size()), MPI_DOUBLE,
                   y_phys_full.data(), plan.counts.data(), plan.displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
}

// the shift used after the calculation
// to decide where to write the already calculated result
// y_phys_full > the fully assembled vec
// x_next > output vec
// row_shift > current shift
// n > dim
// chunk size
static void apply_shift_permutation_omp_for(const std::vector<double>& y_phys_full,
                                            std::vector<double>& x_next,
                                            std::size_t row_shift,
                                            std::size_t n,
                                            std::size_t chunk_size) {
    #pragma omp for schedule(static, chunk_size)
    for (std::size_t p = 0; p < n; ++p) { // iter over the vector y_phys_full
        const std::size_t dest = (p + row_shift) % n; // where the physical value p is to be written
        x_next[dest] = y_phys_full[p]; // writes the value moving it from its physical location to its logical destination
    }
}

// collect output data

struct IterativeResult {
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;
    std::size_t final_row_shift = 0;
};

struct MpiIterativeResult {
    IterativeResult result;
    ExecutionTimers timers;
};

// coordinator
// local matrix L
// dim n
// seed
// plan
// rank
// chunk size
// norm chunk size
// optional pointer where the result is to be stored
static MpiIterativeResult
iterative_spmv_evolving_mpi_omp_for(const LocalMatrix& L, std::size_t n,
                                    std::uint64_t seed,
                                    const AllgatherPlan& plan,
                                    int rank,
                                    std::size_t chunk_size,
                                    std::size_t norm_chunk_size,
                                    std::vector<double>* final_vector_out) {
    // creates obj for stats
    ExecutionTimers timers;
    IterativeResult result;

    const std::size_t shift_rows = compute_shift_rows(n); // how many rows to shift per epoch
    std::size_t row_shift = 0;  // init the total displacement offset

    // pre alloc 4 buffers
    std::vector<double> x_full(n); // complete logical vector
    std::vector<double> x_next(n);  // permutation destination buffer
    std::vector<double> y_phys(L.num_rows); // local SpMV output indexed by phys row
    std::vector<double> y_phys_full(n); // after the Allgatherv, row-indexed phys

    const double t_start = MPI_Wtime();

    // scratch state shared across single blocks potentially executed by different threads
    double t0 = 0.0;
    double sumsq_global = 0.0;

    // parallel -> creates the thread team for this rank
    #pragma omp parallel default(shared)
    {
        // only one thread initializes x_full; the RNG state is sequential
        #pragma omp single
        {
            t0 = MPI_Wtime();
            SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
            for (double& v : x_full) {
                v = rng.next_unit();
            }
        } // every thread waits here until x_full is fully initialized

        // normalize x_full: the vector is already complete on this rank
        normalize_local_omp_for(x_full, 0, n, norm_chunk_size);

        #pragma omp single
        { timers.init_sec = MPI_Wtime() - t0; }

        // main loop
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            // every thread evaluates the same condition
            if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                #pragma omp single
                {
                    const double te0 = MPI_Wtime();
                    row_shift = (row_shift + shift_rows) % n;
                    timers.epoch_transition_sec += MPI_Wtime() - te0;
                } // implicit barrier: everyone waits until row_shift is updated
            }

            // take t0 after the barrier above
            #pragma omp single
            { t0 = MPI_Wtime(); }

            spmv_local_omp_for(L, x_full, y_phys, chunk_size); // whole-team call

            #pragma omp single
            {
                const double spmv_elapsed = MPI_Wtime() - t0;
                timers.computation_sec += spmv_elapsed;
                timers.spmv_sec += spmv_elapsed;
                t0 = MPI_Wtime();
            }

            // local sum of squares over this rank's physical slice
            const double sumsq_local = local_dot_omp_for(y_phys, y_phys, 0, L.num_rows, norm_chunk_size);

            #pragma omp single
            {
                const double dot_elapsed = MPI_Wtime() - t0;
                timers.computation_sec += dot_elapsed;
                timers.vector_ops_sec += dot_elapsed;

                // MPI_Allreduce combines the partial sums from every rank
                const double tr0 = MPI_Wtime();
                MPI_Allreduce(&sumsq_local, &sumsq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                timers.reduction_sec += MPI_Wtime() - tr0;
            } // implicit barrier: no thread reads sumsq_global before it's set

            // every thread now sees the same global norm
            const double inv = 1.0 / std::sqrt(sumsq_global);

            #pragma omp single
            { t0 = MPI_Wtime(); }

            // scale the local slice with the global factor
            local_scale_omp_for(y_phys, 0, L.num_rows, inv, norm_chunk_size); // whole-team call

            #pragma omp single
            {
                const double scale_elapsed = MPI_Wtime() - t0;
                timers.computation_sec += scale_elapsed;
                timers.vector_ops_sec += scale_elapsed;

                // assemble the local pieces into y_phys_full
                const double tcm0 = MPI_Wtime();
                gather_full_vector(y_phys, y_phys_full, plan);
                timers.communication_sec += MPI_Wtime() - tcm0;
                t0 = MPI_Wtime();
            } // implicit barrier: y_phys_full is complete on every thread/rank from here on

            apply_shift_permutation_omp_for(y_phys_full, x_next, row_shift, n, chunk_size); // whole-team call

            #pragma omp single
            {
                const double scatter_elapsed = MPI_Wtime() - t0;
                timers.computation_sec += scatter_elapsed;
                timers.scatter_sec += scatter_elapsed;
                x_full.swap(x_next); // swap done safely, by a single thread
            } // implicit barrier: no one starts the new iteration until the swap is done
        }

        // Rayleigh requires one final multiplication
        #pragma omp single
        { t0 = MPI_Wtime(); }

        spmv_local_omp_for(L, x_full, y_phys, chunk_size); // whole-team call

        #pragma omp single
        {
            const double final_spmv_elapsed = MPI_Wtime() - t0;
            timers.computation_sec += final_spmv_elapsed;
            timers.spmv_sec += final_spmv_elapsed;

            const double tfc0 = MPI_Wtime();
            gather_full_vector(y_phys, y_phys_full, plan); // reassemble the full physical vector on every rank
            timers.communication_sec += MPI_Wtime() - tfc0;
        } // implicit barrier: y_phys_full is complete everywhere before the branch below

        // the final permutation and the rayleigh/checksum computation run on rank 0 only
        if (rank == 0) {
            #pragma omp single
            { t0 = MPI_Wtime(); }

            apply_shift_permutation_omp_for(y_phys_full, x_next, row_shift, n, chunk_size); // whole-team call, rank 0 only
            // x_next now holds the logical y = A_shifted * x_full
            // x_full remains the final vector to report

            #pragma omp single
            {
                const double final_scatter_elapsed = MPI_Wtime() - t0;
                timers.computation_sec += final_scatter_elapsed;
                timers.scatter_sec += final_scatter_elapsed;
                t0 = MPI_Wtime();
            }

            // reduction inside local_dot_omp_for guarantees rayleigh_local is identical on every thread
            const double rayleigh_local = local_dot_omp_for(x_full, x_next, 0, n, norm_chunk_size); // x . (A_shifted * x)

            #pragma omp single
            {
                const double final_dot_elapsed = MPI_Wtime() - t0;
                timers.computation_sec += final_dot_elapsed;
                timers.vector_ops_sec += final_dot_elapsed;

                // computes Rayleigh
                result.rayleigh = rayleigh_local;

                // computes checksum
                const double tchk0 = MPI_Wtime();
                result.checksum = checksum_vector(x_full);
                const double chk_elapsed = MPI_Wtime() - tchk0;
                timers.computation_sec += chk_elapsed;
                timers.vector_ops_sec += chk_elapsed;
                result.final_row_shift = row_shift;

                // save vector
                if (final_vector_out != nullptr) {
                    *final_vector_out = x_full;
                }
            }
        }

        #pragma omp single
        { timers.total_sec = MPI_Wtime() - t_start; }
    } // end of persistent parallel region

    return MpiIterativeResult{result, timers};
}

// timer for process phases
static double reduce_and_print_timer(const char* label, double local_value,
                                     int num_ranks, int rank) {
    double max_value = 0.0;
    double sum_value = 0.0;

    MPI_Reduce(&local_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_value, &sum_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double avg_value = sum_value / static_cast<double>(num_ranks);
        std::cout << label << " = " << max_value << "\n";
        std::cout << "  " << label
                  << " avg=" << avg_value
                  << " max=" << max_value
                  << " imbalance=" << (max_value - avg_value) << "\n";
    }

    return max_value;
}


int main(int argc, char** argv) {
    // tell MPI which level of multithreading support is required
    int provided = MPI_THREAD_SINGLE;
    // program will have multiple threads but only one specific thread will ever make MPI calls
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // process identity
    int rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // verifies that the MPI implementation has actually granted the requested level
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "Error: the MPI implementation does not support MPI_THREAD_FUNNELED "
                      << "(required to safely mix MPI and OpenMP work-sharing)\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    //init var
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t threads_arg, chunk_size, norm_chunk_arg;
    std::string mode;
    std::string dump_vector_path;

    // check args
    const bool args_ok = read_arg_u64(argc, argv, "-n", n64) &&
                         read_arg_u64(argc, argv, "-nz", nz) &&
                         read_arg_str(argc, argv, "-m", mode) &&
                         read_arg_u64(argc, argv, "-t", threads_arg) &&
                         read_arg_u64(argc, argv, "-c", chunk_size) &&
                         read_arg_u64(argc, argv, "-nc", norm_chunk_arg);

    if (!args_ok) {
        if (rank == 0) {
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // optional args
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    // read input as a wide integer then cast to the native type used for the matrix
    const std::size_t n = static_cast<std::size_t>(n64);
    //omp thread count setup
    if (threads_arg > 0) {
        omp_set_num_threads(static_cast<int>(threads_arg));
    }
    const int omp_threads = omp_get_max_threads(); // n threads will actually be used

    // reuse chunk_size otherwise use the user's explicit value
    const std::size_t norm_chunk_size = (norm_chunk_arg == 0)
        ? static_cast<std::size_t>(chunk_size)
        : static_cast<std::size_t>(norm_chunk_arg);

    if (rank == 0) {
        std::cout << "SPARSE_ITERATION_MPI_OMP_WORKSHARING\n";
        std::cout << "MPI Ranks: " << num_ranks << " | OpenMP Threads/rank: " << omp_threads << "\n";
        std::cout << "SpMV Chunk Size: " << chunk_size << " | Norm Chunk Size: " << norm_chunk_size << "\n";
    }

    try {
        // set up var
        std::vector<BlockRange> all_blocks;
        double generation_sec = 0.0;
        double distribution_sec = 0.0;

        // each rank obtains its own LocalMatrix L
        const LocalMatrix L = setup_and_distribute(n, nz, seed, mode, rank, num_ranks,
                                                    all_blocks, generation_sec, distribution_sec);

        // only rank 0 prints the distribution time
        if (rank == 0) {
            std::cout << "distribution_time_sec=" << distribution_sec << "\n\n";
        }

        // development of the communication plan
        const AllgatherPlan plan = build_allgather_plan(all_blocks);

        // prepare final_vector_out pointer
        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out =
            (rank == 0 && !dump_vector_path.empty()) ? &final_vector : nullptr;

        // synchronize all ranks before starting to measure the algorithm's execution time
        MPI_Barrier(MPI_COMM_WORLD);

        // start the func
        const MpiIterativeResult mpi_result =
            iterative_spmv_evolving_mpi_omp_for(L, n, seed, plan, rank,
                                                static_cast<std::size_t>(chunk_size),
                                                norm_chunk_size, final_vector_out);

        if (rank == 0) {
            std::cout << "Time breakdown (seconds):\n";
        }
        reduce_and_print_timer("Init time (sec)", mpi_result.timers.init_sec, num_ranks, rank);
        reduce_and_print_timer("Computation time (sec)", mpi_result.timers.computation_sec, num_ranks, rank);
        reduce_and_print_timer("SpMV time (sec)", mpi_result.timers.spmv_sec, num_ranks, rank);
        reduce_and_print_timer("Vector ops time (sec)", mpi_result.timers.vector_ops_sec, num_ranks, rank);
        reduce_and_print_timer("Scatter time (sec)", mpi_result.timers.scatter_sec, num_ranks, rank);
        reduce_and_print_timer("Reduction time (sec)", mpi_result.timers.reduction_sec, num_ranks, rank);
        reduce_and_print_timer("Communication time (sec)", mpi_result.timers.communication_sec, num_ranks, rank);
        reduce_and_print_timer("Epoch transition (sec)", mpi_result.timers.epoch_transition_sec, num_ranks, rank);
        const double total_sec_max =
            reduce_and_print_timer("Total time (sec)", mpi_result.timers.total_sec, num_ranks, rank);

        if (rank == 0) {
            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << mpi_result.result.rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << mpi_result.result.checksum << std::dec << "\n";

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec) = " << total_sec_max << "\n";

            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, final_vector);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}