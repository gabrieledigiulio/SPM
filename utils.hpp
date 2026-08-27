#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// Deterministic PRNG / mixing

class SplitMix64 {
public:
  explicit SplitMix64(std::uint64_t seed) : state(seed) {}

  std::uint64_t next_u64() {
    state += 0x9e3779b97f4a7c15ULL;
    return mix(state);
  }

  double next_unit() {
    const std::uint64_t x = next_u64();
    return (x >> 11) * (1.0 / 9007199254740992.0);
  }

  static std::uint64_t mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
  }

private:
  std::uint64_t state;
};

// Command-line parsing

static bool read_arg_u64(int argc, char **argv, const std::string &name,
                         std::uint64_t &out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) {
      out = std::strtoull(argv[i + 1], nullptr, 10);
      return true;
    }
  }
  return false;
}

static bool read_arg_str(int argc, char **argv, const std::string &name,
                         std::string &out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

static std::uint64_t checksum_vector(const std::vector<double> &x) {
  std::uint64_t acc = 0;

  for (std::size_t i = 0; i < x.size(); ++i) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x[i], sizeof(double));
    acc ^= SplitMix64::mix(bits ^ SplitMix64::mix(i));
  }

  return acc;
}

static void dump_vector(const std::string &path, const std::vector<double> &x) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("could not open vector dump file: " + path);
  }

  out << std::setprecision(17);
  for (const double v : x) {
    out << v << '\n';
  }

  if (!out) {
    throw std::runtime_error("could not write vector dump file: " + path);
  }
}

static void usage(const char *prog) {
  std::string p(prog);
  bool is_mpi = p.find("mpi") != std::string::npos;
  bool is_iterative = p.find("iterative") != std::string::npos;
  bool is_parallel = !is_iterative;

  std::cerr
      << "Usage:\n"
      << "  " << prog
      << " -n N -nz K -m regular|irregular";
      
  if (is_parallel) {
      if (is_mpi) {
          std::cerr << " -t THREADS_PER_RANK";
      } else {
          std::cerr << " -t THREADS";
      }
      std::cerr << " -c CHUNK_SIZE -nc NORM_CHUNK_SIZE";
  }
  
  std::cerr << " [-s seed] [--dump-vector FILE]\n\n"
            << "Parameters:\n"
            << "  -n   Matrix size, NxN\n"
            << "  -nz  Total number of nonzeros\n"
            << "  -m   Matrix mode: regular or irregular\n";
            
  if (is_parallel) {
      if (is_mpi) {
          std::cerr << "  -t   Number of OpenMP threads per MPI rank\n";
      } else {
          std::cerr << "  -t   Number of working threads\n";
      }
      std::cerr << "  -c   Chunk size for SpMV tasks\n"
                << "  -nc  Chunk size for vector normalization tasks\n";
  }
  
  std::cerr << "  -s   Optional seed, default 111\n"
            << "  --dump-vector FILE\n"
            << "       Optional output file for the final normalized vector\n";
}

struct ExecutionTimers {
    double init_sec = 0.0;
    double computation_sec = 0.0;     
    double spmv_sec = 0.0;            
    double vector_ops_sec = 0.0;      
    double epoch_transition_sec = 0.0;
    double scatter_sec = 0.0;         
    double reduction_sec = 0.0;       
    double communication_sec = 0.0;   
    double total_sec = 0.0;           
};

inline std::chrono::steady_clock::time_point get_time_now() {
    return std::chrono::steady_clock::now();
}

inline double get_elapsed_time(const std::chrono::steady_clock::time_point& start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

inline void print_timer(const char* label, double value) {
    if (value > 0.0) {
        std::cout << "  " << label << " = " << value << "\n";
    }
}

inline void print_all_timers(const ExecutionTimers& t) {
    std::cout << "Time breakdown (seconds):\n";
    print_timer("Init time (sec)", t.init_sec);
    print_timer("Computation time (sec)", t.computation_sec);
    print_timer("SpMV time (sec)", t.spmv_sec);
    print_timer("Vector ops time (sec)", t.vector_ops_sec);
    print_timer("Scatter time (sec)", t.scatter_sec);
    print_timer("Reduction time (sec)", t.reduction_sec);
    print_timer("Communication time (sec)", t.communication_sec);
    print_timer("Epoch transition (sec)", t.epoch_transition_sec);
    std::cout << "Time (sec) = " << t.total_sec << "\n";
}
