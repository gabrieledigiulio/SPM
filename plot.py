import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_breakdown_plot(csv_filepath, title, filename, legend_loc='upper left'):
    try:
        if not os.path.exists(csv_filepath):
            print(f"[SKIP] File {csv_filepath} not found.")
            return

        df = pd.read_csv(csv_filepath)
        nodes = df['Nodes'].astype(str)
        comp = df['Comp_Time_Med'].values
        comm = df['Comm_Time_Med'].values
        red = df['Red_Time_Med'].values
        
        epoch = df['Epoch_Time_Med'].values if 'Epoch_Time_Med' in df.columns else np.zeros(len(nodes))
        scatt = df['Scatt_Time_Med'].values if 'Scatt_Time_Med' in df.columns else np.zeros(len(nodes))

        fig, ax = plt.subplots(figsize=(10, 6))

        p1 = ax.bar(nodes, comp, label='Computation', color='#4c72b0')
        p2 = ax.bar(nodes, comm, bottom=comp, label='Communication', color='#dd8452')
        p3 = ax.bar(nodes, red, bottom=comp+comm, label='Reduction', color='#55a868')
        p4 = ax.bar(nodes, epoch, bottom=comp+comm+red, label='Epoch Transition', color='#c44e52')
        p5 = ax.bar(nodes, scatt, bottom=comp+comm+red+epoch, label='Scatter', color='#9b59b6')

        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 1.0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                            f'{height:.2f}', ha='center', va='center', color='black')

        add_labels(p1)
        add_labels(p2)
        add_labels(p3)
        add_labels(p4)
        add_labels(p5)

        ax.set_title(title)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Time (seconds)')
        ax.legend(loc=legend_loc)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        max_height = max(comp + comm + red + epoch + scatt)
        ax.set_ylim(0, max_height * 1.2)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[OK] Saved breakdown plot: {filename}")

    except Exception as e:
        print(f"[ERROR] Failed to generate {filename}: {e}")

def plot_regular_vs_irregular(csv_filepath, output_filename):
    try:
        if not os.path.exists(csv_filepath):
            print(f"[SKIP] File '{csv_filepath}' not found.")
            return

        df = pd.read_csv(csv_filepath)
        pivot_df = df.pivot(index='Implementation', columns='Mode', values='Total_Time_Med')
        pivot_df = pivot_df.reindex(['CPP_THREADS', 'OMP_TASKS', 'MPI_OMP'])

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(pivot_df.index))

        rects1 = ax.bar(x - width/2, pivot_df['regular'], width, label='Regular', color='#2ca02c')
        rects2 = ax.bar(x + width/2, pivot_df['irregular'], width, label='Irregular', color='#d62728')

        ax.set_ylabel('Total Time (seconds)')
        ax.set_title('Impact of Memory Access Pattern (Total Time)')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"[OK] Saved regular vs irregular plot: {output_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to generate {output_filename}: {e}")

def plot_strong_scaling(csv_filepath, output_filename):
    try:
        if not os.path.exists(csv_filepath):
            print(f"[SKIP] File '{csv_filepath}' not found.")
            return

        df = pd.read_csv(csv_filepath)
        nodes = df['Nodes'].values
        total_times = df['Total_Time_Med'].values
        
        t_1 = total_times[0]
        actual_speedup = t_1 / total_times
        ideal_speedup = nodes

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(nodes, total_times, marker='o', color='blue', label='Total Time')
        ax1.set_title('Strong Scalability: Execution Time')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(nodes)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        ax2.plot(nodes, actual_speedup, marker='s', color='green', label='Actual Relative Speedup')
        ax2.plot(nodes, ideal_speedup, linestyle='--', color='red', label='Ideal Relative Speedup')
        ax2.set_title('Strong Scalability: Relative Speedup')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Speedup')
        ax2.set_xticks(nodes)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"[OK] Saved strong scaling plot: {output_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to generate {output_filename}: {e}")

def plot_weak_scaling(csv_filepath, output_filename, is_constant=False):
    try:
        if not os.path.exists(csv_filepath):
            print(f"[SKIP] File '{csv_filepath}' not found.")
            return

        df = pd.read_csv(csv_filepath)
        nodes = df['Nodes'].values
        total_times = df['Total_Time_Med'].values
        
        t_1 = total_times[0]
        efficiency = t_1 / total_times

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(nodes, total_times, marker='o', color='purple', label='Total Time')
        ax1.axhline(y=t_1, color='red', linestyle='--', label='Ideal Time')
        
        title_suffix = "(Constant N)" if is_constant else "(Scale N & NZ)"
        ax1.set_title(f'Weak Scalability: Execution Time {title_suffix}')
        ax1.set_xlabel('Number of Nodes' + ('' if is_constant else ' (Proportional Problem Size)'))
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(nodes)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        ax2.plot(nodes, efficiency, marker='s', color='green', label='Weak-Scaling Efficiency')
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Ideal Weak-Scaling Efficiency')
        
        ax2.set_title(f'Weak Scalability: Efficiency {title_suffix}')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Weak Scaling Efficiency')
        ax2.set_xticks(nodes)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"[OK] Saved weak scaling plot: {output_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to generate {output_filename}: {e}")

def plot_task_vs_work(omp_csv, mpi_csv, output_filename):
    try:
        if not os.path.exists(omp_csv) or not os.path.exists(mpi_csv):
            print(f"[SKIP] Missing CSV files for Task vs Work comparison: {omp_csv} or {mpi_csv}")
            return

        df_omp = pd.read_csv(omp_csv)
        df_mpi = pd.read_csv(mpi_csv)

        omp_tasks = df_omp[df_omp['Implementation'] == 'OMP_TASKS']
        omp_work = df_omp[df_omp['Implementation'] == 'OMP_WORKSHARING']

        mpi_tasks = df_mpi[df_mpi['Implementation'] == 'MPI_OMP_TASKS']
        mpi_work = df_mpi[df_mpi['Implementation'] == 'MPI_OMP_WORKSHARING']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(omp_tasks['Threads'], omp_tasks['Total_Time_Med'], marker='o', label='OMP Tasks', linewidth=2, color='#1f77b4')
        ax1.plot(omp_work['Threads'], omp_work['Total_Time_Med'], marker='s', label='OMP Work-Sharing', linewidth=2, linestyle='--', color='#ff7f0e')
        ax1.set_title('Single-Node Execution (OpenMP)', fontsize=14, pad=10)
        ax1.set_xlabel('Number of Threads', fontsize=12)
        ax1.set_ylabel('Total Execution Time (s)', fontsize=12)
        ax1.set_xticks(omp_tasks['Threads'])
        ax1.legend(fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(mpi_tasks['Nodes'], mpi_tasks['Total_Time_Med'], marker='o', label='MPI+OMP Tasks', linewidth=2, color='#1f77b4')
        ax2.plot(mpi_work['Nodes'], mpi_work['Total_Time_Med'], marker='s', label='MPI+OMP Work-Sharing', linewidth=2, linestyle='--', color='#ff7f0e')
        ax2.set_title('Distributed Execution (MPI + OpenMP)', fontsize=14, pad=10)
        ax2.set_xlabel('Number of Nodes', fontsize=12)
        ax2.set_ylabel('Total Execution Time (s)', fontsize=12)
        ax2.set_xticks(mpi_tasks['Nodes'])
        ax2.legend(fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved task vs work plot: {output_filename}")

    except Exception as e:
        print(f"[ERROR] Failed to generate {output_filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from experimental CSV results.")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--breakdown", action="store_true", help="Generate breakdown plots")
    parser.add_argument("--regular-vs-irregular", action="store_true", help="Generate regular vs irregular plots")
    parser.add_argument("--strong-scaling", action="store_true", help="Generate strong scaling plots")
    parser.add_argument("--weak-scaling", action="store_true", help="Generate weak scaling plots")
    parser.add_argument("--task-vs-work", action="store_true", help="Generate task vs work-sharing comparison plot")
    args = parser.parse_args()

    if not any([args.all, args.breakdown, args.regular_vs_irregular, args.strong_scaling, args.weak_scaling, args.task_vs_work]):
        args.all = True

    if args.all:
        args.breakdown = True
        args.regular_vs_irregular = True
        args.strong_scaling = True
        args.weak_scaling = True
        args.task_vs_work = True

    os.makedirs("img", exist_ok=True)

    if args.breakdown:
        generate_breakdown_plot(
            "exp/results/strong_scaling_results.csv", 
            "Strong Scalability Time Breakdown", 
            "img/strong_scaling_breakdown.png",
            legend_loc='upper right'
        )
        generate_breakdown_plot(
            "exp/results/weak_scaling_results.csv", 
            "Weak Scalability Time Breakdown (Scale N & NZ)", 
            "img/weak_scaling_breakdown.png",
            legend_loc='upper left'
        )
        generate_breakdown_plot(
            "exp/results/weak_scaling_constant_results.csv", 
            "Weak Scalability Time Breakdown (Constant N)", 
            "img/weak_scaling_constant_breakdown.png",
            legend_loc='upper left'
        )

    if args.regular_vs_irregular:
        plot_regular_vs_irregular(
            "exp/results/regular_vs_irregular.csv", 
            "img/regular_vs_irregular.png"
        )

    if args.strong_scaling:
        plot_strong_scaling(
            "exp/results/strong_scaling_results.csv", 
            "img/strong_scaling_plots.png"
        )

    if args.weak_scaling:
        plot_weak_scaling(
            "exp/results/weak_scaling_results.csv", 
            "img/weak_scalability.png",
            is_constant=False
        )
        plot_weak_scaling(
            "exp/results/weak_scaling_constant_results.csv", 
            "img/weak_scalability_constant.png",
            is_constant=True
        )

    if args.task_vs_work:
        plot_task_vs_work(
            "exp/results/omp_task_vs_work.csv",
            "exp/results/mpi_task_vs_work.csv",
            "img/task_vs_work_comparison.png"
        )
