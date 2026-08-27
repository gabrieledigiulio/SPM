import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. TIME BREAKDOWN PLOTS
# ==========================================
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

# ==========================================
# 2. REGULAR VS IRREGULAR PLOTS
# ==========================================
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

# ==========================================
# 3. STRONG SCALING PLOTS
# ==========================================
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

# ==========================================
# 4. WEAK SCALING PLOTS
# ==========================================
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

# ==========================================
# MAIN CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from experimental CSV results.")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--breakdown", action="store_true", help="Generate breakdown plots")
    parser.add_argument("--regular-vs-irregular", action="store_true", help="Generate regular vs irregular plots")
    parser.add_argument("--strong-scaling", action="store_true", help="Generate strong scaling plots")
    parser.add_argument("--weak-scaling", action="store_true", help="Generate weak scaling plots")
    args = parser.parse_args()

    # If no specific flags are passed, generate everything
    if not any([args.all, args.breakdown, args.regular_vs_irregular, args.strong_scaling, args.weak_scaling]):
        args.all = True

    if args.all:
        args.breakdown = True
        args.regular_vs_irregular = True
        args.strong_scaling = True
        args.weak_scaling = True

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
