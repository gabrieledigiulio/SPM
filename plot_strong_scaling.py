import pandas as pd
import matplotlib.pyplot as plt

def plot_strong_scaling(csv_filepath, output_filename):
    """
    Legge i dati di Strong Scaling da un file CSV e genera un plot affiancato
    per il Tempo di Esecuzione Totale e lo Speedup Relativo.
    """
    try:
        # Legge il CSV
        df = pd.read_csv(csv_filepath)
        
        # Estrae i dati necessari
        nodes = df['Nodes'].values
        total_times = df['Total_Time_Med'].values
        
        # Calcola lo Speedup Relativo: T(1) / T(p)
        t_1 = total_times[0]
        actual_speedup = t_1 / total_times
        
        # Lo Speedup Ideale per lo Strong Scaling è lineare rispetto ai nodi: p
        ideal_speedup = nodes

        # Configura la figura con due subplot affiancati
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Grafico 1: Tempo di Esecuzione ---
        ax1.plot(nodes, total_times, marker='o', color='blue', label='Total Time')
        ax1.set_title('Strong Scalability: Execution Time')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(nodes)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # --- Grafico 2: Speedup Relativo ---
        ax2.plot(nodes, actual_speedup, marker='s', color='green', label='Actual Relative Speedup')
        ax2.plot(nodes, ideal_speedup, linestyle='--', color='red', label='Ideal Relative Speedup')
        ax2.set_title('Strong Scalability: Relative Speedup')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Speedup')
        ax2.set_xticks(nodes)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        # Ottimizza il layout e salva l'immagine
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        print(f"Grafico salvato con successo in {output_filename}")

    except Exception as e:
        print(f"Si è verificato un errore durante la generazione del plot: {e}")

if __name__ == "__main__":
    # Sostituisci "results/strong_scaling_results.csv" con il percorso reale se diverso
    plot_strong_scaling("exp/results/strong_scaling_results.csv", "strong_scaling_plots.png")