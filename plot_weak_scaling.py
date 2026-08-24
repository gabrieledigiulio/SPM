import pandas as pd
import matplotlib.pyplot as plt

def plot_weak_scaling(csv_filepath, output_filename):
    """
    Legge i dati di Weak Scalability da un file CSV e genera un plot affiancato
    per il Tempo di Esecuzione e la Weak-Scaling Efficiency.
    """
    try:
        # Legge il CSV
        df = pd.read_csv(csv_filepath)
        
        # Estrae i dati
        nodes = df['Nodes'].values
        total_times = df['Total_Time_Med'].values
        
        # Calcola la Weak-Scaling Efficiency: E_w = T(1) / T(p)
        t_1 = total_times[0]
        efficiency = t_1 / total_times

        # Configura la figura con due subplot affiancati
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Grafico 1: Execution Time ---
        ax1.plot(nodes, total_times, marker='o', color='purple', label='Total Time')
        # Il tempo ideale nel weak scaling è costante e pari al tempo su 1 nodo (T_1)
        ax1.axhline(y=t_1, color='red', linestyle='--', label='Ideal Time')
        
        ax1.set_title('Weak Scalability: Execution Time')
        ax1.set_xlabel('Number of Nodes (Proportional Problem Size)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(nodes)
        ax1.set_ylim(bottom=0) # Forza l'asse Y a partire da 0 come nell'immagine
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # --- Grafico 2: Weak-Scaling Efficiency ---
        ax2.plot(nodes, efficiency, marker='s', color='green', label='Weak-Scaling Efficiency')
        # L'efficienza ideale nel weak scaling è costantemente 1.0 (100%)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Ideal Weak-Scaling Efficiency')
        
        ax2.set_title('Weak Scalability: Weak-Scaling Efficiency')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Weak Scaling Efficiency')
        ax2.set_xticks(nodes)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        # Ottimizza il layout e salva l'immagine
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        print(f"Grafico salvato con successo in {output_filename}")

    except Exception as e:
        print(f"Si è verificato un errore durante la generazione del plot: {e}")

if __name__ == "__main__":
    # Assicurati che il path punti al primo CSV di Weak Scaling che abbiamo generato
    plot_weak_scaling("exp/results/weak_scaling_results.csv", "weak_scalability.png")