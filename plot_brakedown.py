import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_breakdown_plot(csv_filepath, title, filename, legend_loc='upper left'):
    try:
        if not os.path.exists(csv_filepath):
            print(f"File {csv_filepath} non trovato. Salto la generazione.")
            return

        df = pd.read_csv(csv_filepath)
        nodes = df['Nodes'].astype(str)
        comp = df['Comp_Time_Med'].values
        comm = df['Comm_Time_Med'].values
        red = df['Red_Time_Med'].values
        
        # Gestisce i casi in cui la colonna Scatter o Epoch potrebbe essere assente o NaN
        epoch = df['Epoch_Time_Med'].values if 'Epoch_Time_Med' in df.columns else np.zeros(len(nodes))
        scatt = df['Scatt_Time_Med'].values if 'Scatt_Time_Med' in df.columns else np.zeros(len(nodes))

        fig, ax = plt.subplots(figsize=(10, 6))

        # Creazione dei livelli della barra in pila
        p1 = ax.bar(nodes, comp, label='Computation', color='#4c72b0')
        p2 = ax.bar(nodes, comm, bottom=comp, label='Communication', color='#dd8452')
        p3 = ax.bar(nodes, red, bottom=comp+comm, label='Reduction', color='#55a868')
        p4 = ax.bar(nodes, epoch, bottom=comp+comm+red, label='Epoch Transition', color='#c44e52')
        p5 = ax.bar(nodes, scatt, bottom=comp+comm+red+epoch, label='Scatter', color='#9b59b6')

        # Funzione per inserire i testi dentro le barre
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 1.0: # Mostra label solo se lo strato è alto almeno 1s per non accavallare testi
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
        
        # Alza il limite Y per non far accavallare la legenda con le barre
        max_height = max(comp + comm + red + epoch + scatt)
        ax.set_ylim(0, max_height * 1.2)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Grafico salvato in {filename}")

    except Exception as e:
        print(f"Errore durante la generazione del plot {filename}: {e}")

if __name__ == "__main__":
    
    # 1. Strong Scalability Breakdown
    generate_breakdown_plot(
        "exp/results/strong_scaling_results.csv", 
        "Strong Scalability Time Breakdown", 
        "img/strong_scaling_breakdown.png",
        legend_loc='upper right' # Spostato a destra per non coprire la prima barra
    )
    
    # 2. Weak Scalability (Scale N & NZ)
    generate_breakdown_plot(
        "exp/results/weak_scaling_results.csv", 
        "Weak Scalability Time Breakdown (Scale N & NZ)", 
        "img/weak_scaling_breakdown.png",
        legend_loc='upper left'
    )
    
    # 3. Weak Scalability (Constant N)
    generate_breakdown_plot(
        "exp/results/weak_scaling_constant_results.csv", 
        "Weak Scalability Time Breakdown (Constant N)", 
        "img/weak_scaling_constant_breakdown.png",
        legend_loc='upper left'
    )