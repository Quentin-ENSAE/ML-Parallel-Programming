import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_measures_per_iteration(measure_per_iter):
    """
    Histogramme du nombre de mesures réalisées à chaque itération.
    """
    measures_per_iteration = np.diff([0] + measure_per_iter)
    iterations = np.arange(1, len(measures_per_iteration) + 1)
    plt.figure(figsize=(16, 6))
    plt.bar(iterations, measures_per_iteration, color='skyblue')
    plt.xlabel("Itération")
    plt.ylabel("Nombre de mesures réalisées")
    plt.title("Nombre de mesures réalisées à chaque itération")
    plt.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_best_track(best_track, step, figsize=(24, 6)):
    """
    Affiche l'évolution du meilleur bloc sélectionné à chaque itération.
    """
    iterations = list(range(1, len(best_track) + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, best_track, marker='o')
    ax.set_xlabel("Itération")
    ax.set_ylabel("Meilleur bloc")
    ax.set_title("Évolution du bloc sélectionné")
    ax.grid(True)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    plt.tight_layout()
    plt.show()

def plot_time_distribution(results, blocks, figsize=(24, 6)):
    """
    Affiche la distribution (boxplot) des temps de mesure pour chaque taille de bloc.
    """
    plt.figure(figsize=figsize)
    plt.boxplot([results[b] for b in blocks],
                labels=[str(b) for b in blocks],
                showfliers=False)
    plt.xlabel("Taille de bloc")
    plt.ylabel("Temps mesuré (s)")
    plt.title("Distribution des temps par bloc")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_best_time_colored(best_meas, figsize=(24, 6)):
    """
    Évolution du temps mesuré pour le best, code couleur pour les changements de bloc.
    """
    indices = [idx for idx, _, _ in best_meas]
    blocks_seq = [bsz for _, bsz, _ in best_meas]
    times_seq  = [t   for _, _, t in best_meas]
    colors = ['red' if curr != prev else 'blue'
              for prev, curr in zip(blocks_seq, blocks_seq[1:])]
    plt.figure(figsize=figsize)
    for i in range(len(times_seq)-1):
        plt.plot([indices[i], indices[i+1]],
                 [times_seq[i], times_seq[i+1]],
                 color=colors[i], linewidth=2)
    plt.scatter(indices[0], times_seq[0], color='blue')
    plt.xlabel("Numéro de mesure du best")
    plt.ylabel("Temps mesuré (s)")
    plt.title("Évolution du temps du best\n(rouge = changement de bloc)")
    plt.grid(True)
    plt.legend([
        plt.Line2D([0],[0], color='blue', lw=2),
        plt.Line2D([0],[0], color='red',  lw=2)
    ], ["Même bloc", "Changement"])
    plt.tight_layout()
    plt.show()

def plot_pvalue_evolution(best_track, p_history, step=1, figsize=(24, 6)):
    """
    Affiche l'évolution de la p-value maximale à chaque itération,
    points colorés selon la taille du best et seuil p_stop.
    """
    p_max = [max(d.values()) if d else 0.0 for d in p_history]
    iterations = np.arange(1, len(p_max) + 1)
    unique_blocks = sorted(set(best_track))
    cmap = plt.cm.tab10_r
    color_map = {b: cmap(i % 10) for i, b in enumerate(unique_blocks)}
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, p_max,
            color='gray', linestyle='-', linewidth=1, alpha=0.5,
            zorder=1, label='max p-value')
    for b in unique_blocks:
        xs = [i+1 for i, bb in enumerate(best_track) if bb == b]
        ys = [p_max[i] for i, bb in enumerate(best_track) if bb == b]
        ax.scatter(xs, ys,
                   color=color_map[b],
                   edgecolor='white',
                   s=100,
                   zorder=2,
                   label=f'Bloc {b}')
    ax.axhline(0.5,
               color='blue',
               linestyle='--',
               linewidth=1.5,
               zorder=0,
               label='seuil changement de bloc size optimal = 0.5')

    ax.axhline(0.05,
               color='red',
               linestyle='--',
               linewidth=1.5,
               zorder=0,
               label='seuil p_stop = 0.05')
    ax.set_xlim(0, len(p_max))
    ax.set_xlabel("Itération")
    ax.set_ylabel("max P(b < best)")
    ax.set_title("Évolution de la p-value maximale par itération")
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(),
              by_label.keys(),
              ncol=3,
              loc='upper right',
              framealpha=0.9)
    plt.tight_layout()
    plt.show()

def plot_all_graphs(
    blocks,
    best_track,
    best_measurements,
    results,
    measure_per_iter,
    p_history,
    step=1,
    figsize=(24, 6)
):
    """
    Trace tous les graphiques pertinents du benchmark en appelant les 5 fonctions ci-dessus.
    """
    plot_measures_per_iteration(measure_per_iter)
    plot_best_track(best_track, step, figsize)
    plot_time_distribution(results, blocks, figsize)
    plot_best_time_colored(best_measurements, figsize)
    plot_pvalue_evolution(best_track, p_history, step, figsize)
