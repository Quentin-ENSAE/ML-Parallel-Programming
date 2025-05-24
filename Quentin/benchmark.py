import numpy as np
import time
import math
from math import sqrt
from scipy.stats import t as student_t

# ------------------------
# Fonction d'attention (inchangée)
# ------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def run_attention_once_np(x, block_size):
    seq_len, d_model = x.shape
    output = np.zeros_like(x)
    for i in range(0, seq_len, block_size):
        end = min(i + block_size, seq_len)
        Q = x[i:end]
        K = x[i:end]
        V = x[i:end]
        scores = np.matmul(Q, K.T) / np.sqrt(d_model)
        weights = softmax(scores)
        output[i:end] = np.matmul(weights, V)
    return output



# ------------------------
# Mesure de temps
# ------------------------
def measure_time(x, block_size):
    start = time.perf_counter()
    _ = run_attention_once_np(x, block_size)
    return time.perf_counter() - start


# ------------------------
# Grille exhaustive
# ------------------------

def exhaustive_grid_until_confidence(
    x,
    block_sizes,
    initial_repeats=2,
    free_runs=2,
    confidence=0.95,
    max_iter=100,
    verbose=True
):
    alpha = 1 - confidence
    block_sizes = sorted(set(block_sizes))
    results = {b: [] for b in block_sizes}

    # Pour les tracés
    best_track = []
    best_measurements = []
    p_history = []
    measure_counter = 0           # total global
    measure_per_iter = []         # liste: nombre total de mesures après chaque itération

    # Fonction interne pour mesurer ET incrémenter le compteur
    def measure_and_count(x, b):
        nonlocal measure_counter
        t = measure_time(x, b)
        measure_counter += 1
        return t

    # (0) Free runs pour chauffer les caches / JIT
    if free_runs > 0 and verbose:
        print(f"⚙️  Phase 0 : {free_runs} free runs")
    for _ in range(free_runs):
        for b in block_sizes:
            run_attention_once_np(x, b)
    if verbose and free_runs > 0:
        print("Fin des free runs\n")

    # Phase initiale
    if verbose:
        print("🔍 Phase 1 : mesures initiales")
    for b in block_sizes:
        for _ in range(initial_repeats):
            t = measure_and_count(x, b)
            results[b].append(t)
        med = np.median(results[b])
        if verbose:
            print(f"Bloc {b:<4} → médiane init : {med:.5f} s")

    # Boucle d'itérations
    for it in range(1, max_iter + 1):
        if verbose:
            print(f"\n🔁 Itération {it} : mesure de tous les blocs")

        # Mesures pour tous les blocs
        for b in block_sizes:
            t = measure_and_count(x, b)
            results[b].append(t)
            if verbose:
                print(f" Bloc {b:<4} → {t:.5f} s (n={len(results[b])})")

        # Enregistre le nombre total de mesures à la fin de cette itération
        measure_per_iter.append(measure_counter)

        # Moyennes et variances
        means = {b: np.mean(results[b]) for b in block_sizes}
        vars_  = {b: np.var(results[b], ddof=1) for b in block_sizes}
        ns     = {b: len(results[b]) for b in block_sizes}

        # Choix du best
        best = min(means, key=means.get)
        best_track.append(best)
        if verbose:
            print(f" Meilleur moyen actuel : Bloc {best} (mean={means[best]:.5f})")
        best_measurements.append(
            (measure_counter - (len(block_sizes)-block_sizes.index(best)),
             best,
             results[best][-1])
        )

        # Test de confiance pair à pair et enregistrement p-values
        all_confident = True
        p_vals = {}
        for b in block_sizes:
            if b == best:
                continue
            diff     = means[b] - means[best]
            var_diff = vars_[b] / ns[b] + vars_[best] / ns[best]
            t_stat   = diff / math.sqrt(var_diff)
            df = (var_diff**2) / (
                (vars_[b]**2)/(ns[b]**2*(ns[b]-1)) +
                (vars_[best]**2)/(ns[best]**2*(ns[best]-1))
            )
            p_value = 1 - student_t.cdf(t_stat, df)
            p_vals[b] = p_value
            if verbose:
                print(f"  -> Bloc {best} vs Bloc {b}: p={p_value:.4f}")
            if p_value >= alpha:
                all_confident = False

        p_history.append(p_vals)

        if all_confident:
            if verbose:
                print("\n✅ Confiance 95% atteinte pour le meilleur bloc.")
            break

        if it == max_iter and verbose:
            print(f"\n❌ L’algorithme n’a convergé pour aucune taille de bloc avec une confiance de {confidence*100:.0f} %.")

    if verbose and it < max_iter:
        print(f"\n✅ Résultat final : Bloc {best}, tests totaux = {measure_counter}")

    return best, results, measure_counter, best_track, best_measurements, p_history, measure_per_iter




# ------------------------
# Benchmark adaptatif
# ------------------------
"""
Benchmark adaptatif jusqu'à décision probabiliste,
avec extra repeat du best à CHAQUE itération (plus conservateur).
Compatible avec plot_all_exhaustive_graphs (7 valeurs retournées).
"""

from math import sqrt
import numpy as np
from scipy.stats import t as student_t
import matplotlib.pyplot as plt

def _prob_b_faster_than_a(mean_a, std_a, n_a, mean_b, std_b, n_b):
    var = (std_a ** 2) / n_a + (std_b ** 2) / n_b
    if var == 0:
        return 0.5
    se = sqrt(var)
    diff = mean_b - mean_a
    t_stat = diff / se
    df = var**2 / (
        (std_a**4) / (n_a**2 * (n_a - 1)) +
        (std_b**4) / (n_b**2 * (n_b - 1))
    )
    return student_t.cdf(-t_stat, df)


def smart_benchmark_probabilistic(
    x,
    block_sizes,
    *,
    p_switch: float         = 0.50,
    initial_repeats: int    = 2,
    confidence: float       = 0.90,
    free_runs: int          = 1,
    extra_repeats_best: int = 1,  # Utilisé À CHAQUE ITÉRATION
    max_iter: int           = 100,
    verbose: bool           = True,
):
    block_sizes = sorted(set(block_sizes))
    results = {b: [] for b in block_sizes}
    best_track = []  # Meilleur bloc à chaque itération
    p_stop = 1 - confidence

    # Pour le 3e graphique
    best_measurements = []     # tuples (idx, best, t)
    global_measure_counter = 0 # pour compter chaque mesure du best

    # Pour les graphes par itération
    measure_per_iter = []  # cumul du nombre de mesures à chaque itération
    total_tests = 0        # total de mesures effectuées
    p_history = []         # pour l'évolution de la p-value max

    # (1) réchauffe-caches
    if free_runs > 0 and verbose:
        print(f"⚙️  Phase 0 : {free_runs} free runs")
    for _ in range(free_runs):
        for b in block_sizes:
            run_attention_once_np(x, b)
    if verbose:
        print("Fin des free runs\n")

    # (2) mesures initiales
    if verbose:
        print("🔍 Phase 1 : mesures initiales")
    for b in block_sizes:
        for _ in range(initial_repeats):
            results[b].append(measure_time(x, b))
            total_tests += 1
        if verbose:
            print(f"Bloc {b:<4} → {np.mean(results[b]):.5f} s (n={len(results[b])})")

    # (3) best initial + répétitions bonus (utilisé uniquement à l'init)
    best = min(block_sizes, key=lambda b: np.mean(results[b]))
    if verbose:
        print(f"\nBest initial : {best}")
    for _ in range(extra_repeats_best):
        t = measure_time(x, best)
        global_measure_counter += 1
        best_measurements.append((global_measure_counter, best, t))
        results[best].append(t)
        total_tests += 1

    # Boucle principale
    for it in range(1, max_iter + 1):
        # Stats de tous les blocs
        stats = {
            b: (
                np.mean(results[b]),
                np.std(results[b], ddof=1) if len(results[b]) > 1 else 0.0,
                len(results[b]),
            )
            for b in block_sizes
        }
        mean_best, std_best, n_best = stats[best]

        # Probabilités P(b < best)
        p_better = {}
        for b in block_sizes:
            if b == best:
                continue
            mean_b, std_b, n_b = stats[b]
            p_better[b] = _prob_b_faster_than_a(
                mean_best, std_best, n_best,
                mean_b, std_b, n_b
            )

        # Ajout pour compatibilité p_history
        p_history.append(p_better.copy())
        best_track.append(best)
        measure_per_iter.append(total_tests)  # Nombre total de mesures jusqu'ici

        if verbose:
            print(f"\n🔁 Itération {it}: best = {best}")
            for b, p in sorted(p_better.items()):
                print(f"  P({b} plus rapide que {best}) = {p:.3f}")

        # Décision : stop ? switch ?
        max_p = max(p_better.values(), default=0.0)
        if max_p < p_stop:
            if verbose:
                print(f"\n✅ Arrêt, toutes proba < {p_stop:.2f}")
            break

        if max_p > p_switch:
            new_best = max(p_better, key=p_better.get)
            if verbose:
                print(f"↪️ Switch : {best} → {new_best} (p={max_p:.3f})")
            best = new_best

        # Sélection des blocs à remesurer
        to_measure = [best] + [b for b, p in p_better.items() if p >= p_stop]
        if verbose:
            print("Mesures supplémentaires :", to_measure)

        # Ajout extra_repeats_best mesures du best à CHAQUE itération
        for _ in range(extra_repeats_best):
            t = measure_time(x, best)
            total_tests += 1
            global_measure_counter += 1
            best_measurements.append((global_measure_counter, best, t))
            results[best].append(t)
            if verbose and extra_repeats_best > 0:
                print(f"  (extra) Bloc {best:<4} → {t:.5f} s (n={len(results[best])})")

        # Puis les mesures "normales" (dont une du best qui fera doublon avec l'extra repeat)
        for b in to_measure:
            t = measure_time(x, b)
            total_tests += 1
            if b == best:
                global_measure_counter += 1
                best_measurements.append((global_measure_counter, best, t))
            results[b].append(t)
            if verbose:
                print(f"  Bloc {b:<4} → {t:.5f} s (n={len(results[b])})")
        
        if it == max_iter:
            print(f"\n❌ L’algorithme n’a convergé pour aucune taille de bloc avec une confiance de {confidence*100:.0f} %.")

    if verbose:
        m_final = float(np.median(results[best]))
        print(f"\n🏁 Best final : {best} (médiane={m_final:.5f} s), mesures={total_tests}")

    # Pour compatibilité avec plot_all_exhaustive_graphs
    return best, results, total_tests, best_track, best_measurements, p_history, measure_per_iter