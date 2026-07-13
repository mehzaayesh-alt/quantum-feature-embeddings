"""
Learned quantum feature embeddings - can a trainable encoding actually
beat a fixed/hand picked one?

Instead of training end-to-end on a classification loss (that's what I did
in the credit risk project with the VQC), this trains the embedding itself
against kernel-target alignment (KTA) - basically: adjust the encoding so
that same-class points land close together in Hilbert space and different
class points land far apart, measured via state fidelity. Then a plain SVM
gets dropped on top of whatever kernel comes out.

Circuit is a small data-reuploading embedding: each of the 6 features gets
its own trainable scale + bias, reapplied across a few layers with
entangling gates in between. The scale/bias are exactly the thing being
learned - the untrained version (scale=1, bias=0) is basically the
standard/hand picked feature map you'd use if you didn't bother training
it, so it doubles as the baseline.
"""

import numpy as np
import time, json
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

SEED = 42
rng = np.random.default_rng(SEED)

N_QUBITS = 6
N_LAYERS = 3
N_TRAIN = 80   # used both to fit the kernel/embedding AND to fit the SVM
N_TEST = 150

X = np.load("X.npy")
y = np.load("y.npy")
y_pm = np.where(y == 0, -1, 1)  # +-1 labels for KTA math

with open("feature_names.txt") as f:
    feature_names = f.read().splitlines()

# ---------- circuit ----------
x_p = ParameterVector("x", N_QUBITS)
th_p = ParameterVector("theta", N_LAYERS * N_QUBITS)
ph_p = ParameterVector("phi", N_LAYERS * N_QUBITS)

qc = QuantumCircuit(N_QUBITS)
idx = 0
for l in range(N_LAYERS):
    for q in range(N_QUBITS):
        qc.ry(th_p[idx] * x_p[q] + ph_p[idx], q)
        idx += 1
    for q in range(N_QUBITS - 1):
        qc.cz(q, q + 1)
    qc.cz(N_QUBITS - 1, 0)


def get_statevectors(X_data, params):
    """params = flat [theta(18), phi(18)]"""
    theta = params[: N_LAYERS * N_QUBITS]
    phi = params[N_LAYERS * N_QUBITS :]
    sv_list = np.empty((len(X_data), 2**N_QUBITS), dtype=complex)
    for i, x in enumerate(X_data):
        bind = {}
        for q in range(N_QUBITS):
            bind[x_p[q]] = x[q]
        for k in range(N_LAYERS * N_QUBITS):
            bind[th_p[k]] = theta[k]
            bind[ph_p[k]] = phi[k]
        bound = qc.assign_parameters(bind)
        sv_list[i] = Statevector.from_instruction(bound).data
    return sv_list


def kernel_from_statevectors(sv_a, sv_b):
    # fidelity kernel: |<psi_i|psi_j>|^2, computed directly since we have
    # full statevectors from the simulator (no need to run a swap test
    # like you'd have to on real hardware)
    overlaps = sv_a.conj() @ sv_b.T
    return np.abs(overlaps) ** 2


def kta_score(K, y_labels):
    T = np.outer(y_labels, y_labels)
    num = np.sum(K * T)
    denom = np.sqrt(np.sum(K * K) * np.sum(T * T))
    return num / denom


# ---------- data split ----------
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=SEED
)
X_test, _, y_test, _ = train_test_split(
    X_rest, y_rest, train_size=N_TEST, stratify=y_rest, random_state=SEED
)
y_train_pm = np.where(y_train == 0, -1, 1)

print(f"train={len(X_train)}  test={len(X_test)}  qubits={N_QUBITS}  layers={N_LAYERS}")

# ---------- baseline: untrained / fixed embedding ----------
# scale=1, bias=0 -> features go in exactly as given, no learned adjustment.
# this stands in for "the standard hand picked feature map"
fixed_params = np.concatenate([np.ones(N_LAYERS * N_QUBITS), np.zeros(N_LAYERS * N_QUBITS)])

sv_train_fixed = get_statevectors(X_train, fixed_params)
K_train_fixed = kernel_from_statevectors(sv_train_fixed, sv_train_fixed)
kta_fixed = kta_score(K_train_fixed, y_train_pm)
print(f"\nfixed embedding  | KTA={kta_fixed:.4f}")

# ---------- train the embedding via KTA ----------
def objective(params):
    sv = get_statevectors(X_train, params)
    K = kernel_from_statevectors(sv, sv)
    return -kta_score(K, y_train_pm)  # minimize negative alignment


t0 = time.time()
init_params = np.concatenate(
    [rng.uniform(0.5, 1.5, N_LAYERS * N_QUBITS), rng.uniform(-0.3, 0.3, N_LAYERS * N_QUBITS)]
)
res = minimize(objective, init_params, method="COBYLA", options={"maxiter": 250})
train_time = time.time() - t0
trained_params = res.x

sv_train_trained = get_statevectors(X_train, trained_params)
K_train_trained = kernel_from_statevectors(sv_train_trained, sv_train_trained)
kta_trained = kta_score(K_train_trained, y_train_pm)
print(f"trained embedding | KTA={kta_trained:.4f}  (optimized in {train_time:.1f}s, {res.nfev} evals)")

# ---------- downstream SVM eval, fixed vs trained ----------
def eval_downstream(params):
    sv_tr = get_statevectors(X_train, params)
    sv_te = get_statevectors(X_test, params)
    K_tr = kernel_from_statevectors(sv_tr, sv_tr)
    K_te = kernel_from_statevectors(sv_te, sv_tr)  # test rows vs train cols
    svm = SVC(kernel="precomputed").fit(K_tr, y_train)
    preds = svm.predict(K_te)
    return balanced_accuracy_score(y_test, preds)


acc_fixed = eval_downstream(fixed_params)
acc_trained = eval_downstream(trained_params)
print(f"\ndownstream SVM balanced accuracy")
print(f"  fixed embedding:   {acc_fixed:.3f}")
print(f"  trained embedding: {acc_trained:.3f}")

# classical baseline for reference, same 6 features
rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_train, y_train)
acc_rf = balanced_accuracy_score(y_test, rf.predict(X_test))
print(f"  random forest:     {acc_rf:.3f}")

# ---------- interpretability: what did the embedding learn to emphasize ----------
# average the |theta| scale per feature across layers - a feature the
# optimizer scaled up matters more to the resulting kernel, a feature it
# scaled toward zero got effectively dropped
theta_trained = trained_params[: N_LAYERS * N_QUBITS].reshape(N_LAYERS, N_QUBITS)
learned_importance = np.mean(np.abs(theta_trained), axis=0)

rf_importance = rf.feature_importances_

corr, pval = spearmanr(learned_importance, rf_importance)
print(f"\nlearned feature scale vs random forest importance")
for i, name in enumerate(feature_names):
    print(f"  {name:25s} learned|theta|={learned_importance[i]:.3f}   RF importance={rf_importance[i]:.3f}")
print(f"\nspearman correlation: {corr:.3f} (p={pval:.3f})")

results = {
    "kta_fixed": float(kta_fixed),
    "kta_trained": float(kta_trained),
    "acc_fixed": float(acc_fixed),
    "acc_trained": float(acc_trained),
    "acc_rf": float(acc_rf),
    "learned_importance": learned_importance.tolist(),
    "rf_importance": rf_importance.tolist(),
    "feature_names": feature_names,
    "spearman_corr": float(corr),
    "spearman_p": float(pval),
    "train_time_s": train_time,
    "n_train": N_TRAIN,
    "n_test": N_TEST,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save("K_train_fixed.npy", K_train_fixed)
np.save("K_train_trained.npy", K_train_trained)
np.save("y_train.npy", y_train)

print("\ndone, results in results.json")
