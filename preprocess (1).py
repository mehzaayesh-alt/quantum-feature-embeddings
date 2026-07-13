# prep for the embedding project
# picking breast cancer wisconsin instead of reusing the credit data,
# want a second dataset in the portfolio anyway and this one's a common
# QML benchmark so it's easier to sanity check my numbers against papers

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_full, y = data.data, data.target
feature_names = list(data.feature_names)

print("full data:", X_full.shape)
print("class balance:", np.bincount(y))

# 6 qubits - one trainable RY per feature per layer, want small enough
# that computing statevectors during optimization doesn't take forever
N_FEATURES = 6

mi = mutual_info_classif(X_full, y, random_state=42)
top_idx = np.argsort(mi)[::-1][:N_FEATURES]
top_names = [feature_names[i] for i in top_idx]
print(f"\ntop {N_FEATURES} features by mutual information:")
for i in top_idx:
    print(f"  {feature_names[i]:25s} MI={mi[i]:.4f}")

X_sel = X_full[:, top_idx]

# standardize then squash into [-pi, pi], same as the credit project.
# the trainable scale/bias in the circuit will do further adjustment
# on top of this during training anyway
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)
X_scaled = np.clip(X_scaled, -3, 3) / 3 * np.pi

np.save("X.npy", X_scaled)
np.save("y.npy", y)
with open("feature_names.txt", "w") as f:
    f.write("\n".join(top_names))

print("\nsaved X.npy", X_scaled.shape, "y.npy", y.shape)
