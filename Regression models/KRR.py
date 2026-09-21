#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:18:38 2025

@author: Chadawan Khamdang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, GridSearchCV

# Load data
data = pd.read_csv('Reference_data.csv')
#data = pd.read_csv('DFT_cubic_A2BX6.csv')
#data = pd.read_csv('DFT_rhombohedral_A2BX6.csv')
csvdata = data.to_numpy()

material = csvdata[:, 0]
material_type = csvdata[:, 1]
E_g = csvdata[:, -1]
X = csvdata[:, 2:-1]

# Convert data types
X_fl = np.array(X, dtype=float)
E_g_fl = np.array(E_g, dtype=float)

# Parameter grid for hyperparameter optimization
param_grid = {
    'alpha': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.1, 1]
}

# LOOCV prediction array
Pred_fl = np.zeros(len(E_g_fl))

# Store best parameters
best_params_list = []

# LOOCV loop
for i, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X_fl)):
    print(f'LOOCV Iteration {i+1}/{len(E_g_fl)}')
    
    X_train, X_test = X_fl[train_idx], X_fl[test_idx]
    y_train, y_test = E_g_fl[train_idx], E_g_fl[test_idx]
    
    krr = KernelRidge()
    grid_search = GridSearchCV(estimator=krr, param_grid=param_grid, 
                               cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_krr = grid_search.best_estimator_
    best_krr.fit(X_train, y_train)
    
    # Predict
    Pred_fl[test_idx] = best_krr.predict(X_test)
    
    # Save best params
    best_params_list.append(grid_search.best_params_)
    
    print(f'Predicted: {Pred_fl[test_idx][0]:.3f}, Actual: {y_test[0]:.3f}')
    print('Best params:', grid_search.best_params_)
    print('-' * 50)

# Final performance metrics
r2 = r2_score(E_g_fl, Pred_fl)
rmse = np.sqrt(mean_squared_error(E_g_fl, Pred_fl))
mae = mean_absolute_error(E_g_fl, Pred_fl)

print('\nFinal Model Metrics:')
print('R² Score:', r2)
print('RMSE:', rmse)
print('MAE:', mae)

# Parity plot
custom_colors = {
    'ABX3': '#563A9C',
    'A2BX6': '#8E7DBE',
    "A2BB'X6": '#8576FF',
    'A3B2X9': '#EFBBCF',
    'A4BX6': '#FFD5CD'
}

fig, ax = plt.subplots(figsize=(8, 8))

for typ in np.unique(material_type):
    idx = np.where(material_type == typ)
    ax.scatter(E_g_fl[idx], Pred_fl[idx],
               color=custom_colors[typ],
               label=typ,
               s=150, edgecolor='k', marker='o')

ax.plot([-1, 8], [-1, 8], c='k', ls='-')

ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xlabel('DFT Calculation', fontsize=26)
ax.set_ylabel('ML Prediction', fontsize=26)
ax.set_title('Band gap energy', fontsize=26, pad=10)

plt.tick_params(axis='y', width=2, length=8, labelsize=24)
plt.tick_params(axis='x', width=2, length=8, labelsize=24)

ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.savefig('KRR_LOOCV.png', dpi=450)
plt.show()