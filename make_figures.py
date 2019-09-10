import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.stats import norm

matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'serif'
from filter import christianofitzgerald_filter as cf
#------------------------------------------------------------------------------------------------------------------------
import subprocess
popen = subprocess.Popen("main.py 1", shell=True)
while popen.poll() is None:
    time.sleep(1)
#------------------------------------------------------------------------------------------------------------------------
#   DATASET
#------------------------------------------------------------------------------------------------------------------------
dataset = pd.read_csv('ts.csv')
dataset = dataset.drop(dataset.columns[[0]], axis = 1)

T = np.shape(dataset['p_bank'])[0]
T = list(range(T))
#------------------------------------------------------------------------------------------------------------------------
#   MACROECONOMY
#------------------------------------------------------------------------------------------------------------------------
Y = np.array([np.log(1+(dataset['C'][i] + dataset['I'][i])) for i in range(len(dataset['C']))])
C = np.array([np.log(1+(dataset['C'][i])) for i in range(len(dataset['C']))])
I = np.array([np.log(1+(dataset['I'][i])) for i in range(len(dataset['C']))])
d_C = np.diff(C)
d_I = np.diff(I)
d_Y = np.diff(Y)

cycle_C , trend_C = cf(C, low = 2, high = 24, drift = False)
cycle_I , trend_I = cf(I, low = 2, high = 24, drift = False)
cycle_Y , trend_Y = cf(Y, low = 2, high = 24, drift = False)

credit = np.array([np.log(1+(dataset['r_ell_firms'][i])) for i in range(len(dataset['C']))])
cycle_credit, trend_credit = cf(credit, low = 2, high = 24, drift = False)

price = np.array([np.log(1+(dataset['p'][i])) for i in range(len(dataset['C']))])
wages = np.array([np.log(1+(dataset['w'][i])) for i in range(len(dataset['C']))])
cycle_price, trend_price = cf(price, low = 2, high = 24, drift = False)
cycle_wages, trend_wages = cf(wages, low = 2, high = 24, drift = False)

price_inflation = np.diff(price)
wages_inflation = np.diff(wages)

unemp = np.array([np.log(1+(dataset['unemp'][i])) for i in range(len(dataset['C']))])
cycle_unemp, trend_unemp = cf(unemp, low = 2, high = 24, drift = False)

G = np.array([np.log(1+(dataset['government_budget'][i])) for i in range(len(dataset['C']))])
cycle_G, trend_G = cf(G, low = 2, high = 24, drift = False)
#
#   GDP Decomposition - Trend
#
fig, ax = plt.subplots()

color = 'tab:red'
ax.plot(T, C, color = color, linestyle = '--', label = r'$C_t$: Consumption')
ax.plot(T, trend_C, color = color)

color = 'tab:blue'
ax.plot(T, I, color = color, linestyle = '--', label = r'$I_t$: Investment')
ax.plot(T, trend_I, color = color)

color = 'tab:green'
ax.plot(T, Y, color = color, linestyle = '--', label = r'$Y_t$: GDP')
ax.plot(T, trend_Y, color = color)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=False, shadow=False, ncol=3)
ax.set_xlabel(r'Time')
ax.set_title(r'GDP decomposition - Trend')

fig.tight_layout()
plt.show()

#
#   GDP Decomposition - Cycle
#
fig, ax = plt.subplots()

color = 'tab:red'
ax.plot(T, cycle_C, color = color, label = r'$C_t$: Consumption')

color = 'tab:green'
ax.plot(T, cycle_Y, color = color, label = r'$Y_t$: GDP')

ax.set_xlabel(r'Time')

color = 'tab:blue'
ax.plot(T, cycle_I, color = color, label = r'$I_t$: Investment') # ax2

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=False, shadow=False, ncol=3)
ax.set_title(r'GDP decomposition - Cycle')

fig.tight_layout()
plt.show()

#
#   Credit and Investment - Trend
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.set_ylabel(r'Investment $I_t$', color = color)
ax1.plot(T, I, color=color, linestyle = '--')
ax1.plot(T, trend_I, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'Credit $\ell_t$', color=color)
ax2.plot(T, credit, color=color, linestyle = '--')
ax2.plot(T, trend_credit, color = color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title(r'Credit and Investment - Trend')

fig.tight_layout()
plt.show()

#
#   Credit and Investment - Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_I, color=color, linestyle = '-', label = r'$I_t$')

color = 'tab:blue'
ax1.plot(T, cycle_credit, color=color, linestyle = '-', label = r'$\ell_t$')

ax1.legend()
ax1.set_title(r'Credit and Investment - Cycle')

fig.tight_layout()
plt.show()

#
#   Prices and Wage - Trend
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.set_ylabel(r'Price $p_t$', color = color)
ax1.plot(T, price, color=color, linestyle = '--')
ax1.plot(T, trend_price, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'Wage $w_t$', color=color)
ax2.plot(T, wages, color=color, linestyle = '--')
ax2.plot(T, trend_wages, color = color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title(r'Price and Wage - Trend')

fig.tight_layout()
plt.show()

#
#   Prices and Wage - Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_price, color=color, linestyle = '-', label = r'$p_t$')

color = 'tab:blue'
ax1.plot(T, cycle_wages, color=color, linestyle = '-', label = r'$w_t$')

ax1.legend()
ax1.set_title(r'Price and Wage - Cycle')

fig.tight_layout()
plt.show()

#
#  Price Phillips Curve
#
price_lowess = sm.nonparametric.lowess(price_inflation, dataset['unemp'][1:])
lowess_x = list(zip(*price_lowess))[1]
lowess_y = list(zip(*price_lowess))[0]

fig, ax = plt.subplots()

ax.scatter(price_inflation, dataset['unemp'][1:], color = 'green', marker = 'v')
ax.plot(lowess_x, lowess_y, color = 'r', linestyle = '-', label = 'Lowess Fit')

ax.set_xlabel(r'Inflation $\pi_t$')
ax.set_ylabel(r'Unemployment $U_t$')
ax.set_title(r'Prices Phillips Curve')
ax.legend()

fig.tight_layout()
plt.show()

#
#  Wages Phillips Curve
#
wages_lowess = sm.nonparametric.lowess(wages_inflation, dataset['unemp'][1:])
lowess_x = list(zip(*wages_lowess))[1]
lowess_y = list(zip(*wages_lowess))[0]

fig, ax = plt.subplots()

ax.scatter(wages_inflation, dataset['unemp'][1:], color = 'green', marker = 'v')
ax.plot(lowess_x, lowess_y, color = 'r', linestyle = '-', label = 'Lowess Fit')

ax.set_xlabel(r'Inflation $\pi_t$')
ax.set_ylabel(r'Unemployment $U_t$')
ax.set_title(r'Prices Phillips Curve')
ax.legend()

fig.tight_layout()
plt.show()

#
#  Unemployment and GDP Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_Y, color=color, linestyle = '-', label = r'$u_t$')

color = 'tab:blue'
ax1.plot(T, cycle_unemp, color=color, linestyle = '-', label = r'$y_t$')

ax1.legend()
ax1.set_title(r'Unemployment and GDP - Cycle')

fig.tight_layout()
plt.show()

#
#  Public Expenditure and GDP - Trend
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.set_ylabel(r'Public Consumption $G_t$', color = color)
ax1.plot(T, G, color=color, linestyle = '--')
ax1.plot(T, trend_G, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'GDP $Y_t$', color=color)
ax2.plot(T, Y, color=color, linestyle = '--')
ax2.plot(T, trend_Y, color = color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title(r'GDP and Public Expenditure - Trend')

fig.tight_layout()
plt.show()

#
#  Public Expenditure and GDP - Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_G, color=color, linestyle = '-', label = r'$G_t$')

color = 'tab:blue'
ax1.plot(T, cycle_Y, color=color, linestyle = '-', label = r'$Y_t$')

ax1.legend()
ax1.set_title(r'GDP and Public Expenditure - Cycle')

fig.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------------
#   FINANCE
#------------------------------------------------------------------------------------------------------------------------
p_bank = np.array([np.log(1+(dataset['p_bank'][i])) for i in range(len(dataset['C']))])
cycle_pbank, trend_pbank = cf(p_bank, low = 2, high = 24, drift = False)

d_bank = np.array([np.log(1+(dataset['d_bank'][i])) for i in range(len(dataset['C']))])
cycle_dbank, trend_dbank = cf(d_bank, low = 2, high = 24, drift = False)

credit = np.array([np.log(1+(dataset['r_ell_firms'][i])) for i in range(len(dataset['C']))])
cycle_credit, trend_credit = cf(credit, low = 2, high = 24, drift = False)

leverage = np.array([np.log(1+(dataset['leverage_bank'][i])) for i in range(len(dataset['C']))])
cycle_leverage, trend_leverage = cf(leverage, low = 2, high = 24, drift = False)

interest = np.array([np.log(1+(dataset['realized_interest_rate'][i])) for i in range(len(dataset['C']))])
cycle_interest, trend_interest = cf(interest, low = 2, high = 24, drift = False)

#
#  Price and Dividends - Trend
#
fig, ax = plt.subplots()

color = 'tab:red'
ax.plot(T, p_bank, color = color, linestyle = '--', label = r'$p_t$')
ax.plot(T, trend_pbank, color = color)

color = 'tab:green'
ax.plot(T, d_bank, color = color, linestyle = '--', label = r'$d_t$')
ax.plot(T, trend_dbank, color = color)

ax.legend()
ax.set_xlabel(r'Time')
ax.set_title(r'Prices and Dividends - Trend')

fig.tight_layout()
plt.show()

#
#  Price and Dividends - Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_pbank, color=color, linestyle = '-', label = r'$p_t$')

color = 'tab:green'
ax1.plot(T, cycle_dbank, color=color, linestyle = '-', label = r'$d_t$')

ax1.legend()
ax1.set_title(r'Prices and Dividends - Cycle')

fig.tight_layout()
plt.show()

#
# Credit and Leverage - Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_leverage, color=color, linestyle = '-', label = r'$\lambda_t$')

color = 'tab:green'
ax1.plot(T, cycle_credit, color=color, linestyle = '-', label = r'$\ell_t$')

ax1.legend()
ax1.set_title(r'Credit and Leverage - Cycle')

fig.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------------
#   MICROECONOMICS
#------------------------------------------------------------------------------------------------------------------------
panel_firmsize = pd.read_csv('firmsize.csv')
panel_banksize = pd.read_csv('banksize.csv')
panel_firmproductivity = pd.read_csv('firmproductivity.csv')

firmsize = np.array([np.log(1+(dataset['A_firm'][i])) for i in range(len(dataset['C']))])
cycle_firmsize, trend_firmsize = cf(firmsize, low = 2, high = 24, drift = False)

banksize = np.array([np.log(1+(dataset['A_bank'][i])) for i in range(len(dataset['C']))])
cycle_banksize, trend_banksize = cf(banksize, low = 2, high = 24, drift = False)

firmgrowth = panel_firmsize.diff()
productivitygrowth = panel_firmproductivity.diff()

firmsize_0 = panel_firmsize.iloc[0,:].values.tolist()
firmsize_1 = panel_firmsize.iloc[49,:].values.tolist()
firmsize_2 = panel_firmsize.iloc[99,:].values.tolist()
firmsize_3 = panel_firmsize.iloc[149,:].values.tolist()

firmgrowth_0 = firmgrowth.iloc[1,:].values.tolist()
firmgrowth_1 = firmgrowth.iloc[49,:].values.tolist()
firmgrowth_2 = firmgrowth.iloc[99,:].values.tolist()
firmgrowth_3 = firmgrowth.iloc[149,:].values.tolist()

banksize_0 = panel_banksize.iloc[0,:].values.tolist()
banksize_1 = panel_banksize.iloc[49,:].values.tolist()
banksize_2 = panel_banksize.iloc[99,:].values.tolist()
banksize_3 = panel_banksize.iloc[149,:].values.tolist()

productivity_0 = panel_firmproductivity.iloc[0,:].values.tolist()
productivity_1 = panel_firmproductivity.iloc[49,:].values.tolist()
productivity_2 = panel_firmproductivity.iloc[99,:].values.tolist()
productivity_3 = panel_firmproductivity.iloc[149,:].values.tolist()

productivitygrowth_0 = productivitygrowth.iloc[1,:].values.tolist()
productivitygrowth_1 = productivitygrowth.iloc[49,:].values.tolist()
productivitygrowth_2 = productivitygrowth.iloc[99,:].values.tolist()
productivitygrowth_3 = productivitygrowth.iloc[149,:].values.tolist()
#
# Firm Size and GDP - Trend
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.set_ylabel(r'$A^f_t$', color = color)
ax1.plot(T, firmsize, color=color, linestyle = '--')
ax1.plot(T, trend_firmsize, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(r'$Y_t$', color=color)
ax2.plot(T, Y, color=color, linestyle = '--')
ax2.plot(T, trend_Y, color = color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title(r'GDP and Firm Size - Trend')

fig.tight_layout()
plt.show()

#
# Firm Size and GDP - Cycle
#
fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time')
color = 'tab:red'
ax1.plot(T, cycle_Y, color=color, linestyle = '-', label = r'$Y_t$')

color = 'tab:green'
ax1.plot(T, cycle_firmsize, color=color, linestyle = '-', label = r'$A^f_t$')

ax1.legend()
ax1.set_title(r'GDP and Firm Size - Cycle')

fig.tight_layout()
plt.show()

#
# Firm Size - Density
#
M_0 = max(firmsize_0) ; M_1 = max(firmsize_1) ; M_2 = max(firmsize_2) ; M_3 = max(firmsize_3)
m_0 = min(firmsize_0) ; m_1 = min(firmsize_1) ; m_2 = min(firmsize_2) ; m_3 = min(firmsize_3)
M_0 = min(M_0, 30) ; M_1 = min(M_1, 30) ; M_2 = min(M_2, 30) ; M_3 = min(M_3, 30)

m = min(m_0, m_1, m_2, m_3) ; M = max(M_0, M_1, M_2, M_3)
bins_0 = np.linspace(m,M,1000) ; bins_1 = np.linspace(m,M,1000) ; bins_2 = np.linspace(m,M,1000) ; bins_3 = np.linspace(m,M,1000)
kde_0 = gaussian_kde(firmsize_0)
kde_1 = gaussian_kde(firmsize_1)
kde_2 = gaussian_kde(firmsize_2)
kde_3 = gaussian_kde(firmsize_3)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Size')
ax1.set_ylabel(r'Frequency')

# color = 'yellow'
# ax1.plot(bins_0, kde_0(bins_0), color=color, linestyle = '-', label = r'$A^f_0$')

color = 'green'
ax1.plot(bins_1, kde_1(bins_1), color=color, linestyle = '-', label = r'$A^f_{50}$')

color = 'blue'
ax1.plot(bins_2, kde_2(bins_2), color=color, linestyle = '-', label = r'$A^f_{100}$')

color = 'red'
ax1.plot(bins_3, kde_3(bins_3), color=color, linestyle = '-', label = r'$A^f_{150}$')

ax1.legend()
ax1.set_title(r'Firm Size Density')

fig.tight_layout()
plt.show()
#
# Firm Growth - Density
#
M_0 = max(firmgrowth_0) ; M_1 = max(firmgrowth_1) ; M_2 = max(firmgrowth_2) ; M_3 = max(firmgrowth_3)
m_0 = min(firmgrowth_0) ; m_1 = min(firmgrowth_1) ; m_2 = min(firmgrowth_2) ; m_3 = min(firmgrowth_3)
M_0 = min(M_0, 0.4) ; M_1 = min(M_1, 0.4) ; M_2 = min(M_2, 0.4) ; M_3 = min(M_3, 0.4)

m = min(m_0, m_1, m_2, m_3) ; M = max(M_0, M_1, M_2, M_3)
bins_0 = np.linspace(m,M,1000) ; bins_1 = np.linspace(m,M,1000) ; bins_2 = np.linspace(m,M,1000) ; bins_3 = np.linspace(m,M,1000)
kde_0 = gaussian_kde(firmgrowth_0)
kde_1 = gaussian_kde(firmgrowth_1)
kde_2 = gaussian_kde(firmgrowth_2)
kde_3 = gaussian_kde(firmgrowth_3)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Size Growth')
ax1.set_ylabel(r'Frequency')

# color = 'yellow'
# ax1.plot(bins_0, kde_0(bins_0), color=color, linestyle = '-', label = r'$\Delta A^f_0$')

color = 'green'
ax1.plot(bins_1, kde_1(bins_1), color=color, linestyle = '-', label = r'$\Delta A^f_{50}$')

color = 'blue'
ax1.plot(bins_2, kde_2(bins_2), color=color, linestyle = '-', label = r'$\Delta A^f_{100}$')

color = 'red'
ax1.plot(bins_3, kde_3(bins_3), color=color, linestyle = '-', label = r'$\Delta A^f_{150}$')

ax1.legend()
ax1.set_title(r'Firm Growth Density')

fig.tight_layout()
plt.show()

#
# Firm Productivity - Density
#
M_0 = max(productivity_0) ; M_1 = max(productivity_1) ; M_2 = max(productivity_2) ; M_3 = max(productivity_3)
m_0 = min(productivity_0) ; m_1 = min(productivity_1) ; m_2 = min(productivity_2) ; m_3 = min(productivity_3)
M_0 = min(M_0, 30) ; M_1 = min(M_1, 30) ; M_2 = min(M_2, 30) ; M_3 = min(M_3, 30)

m = min(m_0, m_1, m_2, m_3) ; M = max(M_0, M_1, M_2, M_3)
bins_0 = np.linspace(m,M,1000) ; bins_1 = np.linspace(m,M,1000) ; bins_2 = np.linspace(m,M,1000) ; bins_3 = np.linspace(m,M,1000)
kde_0 = gaussian_kde(productivity_0)
kde_1 = gaussian_kde(productivity_1)
kde_2 = gaussian_kde(productivity_2)
kde_3 = gaussian_kde(productivity_3)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Productivity')
ax1.set_ylabel(r'Frequency')

# color = 'yellow'
# ax1.plot(bins_0, kde_0(bins_0), color=color, linestyle = '-', label = r'$A^f_0$')

color = 'green'
ax1.plot(bins_1, kde_1(bins_1), color=color, linestyle = '-', label = r'$\xi^f_{50}$')

color = 'blue'
ax1.plot(bins_2, kde_2(bins_2), color=color, linestyle = '-', label = r'$\xi^f_{100}$')

color = 'red'
ax1.plot(bins_3, kde_3(bins_3), color=color, linestyle = '-', label = r'$\xi^f_{150}$')

ax1.legend()
ax1.set_title(r'Firm Productivity Density')

fig.tight_layout()
plt.show()

#
# Firm Productivity Growth  Density
#
M_0 = max(productivitygrowth_0) ; M_1 = max(productivitygrowth_1) ; M_2 = max(productivitygrowth_2) ; M_3 = max(productivitygrowth_3)
m_0 = min(productivitygrowth_0) ; m_1 = min(productivitygrowth_1) ; m_2 = min(productivitygrowth_2) ; m_3 = min(productivitygrowth_3)
M_0 = min(M_0, 0.8) ; M_1 = min(M_1, 0.8) ; M_2 = min(M_2, 0.8) ; M_3 = min(M_3, 0.8)

m = min(m_0, m_1, m_2, m_3) ; M = max(M_0, M_1, M_2, M_3)
bins_0 = np.linspace(m,M,1000) ; bins_1 = np.linspace(m,M,1000) ; bins_2 = np.linspace(m,M,1000) ; bins_3 = np.linspace(m,M,1000)
kde_0 = gaussian_kde(productivitygrowth_0)
kde_1 = gaussian_kde(productivitygrowth_1)
kde_2 = gaussian_kde(productivitygrowth_2)
kde_3 = gaussian_kde(productivitygrowth_3)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Productivity Growth')
ax1.set_ylabel(r'Frequency')

# color = 'yellow'
# ax1.plot(bins_0, kde_0(bins_0), color=color, linestyle = '-', label = r'$\Delta A^f_0$')

color = 'green'
ax1.plot(bins_1, kde_1(bins_1), color=color, linestyle = '-', label = r'$\Delta \xi_{50}$')

color = 'blue'
ax1.plot(bins_2, kde_2(bins_2), color=color, linestyle = '-', label = r'$\Delta \xi_{100}$')

color = 'red'
ax1.plot(bins_3, kde_3(bins_3), color=color, linestyle = '-', label = r'$\Delta \xi_{150}$')

ax1.legend()
ax1.set_title(r'Firm Productivity Growth Density')

fig.tight_layout()
plt.show()

#
# Output Growth -  Density
#
d_Y = np.diff(Y)
m_Y = min(d_Y) ; M_Y = max(d_Y) ; bins_Y = np.linspace(m_Y, M_Y, 1000)
kde_Y = gaussian_kde(d_Y)

mu_Y = np.mean(d_Y) ; var_Y = np.std(d_Y)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Size')
color = 'blue'
ax1.set_ylabel(r'Frequency - Kernel', color = color)
ax1.plot(bins_Y, kde_2(bins_Y), color=color, linestyle = '-', label = r'Kernel Density')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel(r'Frequency - Gaussian', color = color)
ax2.plot(bins_Y, norm.pdf(bins_Y,mu_Y,var_Y), color=color, linestyle = '-', label = r'Gaussian Fit')
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title(r'Output Growth Density')

fig.tight_layout()
plt.show()
