import numpy as np
import pandas as pd
import csv
import time
import copy
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller as adf
from filter import christianofitzgerald_filter as cf
#----------------------------------------------------
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'serif'
#----------------------------------------------------
df = pd.read_csv('panel.csv')
df.index = df['Unnamed: 0']
df = df.drop(df.columns[[0]], axis = 1)

Y  = pd.read_csv('Y_panel.csv') ; Y = Y.drop(Y.columns[[0]], axis = 1)
C  = pd.read_csv('C_panel.csv') ; C = C.drop(C.columns[[0]], axis = 1)
I  = pd.read_csv('I_panel.csv') ; I = I.drop(I.columns[[0]], axis = 1)
credit = pd.read_csv('credit_panel.csv') ; credit = credit.drop(credit.columns[[0]], axis = 1)
price = pd.read_csv('price_panel.csv') ; price = price.drop(price.columns[[0]], axis = 1)
wages = pd.read_csv('wages_panel.csv') ; wages = wages.drop(wages.columns[[0]], axis = 1)
price_inflation = pd.read_csv('price_inflation_panel.csv') ; price_inflation = price_inflation.drop(price_inflation.columns[[0]], axis = 1)
wages_inflation = pd.read_csv('wages_inflation_panel.csv') ; wages_inflation = wages_inflation.drop(wages_inflation.columns[[0]], axis = 1)
unemp = pd.read_csv('unemp_panel.csv') ; unemp = unemp.drop(unemp.columns[[0]], axis = 1)
G = pd.read_csv('G_panel.csv') ; G = G.drop(G.columns[[0]], axis = 1)
p_bank = pd.read_csv('p_bank_panel.csv') ; p_bank = p_bank.drop(p_bank.columns[[0]], axis = 1)
d_bank = pd.read_csv('d_bank_panel.csv') ; d_bank = d_bank.drop(d_bank.columns[[0]], axis = 1)
leverage = pd.read_csv('leverage_panel.csv') ; leverage = leverage.drop(leverage.columns[[0]], axis = 1)
interest = pd.read_csv('interest_panel.csv') ; interest = interest.drop(interest.columns[[0]], axis = 1)
productivity = pd.read_csv('productivity_panel.csv') ; productivity = productivity.drop(productivity.columns[[0]], axis = 1)

Y_filter  = pd.read_csv('Y_panel_filter.csv') ; Y_filter = Y_filter.drop(Y_filter.columns[[0]], axis = 1)
C_filter  = pd.read_csv('C_panel_filter.csv') ; C_filter = C_filter.drop(C_filter.columns[[0]], axis = 1)
I_filter  = pd.read_csv('I_panel_filter.csv') ; I_filter = I_filter.drop(I_filter.columns[[0]], axis = 1)
credit_filter = pd.read_csv('credit_panel_filter.csv') ; credit_filter = credit_filter.drop(credit_filter.columns[[0]], axis = 1)
price_filter = pd.read_csv('price_panel_filter.csv') ; price_filter = price_filter.drop(price_filter.columns[[0]], axis = 1)
wages_filter = pd.read_csv('wages_panel_filter.csv') ; wages_filter = wages_filter.drop(wages_filter.columns[[0]], axis = 1)
price_inflation_filter = pd.read_csv('price_inflation_panel_filter.csv') ; price_inflation_filter = price_inflation_filter.drop(price_inflation_filter.columns[[0]], axis = 1)
wages_inflation_filter = pd.read_csv('wages_inflation_panel_filter.csv') ; wages_inflation_filter = wages_inflation_filter.drop(wages_inflation_filter.columns[[0]], axis = 1)
unemp_filter = pd.read_csv('unemp_panel_filter.csv') ; unemp_filter = unemp_filter.drop(unemp_filter.columns[[0]], axis = 1)
G_filter = pd.read_csv('G_panel_filter.csv') ; G_filter = G_filter.drop(G_filter.columns[[0]], axis = 1)
p_bank_filter = pd.read_csv('p_bank_panel_filter.csv') ; p_bank_filter = p_bank_filter.drop(p_bank_filter.columns[[0]], axis = 1)
d_bank_filter = pd.read_csv('d_bank_panel_filter.csv') ; d_bank_filter = d_bank_filter.drop(d_bank_filter.columns[[0]], axis = 1)
leverage_filter = pd.read_csv('leverage_panel_filter.csv') ; leverage_filter = leverage_filter.drop(leverage_filter.columns[[0]], axis = 1)
interest_filter = pd.read_csv('interest_panel_filter.csv') ; interest_filter = interest_filter.drop(interest_filter.columns[[0]], axis = 1)
productivity_filter = pd.read_csv('productivity_panel_filter.csv') ; productivity_filter = productivity_filter.drop(productivity_filter.columns[[0]], axis = 1)

#----------------------------------------------------
#   COMPUTE MOMENTS AND STATISTICS
#----------------------------------------------------
T = Y.shape[0]
#   TABLE I - Average Growth Rate, Variance, Stationarity Tests for GDP components
av_Y = [] ; av_C = [] ; av_I = []
for column in Y:
    av_y = np.mean(np.diff(Y[column]))
    av_Y.append(av_y)
    av_c = np.mean(np.diff(C[column]))
    av_C.append(av_c)
    av_i = np.mean(np.diff(I[column]))
    av_I.append(av_i)

mean_growth_Y = np.mean(av_Y) * 100 ; error_mean_growth_Y = np.std(av_Y) * 100
mean_growth_C = np.mean(av_C) * 100 ; error_mean_growth_C = np.std(av_C) * 100
mean_growth_I = np.mean(av_I) * 100 ; error_mean_growth_I = np.std(av_I) * 100

var_Y = [] ; var_C = [] ; var_I = []
for column in Y_filter:
    var_y = np.std(Y_filter[column]) ** 2
    var_Y.append(var_y)
    var_c = np.std(C_filter[column]) ** 2
    var_C.append(var_c)
    var_i = np.std(I_filter[column]) ** 2
    var_I.append(var_i)

mean_var_Y = np.mean(var_Y) * 100 ; error_mean_var_Y = np.std(var_Y) * 100
mean_var_C = np.mean(var_C) * 100 ; error_mean_var_C = np.std(var_C) * 100
mean_var_I = np.mean(var_I) * 100 ; error_mean_var_I = np.std(var_I) * 100

relative_variance_Y = mean_var_Y / mean_var_Y
relative_variance_C = mean_var_C / mean_var_Y
relative_variance_I = mean_var_I / mean_var_Y

adf_Y = [] ; adf_C = [] ; adf_I = []
for column in Y:
    adf_y = adf(Y[column]) ; adf_c = adf(C[column]) ; adf_i = adf(I[column])
    adf_Y.append(adf_y)
    adf_C.append(adf_c)
    adf_I.append(adf_i)
adf_Y_99 = [] ; adf_C_99 = [] ; adf_I_99 = []
adf_Y_95 = [] ; adf_C_95 = [] ; adf_I_95 = []
adf_Y_90 = [] ; adf_C_90 = [] ; adf_I_90 = []

for t in range(len(adf_Y)):
    if adf_Y[t][0] < adf_Y[t][4]['1%']:
        adf_Y_99.append(1) ; adf_Y_95.append(1) ; adf_Y_90.append(1) #1 means series is Stationary
    elif adf_Y[t][0] > adf_Y[t][4]['1%'] and adf_Y[t][0] < adf_Y[t][4]['5%']:
        adf_Y_99.append(0) ; adf_Y_95.append(1) ; adf_Y_90.append(1)
    elif adf_Y[t][0] > adf_Y[t][4]['1%'] and adf_Y[t][0] > adf_Y[t][4]['5%'] and adf_Y[t][0] < adf_Y[t][4]['10%']:
        adf_Y_99.append(0) ; adf_Y_95.append(0) ; adf_Y_90.append(1)
    else:
        adf_Y_99.append(0) ; adf_Y_95.append(0) ; adf_Y_90.append(0)

    if adf_C[t][0] < adf_C[t][4]['1%']:
        adf_C_99.append(1) ; adf_C_95.append(1) ; adf_C_90.append(1) #1 means series is Stationary
    elif adf_C[t][0] > adf_C[t][4]['1%'] and adf_C[t][0] < adf_C[t][4]['5%']:
        adf_C_99.append(0) ; adf_C_95.append(1) ; adf_C_90.append(1)
    elif adf_C[t][0] > adf_C[t][4]['1%'] and adf_C[t][0] > adf_C[t][4]['5%'] and adf_C[t][0] < adf_C[t][4]['10%']:
        adf_C_99.append(0) ; adf_C_95.append(0) ; adf_C_90.append(1)
    else:
        adf_C_99.append(0) ; adf_C_95.append(0) ; adf_C_90.append(0)

    if adf_I[t][0] < adf_I[t][4]['1%']:
        adf_I_99.append(1) ; adf_I_95.append(1) ; adf_I_90.append(1) #1 means series is Stationary
    elif adf_I[t][0] > adf_I[t][4]['1%'] and adf_I[t][0] < adf_I[t][4]['5%']:
        adf_I_99.append(0) ; adf_I_95.append(1) ; adf_I_90.append(1)
    elif adf_I[t][0] > adf_I[t][4]['1%'] and adf_I[t][0] > adf_I[t][4]['5%'] and adf_I[t][0] < adf_I[t][4]['10%']:
        adf_I_99.append(0) ; adf_I_95.append(0) ; adf_I_90.append(1)
    else:
        adf_I_99.append(0) ; adf_I_95.append(0) ; adf_I_90.append(0)

share_Y_stationary_99 = (np.sum(adf_Y_99) / len(adf_Y_99)) * 100
share_Y_stationary_95 = (np.sum(adf_Y_95) / len(adf_Y_95)) * 100
share_Y_stationary_90 = (np.sum(adf_Y_90) / len(adf_Y_90)) * 100
share_C_stationary_99 = (np.sum(adf_C_99) / len(adf_C_99)) * 100
share_C_stationary_95 = (np.sum(adf_C_95) / len(adf_C_95)) * 100
share_C_stationary_90 = (np.sum(adf_C_90) / len(adf_C_90)) * 100
share_I_stationary_99 = (np.sum(adf_I_99) / len(adf_I_99)) * 100
share_I_stationary_95 = (np.sum(adf_I_95) / len(adf_I_95)) * 100
share_I_stationary_90 = (np.sum(adf_I_90) / len(adf_I_90)) * 100

tests_Y = [share_Y_stationary_99, share_Y_stationary_95, share_Y_stationary_90]
tests_C = [share_C_stationary_99, share_C_stationary_95, share_C_stationary_90]
tests_I = [share_I_stationary_99, share_I_stationary_95, share_I_stationary_90]
mean_growth = [mean_growth_Y, mean_growth_C, mean_growth_I]
error_mean_growth = [error_mean_growth_Y, error_mean_growth_C, error_mean_growth_I]
volatility = [mean_var_Y, mean_var_C, mean_var_I]
error_volatility = [error_mean_var_Y, error_mean_var_C, error_mean_var_I]
relative_volatility = [relative_variance_Y, relative_variance_C, relative_variance_I]

index = ['Growth Rate', '(Std. Error)', 'Volatility', '(Std. Error)', 'Relative Volatility', '(99%)' , '(95%)', '(90%)']
cols = ['Output (Y)', 'Consumption (C)', 'Investment (I)']
table = pd.DataFrame(index = index, columns = cols)
for col in table:
    index_col = cols.index(col)
    column = [mean_growth[index_col] , error_mean_growth[index_col],
              volatility[index_col], error_volatility[index_col], relative_volatility[index_col],
              tests_Y[index_col] , tests_C[index_col] , tests_I[index_col]]
    table[col] = column

with open('TABLE1.tex','w') as tf:
    tf.write(table.to_latex())
#   TABLE II - Normality Tests on GDP and Invesmtent
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import jarque_bera

s_99_Y = [] ; s_95_Y = [] ; s_90_Y = []
d_99_Y = [] ; d_95_Y = [] ; d_90_Y = []
jb_99_Y = [] ; jb_95_Y = [] ; jb_90_Y = []

s_99_I = [] ; s_95_I = [] ; s_90_I = []
d_99_I = [] ; d_95_I = [] ; d_90_I = []
jb_99_I = [] ; jb_95_I = [] ; jb_90_I = []

for column in Y_filter:
    stat , p = shapiro(Y_filter[column])
    if p > 0.1:
        s_99_Y.append(0) ; s_95_Y.append(0) ; s_90_Y.append(0)
    elif p > 0.05 and p < 0.1:
        s_99_Y.append(0) ; s_95_Y.append(0) ; s_90_Y.append(1)
    elif p > 0.01 and p < 0.05:
        s_99_Y.append(0) ; s_95_Y.append(1) ; s_90_Y.append(1)
    else:
        s_99_Y.append(1) ; s_95_Y.append(1) ; s_90_Y.append(1)

    stat , p = normaltest(Y_filter[column])
    if p > 0.1:
        d_99_Y.append(0) ; d_95_Y.append(0) ; d_90_Y.append(0)
    elif p > 0.05 and p < 0.1:
        d_99_Y.append(0) ; d_95_Y.append(0) ; d_90_Y.append(1)
    elif p > 0.01 and p < 0.05:
        d_99_Y.append(0) ; d_95_Y.append(1) ; d_90_Y.append(1)
    else:
        d_99_Y.append(1) ; d_95_Y.append(1) ; d_90_Y.append(1)

    jb = jarque_bera(Y_filter[column]) ; p = jb[1]
    if p > 0.1:
        jb_99_Y.append(0) ; jb_95_Y.append(0) ; jb_90_Y.append(0)
    elif p > 0.05 and p < 0.1:
        jb_99_Y.append(0) ; jb_95_Y.append(0) ; jb_90_Y.append(1)
    elif p > 0.01 and p < 0.05:
        jb_99_Y.append(0) ; jb_95_Y.append(1) ; jb_90_Y.append(1)
    else:
        jb_99_Y.append(1) ; jb_95_Y.append(1) ; jb_90_Y.append(1)

    stat , p = shapiro(I_filter[column])
    if p > 0.1:
        s_99_I.append(0) ; s_95_I.append(0) ; s_90_I.append(0)
    elif p > 0.05 and p < 0.1:
        s_99_I.append(0) ; s_95_I.append(0) ; s_90_I.append(1)
    elif p > 0.01 and p < 0.05:
        s_99_I.append(0) ; s_95_I.append(1) ; s_90_I.append(1)
    else:
        s_99_I.append(1) ; s_95_I.append(1) ; s_90_I.append(1)

    stat , p = normaltest(I_filter[column])
    if p > 0.1:
        d_99_I.append(0) ; d_95_I.append(0) ; d_90_I.append(0)
    elif p > 0.05 and p < 0.1:
        d_99_I.append(0) ; d_95_I.append(0) ; d_90_I.append(1)
    elif p > 0.01 and p < 0.05:
        d_99_I.append(0) ; d_95_I.append(1) ; d_90_I.append(1)
    else:
        d_99_I.append(1) ; d_95_I.append(1) ; d_90_I.append(1)

    jb = jarque_bera(I_filter[column]) ; p = jb[1]
    if p > 0.1:
        jb_99_I.append(0) ; jb_95_I.append(0) ; jb_90_I.append(0)
    elif p > 0.05 and p < 0.1:
        jb_99_I.append(0) ; jb_95_I.append(0) ; jb_90_I.append(1)
    elif p > 0.01 and p < 0.05:
        jb_99_I.append(0) ; jb_95_I.append(1) ; jb_90_I.append(1)
    else:
        jb_99_I.append(1) ; jb_95_I.append(1) ; jb_90_I.append(1)

share_shap_99_Y_normal = (np.sum(s_99_Y) / len(s_99_Y)) * 100
share_shap_95_Y_normal = (np.sum(s_95_Y) / len(s_95_Y)) * 100
share_shap_90_Y_normal = (np.sum(s_90_Y) / len(s_90_Y)) * 100

share_dag_99_Y_normal = (np.sum(d_99_Y) / len(d_99_Y)) * 100
share_dag_95_Y_normal = (np.sum(d_95_Y) / len(d_95_Y)) * 100
share_dag_90_Y_normal = (np.sum(d_90_Y) / len(d_90_Y)) * 100

share_jb_99_Y_normal = (np.sum(jb_99_Y) / len(jb_99_Y)) * 100
share_jb_95_Y_normal = (np.sum(jb_95_Y) / len(jb_95_Y)) * 100
share_jb_90_Y_normal = (np.sum(jb_90_Y) / len(jb_90_Y)) * 100

tests_Y = [[share_shap_99_Y_normal, share_shap_95_Y_normal, share_shap_90_Y_normal],
           [share_dag_99_Y_normal, share_dag_95_Y_normal, share_dag_90_Y_normal],
           [share_jb_99_Y_normal, share_jb_95_Y_normal, share_jb_90_Y_normal]]

share_shap_99_I_normal = (np.sum(s_99_I) / len(s_99_I)) * 100
share_shap_95_I_normal = (np.sum(s_95_I) / len(s_95_I)) * 100
share_shap_90_I_normal = (np.sum(s_90_I) / len(s_90_I)) * 100

share_dag_99_I_normal = (np.sum(d_99_I) / len(d_99_I)) * 100
share_dag_95_I_normal = (np.sum(d_95_I) / len(d_95_I)) * 100
share_dag_90_I_normal = (np.sum(d_90_I) / len(d_90_I)) * 100

share_jb_99_I_normal = (np.sum(jb_99_I) / len(jb_99_I)) * 100
share_jb_95_I_normal = (np.sum(jb_95_I) / len(jb_95_I)) * 100
share_jb_90_I_normal = (np.sum(jb_90_I) / len(jb_90_I)) * 100

tests_I = [[share_shap_99_I_normal, share_shap_95_I_normal, share_shap_90_I_normal],
           [share_dag_99_I_normal, share_dag_95_I_normal, share_dag_90_I_normal],
           [share_jb_99_I_normal, share_jb_95_I_normal, share_jb_90_I_normal]]

cols = ['Shapiro-Wilk (%)', 'DAgostino (%)', 'Jarque-Bera (Y)']
index  = ['Y (99%)', 'Y (95%)', 'Y (90%)', 'I (99%)', 'I (95%)', 'I (90%)']

table = pd.DataFrame(index = index, columns = cols)
for column in table:
    col = []
    index_col = cols.index(column)
    col = tests_Y[index_col] + tests_I[index_col]
    table[column] = col

with open('TABLE2.tex','w') as tf:
    tf.write(table.to_latex())
#----------------------------------------------------
#   CROSS-CORRELATION TABLE
#----------------------------------------------------
#   TABLE III - Cross correlation
id = df.index.tolist() ; id = list(dict.fromkeys(id)) ; T = Y.shape[0]
T_shift = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
Corr = []
for i in id:
    data = df.loc[i]
    corr = []
    for col in data:
        corr_X = []
        Y = np.array(data['Y'].tolist())
        X = np.array(data[col].tolist())
        for t in T_shift:
            if t > 0:
                YY = Y[0:T-t]
                XX = X[t:]
            else:
                YY = Y[-t:]
                XX = X[0:T+t]
            corrcoef = np.corrcoef(YY,XX)[0][1]
            corr_X.append(corrcoef)
        corr.append(corr_X)
    Corr.append(corr)

mean = [] ; stde = [] # dim = 15 * 9
for x in range(np.shape(Corr)[1]):
    mean_X = []
    stde_X = []
    for y in range(np.shape(Corr)[2]):
        mean_XX = []
        stde_XX = []
        for z in range(np.shape(Corr)[0]):
            mean_XX.append(Corr[z][x][y])
            stde_XX.append(Corr[z][x][y])
        mean_X.append(np.mean(mean_XX))
        stde_X.append(np.std(stde_XX))
    mean.append(mean_X)
    stde.append(stde_X)

id = ['Output', 'Output', 'Consumption','Consumption', 'Investment','Investment', 'Credit','Credit', 'Price','Price', 'Wage','Wage', 'Price Inflation','Price Inflation',
      'Wage Inflation','Wage Inflation', 'Unemployment', 'Unemployment', 'Public Expenditure', 'Public Expenditure', 'Bank Price', 'Bank Price',
      'Bank Dividend', 'Bank Dividend', 'Leverage', 'Leverage', 'Interest rate', 'Interest rate', 'Productivity', 'Productivity']

cols = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
CorrelationTable = pd.DataFrame(index = id, columns = cols)

for col in CorrelationTable:
    column = []
    index_col = cols.index(col)
    for i in range(len(mean)):
        column.append(mean[i][index_col])
        column.append(stde[i][index_col])
    CorrelationTable[col] = column

with open('TABLE4.tex','w') as tf:
    tf.write(CorrelationTable.to_latex())
#----------------------------------------------------
#   CREDIT MARKET AND REAL ACTIVITY
#----------------------------------------------------
Y  = pd.read_csv('Y_panel.csv') ; Y = Y.drop(Y.columns[[0]], axis = 1)
C  = pd.read_csv('C_panel.csv') ; C = C.drop(C.columns[[0]], axis = 1)
I  = pd.read_csv('I_panel.csv') ; I = I.drop(I.columns[[0]], axis = 1)

index = Y_filter.index ; cols = Y_filter.columns
df_CrY = pd.DataFrame(index = index, columns = cols)
for col in df_CrY:
    YY = Y_filter[col].tolist()
    CC = credit_filter[col].tolist()
    T = len(YY)
    column = [CC[i] / YY[i] for i in range(T)]

    df_CrY[col] = column

mean_CY = [] ; dropped_index = []
for col in df_CrY:
    mean_CY.append(np.mean(df_CrY[col]))
for i in range(len(mean_CY)):
    if mean_CY[i] > 5*np.mean(mean_CY) or mean_CY[i] < 0.1*np.mean(mean_CY):
        dropped_index.append(i)

growth_Y = [] ; var_Y = [] ; skew_Y = [] ; kurt_Y = []
growth_C = [] ; var_C = [] ; skew_C = [] ; kurt_C = []
growth_I = [] ; var_I = [] ; skew_I = [] ; kurt_I = []
CrY = []
for col in df_CrY:
    index_col = df_CrY.columns.tolist().index(col)
    if index_col not in dropped_index:
        ggrowth_Y = np.mean(np.diff(Y[col])) * 100
        growth_Y.append(ggrowth_Y)
        vvar_Y = (np.std(Y_filter[col]) ** 2) * 100
        var_Y.append(vvar_Y)
        sskew_Y = skew(Y_filter[col])
        skew_Y.append(sskew_Y)
        kkurt_Y = kurtosis(Y_filter[col])
        kurt_Y.append(kkurt_Y)

        ggrowth_C = (np.mean(np.diff(C[col]))) * 100
        growth_C.append(ggrowth_C)
        vvar_C = (np.std(C_filter[col]) ** 2) * 100
        var_C.append(vvar_C)
        sskew_C = skew(C_filter[col])
        skew_C.append(sskew_C)
        kkurt_C = kurtosis(C_filter[col])
        kurt_C.append(kkurt_C)

        ggrowth_I = (np.mean(np.diff(I[col]))) * 100
        growth_I.append(ggrowth_I)
        vvar_I = (np.std(I_filter[col]) ** 2) * 100
        var_I.append(vvar_I)
        sskew_I = skew(I_filter[col])
        skew_I.append(sskew_I)
        kkurt_I = kurtosis(I_filter[col])
        kurt_I.append(kkurt_I)

        CCrY = np.mean(df_CrY[col])
        CrY.append(CCrY)

plt.subplot(231)
plt.scatter(CrY, growth_Y)
z = np.polyfit(CrY, growth_Y, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('GDP - 1st Moment')

plt.subplot(232)
plt.scatter(CrY, growth_C)
z = np.polyfit(CrY, growth_C, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('C - 1st Moment')

plt.subplot(233)
plt.scatter(CrY, growth_I)
z = np.polyfit(CrY, growth_I, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('I - 1st Moment')

plt.subplot(234)
plt.scatter(CrY, var_Y)
z = np.polyfit(CrY, var_Y, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('GDP - 2nd Moment')

plt.subplot(235)
plt.scatter(CrY, var_C)
z = np.polyfit(CrY, var_C, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('C - 2nd Moment')

plt.subplot(236)
plt.scatter(CrY, var_I)
z = np.polyfit(CrY, var_I, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('I - 2nd Moment')

plt.tight_layout()
plt.show()

plt.subplot(231)
plt.scatter(CrY, skew_Y)
z = np.polyfit(CrY, skew_Y, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('GDP - 3rd Moment')

plt.subplot(232)
plt.scatter(CrY, skew_C)
z = np.polyfit(CrY, skew_C, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('C - 3rd Moment')

plt.subplot(233)
plt.scatter(CrY, skew_I)
z = np.polyfit(CrY, skew_I, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('I - 3rd Moment')

plt.subplot(234)
plt.scatter(CrY, kurt_Y)
z = np.polyfit(CrY, kurt_Y, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('GDP - 4th Moment')

plt.subplot(235)
plt.scatter(CrY, kurt_C)
z = np.polyfit(CrY, kurt_C, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('C - 4th Moment')

plt.subplot(236)
plt.scatter(CrY, kurt_I)
z = np.polyfit(CrY, kurt_I, 1)
p = np.poly1d(z)
plt.plot(CrY,p(CrY),"r-")
plt.xlabel(r'Credit-to-GDP')
plt.title('I - 4th Moment')

plt.tight_layout()
plt.show()
#----------------------------------------------------
#   FIRM DYNAMICS
#----------------------------------------------------
FSD = pd.read_csv('FSD.csv')
FSD.index = FSD['id']
FSD = FSD.drop(FSD.columns[[0,1,3]], axis = 1)
FGD = FSD.diff()
time = FSD['time'].tolist()
FGD['time'] = time

checkpoint = [50, 100, 150, 200, 250]
#   TABLE VI - Mean, Variance, Skewness, Kurtosis and Normality on FSD
mean_50 = [] ; mean_100 = [] ; mean_150 = [] ; mean_200 = [] ; mean_250 = []
stde_50 = [] ; stde_100 = [] ; stde_150 = [] ; stde_200 = [] ; stde_250 = []
skew_50 = [] ; skew_100 = [] ; skew_150 = [] ; skew_200 = [] ; skew_250 = []
kurt_50 = [] ; kurt_100 = [] ; kurt_150 = [] ; kurt_200 = [] ; kurt_250 = []

jb_50 = [] ; jb_100 = [] ; jb_150 = [] ; jb_200 = [] ; jb_250 = []
sw_50 = [] ; sw_100 = [] ; sw_150 = [] ; sw_200 = [] ; sw_250 = []
for index, row in FSD.iterrows():
    row = row.tolist()
    if row[0] == 50:
        mmean_50 = np.mean(row[1:]) ; mean_50.append(mmean_50)
        sstde_50 = np.std(row[1:]) ; stde_50.append(sstde_50)
        sskew_50 = skew(row[1:]) ; skew_50.append(sskew_50)
        kkurt_50 = kurtosis(row[1:]) ; kurt_50.append(kkurt_50)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_50.append(0)
        else:
            jb_50.append(1)
        if p_sw > 0.05:
            sw_50.append(0)
        else:
            sw_50.append(1)

    elif row[0] == 100:
        mmean_100 = np.mean(row[1:]) ; mean_100.append(mmean_100)
        sstde_100 = np.std(row[1:]) ; stde_100.append(sstde_100)
        sskew_100 = skew(row[1:]) ; skew_100.append(sskew_100)
        kkurt_100 = kurtosis(row[1:]) ; kurt_100.append(kkurt_100)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_100.append(0)
        else:
            jb_100.append(1)
        if p_sw > 0.05:
            sw_100.append(0)
        else:
            sw_100.append(1)

    elif row[0] == 150:
        mmean_150 = np.mean(row[1:]) ; mean_150.append(mmean_150)
        sstde_150 = np.std(row[1:]) ; stde_150.append(sstde_150)
        sskew_150 = skew(row[1:]) ; skew_150.append(sskew_150)
        kkurt_150 = kurtosis(row[1:]) ; kurt_150.append(kkurt_150)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_150.append(0)
        else:
            jb_150.append(1)
        if p_sw > 0.05:
            sw_150.append(0)
        else:
            sw_150.append(1)

    elif row[0] == 200:
        mmean_200 = np.mean(row[1:]) ; mean_200.append(mmean_200)
        sstde_200 = np.std(row[1:]) ; stde_200.append(sstde_200)
        sskew_200 = skew(row[1:]) ; skew_200.append(sskew_200)
        kkurt_200 = kurtosis(row[1:]) ; kurt_200.append(kkurt_200)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_200.append(0)
        else:
            jb_200.append(1)
        if p_sw > 0.05:
            sw_200.append(0)
        else:
            sw_200.append(1)

    elif row[0] == 250:
        mmean_250 = np.mean(row[1:]) ; mean_250.append(mmean_250)
        sstde_250 = np.std(row[1:]) ; stde_250.append(sstde_250)
        sskew_250 = skew(row[1:]) ; skew_250.append(sskew_250)
        kkurt_250 = kurtosis(row[1:]) ; kurt_250.append(kkurt_250)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_250.append(0)
        else:
            jb_250.append(1)
        if p_sw > 0.05:
            sw_250.append(0)
        else:
            sw_250.append(1)

mean_50 = np.mean(mean_50) ; mean_100 = np.mean(mean_100) ; mean_150 = np.mean(mean_150) ; mean_200 = np.mean(mean_200)  ; mean_250 = np.mean(mean_250)
sd_mean_50 = np.std(mean_50) ; sd_mean_100 = np.std(mean_100) ; sd_mean_150 = np.std(mean_150) ; sd_mean_200 = np.std(mean_200)  ; sd_mean_250 = np.std(mean_250)

stde_50 = np.mean(stde_50) ; stde_100 = np.mean(stde_100) ; stde_150 = np.mean(stde_150) ; stde_200 = np.mean(stde_200)  ; stde_250 = np.mean(stde_250)
sd_stde_50 = np.std(stde_50) ; sd_stde_100 = np.std(stde_100) ; sd_stde_150 = np.std(stde_150) ; sd_stde_200 = np.std(stde_200)  ; sd_stde_250 = np.std(stde_250)

skew_50 = np.mean(skew_50) ; skew_100 = np.mean(skew_100) ; skew_150 = np.mean(skew_150) ; skew_200 = np.mean(skew_200)  ; skew_250 = np.mean(skew_250)
sd_skew_50 = np.std(skew_50) ; sd_skew_100 = np.std(skew_100) ; sd_skew_150 = np.std(skew_150) ; sd_skew_200 = np.std(skew_200)  ; sd_skew_250 = np.std(skew_250)

kurt_50 = np.mean(kurt_50) ; kurt_100 = np.mean(kurt_100) ; kurt_150 = np.mean(kurt_150) ; kurt_200 = np.mean(kurt_200)  ; kurt_250 = np.mean(kurt_250)
sd_kurt_50 = np.std(kurt_50) ; sd_kurt_100 = np.std(kurt_100) ; sd_kurt_150 = np.std(kurt_150) ; sd_kurt_200 = np.std(kurt_200)  ; sd_kurt_250 = np.std(kurt_250)

mean = [mean_50, mean_100, mean_150, mean_200, mean_250]
sd_mean = [sd_mean_50, sd_mean_100, sd_mean_150, sd_mean_200, sd_mean_250]

stde = [stde_50, stde_100, stde_150, stde_200, stde_250]
sd_stde = [sd_stde_50, sd_stde_100, sd_stde_150, sd_stde_200, sd_stde_250]

skeww = [skew_50, skew_100, skew_150, skew_200, skew_250]
sd_skew = [sd_skew_50, sd_skew_100, sd_skew_150, sd_skew_200, sd_skew_250]

kurtt = [kurt_50, kurt_100, kurt_150, kurt_200, kurt_250]
sd_kurt = [sd_kurt_50, sd_kurt_100, sd_kurt_150, sd_kurt_200, sd_kurt_250]

jb = [np.sum(jb_50)*100/len(jb_50) , np.sum(jb_100)*100/len(jb_100) , np.sum(jb_150)*100/len(jb_150) , np.sum(jb_200)*100/len(jb_200) , np.sum(jb_250)*100/len(jb_250)]
sw = [np.sum(sw_50)*100/len(sw_50) , np.sum(sw_100)*100/len(sw_100) , np.sum(sw_150)*100/len(sw_150) , np.sum(sw_200)*100/len(sw_200) , np.sum(sw_250)*100/len(sw_250)]

index = ['Mean', '(Mean Std.)', 'Variance', '(Variance Std.)', 'Skewness', '(Skewness Std.)', 'Kurtosis', '(Kurtosis Std.)', 'Jarque-Bera (%)', 'Shapiro-Wilk (%)']
columns = [50, 100, 150, 200, 250]

table = pd.DataFrame(index = index, columns = columns)
for col in table:
    index_col = columns.index(col)
    column = []
    column.append(mean[index_col]) ; column.append(sd_mean[index_col])
    column.append(stde[index_col]) ; column.append(sd_stde[index_col])
    column.append(skeww[index_col]) ; column.append(sd_skew[index_col])
    column.append(kurtt[index_col]) ; column.append(sd_kurt[index_col])
    column.append(jb[index_col])
    column.append(sw[index_col])

    table[col] = column

with open('TABLE VI.tex','w') as tf:
    tf.write(table.to_latex())
#   TABLE VII - Mean, Variance, Skewness, Kurtosis and Normality on FGD
mean_50 = [] ; mean_100 = [] ; mean_150 = [] ; mean_200 = [] ; mean_250 = []
stde_50 = [] ; stde_100 = [] ; stde_150 = [] ; stde_200 = [] ; stde_250 = []
skew_50 = [] ; skew_100 = [] ; skew_150 = [] ; skew_200 = [] ; skew_250 = []
kurt_50 = [] ; kurt_100 = [] ; kurt_150 = [] ; kurt_200 = [] ; kurt_250 = []

jb_50 = [] ; jb_100 = [] ; jb_150 = [] ; jb_200 = [] ; jb_250 = []
sw_50 = [] ; sw_100 = [] ; sw_150 = [] ; sw_200 = [] ; sw_250 = []
for index, row in FGD.iterrows():
    row = row.tolist()
    if row[0] == 50:
        mmean_50 = np.mean(row[1:]) ; mean_50.append(mmean_50)
        sstde_50 = np.std(row[1:]) ; stde_50.append(sstde_50)
        sskew_50 = skew(row[1:]) ; skew_50.append(sskew_50)
        kkurt_50 = kurtosis(row[1:]) ; kurt_50.append(kkurt_50)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_50.append(0)
        else:
            jb_50.append(1)
        if p_sw > 0.05:
            sw_50.append(0)
        else:
            sw_50.append(1)

    elif row[0] == 100:
        mmean_100 = np.mean(row[1:]) ; mean_100.append(mmean_100)
        sstde_100 = np.std(row[1:]) ; stde_100.append(sstde_100)
        sskew_100 = skew(row[1:]) ; skew_100.append(sskew_100)
        kkurt_100 = kurtosis(row[1:]) ; kurt_100.append(kkurt_100)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_100.append(0)
        else:
            jb_100.append(1)
        if p_sw > 0.05:
            sw_100.append(0)
        else:
            sw_100.append(1)

    elif row[0] == 150:
        mmean_150 = np.mean(row[1:]) ; mean_150.append(mmean_150)
        sstde_150 = np.std(row[1:]) ; stde_150.append(sstde_150)
        sskew_150 = skew(row[1:]) ; skew_150.append(sskew_150)
        kkurt_150 = kurtosis(row[1:]) ; kurt_150.append(kkurt_150)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_150.append(0)
        else:
            jb_150.append(1)
        if p_sw > 0.05:
            sw_150.append(0)
        else:
            sw_150.append(1)

    elif row[0] == 200:
        mmean_200 = np.mean(row[1:]) ; mean_200.append(mmean_200)
        sstde_200 = np.std(row[1:]) ; stde_200.append(sstde_200)
        sskew_200 = skew(row[1:]) ; skew_200.append(sskew_200)
        kkurt_200 = kurtosis(row[1:]) ; kurt_200.append(kkurt_200)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_200.append(0)
        else:
            jb_200.append(1)
        if p_sw > 0.05:
            sw_200.append(0)
        else:
            sw_200.append(1)

    elif row[0] == 250:
        mmean_250 = np.mean(row[1:]) ; mean_250.append(mmean_250)
        sstde_250 = np.std(row[1:]) ; stde_250.append(sstde_250)
        sskew_250 = skew(row[1:]) ; skew_250.append(sskew_250)
        kkurt_250 = kurtosis(row[1:]) ; kurt_250.append(kkurt_250)

        jjb = jarque_bera(row[1:]) ; p_jb = jjb[1]
        ssw, p_sw = shapiro(row[1:])
        if p_jb > 0.05:
            jb_250.append(0)
        else:
            jb_250.append(1)
        if p_sw > 0.05:
            sw_250.append(0)
        else:
            sw_250.append(1)

mean_50 = np.mean(mean_50) ; mean_100 = np.mean(mean_100) ; mean_150 = np.mean(mean_150) ; mean_200 = np.mean(mean_200)  ; mean_250 = np.mean(mean_250)
sd_mean_50 = np.std(mean_50) ; sd_mean_100 = np.std(mean_100) ; sd_mean_150 = np.std(mean_150) ; sd_mean_200 = np.std(mean_200)  ; sd_mean_250 = np.std(mean_250)

stde_50 = np.mean(stde_50) ; stde_100 = np.mean(stde_100) ; stde_150 = np.mean(stde_150) ; stde_200 = np.mean(stde_200)  ; stde_250 = np.mean(stde_250)
sd_stde_50 = np.std(stde_50) ; sd_stde_100 = np.std(stde_100) ; sd_stde_150 = np.std(stde_150) ; sd_stde_200 = np.std(stde_200)  ; sd_stde_250 = np.std(stde_250)

skew_50 = np.mean(skew_50) ; skew_100 = np.mean(skew_100) ; skew_150 = np.mean(skew_150) ; skew_200 = np.mean(skew_200)  ; skew_250 = np.mean(skew_250)
sd_skew_50 = np.std(skew_50) ; sd_skew_100 = np.std(skew_100) ; sd_skew_150 = np.std(skew_150) ; sd_skew_200 = np.std(skew_200)  ; sd_skew_250 = np.std(skew_250)

kurt_50 = np.mean(kurt_50) ; kurt_100 = np.mean(kurt_100) ; kurt_150 = np.mean(kurt_150) ; kurt_200 = np.mean(kurt_200)  ; kurt_250 = np.mean(kurt_250)
sd_kurt_50 = np.std(kurt_50) ; sd_kurt_100 = np.std(kurt_100) ; sd_kurt_150 = np.std(kurt_150) ; sd_kurt_200 = np.std(kurt_200)  ; sd_kurt_250 = np.std(kurt_250)

mean = [mean_50, mean_100, mean_150, mean_200, mean_250]
sd_mean = [sd_mean_50, sd_mean_100, sd_mean_150, sd_mean_200, sd_mean_250]

stde = [stde_50, stde_100, stde_150, stde_200, stde_250]
sd_stde = [sd_stde_50, sd_stde_100, sd_stde_150, sd_stde_200, sd_stde_250]

skeww = [skew_50, skew_100, skew_150, skew_200, skew_250]
sd_skew = [sd_skew_50, sd_skew_100, sd_skew_150, sd_skew_200, sd_skew_250]

kurtt = [kurt_50, kurt_100, kurt_150, kurt_200, kurt_250]
sd_kurt = [sd_kurt_50, sd_kurt_100, sd_kurt_150, sd_kurt_200, sd_kurt_250]

jb = [np.sum(jb_50)*100/len(jb_50) , np.sum(jb_100)*100/len(jb_100) , np.sum(jb_150)*100/len(jb_150) , np.sum(jb_200)*100/len(jb_200) , np.sum(jb_250)*100/len(jb_250)]
sw = [np.sum(sw_50)*100/len(sw_50) , np.sum(sw_100)*100/len(sw_100) , np.sum(sw_150)*100/len(sw_150) , np.sum(sw_200)*100/len(sw_200) , np.sum(sw_250)*100/len(sw_250)]

index = ['Mean', '(Mean Std.)', 'Variance', '(Variance Std.)', 'Skewness', '(Skewness Std.)', 'Kurtosis', '(Kurtosis Std.)', 'Jarque-Bera (%)', 'Shapiro-Wilk (%)']
columns = [50, 100, 150, 200, 250]

table = pd.DataFrame(index = index, columns = columns)
for col in table:
    index_col = columns.index(col)
    column = []
    column.append(mean[index_col]) ; column.append(sd_mean[index_col])
    column.append(stde[index_col]) ; column.append(sd_stde[index_col])
    column.append(skeww[index_col]) ; column.append(sd_skew[index_col])
    column.append(kurtt[index_col]) ; column.append(sd_kurt[index_col])
    column.append(jb[index_col])
    column.append(sw[index_col])

    table[col] = column

with open('TABLE6.tex','w') as tf:
    tf.write(table.to_latex())
#----------------------------------------------------
Y  = pd.read_csv('Y_panel.csv') ; Y = Y.drop(Y.columns[[0]], axis = 1)
C  = pd.read_csv('C_panel.csv') ; C = C.drop(C.columns[[0]], axis = 1)
I  = pd.read_csv('I_panel.csv') ; I = I.drop(I.columns[[0]], axis = 1)

Y_trend = np.array(Y['16'].tolist()) ; Y_trend, YY = cf(Y_trend, low = 2, high = 24, drift = False)
Y_cycle = np.array(Y_filter['16'].tolist())
I_trend = np.array(I['16'].tolist()) ; I_trend, II = cf(I_trend, low = 2, high = 24, drift = False)
I_cycle = np.array(I_filter['16'].tolist())
C_trend = np.array(C['16'].tolist()) ; C_trend, CC = cf(C_trend, low = 2, high = 24, drift = False)
C_cycle = np.array(C_filter['16'].tolist())

std_Y_trend = [] ; std_Y_cycle = []
std_I_trend = [] ; std_I_cycle = []
std_C_trend = [] ; std_C_cycle = []
for index, row in Y.iterrows():
    row_Y_trend = row.tolist()
    std_Y_trend.append(np.std(row_Y_trend[1:]))
for index, row in Y_filter.iterrows():
    row_Y_cycle = row.tolist()
    std_Y_cycle.append(np.std(row_Y_cycle[1:]))
for index, row in I.iterrows():
    row_I_trend = row.tolist()
    std_I_trend.append(np.std(row_I_trend[1:]))
for index, row in I_filter.iterrows():
    row_I_cycle = row.tolist()
    std_I_cycle.append(np.std(row_I_cycle[1:]))
for index, row in C.iterrows():
    row_C_trend = row.tolist()
    std_C_trend.append(np.std(row_C_trend[1:]))
for index, row in C_filter.iterrows():
    row_C_cycle = row.tolist()
    std_C_cycle.append(np.std(row_C_cycle[1:]))

T = list(range(len(Y_trend)))
Y_trend_plus = [Y_trend[i] + 1.96 * std_Y_trend[i] for i in T]
Y_trend_minus = [Y_trend[i] - 1.96 * std_Y_trend[i] for i in T]
I_trend_plus = [I_trend[i] + 1.96 * std_I_trend[i] for i in T]
I_trend_minus = [I_trend[i] - 1.96 * std_I_trend[i] for i in T]
C_trend_plus = [C_trend[i] + 1.96 * std_C_trend[i] for i in T]
C_trend_minus = [C_trend[i] - 1.96 * std_C_trend[i] for i in T]

Y_cycle_plus = [Y_cycle[i] + 1.96 * std_Y_cycle[i] for i in T]
Y_cycle_minus = [Y_cycle[i] - 1.96 * std_Y_cycle[i] for i in T]
I_cycle_plus = [I_cycle[i] + 1.96 * std_I_cycle[i] for i in T]
I_cycle_minus = [I_cycle[i] - 1.96 * std_I_cycle[i] for i in T]
C_cycle_plus = [C_cycle[i] + 1.96 * std_C_cycle[i] for i in T]
C_cycle_minus = [C_cycle[i] - 1.96 * std_C_cycle[i] for i in T]

fig, ax = plt.subplots()

color = 'tab:blue'
ax.plot(T, Y_trend, color = color, linestyle = '-')
ax.plot(T, Y_trend_plus, color = color, linestyle = '--')
ax.plot(T, Y_trend_minus, color = color, linestyle = '--')

ax.set_xlabel(r'Time')
ax.set_title(r'GDP - Trend')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()

color = 'tab:red'
ax.plot(T, C_trend, color = color, linestyle = '-')
ax.plot(T, C_trend_plus, color = color, linestyle = '--')
ax.plot(T, C_trend_minus, color = color, linestyle = '--')

ax.set_xlabel(r'Time')
ax.set_title(r'Consumption - Trend')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()

color = 'tab:green'
ax.plot(T, I_trend, color = color, linestyle = '-')
ax.plot(T, I_trend_plus, color = color, linestyle = '--')
ax.plot(T, I_trend_minus, color = color, linestyle = '--')

ax.set_xlabel(r'Time')
ax.set_title(r'Investment - Trend')
fig.tight_layout()
plt.show()
#
fig, ax = plt.subplots()

color = 'tab:blue'
ax.plot(T, Y_cycle, color = color, linestyle = '-')
ax.plot(T, Y_cycle_plus, color = color, linestyle = '--')
ax.plot(T, Y_cycle_minus, color = color, linestyle = '--')

ax.set_xlabel(r'Time')
ax.set_title(r'GDP - Cycle')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()

color = 'tab:red'
ax.plot(T, C_cycle, color = color, linestyle = '-')
ax.plot(T, C_cycle_plus, color = color, linestyle = '--')
ax.plot(T, C_cycle_minus, color = color, linestyle = '--')

ax.set_xlabel(r'Time')
ax.set_title(r'Consumption - Cycle')
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()

color = 'tab:green'
ax.plot(T, I_cycle, color = color, linestyle = '-')
ax.plot(T, I_cycle_plus, color = color, linestyle = '--')
ax.plot(T, I_cycle_minus, color = color, linestyle = '--')

ax.set_xlabel(r'Time')
ax.set_title(r'Investment - Cycle')
fig.tight_layout()
plt.show()
#----------------------------------------------------
YY = []
II = []
for col in Y_filter:
    YY.append(Y_filter[col].tolist())
    II.append(I_filter[col].tolist())
YY = np.array(YY) ; II = np.array(II)
YY = YY.flatten() ; II = II.flatten()
m_Y = min(YY) ; M_Y = max(YY)
m_I = min(II) ; M_I = max(II)

kde_Y = gaussian_kde(YY) ; x_Y = np.linspace(m_Y, M_Y, 1000)
kde_I = gaussian_kde(II) ; x_I = np.linspace(m_I, M_I, 1000)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Output')
ax1.set_ylabel(r'Frequency')

ax1.set_ylabel(r'Frequency - Kernel', color = 'blue')
ax1.tick_params(axis='y', color = 'blue')
ax1.plot(x_Y, kde_Y(x_Y), linestyle = '-', color = 'blue')

ax2 = ax1.twinx()
ax2.set_ylabel(r'Frequency - Gaussian', color = 'red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.plot(x_Y,norm.pdf(x_Y,np.mean(YY),np.sqrt(np.var(YY))), color = 'red')
ax1.set_title(r'GDP Cycle')

fig.tight_layout()
plt.show()


fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Investment')

ax1.set_ylabel(r'Frequency', color = 'blue')
ax1.set_ylabel(r'Frequency - Kernel', color = 'blue')
ax1.tick_params(axis='y', color = 'blue')
ax1.plot(x_I, kde_Y(x_I), linestyle = '-', color = 'blue')

ax2 = ax1.twinx()
ax2.set_ylabel(r'Frequency - Gaussian', color = 'red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.plot(x_I,norm.pdf(x_I,np.mean(II),np.sqrt(np.var(II))), color = 'red')

ax1.set_title(r'Investment Cycle')

fig.tight_layout()
plt.show()

sm.qqplot(YY, line = 's')
plt.show()

sm.qqplot(II, line = 's')
plt.show()
