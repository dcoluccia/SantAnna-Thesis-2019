import numpy as np
import time
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.stats import norm
#----------------------------------------------------
from filter import christianofitzgerald_filter as cf
#----------------------------------------------------
import subprocess
#----------------------------------------------------
#   FIRST SIMULATION
#----------------------------------------------------
popen           = subprocess.Popen("main.py 1", shell=True)
while popen.poll() is None:
    time.sleep(1)
dataset         = pd.read_csv('ts.csv')
dataset         = dataset.drop(dataset.columns[[0]], axis = 1)
T = len(dataset['C'])

Y               = np.array([np.log(1+(dataset['C'][i] + dataset['I'][i])) for i in range(T)])  ; Y = Y[:,None]
C               = np.array([np.log(1+(dataset['C'][i])) for i in range(T)]) ; C = C[:,None]
I               = np.array([np.log(1+(dataset['I'][i])) for i in range(T)]) ; I = I[:,None]
credit          = np.array([np.log(1+(dataset['r_ell_firms'][i])) for i in range(T)]) ; credit = credit[:,None]
price           = np.array([np.log(1+(dataset['p'][i])) for i in range(T)]) ; price = price[:,None]
wages           = np.array([np.log(1+(dataset['w'][i])) for i in range(T)]) ; wages = wages[:,None]
price_inflation = np.diff(price, axis = 0) ; price_inflation = np.insert(price_inflation, 0, 0) ; price_inflation = price_inflation[:,None]
wages_inflation = np.diff(wages, axis = 0) ; wages_inflation = np.insert(wages_inflation, 0, 0) ; wages_inflation = wages_inflation[:,None]
unemp           = np.array([np.log(1+(dataset['unemp'][i])) for i in range(T)]) ; unemp = unemp[:,None]
G               = np.array([np.log(1+(dataset['government_budget'][i])) for i in range(T)]) ; G = G[:,None]
p_bank          = np.array([np.log(1+(dataset['p_bank'][i])) for i in range(T)]) ; p_bank = p_bank[:,None]
d_bank          = np.array([np.log(1+(dataset['d_bank'][i])) for i in range(T)]) ; d_bank = d_bank[:,None]
leverage        = np.array([np.log(1+(dataset['leverage_bank'][i])) for i in range(T)]) ; leverage = leverage[:,None]
interest        = np.array([np.log(1+(dataset['realized_interest_rate'][i])) for i in range(T)]) ; interest = interest[:,None]
productivity    = np.array([np.log(1+(dataset['productivity'][i])) for i in range(T)]) ; productivity = productivity[:,None]

cycle_Y               , trend_Y               = cf(Y               , low = 2 , high = 24 , drift = False) ; cycle_Y = cycle_Y[:,None]
cycle_C               , trend_C               = cf(C               , low = 2 , high = 24 , drift = False) ; cycle_C = cycle_C[:,None]
cycle_I               , trend_I               = cf(I               , low = 2 , high = 24 , drift = False) ; cycle_I = cycle_I[:,None]
cycle_credit          , trend_credit          = cf(credit          , low = 2 , high = 24 , drift = False) ; cycle_credit = cycle_credit[:,None]
cycle_price           , trend_price           = cf(price           , low = 2 , high = 24 , drift = False) ; cycle_price = cycle_price[:,None]
cycle_wages           , trend_wages           = cf(wages           , low = 2 , high = 24 , drift = False) ; cycle_wages = cycle_wages[:,None]
cycle_price_inflation , trend_price_inflation = cf(price_inflation , low = 2 , high = 24 , drift = False) ; cycle_price_inflation = cycle_price_inflation[:,None]
cycle_wages_inflation , trend_wages_inflation = cf(wages_inflation , low = 2 , high = 24 , drift = False) ; cycle_wages_inflation = cycle_wages_inflation[:,None]
cycle_unemp           , trend_unemp           = cf(unemp           , low = 2 , high = 24 , drift = False) ; cycle_unemp = cycle_unemp[:,None]
cycle_G               , trend_G               = cf(G               , low = 2 , high = 24 , drift = False) ; cycle_G = cycle_G[:,None]
cycle_p_bank          , trend_p_bank          = cf(p_bank          , low = 2 , high = 24 , drift = False) ; cycle_p_bank = cycle_p_bank[:,None]
cycle_d_bank          , trend_d_bank          = cf(d_bank          , low = 2 , high = 24 , drift = False) ; cycle_d_bank = cycle_d_bank[:,None]
cycle_leverage        , trend_leverage        = cf(leverage        , low = 2 , high = 24 , drift = False) ; cycle_leverage = cycle_leverage[:,None]
cycle_interest        , trend_interest        = cf(interest        , low = 2 , high = 24 , drift = False) ; cycle_interest = cycle_interest[:,None]
cycle_productivity    , trend_productivity    = cf(productivity    , low = 2 , high = 24 , drift = False) ; cycle_productivity = cycle_productivity[:,None]

#----------------------------------------------------
#   POPULATE FIRST DATASET
#----------------------------------------------------
KEY = 1
key = [KEY]

Y_panel               = pd.DataFrame(data = Y , columns = key)
C_panel               = pd.DataFrame(data = C , columns = key)
I_panel               = pd.DataFrame(data = I , columns = key)
credit_panel          = pd.DataFrame(data = credit , columns = key)
price_panel           = pd.DataFrame(data = price , columns = key)
wages_panel           = pd.DataFrame(data = wages , columns = key)
price_inflation_panel = pd.DataFrame(data = price_inflation , columns = key)
wages_inflation_panel = pd.DataFrame(data = wages_inflation , columns = key)
unemp_panel           = pd.DataFrame(data = unemp , columns = key)
G_panel               = pd.DataFrame(data = G , columns = key)
p_bank_panel          = pd.DataFrame(data = p_bank , columns = key)
d_bank_panel          = pd.DataFrame(data = d_bank , columns = key)
leverage_panel        = pd.DataFrame(data = leverage , columns = key)
interest_panel        = pd.DataFrame(data = interest , columns = key)
productivity_panel    = pd.DataFrame(data = productivity , columns = key)

Y_panel_filter               = pd.DataFrame(data = cycle_Y , columns = key)
C_panel_filter               = pd.DataFrame(data = cycle_C , columns = key)
I_panel_filter               = pd.DataFrame(data = cycle_I , columns = key)
credit_panel_filter          = pd.DataFrame(data = cycle_credit , columns = key)
price_panel_filter           = pd.DataFrame(data = cycle_price , columns = key)
wages_panel_filter           = pd.DataFrame(data = cycle_wages , columns = key)
price_inflation_panel_filter = pd.DataFrame(data = cycle_price_inflation , columns = key)
wages_inflation_panel_filter = pd.DataFrame(data = cycle_wages_inflation , columns = key)
unemp_panel_filter           = pd.DataFrame(data = cycle_unemp , columns = key)
G_panel_filter               = pd.DataFrame(data = cycle_G , columns = key)
p_bank_panel_filter          = pd.DataFrame(data = cycle_p_bank , columns = key)
d_bank_panel_filter          = pd.DataFrame(data = cycle_d_bank , columns = key)
leverage_panel_filter        = pd.DataFrame(data = cycle_leverage , columns = key)
interest_panel_filter        = pd.DataFrame(data = cycle_interest , columns = key)
productivity_panel_filter    = pd.DataFrame(data = cycle_productivity , columns = key)

keys    = ['Y', 'C', 'I', 'credit', 'price', 'wages', 'price_inflation', 'wages_inflation', 'unemp', 'G', 'p_bank', 'd_bank', 'leverage', 'interest', 'productivity']
id      = [KEY for i in range(T)]

data = [Y, C, I, credit, price, wages, price_inflation,
        wages_inflation, unemp, G,
        p_bank, d_bank, leverage, interest, productivity]
panel = pd.DataFrame(columns = keys, index = id)
for column in panel:
    index_column  = keys.index(column)
    panel[column] = data[index_column]

data = [cycle_Y, cycle_C, cycle_I, cycle_credit, cycle_price, cycle_wages,
        cycle_price_inflation, cycle_wages_inflation, cycle_unemp, cycle_G,
        cycle_p_bank, cycle_d_bank, cycle_leverage, cycle_interest, cycle_productivity]
panel_filter = pd.DataFrame(columns = keys, index = id)
for column in panel:
    index_column  = keys.index(column)
    panel_filter[column] = data[index_column]

panel_firms = pd.read_csv('firmsize.csv')
panel_firms['id'] = id
panel_firms.set_index('id')
times = [i for i in range(panel_firms.shape[0])]
panel_firms['time'] = times
cols = panel_firms.columns.tolist()
cols = cols[-1:] + cols[:-1]
panel_firms = panel_firms[cols]
cols = panel_firms.columns.tolist()
cols = cols[-1:] + cols[:-1]
panel_firms = panel_firms[cols]
#----------------------------------------------------
#   SIMULATIONS
#----------------------------------------------------
N = 50 # Number of MonteCarlo simulations

counter = 0
while counter < N:
    popen           = subprocess.Popen("main.py 1", shell=True)
    while popen.poll() is None:
        time.sleep(1)
    dataset         = pd.read_csv('ts.csv')
    dataset         = dataset.drop(dataset.columns[[0]], axis = 1)
    T = len(dataset['C'])

    Y               = np.array([np.log(1+(dataset['C'][i] + dataset['I'][i])) for i in range(T)]) ; Y = Y[:,None]
    C               = np.array([np.log(1+(dataset['C'][i])) for i in range(T)]) ; C = C[:,None]
    I               = np.array([np.log(1+(dataset['I'][i])) for i in range(T)]) ; I = I[:,None]
    credit          = np.array([np.log(1+(dataset['r_ell_firms'][i])) for i in range(T)]) ; credit = credit[:,None]
    price           = np.array([np.log(1+(dataset['p'][i])) for i in range(T)]) ; price = price[:,None]
    wages           = np.array([np.log(1+(dataset['w'][i])) for i in range(T)]) ; wages = wages[:,None]
    price_inflation = np.diff(price, axis = 0) ; price_inflation = np.insert(price_inflation, 0, 0) ; price_inflation = price_inflation[:,None]
    wages_inflation = np.diff(wages, axis = 0) ; wages_inflation = np.insert(wages_inflation, 0, 0) ; wages_inflation = wages_inflation[:,None]
    unemp           = np.array([np.log(1+(dataset['unemp'][i])) for i in range(T)]) ; unemp = unemp[:,None]
    G               = np.array([np.log(1+(dataset['government_budget'][i])) for i in range(T)]) ; G = G[:,None]
    p_bank          = np.array([np.log(1+(dataset['p_bank'][i])) for i in range(T)]) ; p_bank = p_bank[:,None]
    d_bank          = np.array([np.log(1+(dataset['d_bank'][i])) for i in range(T)]) ; d_bank = d_bank[:,None]
    leverage        = np.array([np.log(1+(dataset['leverage_bank'][i])) for i in range(T)]) ; leverage = leverage[:,None]
    interest        = np.array([np.log(1+(dataset['realized_interest_rate'][i])) for i in range(T)]) ; interest = interest[:,None]
    productivity    = np.array([np.log(1+(dataset['productivity'][i])) for i in range(T)]) ; productivity = productivity[:,None]

    cycle_Y               , trend_Y               = cf(Y               , low = 2 , high = 24 , drift = False) ; cycle_Y               = cycle_Y[:,None]
    cycle_C               , trend_C               = cf(C               , low = 2 , high = 24 , drift = False) ; cycle_C               = cycle_C[:,None]
    cycle_I               , trend_I               = cf(I               , low = 2 , high = 24 , drift = False) ; cycle_I               = cycle_I[:,None]
    cycle_credit          , trend_credit          = cf(credit          , low = 2 , high = 24 , drift = False) ; cycle_credit          = cycle_credit[:,None]
    cycle_price           , trend_price           = cf(price           , low = 2 , high = 24 , drift = False) ; cycle_price           = cycle_price[:,None]
    cycle_wages           , trend_wages           = cf(wages           , low = 2 , high = 24 , drift = False) ; cycle_wages           = cycle_wages[:,None]
    cycle_price_inflation , trend_price_inflation = cf(price_inflation , low = 2 , high = 24 , drift = False) ; cycle_price_inflation = cycle_price_inflation[:,None]
    cycle_wages_inflation , trend_wages_inflation = cf(wages_inflation , low = 2 , high = 24 , drift = False) ; cycle_wages_inflation = cycle_wages_inflation[:,None]
    cycle_unemp           , trend_unemp           = cf(unemp           , low = 2 , high = 24 , drift = False) ; cycle_unemp           = cycle_unemp[:,None]
    cycle_G               , trend_G               = cf(G               , low = 2 , high = 24 , drift = False) ; cycle_G               = cycle_G[:,None]
    cycle_p_bank          , trend_p_bank          = cf(p_bank          , low = 2 , high = 24 , drift = False) ; cycle_p_bank          = cycle_p_bank[:,None]
    cycle_d_bank          , trend_d_bank          = cf(d_bank          , low = 2 , high = 24 , drift = False) ; cycle_d_bank          = cycle_d_bank[:,None]
    cycle_leverage        , trend_leverage        = cf(leverage        , low = 2 , high = 24 , drift = False) ; cycle_leverage        = cycle_leverage[:,None]
    cycle_interest        , trend_interest        = cf(interest        , low = 2 , high = 24 , drift = False) ; cycle_interest        = cycle_interest[:,None]
    cycle_productivity    , trend_productivity    = cf(productivity    , low = 2 , high = 24 , drift = False) ; cycle_productivity    = cycle_productivity[:,None]

    if wages[-1] > wages[0] :
        KEY += 1
        key = [KEY]
        Y_panel[KEY]                = Y
        C_panel[KEY]                = C
        I_panel[KEY]                = I
        credit_panel[KEY]           = credit
        price_panel[KEY]            = price
        wages_panel[KEY]            = wages
        price_inflation_panel[KEY]  = price_inflation
        wages_inflation_panel[KEY]  = wages_inflation
        unemp_panel[KEY]            = unemp
        G_panel[KEY]                = G
        p_bank_panel[KEY]           = p_bank
        d_bank_panel[KEY]           = d_bank
        leverage_panel[KEY]         = leverage
        interest_panel[KEY]         = interest
        productivity_panel[KEY]     = productivity

        Y_panel_filter[KEY]                = cycle_Y
        C_panel_filter[KEY]                = cycle_C
        I_panel_filter[KEY]                = cycle_I
        credit_panel_filter[KEY]           = cycle_credit
        price_panel_filter[KEY]            = cycle_price
        wages_panel_filter[KEY]            = cycle_wages
        price_inflation_panel_filter[KEY]  = cycle_price_inflation
        wages_inflation_panel_filter[KEY]  = cycle_wages_inflation
        unemp_panel_filter[KEY]            = cycle_unemp
        G_panel_filter[KEY]                = cycle_G
        p_bank_panel_filter[KEY]           = cycle_p_bank
        d_bank_panel_filter[KEY]           = cycle_d_bank
        leverage_panel_filter[KEY]         = cycle_leverage
        interest_panel_filter[KEY]         = cycle_interest
        productivity_panel_filter[KEY]     = cycle_productivity

        keys    = ['Y', 'C', 'I', 'credit', 'price', 'wages', 'price_inflation', 'wages_inflation', 'unemp', 'G', 'p_bank', 'd_bank', 'leverage', 'interest', 'productivity']
        id      = [KEY for i in range(T)]

        data = [Y, C, I, credit, price, wages, price_inflation,
                wages_inflation, unemp, G,
                p_bank, d_bank, leverage, interest, productivity]
        panel_interim = pd.DataFrame(columns = keys, index = id)
        for column in panel_interim:
            index_column  = keys.index(column)
            panel_interim[column] = data[index_column]

        data = [cycle_Y, cycle_C, cycle_I, cycle_credit, cycle_price, cycle_wages,
                cycle_price_inflation, cycle_wages_inflation, cycle_unemp, cycle_G,
                cycle_p_bank, cycle_d_bank, cycle_leverage, cycle_interest, cycle_productivity]
        panel_filter_interim = pd.DataFrame(columns = keys, index = id)
        for column in panel_filter_interim:
            index_column  = keys.index(column)
            panel_filter_interim[column] = data[index_column]

        panel = pd.concat([panel, panel_interim], axis = 0)
        panel_filter = pd.concat([panel_filter, panel_filter_interim], axis = 0)

        panel_firms_interim = pd.read_csv('firmsize.csv')
        panel_firms_interim['id'] = id
        panel_firms_interim.set_index('id')
        times = [i for i in range(panel_firms_interim.shape[0])]
        panel_firms_interim['time'] = times
        cols = panel_firms_interim.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        panel_firms_interim = panel_firms_interim[cols]
        cols = panel_firms_interim.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        panel_firms_interim = panel_firms_interim[cols]

        panel_firms = pd.concat([panel_firms, panel_firms_interim], axis = 0)

        counter += 1

Y_panel.to_csv('Y_panel.csv')
C_panel.to_csv('C_panel.csv')
I_panel.to_csv('I_panel.csv')
credit_panel.to_csv('credit_panel.csv')
price_panel.to_csv('price_panel.csv')
wages_panel.to_csv('wages_panel.csv')
price_inflation_panel.to_csv('price_inflation_panel.csv')
wages_inflation_panel.to_csv('wages_inflation_panel.csv')
unemp_panel.to_csv('unemp_panel.csv')
G_panel.to_csv('G_panel.csv')
p_bank_panel.to_csv('p_bank_panel.csv')
d_bank_panel.to_csv('d_bank_panel.csv')
leverage_panel.to_csv('leverage_panel.csv')
interest_panel.to_csv('interest_panel.csv')
productivity_panel.to_csv('productivity_panel.csv')
panel.to_csv('panel.csv')

Y_panel_filter.to_csv('Y_panel_filter.csv')
C_panel_filter.to_csv('C_panel_filter.csv')
I_panel_filter.to_csv('I_panel_filter.csv')
credit_panel_filter.to_csv('credit_panel_filter.csv')
price_panel_filter.to_csv('price_panel_filter.csv')
wages_panel_filter.to_csv('wages_panel_filter.csv')
price_inflation_panel_filter.to_csv('price_inflation_panel_filter.csv')
wages_inflation_panel_filter.to_csv('wages_inflation_panel_filter.csv')
unemp_panel_filter.to_csv('unemp_panel_filter.csv')
G_panel_filter.to_csv('G_panel_filter.csv')
p_bank_panel_filter.to_csv('p_bank_panel_filter.csv')
d_bank_panel_filter.to_csv('d_bank_panel_filter.csv')
leverage_panel_filter.to_csv('leverage_panel_filter.csv')
interest_panel_filter.to_csv('interest_panel_filter.csv')
productivity_panel_filter.to_csv('productivity_panel_filter.csv')
panel_filter.to_csv('panel_filter.csv')

panel_firms.to_csv('FSD.csv')
