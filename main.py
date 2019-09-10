import numpy as np
from economy import economy
import pandas as pd
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω, Δ
#------------------------------------------------------------------------------------------------------------------------
#                                                  PARAMETERS
#------------------------------------------------------------------------------------------------------------------------
#H >> I > K > J
I = 100
J = 10
K = 50
H = 1000

δ_equity = [np.random.triangular(0.1, 0.2, 0.3) for i in range(I)] # 0.8, 1, 1.2 # 0.1, 0.3, 0.5 # 0.2, 0.5, 0.8 #0.5, 0.7, 0.9
γ_equity = [np.random.triangular(0.2,0.3,0.4) for i in range(I)]
δ_bank = [np.random.triangular(0.2,0.5,0.8) for j in range(J)] #
γ = [np.random.triangular(0.8,1.0,1.2) for j in range(J)]
φ_bank = 0.6#6
ρ_firm = 0.95#85

β_equity = 0.9
ρ_1 = [np.random.triangular(0.2,0.3,0.4) for j in range(J)]
ρ_2 = [np.random.triangular(0.2,0.3,0.4) for j in range(J)]
ρ_3 = [np.random.triangular(0.2,0.3,0.4) for j in range(J)]
ρ_4 = [np.random.triangular(0.2,0.3,0.4) for j in range(J)]
bar_λ = 30
a_VAR = 0.99
E = 1
μ = 0.15
σ = 0.4
ω_1 = 0.5
ω_2 = 0.5
φ_firm = 0.9
λ = 0.7#45
ζ = 0.5
η = 0.3
ι_1 = 0.02
ι_2 = 0.02
χ = 0.2
a = 0.3
b = 0.9
κ = 0.0
b_h = 0.6
τ_house = 0.3
ψ_1 = 0.3
ψ_2 = 0.3
ψ_3 = 0.3
υ_1 = 0.0
υ_2 = 0.9
δ = 0.2
bar_w = 0
#------------------------------------------------------------------------------------------------------------------------
#                                              INITIAL CONDITIONS
#------------------------------------------------------------------------------------------------------------------------
p_bank = [np.random.triangular(7.8,7.9,8) for j in range(J)]
μ_equity = [[np.random.triangular(7.8,7.9,8) for j in range(J)] for i in range(I)]
σ_equity = [[0 for j in range(J)] for i in range(I)]
d_bank = [np.random.triangular(7.8,7.9,8) for j in range(J)]
f_bank = [np.random.triangular(7.8,7.9,8) for j in range(J)]
Y_equity = [np.random.triangular(6,7,8) for i in range(I)]
w_equity = [[1/J for j in range(J)] for i in range(I)]
n_equity = [[E/I for j in range(J)] for i in range(I)]
Y_D = [np.random.triangular(5,6,7) for k in range(K)] #5,6,7
Y_S = [np.random.triangular(5,6,7) for k in range(K)] #5,6,7
Π = [np.random.triangular(0.8,0.9,1) for k in range(K)]
Ka = [np.random.triangular(5,5.1,5.2) for k in range(K)]

ξ_firms = [np.random.triangular(1.0,1.05,1.1) for k in range(K)] # 0.5,0.6,0.7 # 1.0,1.05,1.1
L_ξ_firms = [np.random.triangular(1.0,1.05,1.1) for k in range(K)]
s = [1/K for k in range(K)]
d_firm = [np.random.triangular(0.8,0.9,1) for k in range(K)]
p_firm = [np.random.triangular(0.9,1,1.1) for k in range(K)]
L_p_firm = [np.random.triangular(0.9,1,1.1) for k in range(K)]
μ_firm = [np.random.triangular(0.9,1.0,1.1) for k in range(K)]

Ω_bank = []
client = list(range(K))
for j in range(J):
    client_j = np.random.choice(client, int(K/J), replace = False).tolist()
    for client_jj in client_j:
        client.remove(client_jj)
    Ω_bank.append(client_j)

μ_bank = [[np.random.triangular(0.9,1,1.1) for k in range(len(Ω_bank[j]))] for j in range(J)]
σ_bank = [[np.random.triangular(0.01,0.02,0.03) for k in range(len(Ω_bank[j]))] for j in range(J)]
r_bank = [[np.random.triangular(0.01,0.015,0.02) for k in range(len(Ω_bank[j]))] for j in range(J)]
rr_bank = [[np.random.triangular(0.01,0.015,0.02) for k in range(len(Ω_bank[j]))] for j in range(J)]

A_bank = [np.random.triangular(9,10,11) for j in range(J)]
L_A_bank = [np.random.triangular(9,10,11) for j in range(J)]
A_firm = [np.random.triangular(10,15,20) for k in range(K)]
L_A_firm = [np.random.triangular(10,15,20) for k in range(K)]
π = 0.01
L_π = 0.01

Ω_emp = []
emp = list(range(H))
for k in range(K):
    workers_k = np.random.choice(emp, int(0.1*H / K), replace = False).tolist()
    for worker_k in workers_k:
        emp.remove(worker_k)
    Ω_emp.append(workers_k)

N = [len(Ω_emp[k]) for k in range(K)]
L_N = [np.sum(Ω_emp[k]) for k in range(K)]
U = emp

Ω_firm = []
cons = list(range(H))
for k in range(K):
    cons_k = np.random.choice(cons, int(H/K), replace = False).tolist()
    for cons_kk in cons_k:
        cons.remove(cons_kk)
    Ω_firm.append(cons_k)

σ_firm = [[1/H for k in range(K)] for i in range(H)]
w = 0.5
G = len(emp) * b_h * w
#
T = 700 # 750
#------------------------------------------------------------------------------------------------------------------------
#                                                   OUTPUT
#------------------------------------------------------------------------------------------------------------------------
model = economy(δ_equity, γ_equity, δ_bank, γ, φ_bank, ρ_firm, ξ_firms, L_ξ_firms,
                β_equity,
                ρ_1, ρ_2, ρ_3, ρ_4, bar_λ, a_VAR, E,
                μ, σ, ω_1, ω_2, φ_firm, ζ, η, ι_1, ι_2, χ, a, b, κ,
                b_h, τ_house, ψ_1, ψ_2, ψ_3,
                υ_1, υ_2, δ, bar_w,
                p_bank, μ_equity, σ_equity, d_bank, f_bank, Y_equity, w_equity, n_equity,
                Y_D, Y_S, Π, Ka, λ, s, d_firm, p_firm, L_p_firm, μ_firm,
                μ_bank, σ_bank, r_bank, rr_bank, A_bank, L_A_bank, A_firm, L_A_firm, π, L_π,
                N, L_N, U, Ω_emp, Ω_bank, Ω_firm, σ_firm, w, G,
                T)

d_time_series = []
firmsize_panel = []
firmproductivity_panel = [] #450
banksize_panel = []
#   Equity Funds
d_bank_ts = model.d_bank_ts[550:]           ; d_time_series.append(d_bank_ts)
p_bank_ts = model.p_bank_ts[550:]           ; d_time_series.append(p_bank_ts)
#   Banks
# r_bank_ts = model.r_bank_ts[550:]           ; d_panel.append(r_bank_ts)
r_ts      = model.r_ts[550:]                ; d_time_series.append(r_ts)
rr_ts = model.rr_ts[550:]                   ; d_time_series.append(rr_ts)
ell_ts = model.ell_ts[550:]                 ; d_time_series.append(ell_ts)
r_ell_firms_ts = model.r_ell_firms_ts[550:] ; d_time_series.append(r_ell_firms_ts)
λ_banks_ts = model.λ_banks_ts[550:]         ; d_time_series.append(λ_banks_ts)
A_bank_ts = model.A_bank_ts[550:]           ; banksize_panel.append(A_bank_ts)
#   Firms
# ell_d_ts = model.ell_d_ts[550:]             ; d_panel.append(ell_d_ts)
s_ts = model.s_ts[550:]                     ; d_time_series.append(s_ts)
K_ts = model.K_ts[550:]                     ; d_time_series.append(K_ts)
I_ts = model.I_ts[550:]                     ; d_time_series.append(I_ts)
A_firm_ts = model.A_firm_ts[550:]           ; firmsize_panel.append(A_firm_ts)
ξ_firms_ts = model.ξ_firms_ts[550:]         ; firmproductivity_panel.append(ξ_firms_ts)
p_firm_ts = model.p_firm_ts[550:]           ; d_time_series.append(p_firm_ts)
#   Job Market
w_ts = model.w_ts[550:]                      ; d_time_series.append(w_ts)
unemp_rate_ts = model.unemp_rate_ts[550:]    ; d_time_series.append(unemp_rate_ts)
emp_rate_ts = model.emp_rate_ts[550:]        ; d_time_series.append(emp_rate_ts)
#   Macroeconomy
C_ts = model.C_ts[550:]                      ; d_time_series.append(C_ts)
Y_ts = model.Y_ts[550:]                      ; d_time_series.append(Y_ts)
π_ts = model.π_ts[550:]                      ; d_time_series.append(π_ts)
ξ_ts = model.ξ_ts[550:]                      ; d_time_series.append(ξ_ts)
G_ts = model.G_ts[550:]                      ; d_time_series.append(G_ts)

AA_bank_ts = model.AA_bank_ts[550:]         ; d_time_series.append(AA_bank_ts)
AA_firm_ts = model.AA_firm_ts[550:]         ; d_time_series.append(AA_firm_ts)

key_ts = ['d_bank', 'p_bank', 'interest_rate', 'realized_interest_rate', 'ell_bank', 'r_ell_firms', 'leverage_bank', 'market_share', 'K', 'I', 'p','w', 'unemp', 'emp', 'C', 'Y', 'inflation', 'productivity', 'government_budget', 'A_bank', 'A_firm']
df_ts = pd.DataFrame(data = np.transpose(d_time_series), columns = key_ts)
df_ts.to_csv('ts.csv')

key = list(range(len(firmsize_panel[0][0])))
df_firmsize = pd.DataFrame(data = firmsize_panel[0], columns = key)
df_firmsize.to_csv('firmsize.csv')

key = list(range(len(banksize_panel[0][0])))
df_banksize = pd.DataFrame(data = banksize_panel[0], columns = key)
df_banksize.to_csv('banksize.csv')

key = list(range(len(firmproductivity_panel[0][0])))
df_firmproductivity = pd.DataFrame(data = firmproductivity_panel[0], columns = key)
df_firmproductivity.to_csv('firmproductivity.csv')
