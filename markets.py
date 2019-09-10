import numpy as np
import copy
from scipy.stats import norm
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω, Δ
#------------------------------------------------------------------------------------------------------------------------
class JobMarket:
    """
    The Job Market is rigid wrt demand for workers by firms. Random firing/matching updates
    the employment sets for each given firm.
    What this does:
        1 - Compute the wage
        2 - Fire all those workers whose employer demands less labor
        3 - (Try to) hire workers for all those firms demanding more labor
        4 - Compute the new supply of labor

    Inputs - Lists
    --------------------
        N       :   firm-level employment
                    [J*1]
        L_N     :   lagged firm level employment
                    [J*1]
        U       :   Unemployment pool
                    [H*1]
        N_e     :   expected (desired) employment
                    [J*1]
        Ω_emp   :   Employment set of firms
                    [J*H]
        w       :   wage
                    [1*1]
        π       :   inflation
                    [1*1]
        L_π     :   lagged inflation
                    [1*1]
        ξ       :   average productivity
                    [1*1]
        L_ξ     :   lagged average productivity
                    [1*1]

    Inputs - Parameters
    --------------------
        ψ_1     :   sensitivity of wage wrt unemployment
        ψ_2     :   sensitivity of wage wrt inflation
        ψ_3     :   sensitivity of wage wrt productivity

    Output
    --------------------
        w       :   New wage
        Ω_emp   :   New employment set
        N       :   New employment level
        L_N     :   New lagged employment level

    @author: ColucciaDM
    """

    def __init__(self, N, L_N, U, N_e, Ω_emp, w, π, L_π, p_firm, L_p_firm, Π, ξ_firms, L_ξ_firms, ψ_1, ψ_2, ψ_3, bar_w):
        #   INPUT - List
        self.N = N
        self.L_N = L_N
        self.U = U
        self.N_e = N_e
        self.Ω_emp = Ω_emp
        self.w = w
        self.π = π
        self.L_π = L_π
        self.p_firm = p_firm
        self.L_p_firm = L_p_firm
        self.Π = Π
        self.ξ_firms = ξ_firms
        self.L_ξ_firms = L_ξ_firms

        self.ξ = np.mean(self.ξ_firms)
        self.L_ξ = np.mean(self.L_ξ_firms)
        #   INPUT - Params
        self.ψ_1 = ψ_1
        self.ψ_2 = ψ_2
        self.ψ_3 = ψ_3
        self.bar_w = bar_w
        #   OUTPUT
        self.L_N = copy.deepcopy(self.N)

        self.H = np.sum(self.N) + len(self.U)
        self.K = len(self.ξ_firms)
        #   WorkFlow
        self.exceeding_employment()
        self.market()
        self.update_wage()

    def exceeding_employment(self):
        'Put workers into the unemployment pool'

        for k in range(self.K):
            if self.N_e[k] < self.N[k]:
                if self.N[k] - self.N_e[k] >= 0:
                    if len(self.Ω_emp[k]) > 0:
                        ex = int(1*(self.N[k] - self.N_e[k]))
                        if len(self.Ω_emp[k]) > ex:
                            u = np.random.choice(self.Ω_emp[k], ex, replace = False).tolist()
                            e = [i for i in self.Ω_emp[k] if i not in u]

                            self.U.extend(u)
                            self.Ω_emp[k] = e
                            self.N[k] -= len(u)
                        # else:
                        #     self.U.extend(self.Ω_emp[k])
                        #     self.Ω_emp[k] = []
                        #     self.N[k] = 0

    def demanding_firms(self):
        'Populate a list of k firm demanding labor'
        e_firm = []
        for k in range(self.K):
            if self.N[k] - self.N_e[k] < 0:
                e_firm.append(k)

        return e_firm

    def exceeding_unemployment(self):
        'Put workers into employment set of firms'
        e_firm = self.demanding_firms() ; np.random.shuffle(e_firm)

        e = 0
        for k in e_firm:
            if len(self.U) > 0:
                e = np.random.choice(self.U, 1).tolist()[0]
                self.Ω_emp[k].append(e)
                self.U.remove(e)
                self.N[k] += 1

    def exceeding_unemployment_fast(self):
        'Solve the one shot iteration job market more fastly'
        e_firm = self.demanding_firms() ; np.random.shuffle(e_firm)

        for k in e_firm:
            if len(self.U) > 0:
                demanded_workers = int((self.N_e[k] - self.N[k]))
                amount_workers = min(len(self.U), demanded_workers)
                e = np.random.choice(self.U, amount_workers, replace = False).tolist()
                for e_h in e:
                    self.U.remove(e_h)
                    self.Ω_emp[k].append(e_h)
                self.N[k] += amount_workers

    def market(self):
        'Run exceeding_unemployment until either no firm has vacancies or there are no unemployed'

        emp = min(len(self.demanding_firms()),len(self.U))
        while emp > 0:
            self.exceeding_unemployment()
            emp = min(len(self.demanding_firms()),len(self.U))

    def update_wage(self):
        'Get the new wage level'
        if (1 - np.sum(self.L_N) / self.H) != 0:
            unemp = (( 1 - np.sum(self.N) / self.H ) - ( 1 - np.sum(self.L_N) / self.H )) / ( 1 - np.sum(self.L_N) / self.H )
        else:
            unemp = 0
        infl = self.π
        prod = (self.ξ - self.L_ξ) / self.L_ξ
        prof = np.mean(self.Π)
        # print('------------')
        # print('Unemployment') ; print(unemp)
        # print('Inflation') ; print(infl)
        # print('Productivity') ; print(prod)
        # print('Profits') ; print(prof)
        # print('------------')
        μ =  1 - 10*self.ψ_1*unemp + 0.35*self.ψ_2*infl + 0.1*self.ψ_3*prod + 0.05*self.ψ_3 * prof  #0.005*self.ψ_3 * prof
        W = self.w * μ
        #W = self.w * (1 + np.random.triangular(-0.05,0,0.06))
        self.w = max(max(W , self.bar_w * self.w) , 0.1) # max(W , 0.1) 1.001*self.w#

#------------------------------------------------------------------------------------------------------------------------
class GoodsMarket:
    """
    The Goods Market feeds into the realized output of firms given demand and supply on a
    rolling basis to avoid exceeding rigidity.
    What this does:
        1 - Match demand and supply for each firm
        2 - Update commodity market inflation

    Inputs - Lists
    --------------------
        Y_D     :   Demand for good
                    [K*1]
        Y_S     :   Supply of good
                    [K*1]
        π       :   inflation
                    [1*1]
        L_π     :   lagged inflation
                    [1*1]
        p_firm  :   Price of consumption good
                    (K*1)
        L_p_firm:   Lagged price of consumption good
                    (K*1)

    Inputs - Parameters
    --------------------

    Output
    --------------------
        r_Y     :   Realized output sold
                    [K*1]
        π       :   inflation
                    [1*1]
        L_π     :   lagged inflation
                    [1*1]

    @author: ColucciaDM
    """

    def __init__(self, Y_D, Y_S, p_firm, L_p_firm, π, L_π):
        #   INPUT
        self.Y_D = Y_D
        self.Y_S = Y_S
        self.p_firm = p_firm
        self.L_p_firm = L_p_firm
        self.π = π
        self.L_π = L_π
        self.K = len(self.Y_D)
        #   OUTPUT
        self.r_Y = []
        #   WorkFlow
        self.market()
        self.update_inflation()

    def market_k(self, k):
        'Match demand and supply for k-th firm'
        if self.Y_D[k] < self.Y_S[k]:
            r_y = self.Y_D[k]
        elif self.Y_D[k] > self.Y_S[k]:
            r_y = self.Y_S[k]
        else:
            r_y = self.Y_D[k]

        r_y = max(r_y,0)
        return r_y

    def market(self):
        'Market wide matching'
        r_y = []
        for k in range(self.K):
            rr_y = self.market_k(k)
            r_y.append(rr_y)

        self.r_Y = r_y

    def update_inflation(self):
        'Update inflation and past inflation'
        self.L_π = self.π
        p = np.mean(self.p_firm)
        L_p = np.mean(self.L_p_firm)
        self.π = (p - L_p) / L_p

#------------------------------------------------------------------------------------------------------------------------
class CreditMarket:
    """
    Allocate credit across firms, for each bank, given the portfolio composition
    thus computed.
    Allocation is rolling over, meaning that after each round, residual credit is
    re-allocated across residual firms demanding credit after previous round, according
    to previous portfolio weights.
    What this does:
        1 - Evaluate ideal weights in portfolios for banks given credit supply
        2 - Match demand and supply for each j, across k
        3 - Roll over matching across rounds

    Inputs - Lists
    --------------------
        ell    :    credit supply
                    [J*K]
        ell_d  :    Demand for credit
                    [K*1]
        Ω_bank :   list of clients of banks
                    [J*K]

    Inputs - Parameters
    --------------------

    Output
    --------------------
        r_ell_firms :   Realized credit supply
                        (K*1)
        r_ell       :   realized gross loan returns
                        (J*1)
        credit_full :   firm-bank matching credit supply
                        (J*K)

    @author: ColucciaDM
    """

    def __init__(self, ell, ell_d, Ω_bank):
        #   INPUT
        self.ell = ell
        self.ell_d = ell_d
        self.Ω_bank = Ω_bank
        self.ell_s = [np.sum(self.ell[j]) for j in range(len(self.ell))] #credi supply for each bank

        self.w = [] #bank portfolios
        self.J = len(self.ell)
        self.K = len(self.ell_d)
        #   OUTPUT
        self.credit_full = [[] for j in range(self.J)]
        self.r_ell_firms = [0 for j in range(self.K)]
        self.r_ell = []
        self.indexed_ell = []

        #   WorkFlow
        self.compute_weights(self.Ω_bank)
        self.credit_market()
        self.update()

    def compute_weight(self, j, clients):
        'Compute bank j-th portfolio weights (ideal) across k in clients'
        index_k = []
        for k in clients:
            index_k.append(self.Ω_bank[j].index(k))

        sum = 0
        for k in index_k:
            sum += self.ell[j][k]

        W = []
        for k in index_k:
            w = self.ell[j][k] / sum
            W.append(w)

        return W

    def compute_weights(self, client_set):
        'Compute portfolio composition for each bank'
        W = []
        for j in range(self.J):
            clients = client_set[j]
            w = self.compute_weight(j, clients)
            W.append(w)

        self.w = W

    def credit_market_j(self, j):
        'Run the credit market for the j-th bank'
        client_set = self.Ω_bank[j] ; client_number = len(client_set)
        residual_demand = [self.ell_d[k] for k in client_set]
        residual_supply = self.ell[j]
        residual_exposure = np.sum(residual_supply)
        initial_exposure = np.sum(residual_supply)
        allocate_credit = []

        count = 1
        while count >= 1:
            if count == 1:
                match = [min(residual_demand[k], residual_supply[k]) for k in range(client_number)]
                residual_demand = [residual_demand[k] - match[k] for k in range(client_number)]
                residual_supply = [residual_supply[k] - match[k] for k in range(client_number)]

                residual_firms = self.Ω_bank[j]
                allocate_credit = match
                residual_exposure -= np.sum(match)

            else:
                residual_firms = []
                for k in client_set:
                    index_k = client_set.index(k)
                    if residual_demand[index_k] > 0 and residual_supply[index_k] > 0:
                        residual_firms.append(k)

                new_weights = self.compute_weight(j, residual_firms)
                residual_supply_new = []
                residual_demand_new = []
                for k in client_set:
                    if k in residual_firms:
                        index_k = client_set.index(k)
                        residual_demand_new.append(residual_demand[index_k])
                        index_k = residual_firms.index(k)
                        residual_supply_new.append(new_weights[index_k] * residual_exposure)
                    else:
                        residual_demand_new.append(0.0)
                        residual_supply_new.append(0.0)
                match = [min(residual_demand_new[k], residual_demand_new[k]) for k in range(client_number)]
                residual_demand = [residual_demand_new[k] - match[k] for k in range(client_number)]
                residual_supply = [residual_supply_new[k] - match[k] for k in range(client_number)]

                for k in range(client_number):
                    allocate_credit[k] += match[k]
                residual_exposure -= np.sum(match)

            if residual_exposure == 0:
                count = 0
            elif len(residual_firms) == 0:
                count = 0
            elif np.sum(match) < (0.001) * initial_exposure:
                count = 0
            else:
                count += 1

        self.credit_full[j] = allocate_credit
        self.ell_s[j] = np.sum(allocate_credit)

        # credit_d = copy.deepcopy(self.ell_d)
        # count = 1
        # while count >= 1:
        #     if count == 1:
        #         active_firms = self.Ω_bank[j]
        #         supply = self.ell[j]
        #         demand = [credit_d[k] for k in active_firms]
        #         match  = [min(supply[k], demand[k]) for k in range(len(supply))]
        #
        #         self.ell_s[j] -= np.sum(match)
        #         self.credit_full[j] = match
        #
        #         net = [demand[k] - match[k] for k in range(len(demand))]
        #     else:
        #         active_firms = []
        #         for k in range(len(net)):
        #             if net[k] > 0:
        #                 index_k = self.Ω_bank[j][k]
        #                 active_firms.append(index_k)
        #
        #         if len(active_firms) > 0:
        #             portfolio = self.compute_weight(j, active_firms)
        #
        #             supply = [self.ell_s[j] * portfolio[k] for k in range(len(portfolio))]
        #             demand = [net[k] for k in range(len(portfolio))]
        #             match  = [min(supply[k], demand[k]) for k in range(len(supply))]
        #
        #             self.ell_s[j] -= np.sum(match)
        #
        #             #new_match = []
        #             for k in range(len(net)):
        #                 if net[k] > 0:
        #                     index_k = self.Ω_bank[j][k]
        #                     index_k = active_firms.index(index_k)
        #                     new_net = demand[index_k] - match[index_k]
        #                     net[k] += new_net
        #                     #new_match.append(match[active_firms.index(k)])
        #                 else:
        #                     new_net = 0.0
        #                     net[k] += new_net
        #                     #new_match.append(0.0)
        #
        #             for k in range(len(self.credit_full[j])):
        #                 self.credit_full[j][k] += net[k]
        #
        #     if self.ell_s[j] == 0:
        #         count = 0
        #     elif np.sum(match) < 0.1:
        #         count = 0
        #     elif len(active_firms) == 0:
        #         count = 0
        #     else:
        #         count += 1

    def credit_market(self):
        'Run credit market'
        for j in range(self.J):
            self.credit_market_j(j)

    def update(self):
        'Populate the two lists for banks and firms'
        r_ell = []
        for j in range(self.J):
            rr_ell = np.sum(self.credit_full[j])
            r_ell.append(rr_ell)
        self.r_ell = r_ell

        for k in range(self.K):
            for j in range(self.J):
                if k in self.Ω_bank[j]:
                    index = self.Ω_bank[j].index(k)
                    self.r_ell_firms[k] = self.credit_full[j][index]
#------------------------------------------------------------------------------------------------------------------------
# I = 100
# J = 10
# K = 50
# H = 1000
#
# Ω_emp = []
# emp = list(range(H))
# for k in range(K):
#     workers_k = np.random.choice(emp, int((H*0.5) / K), replace = False).tolist()
#     for worker_k in workers_k:
#         emp.remove(worker_k)
#     Ω_emp.append(workers_k)
#
# N = [np.sum(Ω_emp[k]) for k in range(K)]
# L_N = [np.sum(Ω_emp[k]) for k in range(K)]
# U = emp
#
# N_e = [np.sum(Ω_emp[k]) - 1 for k in range(K)]
# w = 0.2
# π = 0.01
# L_π = 0.01
# ξ_firms = [1 for k in range(K)]
# L_ξ_firms = [1 for k in range(K)]
# ψ_1 = 0.3
# ψ_2 = 0.3
# ψ_3 = 0.3
#
# Job = JobMarket(N, L_N, U, N_e, Ω_emp, w, π, L_π, ξ_firms, L_ξ_firms, ψ_1, ψ_2, ψ_3)
#
# Y_D = [1 for k in range(K)]
# Y_S = [0.5 for k in range(K)]
# p_firm = [0.1 for k in range(K)]
# L_p_firm = [0.08 for k in range(K)]
# π = 0.01
# L_π = 0.01
#
# Goods = GoodsMarket(Y_D, Y_S, p_firm, L_p_firm, π, L_π)
#
# Ω_bank = []
# client = list(range(K))
# for j in range(J):
#     client_j = np.random.choice(client, int(K/J), replace = False).tolist()
#     for client_jj in client_j:
#         client.remove(client_jj)
#     Ω_bank.append(client_j)
# ell = [[0.1 for k in range(len(Ω_bank[j]))] for j in range(J)]
# ell_d =  [0.3 for k in range(K)]
#
# Credit = CreditMarket(ell, ell_d, Ω_bank)
