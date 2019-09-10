import numpy as np
import scipy.stats as stat
import math
import copy
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω
#------------------------------------------------------------------------------------------------------------------------
class pre_firms:
    """
    Set the demand for loans from the real sector, composed of (i) employment,
    (ii) RD and (iii) physical investment, based on past demand. It is corrected
    by the internal funding adjustment buffer.
    What this does:
        1 - Evaluate current desired level of production
        2 - Evaluate the resulting desired physical investment, RD expenditure and
            employment decision
        3 - Evaluate the buffer level of net worth
        4 - Evaluate the resulting demand for credit

    Inputs - Lists
    --------------------
        ξ_firms     :   firm-level productivity
                        (K*1)
        Y_D         :   firm-level past demand
                        (K*1)
        Y_S         :   firm-level past supply
                        (K*1)
        A_firm      :   Firm net worth
                        (K*1)
        Π           :   past profit
                        (K*1)
        K           :   past capital level
                        (K*1)

    Inputs - Parameters
    --------------------
        λ           :   Adjustment to past excess demand
        χ           :   RD share of profit
        φ_firm      :   Net worth buffer
        ζ           :   Labor productivity

    Output
    --------------------
        ell_d       :   Demand for credit
        α_RD        :   Share of expenditure in RD
        α_I         :   Share of expenditure in investment
        α_N         :   Share of expenditure on employment
        N_e         :   Desired employment

    @author: ColucciaDM
    """

    def __init__(self, ξ_firms, Y_D, Y_S, Π, Ka, A_firm, N, Ω_emp, U, w, λ, χ, φ_firm, ζ, δ):
        #   Input - Lists
        self.ξ_firms = ξ_firms
        self.Y_D = Y_D
        self.Y_S = Y_S
        self.Π   = Π
        self.Ka   = Ka
        self.A_firm = A_firm
        self.N = N
        self.Ω_emp = Ω_emp
        self.w = w
        self.U = U
        #   Input - Params
        self.λ = λ
        self.χ = χ
        self.φ_firm = φ_firm
        self.ζ = ζ
        self.δ = δ
        #   Output
        self.ell_d = []
        self.α_RD  = []
        self.α_I   = []
        self.α_N   = []
        self.N_e   = []

        self.K = len(self.ξ_firms)

        #   Work flow
        self.compute_credit()
        self.employment()

    def compute_desired(self,k):
        'Compute k-th firm desired level of production and expenditures'
        ξ = self.ξ_firms[k]
        dY = self.Y_D[k] - self.Y_S[k]#max(self.Y_D[k] - self.Y_S[k] , self.Π[k])

        Y_e = max(self.Y_S[k] + self.λ*dY, 0)
        K_e = Y_e / ξ
        RD_e = self.χ*self.Π[k]
        I_e = max(0,K_e - self.Ka[k]*(1-self.δ))
        self.Ka[k] = self.Ka[k]*(1-self.δ)
        N_e = Y_e / (self.ζ*ξ)

        return Y_e, I_e, N_e, RD_e

    def compute_loan_demand(self,k):
        'Compute k-th firm demand for credit and expenditure shares'
        Y_e, I_e, N_e, RD_e = self.compute_desired(k)

        overall_investment = N_e + I_e + RD_e
        overall_internal = (1-self.φ_firm)*self.A_firm[k]
        ell_k = max(overall_investment - overall_internal, 0)

        if Y_e != 0:
            α_I = I_e / Y_e
            α_N = N_e / Y_e
            α_RD = RD_e / Y_e
            α = α_I + α_N + α_RD
            if α != 0:
                α_I = α_I / α
                α_N = α_N / α
                α_RD = α_RD / α
            else:
                α_I = 0
                α_N = 0
                α_RD = 0
        else:
            α_I = 0
            α_N = 0
            α_RD = 0

        return ell_k, α_I, α_N, α_RD, I_e

    def compute_credit(self):
        'Compute overall demand for credit and expenditure shares'
        ell_d = []
        α_I = []
        α_N = []
        α_RD = []
        I_e = []
        for k in range(self.K):
            L, a_I, a_N, a_RD, i_e = self.compute_loan_demand(k)
            ell_d.append(L)
            α_I.append(a_I)
            α_N.append(a_N)
            α_RD.append(a_RD)
            I_e.append(i_e)
        self.ell_d = ell_d
        self.α_I = α_I
        self.α_N = α_N
        self.α_RD = α_RD
        self.I_e = I_e

    def employment(self):
        'Compute desired employment'
        N = []
        for k in range(self.K):
            Y_e, I_e, N_e, RD_e = self.compute_desired(k)
            N.append(N_e)
        self.N_e = N

#------------------------------------------------------------------------------------------------------------------------
class firms_pricing:
    """
    Following the closure of the market for credit and work,
    firms decide how much to charge for the consumption good.
    What this does:
        1 - Compute Competitiveness
        2 - Compute Market Shares
        3 - Compute Mark-ups

    Inputs - Lists
    --------------------
        Y_D         :   firm-level past demand
                        (K*1)
        Y_S         :   firm-level past supply
                        (K*1)
        s           :   Market share
                        (K*1)
        p_firm      :   Price of consumption good
                        (K*1)
        μ_firm      :   Mark-Ups of firms
                        (K*1)
        r_bank      :   Interest rate charged on firms
                        ()
        w           :   Wage
                        (1*1)
        r_ell_firms :   Realized credit supply
                        (K*1)
        A_firm      :   Firm net worth
                        (K*1)
        α_N         :   Share of expenditure in capital
                        (K*1)
        Ω_bank      :   list of clients of banks
                        [J*K]
        ξ_firms     :   firm-level productivity
                        (K*1)
        K           :   past capital level
                        (K*1)
        N           :   employment level
                        (K*1)

    Inputs - Parameters
    --------------------
        μ           :   Mark up sensitivity
        σ           :   Market share sensitivity
        ω_1, ω_2    :   Competitiveness parameters
        φ_firm      :   Regulatory capital buffer
        ζ           :   Labor productivity

    Output
    --------------------
        p_firm      :   Price of consumption good
                        (K*1)
        L_p_firm    :   Lagged price of consumption good
                        (K*1)
        μ_firm      :   Mark-Ups of firms
                        (K*1)
        s           :   Market share
                        (K*1)

    @author: ColucciaDM
    """

    def __init__(self, Y_D, Y_S, s, p_firm, L_p_firm, μ_firm, r_bank, w, r_ell_firms, A_firm, α_N, Ω_bank, ξ_firms, Ka, N, Ω_emp, μ, σ, ω_1, ω_2, φ_firm, ζ):
        #   Input - Lists
        self.Y_D = Y_D
        self.Y_S = Y_S
        self.s   = s
        self.p_firm = p_firm
        self.L_p_firm = L_p_firm
        self.μ_firm = μ_firm
        self.r_bank = r_bank
        self.w = w
        self.r_ell_firms = r_ell_firms
        self.A_firm = A_firm
        self.α_N = α_N
        self.Ω_bank = Ω_bank
        self.ξ_firms = ξ_firms
        self.Ka = Ka
        self.N = N
        #   Input - Params
        self.μ = μ
        self.σ = σ
        self.ω_1 = ω_1
        self.ω_2 = ω_2
        self.φ_firm = φ_firm
        self.ζ = ζ
        #   Output
        #   Updated therehence
        self.K = len(self.Y_D)
        self.a_e = 0
        #   Workflow
        self.compute_average_competitiveness()
        self.compute_market_shares()
        self.compute_prices()

    def compute_competitiveness(self,k):
        'Compute k-th firm competitiveness'
        dY = self.Y_D[k] - self.Y_S[k]
        e_k = - self.ω_1*self.p_firm[k] - self.ω_2*dY

        return e_k

    def compute_average_competitiveness(self):
        'Compute the average competitiveness'
        a_e = 0
        for k in range(self.K):
            e = self.compute_competitiveness(k)
            es = e*self.s[k]
            a_e += es

        self.a_e = a_e

    def compute_market_shares(self):
        'Compute the new market shares implied by the levels of competitiveness'
        e = []
        for k in range(self.K):
            e_k = self.compute_competitiveness(k)
            e.append(e_k)
        a_e = self.a_e
        e = [np.divide(e[k], a_e) for k in range(self.K)]

        s = []
        for k in range(self.K):
            s_k = self.s[k] * (1 + self.σ * e[k])
            s.append(s_k)
        sum_s = np.sum(s)
        s = [np.divide(s[k] , sum_s) for k in range(self.K)]

        Y = [min(self.Y_D[k], self.Y_S[k]) / self.p_firm[k] for k in range(self.K)] ; sY = np.sum(Y)
        s = [np.divide(Y[k] - np.mean(Y), sY) for k in range(self.K)]
        self.s = s

    def compute_cost(self,k):
        'Compute unit cost of production of firm k'
        ell = self.r_ell_firms[k]
        A = self.A_firm[k]

        bank = 0 ; firm = 0
        for j in range(len(self.Ω_bank)):
            if k in self.Ω_bank[j]:
                bank = j
                firm = self.Ω_bank[bank].index(k)
        r = self.r_bank[bank][firm]

        C = r*ell + self.w*self.α_N[k]*(ell + (1-self.φ_firm)*A)
        Y = self.ξ_firms[k]*min(self.ζ*self.α_N[k]*(ell + (1-self.φ_firm)*A), self.Ka[k])

        if Y > 0:
            c = C / Y
        else:
            c = 0

        return c

    def compute_price(self, k):
        'Compute the price charged by the k-th firm'
        c_k = self.compute_cost(k)

        s = self.s[k]
        μ_k = self.μ_firm[k] * (1 + self.μ * s)#min(self.μ_firm[k] * (1 + self.μ * s) , 1.5)
        p_k = max(μ_k*c_k, 0.1) #max(μ_k*c_k, self.p_firm[k] * 0.5) #

        return p_k, μ_k

    def compute_prices(self):
        'Compute prices for all the firms, update mark-ups and market shares'
        p_firm = []
        μ_firm = []
        for k in range(self.K):
            p, μ = self.compute_price(k)
            p_firm.append(p)
            μ_firm.append(μ)
        self.L_p_firm = self.p_firm
        self.p_firm = p_firm
        self.μ_firm = μ_firm
#------------------------------------------------------------------------------------------------------------------------
class firms_production:
    """
    Given the credit market and the labor market, the supply Y_S is set, investment is undertaken.
    What this does:
        1 - Evaluate supply of goods market
        2 - Evaluate investment

    Inputs - Lists
    --------------------
        K           :   past capital level
                        (K*1)
        N           :   employment level
                        (K*1)
        ξ_firms     :   firm-level productivity
                        (K*1)
        r_ell_firms :   Realized credit supply
                        (K*1)
        A_firm      :   Firm net worth
                        (K*1)
        α_I         :   Share of expenditure in investment
                        (K*1)

    Inputs - Parameters
    --------------------
        φ_firm      :   Regulatory capital buffer
        ζ           :   Labor productivity

    Output
    --------------------
        Y_S         :   Output supply
        K           :   New Capital
        I           :   Investment

    @author: ColucciaDM
    """

    def __init__(self, Ka, N, ξ_firms, r_ell_firms, A_firm, I_e, α_I, Ω_bank, φ_firm, ζ):
        #   INPUT - Lists
        self.Ka = Ka
        self.N = N
        self.ξ_firms = ξ_firms
        self.r_ell_firms = r_ell_firms
        self.A_firm = A_firm
        self.I_e = I_e
        self.α_I = α_I
        #   INPUT - Params
        self.φ_firm = φ_firm
        self.ζ = ζ
        #   OUTPUT
        self.I = []
        self.Y_S = []

        self.K = len(self.Ka)
        #   WorkFlow
        self.update_K()
        self.production()

    def compute_investment(self, k):
        'Compute the investment level for the k-th firm'
        ell = self.r_ell_firms[k]
        a = (1-self.φ_firm) * self.A_firm[k]
        I = max(min(self.α_I[k] * (ell + a) , self.I_e[k]), 0)
        down_capital = self.α_I[k] * a

        return I, down_capital

    def update_K(self):
        'Run investment and update capital'
        I = []
        for k in range(self.K):
            i, a = self.compute_investment(k)
            I.append(i)
            self.Ka[k] += i
            self.A_firm[k] -= a

        self.I = I

    def production_k(self, k):
        'Compute the production level for k-th firm'

        y_s = self.ξ_firms[k] * min(self.Ka[k], self.ζ * self.N[k])
        # if min(self.Ka[k], self.ζ * self.N[k]) == self.Ka[k]:
        #     print('capital')
        # else:
        #     print('labor')
        return y_s

    def production(self):
        'Run production'
        Y_S = []
        for k in range(self.K):
            y_s = self.production_k(k)
            Y_S.append(y_s)

        self.Y_S = Y_S

#------------------------------------------------------------------------------------------------------------------------
class post_firms:
    """
    Given the outcome of the credit market, labor market and goods market firms perform investment in
    physical capital, RD and employment decision.
    What this does:
        1 - Given credit supply and internal funding, evaluate actual performance
        2 - Dividends and LoM of wealth
        3 - RD

    Inputs - Lists
    --------------------
        r_Y         :   effective output
                        (K*1)
        K           :   past capital level
                        (K*1)
        N           :   employment level
                        (K*1)
        w           :   Wage
                        (1*1)
        Ω_bank      :   list of clients of banks
                        [J*K]
        r_bank      :   interest rate on loans
                        [J*K]
        r_ell_firms :   Realized credit supply
                        (K*1)
        p_firm      :   Price of consumption good
                        (K*1)
        A_firm      :   Firm net worth
                        (K*1)
        ξ_firms     :   firm-level productivity
                        (K*1)
        α_RD        :   Share of expenditure in RD
                        (K*1)
        α_I         :   Share of expenditure in investment
                        (K*1)
        α_N         :   Share of expenditure on employment
                        (K*1)

    Inputs - Parameters
    --------------------
        φ_firm      :   Net worth buffer
        ρ_firm      :   Share of profit divided to households
        η           :   Share of expenditure to innovation
        ι_1         :   Parameter of innovation expenditure
        ι_2         :   Parameter of imitation expenditure
        χ           :   Share of RD expenditure
        a,b         :   Support of the Beta
        κ           :   Exogenous technological progress rate

    Output
    --------------------
        A_firm      :   New net worth
        d_firm      :   Net return of firms
        ξ_firms     :   New productivity levels
        Π           :   Profit
        rr_bank     :   realized interest rate on loans

    @author: ColucciaDM
    """

    def __init__(self, r_Y, Ka, N, w, Ω_bank, r_bank, r_ell_firms, p_firm, A_firm, ξ_firms, α_RD, α_I, α_N, φ_firm, ρ_firm, η, ι_1, ι_2, χ, a, b, κ):
        #   INPUT - Lists
        self.r_Y = r_Y
        self.Ka = Ka
        self.N = N
        self.w = w
        self.Ω_bank = Ω_bank
        self.r_bank = r_bank
        self.r_ell_firms = r_ell_firms
        self.p_firm = p_firm
        self.A_firm = A_firm
        self.ξ_firms = ξ_firms
        self.α_RD = α_RD
        self.α_I = α_I
        self.α_N = α_N
        #   INPUT - Parameters
        self.φ_firm = φ_firm
        self.ρ_firm = ρ_firm
        self.η = η
        self.ι_1 = ι_1
        self.ι_2 = ι_2
        self.χ = χ
        self.a = a
        self.b = b
        self.κ = κ

        self.K = len(A_firm)
        self.J = len(Ω_bank)
        self.ξ_old = ξ_firms
        #   OUTPUT
        self.d_firm = []
        self.Π = []
        self.rr_bank = [[0 for k in range(len(self.Ω_bank[j]))] for j in range(self.J)]
        #   Work flow
        self.compute_profits()
        self.new_productivity()
        self.returns()

    def compute_profit(self, k):
        'Compute ex-post profits of k-th firm'
        p = self.p_firm[k] ; Y = self.r_Y[k] ; w = self.w ; l = self.r_ell_firms[k] ; n = self.N[k]
        bank = 0 ; firm = 0
        for j in range(len(self.Ω_bank)):
            if k in self.Ω_bank[j]:
                bank = j
                firm = self.Ω_bank[bank].index(k)
        r = 1 + self.r_bank[bank][firm]

        π = p*Y - w*n - (r-1)*l
        rr = 0
        if  self.A_firm[k] + π >= (r-1) * l:
            rr = r
        else:
            if l != 0:
                rr = max(0, (self.A_firm[k] + π) / l)
            else:
                rr = r

        return π, rr

    def compute_profits(self):
        'Compute profits and realized interest rates'
        ΠΠ = []
        RR = []
        for k in range(self.K):
            ππ, rr = self.compute_profit(k)
            ΠΠ.append(ππ)
            RR.append(rr)

        for k in range(self.K):
            bank = 0
            for j in range(self.J):
                if k in self.Ω_bank[j]:
                    bank = j
            index_k = self.Ω_bank[bank].index(k)
            realized_r = RR[k]
            self.rr_bank[bank][index_k] = realized_r

        self.Π = ΠΠ

    def returns_j(self, j):
        'Compute updated net worth and dividends of j-th firm'
        if self.Π[j] >= 0:
            d_A = (1 - self.ρ_firm) * self.Π[j]
            d = self.ρ_firm * self.Π[j]
        else:
            d_A = self.Π[j]
            d = 0

        return d_A, d

    def returns(self):
        'Compute updated net worth and dividends of j-th firm'
        d_A = []
        d = []
        for k in range(self.K):
            dd_A, dd = self.returns_j(k)
            d_A.append(dd_A)
            d.append(dd)

        self.d_firm = d
        for k in range(self.K):
            self.A_firm[k] += d_A[k]
            if self.A_firm[k] < 0:
                self.A_firm[k] = np.mean(self.A_firm)

    def technological_progress(self, k):
        'Obtain the new productivity for k-th firm given profits and credit'
        ι_1 = self.ι_1 ; ι_2 = self.ι_2 ; η = self.η ; ξ = self.ξ_firms[k]
        RD = self.χ * self.Π[k] # self.A_firm[k] -= RD

        IN = η * RD ; IM = (1-η) * RD

        if IN < - 100:
            θ_IN = 0.0
        else:
            θ_IN = max(0.0 , 1 - np.exp(-ι_1 * IN))
        θ_INN = np.random.binomial(1,θ_IN)
        if IM < - 100:
            θ_IM = 0.0
        else:
            θ_IM = max(0.0 , 1 - np.exp(-ι_2 * IM))
        θ_IMM = np.random.binomial(1,θ_IM)

        if θ_INN == 1:
            x_IN = np.random.beta(self.a, self.b, 1).tolist()[0]
            ξ_IN = ξ * (1 + x_IN)

        if θ_IMM == 1:
            dist = []
            sum = 0
            for i in range(len(self.ξ_firms)):
                d = (self.ξ_old[i] - ξ)**2
                dist.append(d)
                sum += d
            if sum > 0:
                for i in range(len(dist)):
                    dist[i] = dist[i] / sum
            else:
                for i in range(len(dist)):
                    dist[i] = 1 / len(dist)
            ξ_IM = np.random.choice(self.ξ_old, size = 1, p = dist).tolist()[0]
        ξ_exo = (1 + self.κ) * ξ

        if θ_INN == 1 and θ_IMM == 1:
            ξ_new = max(ξ_exo, ξ_IN, ξ_IM)
        elif θ_INN == 1 and θ_IMM == 0:
            ξ_new = max(ξ_exo, ξ_IN)
        elif θ_INN == 0 and θ_IMM == 1:
            ξ_new = max(ξ_exo, ξ_IM)
        else:
            ξ_new = ξ_exo

        return ξ_new

    def new_productivity(self):
        'Compute the productivity stemming from RD and exogenous process'
        new_ξ = []
        for k in range(self.K):
            ξ = self.technological_progress(k)
            new_ξ.append(ξ)
        self.ξ_firms = new_ξ
#------------------------------------------------------------------------------------------------------------------------
# I = 100
# J = 10
# K = 50
# H = 1000
#
# ξ_firms = [1 for k in range(K)]
# Y_D = [1 for k in range(K)]
# Y_S = [0.5 for k in range(K)]
# Π = [0.1 for k in range(K)]
# Ka = [0.5 for k in range(K)]
# A_firm = [5 for k in range(K)]
# λ = 0.2
# χ = 0.2
# φ_firm = 0.8
# ζ = 0.4
#
# model = pre_firms(ξ_firms, Y_D, Y_S, Π, Ka, A_firm, λ, χ, φ_firm, ζ)
#
# s = [1/K for k in range(K)]
# p_firm = [0.1 for k in range(K)]
# L_p_firm = [0.08 for k in range(K)]
# μ_firm = [0.1 for k in range(K)]
# w = 0.2
# r_ell_firms = [1 for k in range(K)]
# α_N = model.α_N
# Ω_bank = []
# client = list(range(K))
# for j in range(J):
#     client_j = np.random.choice(client, int(K/J), replace = False).tolist()
#     for client_jj in client_j:
#         client.remove(client_jj)
#     Ω_bank.append(client_j)
# r_bank = [[0.08 for k in range(len(Ω_bank[j]))] for j in range(J)]
# Ω_emp = []
# emp = list(range(H))
# for k in range(K):
#     workers_k = np.random.choice(emp, int((H*0.5) / K), replace = False).tolist()
#     for worker_k in workers_k:
#         emp.remove(worker_k)
#     Ω_emp.append(workers_k)
# N = [np.sum(Ω_emp[j]) for j in range(K)]
# μ = 0.7
# σ = 2
# ω_1 = 0.3
# ω_2 = 0.2
#
# model2 = firms_pricing(Y_D, Y_S, s, p_firm, L_p_firm, μ_firm, r_bank, w, r_ell_firms, A_firm, α_N, Ω_bank, ξ_firms, Ka, N, μ, σ, ω_1, ω_2, φ_firm, ζ)
#
# α_I = model.α_I
#
# model3 = firms_production(Ka, N, ξ_firms, r_ell_firms, A_firm, α_I, Ω_bank, φ_firm, ζ)
#
# r_Y = model3.Y_S
# p_firm = model2.p_firm
# α_RD = model.α_RD
# η = 0.3
# ι_1 = 0.1
# ι_2 = 0.2
# χ = 0.2
# a = 0
# b = 1
# κ = 0.1
# ρ_firm = 0.5
#
# model4 = post_firms(r_Y, Ka, N, w, Ω_bank, r_bank, r_ell_firms, p_firm, A_firm, ξ_firms, α_RD, α_I, α_N, φ_firm, ρ_firm, η, ι_1, ι_2, χ, a, b, κ)
