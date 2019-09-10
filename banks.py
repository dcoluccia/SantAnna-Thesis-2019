import numpy as np
from scipy.special import erfinv
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω
#------------------------------------------------------------------------------------------------------------------------
class pre_banks:
    """
    Sets the supply of loans and the interest rate matrix banks charge to firms.
    What this does:
        1 - Banks set the interest rate to each firm in their client set
        2 - Banks optimize portfolio weights among clients
        3 - Banks evaluate their Value-at-Risk exposure and resulting total loan supply
        4 - Banks allocate credit among clients

    Input - Lists
    --------------------
        p_bank :    price of bank equity
                    [J*1]
        μ_bank :    old MA expected return of firm loans
                    [J*K]
        σ_bank :    old MA expected variances of firm loans
                    [J*K]
        r_bank :    old interest rate on loans
                    [J*K]
        rr_bank :   realized old interest rate on loans
                    [J*K]
        A_bank :    net worth of banks
                    [J*1]
        L_A_bank :  old net worth of banks
                    [J*1]
        A_firm :    net worth of firms
                    [K*1]
        L_A_firm :  old  net worth of firms
                    [K*1]
        Ω_bank :    list of clients of banks
                    [J*K]
        π :         inflation
                    [1*1]
        L_π :       lagged inflation
                    [1*1]
        ξ :         average productivity
                    [1*1]
        L_ξ :       lagged average productivity
                    [1*1]

    Input - Parameters
    --------------------
        ρ_1 :       wage - firm net worth
        ρ_2         wage - bank net worth
        ρ_3         wage - inflation
        ρ_4         wage - productivity
        δ_bank      MA expectation formation
        γ           Portfolio parameter
        bar_λ       Regulatory maximal leverage
        a           VaR exposure parameter
        E           Nominal equity

    Output
    --------------------
        μ_bank :    new MA expected return of firm loans
                    [J*K]
        σ_bank :    new MA expected variances of firm loans
                    [J*K]
        r_bank :    new interest rate on loans
                    [J*K]
        ell    :    credit supply
                    [J*K]

    @author: ColucciaDM
    """

    def __init__(self, p_bank, μ_bank, σ_bank, r_bank, rr_bank, A_bank, L_A_bank, A_firm, L_A_firm, Ω_bank, π, L_π, ξ_firms, L_ξ_firms, ρ_1, ρ_2, ρ_3, ρ_4, δ_bank, γ, bar_λ, a_VAR, E):
        # INPUT - Lists#
        self.p_bank     = p_bank
        self.μ_bank     = μ_bank
        self.σ_bank     = σ_bank
        self.r_bank     = r_bank
        self.rr_bank    = rr_bank
        self.A_bank     = A_bank
        self.L_A_bank   = L_A_bank
        self.A_firm     = A_firm
        self.L_A_firm   = L_A_firm
        self.Ω_bank     = Ω_bank
        self.π          = π
        self.L_π        = L_π
        self.ξ_firms    = ξ_firms
        self.L_ξ_firms  = L_ξ_firms

        self.ξ = np.mean(self.ξ_firms)
        self.L_ξ = np.mean(self.L_ξ_firms)

        self.λ_banks = []
        # INPUT - Parameters #
        self.ρ_1        = ρ_1
        self.ρ_2        = ρ_2
        self.ρ_3        = ρ_3
        self.ρ_4        = ρ_4
        self.δ_bank     = δ_bank
        self.γ          = γ
        self.bar_λ      = bar_λ
        self.a_VAR      = a_VAR
        self.E          = E
        # OUTPUT #
        self.ell = []
        self.λ_banks = []
        #other output is overwritten to existing input

        # Local variables
        self.J = len(self.A_bank)
        self.K = len(self.A_firm)

        # Work Flow
        self.compute_return_matrix()
        self.compute_loans()

    def compute_r(self, r, A_k, A_j, LA_k, LA_j, j):
        'Compute interest rate for a given firm'
        π = self.π ; L_π = self.L_π
        ξ = self.ξ ; L_ξ = self.L_ξ
        ρ_1 = 0.1*self.ρ_1[j] ; ρ_2 = 0.1*self.ρ_2[j] ; ρ_3 = 0.1*self.ρ_3[j] ; ρ_4 = 0.1*self.ρ_4[j]

        if LA_j != 0:
            dA_j = (A_j - LA_j) / LA_j
        else:
            dA_j = 0
        if LA_k != 0:
            dA_k = (A_k - LA_k) / LA_k
        else:
            dA_k = 0
        if L_π != 0:
            dπ =  (π - L_π) #/ L_π
        else:
            dπ = 0
        dξ = (ξ - L_ξ) / L_ξ

        new_r = r * (1 + ρ_1*dA_k - 0.25*ρ_2*dA_j + ρ_3*dπ + ρ_4*dξ) #( 1 - 0.8*ρ_1*dA_k - 0.3*ρ_2*dA_j + ρ_3*dπ + ρ_4*dξ) # (1 + np.random.triangular(-0.1, 0, 0.1))#

        return new_r

    def compute_return_matrix(self):
        'Use compute_r to evaluate the interest rate on loans for all firms'
        J = self.J

        R = []
        for j in range(J):
            row_R = []
            for k in self.Ω_bank[j]:
                index_client = self.Ω_bank[j].index(k)
                A_k = self.A_firm[k]
                A_j = self.A_bank[j]
                LA_k = self.L_A_firm[k]
                LA_j = self.L_A_bank[j]
                r = self.r_bank[j][index_client]
                new_r = max(self.compute_r(r, A_k, A_j, LA_k, LA_j, j) , 0.01)

                row_R.append(new_r)
            R.append(row_R)

        self.r_bank = R

    def compute_portfolio(self, j):
        'Compute portfolios of j-th bank'
        δ = self.δ_bank[j] ; r = self.r_bank[j] ; old_μ = self.μ_bank[j] ; γ = self.γ

        μ = [] ; A = []
        for k in self.Ω_bank[j]:
            k_index = self.Ω_bank[j].index(k)
            μ_k = δ*r[k_index] + (1-δ)*old_μ[k_index]
            A.append(self.A_firm[k])
            μ.append(μ_k)

        s = []
        for i in range(len(A)):
            if A[i] > 0:
                s_i = (μ[i]**(γ[j]))*(A[i]**(1-γ[j]))
            else:
                s_i = 0
            s.append(s_i)

        weight = [] ; sum = 0
        for i in range(len(s)):
            sum += np.exp(γ[j]*s[i])

        for i in range(len(s)):
            w = np.exp(γ[j]*s[i]) / sum
            weight.append(w)

        return weight

    def compute_loan_supply(self,j):
        'Compute leverage exposure of j-th bank'
        δ = self.δ_bank[j] ; r = self.rr_bank[j] ; old_μ = self.μ_bank[j] ; old_σ = self.σ_bank[j] ; γ = self.γ

        σ = []
        for k in self.Ω_bank[j]:
            k_index = self.Ω_bank[j].index(k)
            μ_k = δ*r[k_index] + (1-δ)*old_μ[k_index]
            σ_k = δ*(r[k_index] - μ_k)**2 + (1-δ)*old_σ[k_index]

            σ.append(σ_k)
        w = self.compute_portfolio(j)
        av_σ = 0
        for k in range(len(w)):
            av_σ += (w[k]**2)*σ[k]

        α = (np.sqrt(2)*erfinv(2*self.a_VAR - 1))**(-1)
        λ = min(α/av_σ, self.bar_λ)
        loan_supply = λ * self.p_bank[j] * self.E# + 0.1 * self.A_bank[j]

        return loan_supply, λ

    def compute_loans(self):
        'Compute loans to banks, update output'
        J = self.J

        ell = [] ; λλ = []
        for j in range(J):
            loan_supply, λ = self.compute_loan_supply(j)
            weights = self.compute_portfolio(j)
            loan = [weights[i]*loan_supply for i in range(len(weights))]
            ell.append(loan)
            λλ.append(λ)

        for j in range(J):
            μ = [] ; σ = []
            δ = self.δ_bank[j] ; r = self.r_bank[j]
            old_μ = self.μ_bank[j]
            old_σ = self.σ_bank[j]
            for k in self.Ω_bank[j]:
                k_index = self.Ω_bank[j].index(k)
                μ_k = δ*r[k_index] + (1-δ)*old_μ[k_index]
                σ_k = δ*(r[k_index] - μ_k)**2 + (1-δ)*old_σ[k_index]

                μ.append(μ_k) ; σ.append(σ_k)
            self.μ_bank[j] = μ ; self.σ_bank[j] = σ

        self.ell = ell ; self.λ_banks = λλ

#------------------------------------------------------------------------------------------------------------------------
class post_banks:
    """
    Post-credit market class: what this does:
        1 - Evaluate returns on loans
        2 - Distribute dividends

    Input - Lists
    --------------------
        r_ell   :   realized gross loan returns
        rr_bank:   realized interest rate on loans
        p_bank  :   price of equity
        A_bank  :   bank net worth

    Input - Parameters
    --------------------
        φ_bank  :   bank regulatory buffer
        E       :   nominal equity

    Output
    --------------------
        d_bank  :   net returns on bank equity
        A_bank  :   updated bank net worth

    @author: ColucciaDM
    """

    def __init__(self, r_ell, rr_bank, p_bank, A_bank, Ω_bank, credit_full, φ_bank, E):
        #   INPUTS - list
        self.r_ell = r_ell
        self.rr_bank = rr_bank
        self.p_bank = p_bank
        self.A_bank = A_bank
        self.Ω_bank = Ω_bank
        self.credit_full = credit_full

        self.J = len(self.A_bank)
        #   INPUTS - parameters
        self.φ_bank = φ_bank
        self.E = E
        #   OUTPUT
        self.d_bank = []

        #   Work flow
        self.compute_returns()

    def compute_gross_dividends(self,j):
        'Compute gross dividends and new net worth for the j-th bank'
        φ = self.φ_bank ; J = self.J

        returns = 0
        for k in self.Ω_bank[j]:
            index_k = self.Ω_bank[j].index(k)
            returns += self.credit_full[j][index_k] * (1 + self.rr_bank[j][index_k])

        returns += 0.002 * self.A_bank[j]
        #self.A_bank[j] -= 0.001 * self.A_bank[j]
        dG = φ*returns
        A = self.A_bank[j] + (1-φ)*returns
        D = dG/(self.E * self.p_bank[j])

        return D, A

    def compute_returns(self):
        'Compute the (J*1) return vector of banks'
        d_bank = [] ; a_bank = []
        for j in range(self.J):
            d,a = self.compute_gross_dividends(j)
            d_bank.append(d)
            a_bank.append(a)

        self.d_bank = d_bank
        self.A_bank = a_bank
#------------------------------------------------------------------------------------------------------------------------
# I = 100
# J = 10
# K = 50
# H = 1000
#
# Ω_bank = []
# client = list(range(K))
# for j in range(J):
#     client_j = np.random.choice(client, int(K/J), replace = False).tolist()
#     for client_jj in client_j:
#         client.remove(client_jj)
#     Ω_bank.append(client_j)
#
# p_bank = [0.01 for j in range(J)]
# μ_bank = [[0.01 for k in range(len(Ω_bank[j]))] for j in range(J)]
# σ_bank = [[0.5 for k in range(len(Ω_bank[j]))] for j in range(J)]
# r_bank = [[0.08 for k in range(len(Ω_bank[j]))] for j in range(J)]
# rr_bank = [[0.08 for k in range(len(Ω_bank[j]))] for j in range(J)]
# A_bank = [10 for j in range(J)]
# L_A_bank = [10 for j in range(J)]
# A_firm = [5 for k in range(K)]
# L_A_firm = [5 for k in range(K)]
# π = 0.01
# L_π = 0.01
# ξ_firms = [1 for k in range(K)]
# L_ξ_firms = [1 for k in range(K)]
# ρ_1 = [np.random.triangular(0.01,0.3,0.4) for j in range(J)]
# ρ_2 = [np.random.triangular(0.01,0.3,0.4) for j in range(J)]
# ρ_3 = [np.random.triangular(0.01,0.3,0.4) for j in range(J)]
# ρ_4 = [np.random.triangular(0.01,0.3,0.4) for j in range(J)]
# δ_bank = [np.random.triangular(0.5,0.6,0.7) for j in range(J)]
# γ = [np.random.triangular(0.8,1.0,1.2) for j in range(J)]
# bar_λ = 30
# a_VAR = 0.95
# E = 1
#
# model1 = pre_banks(p_bank, μ_bank, σ_bank, r_bank, rr_bank, A_bank, L_A_bank, A_firm, L_A_firm, Ω_bank, π, L_π, ξ_firms, L_ξ_firms, ρ_1, ρ_2, ρ_3, ρ_4, δ_bank, γ, bar_λ, a_VAR, E)
#
# r_ell = [2 for j in range(J)]
# credit_full = model1.ell
# φ_bank = 0.8
#
# model2 = post_banks(r_ell, rr_bank, p_bank, A_bank, Ω_bank, credit_full, φ_bank, E)
