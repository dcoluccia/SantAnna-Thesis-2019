import numpy as np
from equity import pre_equity, post_equity
from banks import pre_banks, post_banks
from firms import pre_firms, firms_production, firms_pricing, post_firms
from households import households
from markets import JobMarket, CreditMarket, GoodsMarket
from transitions import firm_transitions, household_transitions

# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω, Δ
#------------------------------------------------------------------------------------------------------------------------
class economy:
    """
    Main file: call in sequence classes.
    Timeline:
        1 - PRE MARKET
            1.1 - pre_equity
            1.2 - pre_firms
            1.3 - pre_banks
        2 - MARKET
            2.1 CreditMarket
            2.2 firms_pricing
            2.3 JobMarket
            2.4 firms_production
            2.5 GoodsMarket
        3 - POST MARKET
            3.1 post_firms
            3.2 post_banks
            3.3 post_equity
        4 - TRANSITIONS
            4.1 transitions_k
            4.2 transitions_h

    Inputs - Lists
    --------------------
        > Equity Funds
            δ_equity    :       MA parameter
            γ_equity    :       Chartism expectation
        > Banks
            δ_bank      :       MA expectation formation
            γ           :       Portfolio parameter
            φ_bank      :       Bank regulatory buffer
        > Firms
            ρ_firm      :   Share of profit divided to households
        > Households / Government
        > Markets
        > Transitions


    Inputs - Scalars
    --------------------
        > Equity Funds
            β_equity    :       Portfolio responsiveness
        > Banks
            ρ_1         :       wage - firm net worth
            ρ_2         :       wage - bank net worth
            ρ_3         :       wage - inflation
            ρ_4         :       wage - productivity
            bar_λ       :       Regulatory maximal leverage
            a_VAR       :       VaR exposure parameter
            E           :       Nominal equity
        > Firms
            μ           :   Mark up sensitivity
            σ           :   Market share sensitivity
            ω_1, ω_2    :   Competitiveness parameters
            φ_firm      :   Regulatory capital buffer
            ζ           :   Labor productivity
            η           :   Share of expenditure to innovation
            ι_1         :   Parameter of innovation expenditure
            ι_2         :   Parameter of imitation expenditure
            χ           :   Share of RD expenditure
            a,b         :   Support of the Beta
            κ           :   Exogenous technological progress rate
            δ           :   Capital depreciation
        > Households / Government
            b            :   social security wage (to be made endogenous)
            τ_house      :   tax rate (tba)
        > Markets
            ψ_1          :   sensitivity of wage wrt unemployment
            ψ_2          :   sensitivity of wage wrt inflation
            ψ_3          :   sensitivity of wage wrt productivity
        > Transitions
            υ_1          :   sensitivity to interest rate mark-ups
            υ_2          :   sensitivity to price mark-ups

    Inputs - Initial conditions
    --------------------
        self.p_bank          = p_bank
        self.μ_equity        = μ_equity
        self.σ_equity        = σ_equity
        self.d_bank          = d_bank
        self.f_bank          = f_bank
        self.Y_equity        = Y_equity
        self.w_equity        = w_equity
        self.n_equity        = n_equity
        self.Y_D             = Y_D
        self.Y_S             = Y_S
        self.Π               = Π
        self.K               = K
        self.λ               = λ
        self.s               = s
        self.p_firm          = p_firm
        self.L_p_firm        = L_p_firm
        self.μ_firm          = μ_firm
        self.μ_bank          = μ_bank
        self.σ_bank          = σ_bank
        self.r_bank          = r_bank
        self.rr_bank         = rr_bank
        self.A_bank          = A_bank
        self.L_A_bank        = L_A_bank
        self.A_firm          = A_firm
        self.L_A_firm        = L_A_firm
        self.Ω_bank          = Ω_bank
        self.π               = π
        self.L_π             = L_π
        self.N               = N
        self.L_N             = L_N
        self.U               = U
        self.Ω_emp           = Ω_emp
        self.Ω_bank          = Ω_bank
        self.Ω_firm          = Ω_firm
        self.σ_firm          = σ_firm
        self.w               = w

    Output
    --------------------
        > Equity funds
            p_bank      :       Price of bank equity
            d_bank      :       Return on bank equity
        > Banks
            r_bank      :       Lending rate
            ell         :       Supply of credit
            r_ell_firms :       Realized credit supply
            λ_banks    :       Leverage ratio
            A_bank      :       Net worth of banks
        > Firms
            ell_d       :       Demand for credit
            s           :       Market share
            K           :       Capital
            I           :       Investment
            A_firm      :       Firm net worth
            ξ_firms     :       New productivity levels
        > Household
            w           :       Wage (outcome of job market)
            len(U)      :       Absolute unemployment
            np.sum(L)   :       Absolute employment
        > Macro-indicators
            np.sum(r_Y) :       Consumption
            np.sum(I)   :       Investment
            π           :       Inflation
            ξ_firms     :       Firm productivities

    @author : ColucciaDM
    """

    def __init__(self,
                 δ_equity, γ_equity, δ_bank, γ, φ_bank, ρ_firm, ξ_firms, L_ξ_firms, # Lists
                 β_equity, #Equity
                 ρ_1, ρ_2, ρ_3, ρ_4, bar_λ, a_VAR, E, #Banks
                 μ, σ, ω_1, ω_2, φ_firm, ζ, η, ι_1, ι_2, χ, a, b, κ,  #Firms
                 b_h, τ_house, ψ_1, ψ_2, ψ_3,  #Households - Job market
                 υ_1, υ_2, δ, bar_w,#Transitions
                 p_bank, μ_equity, σ_equity, d_bank, f_bank, Y_equity, w_equity, n_equity, #Initial conditions
                 Y_D, Y_S, Π, K, λ, s, d_firm, p_firm, L_p_firm, μ_firm,
                 μ_bank, σ_bank, r_bank, rr_bank, A_bank, L_A_bank, A_firm, L_A_firm, π, L_π,
                 N, L_N, U, Ω_emp, Ω_bank, Ω_firm, σ_firm, w, G,
                 T): #Number of iterations
        #   INPUT - Lists
        self.δ_equity   = δ_equity
        self.γ_equity   = γ_equity
        self.δ_bank     = δ_bank
        self.γ          = γ
        self.φ_bank     = φ_bank
        self.ρ_firm     = ρ_firm
        self.ξ_firms    = ξ_firms
        self.L_ξ_firms  = L_ξ_firms
        #   INPUT - Params
        self.β_equity   = β_equity
        self.ρ_1        = ρ_1
        self.ρ_2        = ρ_2
        self.ρ_3        = ρ_3
        self.ρ_4        = ρ_4
        self.bar_λ      = bar_λ
        self.a_VAR      = a_VAR
        self.E          = E
        self.μ          = μ
        self.σ          = σ
        self.ω_1        = ω_1
        self.ω_2        = ω_2
        self.φ_firm     = φ_firm
        self.ζ          = ζ
        self.η          = η
        self.ι_1        = ι_1
        self.ι_2        = ι_2
        self.χ          = χ
        self.a          = a
        self.b          = b
        self.κ          = κ
        self.b_h        = b_h
        self.τ_house    = τ_house
        self.ψ_1        = ψ_1
        self.ψ_2        = ψ_2
        self.ψ_3        = ψ_3
        self.υ_1        = υ_1
        self.υ_2        = υ_2
        self.δ          = δ
        self.bar_w      = bar_w
        #   INPUT - Initial Conditions
        self.p_bank          = p_bank
        self.μ_equity        = μ_equity
        self.σ_equity        = σ_equity
        self.d_bank          = d_bank
        self.f_bank          = f_bank
        self.Y_equity        = Y_equity
        self.w_equity        = w_equity
        self.n_equity        = n_equity
        self.Y_D             = Y_D
        self.Y_S             = Y_S
        self.Π               = Π
        self.K               = K
        self.λ               = λ
        self.s               = s
        self.d_firm          = d_firm
        self.p_firm          = p_firm
        self.L_p_firm        = L_p_firm
        self.μ_firm          = μ_firm
        self.μ_bank          = μ_bank
        self.σ_bank          = σ_bank
        self.r_bank          = r_bank
        self.rr_bank         = rr_bank
        self.A_bank          = A_bank
        self.L_A_bank        = L_A_bank
        self.A_firm          = A_firm
        self.L_A_firm        = L_A_firm
        self.π               = π
        self.L_π             = L_π
        self.N               = N
        self.L_N             = L_N
        self.U               = U
        self.Ω_emp           = Ω_emp
        self.Ω_bank          = Ω_bank
        self.Ω_firm          = Ω_firm
        self.σ_firm          = σ_firm
        self.w               = w
        self.G               = G
        #
        self.T               = T
        #
        self.H = 0
        for j in range(len(self.Ω_firm)):
            self.H += len(self.Ω_firm[j])
        #   OUTPUT
        #   Equity Funds
        self.p_bank_ts      = []
        self.d_bank_ts      = []
        #   Banks
        self.r_bank_ts       = []
        self.r_ts            = []
        self.rr_ts           = []
        self.ell_ts          = []
        self.r_ell_firms_ts  = []
        self.λ_banks_ts      = []
        self.A_bank_ts       = []
        self.AA_bank_ts      = []
        #   Firms
        self.p_firm_ts      = []
        self.ell_d_ts       = []
        self.s_ts           = []
        self.K_ts           = []
        self.I_ts           = []
        self.A_firm_ts      = []
        self.AA_firm_ts     = []
        self.ξ_firms_ts     = []
        #   Job Market
        self.w_ts           = []
        self.unemp_rate_ts  = []
        self.emp_rate_ts    = []
        #   Macroeconomy
        self.C_ts           = []
        self.Y_ts           = []
        self.π_ts           = []
        self.ξ_ts           = []
        self.G_ts           = []

        #   WORKFLOW
        self.simulate()

    def run_economy(self):
        'Run one single shot of the economy'
        #   1   #
        a1 = pre_equity(self.p_bank, self.μ_equity, self.σ_equity, self.d_bank, self.f_bank, self.Y_equity, self.w_equity, self.n_equity, self.δ_equity, self.β_equity, self.γ_equity, self.E)
        p_bank = a1.p_bank ; μ_equity = a1.μ_equity ; σ_equity = a1.σ_equity ; w_equity = a1.w_equity; new_e_Y = a1.new_e_Y
        # print('a1')
        a2 = pre_firms(self.ξ_firms, self.Y_D, self.Y_S, self.Π, self.K, self.A_firm, self.N, self.Ω_emp, self.U, self.w, self.λ, self.χ, self.φ_firm, self.ζ, self.δ)
        ell_d = a2.ell_d ; α_RD = a2.α_RD ; α_I = a2.α_I ; α_N = a2.α_N ; N_e = a2.N_e ; N = a2.N ; Ω_emp = a2.Ω_emp ; K = self.K ; I_e = a2.I_e
        # print('a2')
        a3 = pre_banks(p_bank, self.μ_bank, self.σ_bank, self.r_bank, self.rr_bank, self.A_bank, self.L_A_bank, self.A_firm, self.L_A_firm, self.Ω_bank, self.π, self.L_π, self.ξ_firms, self.L_ξ_firms, self.ρ_1, self.ρ_2, self.ρ_3, self.ρ_4, self.δ_bank, self.γ, self.bar_λ, self.a_VAR, self.E)
        μ_bank = a3.μ_bank ; σ_bank = a3.σ_bank ; r_bank = a3.r_bank ; ell = a3.ell ; λ_banks = a3.λ_banks
        # print('a3')
        #   2   #
        b1 = CreditMarket(ell, ell_d, self.Ω_bank)
        r_ell_firms = b1.r_ell_firms ; r_ell = b1.r_ell ; credit_full = b1.credit_full
        # print('b1')
        b2 = firms_pricing(self.Y_D, self.Y_S, self.s, self.p_firm, self.L_p_firm, self.μ_firm, r_bank, self.w, r_ell_firms, self.A_firm, α_N, self.Ω_bank, self.ξ_firms, self.K, self.N, Ω_emp, self.μ, self.σ, self.ω_1, self.ω_2, self.φ_firm, self.ζ)
        p_firm = b2.p_firm ; L_p_firm = b2.L_p_firm ; μ_firm = b2.μ_firm ; s = b2.s
        # print('b2')
        b3 = JobMarket(self.N, self.L_N, self.U, N_e, Ω_emp, self.w, self.π, self.L_π, p_firm, L_p_firm, self.Π, self.ξ_firms, self.L_ξ_firms, self.ψ_1, self.ψ_2, self.ψ_3, self.bar_w)
        w = b3.w ; Ω_emp = b3.Ω_emp ; N = b3.N ; L_N = b3.L_N ; U = b3.U
        # print('b3')
        b4 = households(self.G, self.d_firm, p_firm, Ω_emp, self.Ω_firm, self.σ_firm, w, self.b_h, self.τ_house)
        Y_D = b4.Y_D ; G = b4.G
        # print('b4')
        b5 = firms_production(K, N, self.ξ_firms, r_ell_firms, self.A_firm, I_e, α_I, self.Ω_bank, self.φ_firm, self.ζ)
        Y_S = b5.Y_S ; Ka = b5.Ka ; I = b5.I ; A_firm = b5.A_firm
        # print('b5')
        b6 = GoodsMarket(Y_D, Y_S, p_firm, L_p_firm, self.π, self.L_π)
        r_Y = b6.r_Y ; π = b6.π ; L_π = b6.L_π
        # print('b6')
        #   3   #
        c1 = post_firms(r_Y, Ka, N, w, self.Ω_bank, r_bank, r_ell_firms, p_firm, A_firm, self.ξ_firms, α_RD, α_I, α_N, self.φ_firm, self.ρ_firm, self.η, self.ι_1, self.ι_2, self.χ, self.a, self.b, self.κ)
        A_firm = c1.A_firm ; d_firm = c1.d_firm ; ξ_firms = c1.ξ_firms ; Π = c1.Π ; rr_bank = c1.rr_bank
        # print('c1')
        c2 = post_banks(r_ell, rr_bank, p_bank, self.A_bank, self.Ω_bank, credit_full, self.φ_bank, self.E)
        d_bank = c2.d_bank ; A_bank = c2.A_bank
        # print('c2')
        c3 = post_equity(p_bank, d_bank, new_e_Y, w_equity, self.n_equity, self.E)
        Y_equity = c3.Y_equity ; n_equity = c3.n_equity ; f_bank = c3.f_bank
        # print('c3')
        #   4   #
        d1 = firm_transitions(r_bank, self.Ω_bank, μ_bank, σ_bank, rr_bank, self.υ_1)
        Ω_bank = d1.Ω_bank ; r_bank = d1.r_bank ; rr_bank = d1.rr_bank ; μ_bank = d1.μ_bank ; σ_bank = d1.σ_bank
        # print('d1')
        d2 = household_transitions(self.Ω_firm, p_firm, self.υ_2)
        Ω_firm = d2.Ω_firm
        # print('d2')

        # #   Check Print
        # print('--------------------------')
        # print('d_bank') ; print(np.mean(d_bank))
        # print('d_firm'); print(np.mean(d_firm))
        # print('Profit'); print(np.mean(Π))
        # print('Demand') ; print(np.sum(Y_D)) ; print('Supply') ; print(np.sum(Y_S))
        # print('p_firm') ;print(np.mean(p_firm))
        # print('Unemp'); print(len(U))
        # print('wage') ; print(w)
        # print('Employment') ; print(np.sum(N))
        # print('Inflation') ; print(π)
        # print('Bank Credit') ; print(np.sum([np.sum(credit_full[j]) for j in range(len(ell))]))
        # print('Firm Credit') ; print(np.sum(r_ell_firms))
        # print('Offered Credit') ; print(np.sum([np.sum(ell[j]) for j in range(len(ell))]))
        # print('Demanded Credit') ; print(np.sum(ell_d))
        # # print('Consumption') ; print(np.sum(r_Y))
        # # print('Investment') ; print(np.sum(I))
        # print('---------------------------')
        # #print(K) ; print(N)

        #   UPDATE self INSTATIATION
        self.p_bank = p_bank
        self.μ_equity = μ_equity
        self.σ_equity = σ_equity
        self.d_bank = d_bank
        self.f_bank = [max(f_bank[i] + np.random.normal(0,0.2),0) for i in range(len(f_bank))]
        self.Y_equity = Y_equity
        self.w_equity = w_equity
        self.n_equity = n_equity

        self.L_ξ_firms = self.ξ_firms
        self.ξ_firms = ξ_firms
        self.Y_D = Y_D
        self.Y_S = Y_S
        self.Π = Π
        self.Ka = Ka

        self.μ_bank = μ_bank
        self.σ_bank = σ_bank
        self.r_bank = r_bank
        self.rr_bank = rr_bank
        self.L_A_bank = self.A_bank
        self.A_bank = A_bank
        self.L_A_firm = self.A_firm
        self.A_firm = A_firm
        self.Ω_bank = Ω_bank
        self.L_π = self.π
        self.π = π

        self.ell = ell
        self.s = s
        self.L_p_firm = self.p_firm
        self.p_firm = p_firm
        self.μ_firm = μ_firm
        self.w = w
        self.L_N = self.N
        self.N = N
        self.U = U
        self.Ω_emp = Ω_emp
        self.d_firm = d_firm
        self.Ω_firm = Ω_firm
        self.G = G

        #   Append TS   #
        #   Equity Funds
        self.p_bank_ts.append(np.mean(p_bank))
        self.d_bank_ts.append(np.mean(d_bank))
        #   Banks
        r_bank_ts = [] ; r_ts = []
        for j in range(len(r_bank)):
            if len(r_bank[j]) > 0:
                r_bank_ts.append(np.mean(r_bank[j]))
                r_ts.append(np.mean(r_bank[j]))
            else:
                r_bank_ts.append(0)
                r_ts.append(0)
        self.r_bank_ts.append(np.mean(r_bank_ts))
        self.r_ts.append(np.mean(r_ts))
        rr_ts = []
        for j in range(len(r_bank)):
            if len(rr_bank[j]) > 0:
                rr_ts.append(np.mean(rr_bank[j]))
            else:
                rr_ts.append(0)
        self.rr_ts.append(np.mean(rr_ts))
        self.ell_ts.append(np.mean([np.sum(ell[j]) for j in range(len(ell))]))
        self.r_ell_firms_ts.append(np.sum(r_ell_firms))
        self.λ_banks_ts.append(np.mean(λ_banks))
        self.A_bank_ts.append(A_bank)
        self.AA_bank_ts.append(np.mean(A_bank))
        #   Firms
        self.p_firm_ts.append(np.mean(self.p_firm))
        self.ell_d_ts.append(np.sum(ell_d))
        self.s_ts.append(np.mean(s))
        self.K_ts.append(np.sum(Ka))
        self.I_ts.append(np.sum(I))
        self.A_firm_ts.append(A_firm[:])
        self.AA_firm_ts.append(np.sum(A_firm))
        self.ξ_firms_ts.append(ξ_firms)
        #   Job Market
        self.w_ts.append(w)
        self.unemp_rate_ts.append(len(U) / self.H)
        self.emp_rate_ts.append(np.sum(N) / self.H)
        #   Macroeconomy
        self.C_ts.append(np.sum(r_Y))
        self.Y_ts.append(np.sum(r_Y) + np.sum(I))
        self.π_ts.append(π)
        self.ξ_ts.append(np.mean(ξ_firms))
        self.G_ts.append(G)

    def shock(self, t):
        if t == 750:
            self.bar_λ = 3

    def simulate(self):
        'Simulate the economy T times. Obtain resulting time series.'
        for t in range(self.T):
            print(t)
            self.shock(t)
            self.run_economy()
#------------------------------------------------------------------------------------------------------------------------
