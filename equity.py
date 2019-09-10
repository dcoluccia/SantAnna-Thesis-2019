import numpy as np
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω
#------------------------------------------------------------------------------------------------------------------------
class pre_equity:
    """
    Before the credit market, equity funds price bank equity according to a quasi
    market clearing mechanism.
    What this does:
        1 - Given current MA expectations, update
        2 - Compute portfolio weights
        3 - Compute market excess demand
        4 - Obtain the market clearing price

    Inputs - Lists
    --------------------
        p_bank      :       Price of bank equity
                            [J*1]
        μ_equity    :       MA expected return
                            [I*J]
        σ_equity    :       MA expected return variance
                            [I*J]
        d_bank      :       Return on bank equity
                            [J*1]
        f_bank      :       Forward return on bank equity
                            [J*1]
        Y_equity    :       Past fund net worth
                            [I*1]
        w_equity    :       Past portfolio weights
                            [I*J]
        n_equity    :       Past equity holdings
                            [I*J]
    Inputs - Parameters
    --------------------
        δ_equity    :       MA parameter
        β_equity    :       Portfolio responsiveness
        γ_equity    :       Chartism expectation

    Output
    --------------------
        p_bank      :       NEW price of bank equity
        μ_equity    :       NEW expected return on equity
        σ_equity    :       NEW expected variance
        w_equity    :       NEW weights

    @author: ColucciaDM
    """

    def __init__(self, p_bank, μ_equity, σ_equity, d_bank, f_bank, Y_equity, w_equity, n_equity, δ_equity, β_equity, γ_equity, E):
        #   INPUT - Lists
        self.p_bank = p_bank
        self.μ_equity = μ_equity
        self.σ_equity = σ_equity
        self.d_bank = d_bank
        self.f_bank = f_bank
        self.Y_equity = Y_equity
        self.w_equity = w_equity
        self.n_equity = n_equity
        #   INPUT - parameters
        self.δ_equity = δ_equity
        self.β_equity = β_equity
        self.γ_equity = γ_equity
        self.E = E

        self.new_w = []
        self.new_e_n = []
        self.new_e_Y = []
        self.I = len(Y_equity)
        self.J = len(p_bank)
        #   OUTPUT

        #   Workflow
        self.compute_portfolios()
        self.compute_expected_incomes()
        self.compute_price()
        self.update()

    def compute_MA(self, i):
        'Update MA components for i-th fund'
        δ = self.δ_equity[i] ; μ = self.μ_equity[i] ; σ = self.σ_equity[i]

        μ_i = [] ; σ_i = []
        for j in range(self.J):
            μ_j = δ*self.d_bank[j] + (1-δ)*μ[j]
            σ_j = δ*(self.d_bank[j] - μ_j)**2 + (1-δ)*σ[j]
            μ_i.append(μ_j)
            σ_i.append(σ_j)

        return μ_i, σ_i

    def compute_portfolio(self, i):
        'Run portfolio optimization to derive portfolio weights for the i-th firm'
        μ_i, σ_i = self.compute_MA(i)
        s = []
        for j in range(len(μ_i)):
            s.append(np.log(1+μ_i[j]))#/σ_i[j]) #<----------------------------s.append(μ_i[j])

        for j in range(len(s)):
            s[j] = np.exp(self.β_equity * s[j])

        sum = 0
        for j in range(len(s)):
            sum += s[j]

        for j in range(len(s)):
            s[j] = s[j] / sum
        w_i = s

        return w_i, μ_i, σ_i

    def compute_portfolios(self):
        'Compute portfolios among all equity funds'
        new_w = [] ; new_μ = [] ; new_σ = []
        for j in range(self.I):
            w , μ, σ = self.compute_portfolio(j)
            new_w.append(w)
            new_μ.append(μ)
            new_σ.append(σ)

        self.new_w = new_w
        self.μ_equity = new_μ
        self.σ_equity = new_σ

    def compute_expected_income(self, i):
        'Compute the expected income of the i-th fund given standing portfolio'
        Y = 0
        for j in range(self.J):
            d_e = self.γ_equity[i] * self.f_bank[j] + (1-self.γ_equity[i]) * self.μ_equity[i][j]
            Y  += d_e * self.n_equity[i][j]

        return Y

    def compute_expected_incomes(self):
        'Compute overall expected income across funds'
        Y = []
        for i in range(self.I):
            y = self.compute_expected_income(i)
            Y.append(y)

        self.new_e_Y = Y

    def compute_price(self):
        'Compute market clearing price of bank equity'
        PP = []
        for j in range(self.J):
            P = self.p_bank[j]
            new_d = 0
            old_d = 0

            for i in range(self.I):
                new_dd = self.new_w[i][j] * self.new_e_Y[i]
                old_dd = self.w_equity[i][j] * self.Y_equity[i]

                new_d += new_dd
                old_d += old_dd

            pp = new_d
            #P * np.divide(new_d, old_d)
            PP.append(pp)

        self.p_bank = PP

    def update(self):
        'Update interim lists'
        self.w_equity = self.new_w
#------------------------------------------------------------------------------------------------------------------------
class post_equity:
    """
    Post-credit, good market: returns on equity are realized, actual income is computed and
    orders are adjusted to clear.
    What this does:
        1 - Compute realized income
        2 - Compute fraction of orders that can actually be matched
        3 - Update the new composition of portfolios

    Inputs - Lists
    --------------------
        p_bank      :       Price of bank equity
                            [J*1]
        d_bank      :       Return on bank equity
                            [J*1]
        new_e_Y     :       New expected fund net worth
                            [I*1]
        w_equity    :       Past portfolio weights
                            [I*J]
        n_equity    :       Past equity holdings
                            [I*J]

    Output
    --------------------
        Y_equity    :       NEW fund net worth
                            [I*1]
        n_equity    :       NEW equity holdings
                            [I*J]
        f_bank      :       NEW Forward return on bank equity
                            [J*1]

    @author: ColucciaDM
    """

    def __init__(self, p_bank, d_bank, new_e_Y, w_equity, n_equity, E):
        #   INPUT
        self.p_bank = p_bank
        self.d_bank = d_bank
        self.new_e_Y = new_e_Y
        self.w_equity = w_equity
        self.n_equity = n_equity
        self.E = E

        self.I = len(n_equity)
        self.J = len(d_bank)
        #   OUTPUT
        self.f_bank = []
        self.Y_equity = []

        #   WorkFlow
        self.compute_Y_overall()
        self.update_equity()

    def compute_Y(self, i):
        'Compute net worth given realized return'
        Y = 0
        for j in range(self.J):
            y = self.d_bank[j] * self.n_equity[i][j]
            Y += y

        return Y

    def compute_Y_overall(self):
        'Compute new net worth'
        YY = []
        for i in range(self.I):
            yy = self.compute_Y(i)
            YY.append(yy)

        self.Y_equity = YY

    def expected_n(self, j):
        'Compute the expected j-th bank equity exposure'
        num = 0
        for i in range(self.I):
            nnum = self.w_equity[i][j] * self.new_e_Y[i]
            num += nnum
        # num = self.E * self.p_bank[j]

        Num = num
        return Num

    def expected_d(self, j):
        'Compute the realized market clearing j-th bank equity exposure'
        denum = 0
        for i in range(self.I):
            ddenum = self.w_equity[i][j] * self.Y_equity[i]
            denum += ddenum

        Denum = denum
        return Denum

    def compute_φ(self, j):
        'Compute the adjustment parameter for j-th bank'
        N = self.expected_n(j)
        D = self.expected_d(j)
        φ_j = np.divide(N , D)

        return φ_j

    def compute_adjustment(self):
        'Compute the overall adjustment coefficients'
        φ = []
        for j in range(self.J):
            φ.append(self.compute_φ(j))

        return φ

    def update_equity(self):
        'Update the portfolios consistent with realized market clearing'
        φ = self.compute_adjustment()
        for j in range(self.J):
            for i in range(self.I):
                self.n_equity[i][j] = φ[j] * self.w_equity[i][j] * np.divide(self.Y_equity[i] , self.p_bank[j])
        self.f_bank = [self.d_bank[k] + np.random.normal(0,0.1) for k in range(self.J)]
#------------------------------------------------------------------------------------------------------------------------
# I = 100
# J = 10
# E = 1
# p_bank = [np.random.triangular(0.5,1,1.5) for j in range(J)]
# μ_equity = [[1 for j in range(J)] for i in range(I)]
# σ_equity = [[5 for j in range(J)] for i in range(I)]
# d_bank = [1 for j in range(J)]
# f_bank = [1 for j in range(J)]
# Y_equity = [0.12 for i in range(I)]
# w_equity = [[1/J for j in range(J)] for i in range(I)]
# n_equity = [[E/I for j in range(J)] for i in range(I)]
# δ_equity = [np.random.triangular(0.1,0.2,0.3) for i in range(I)]
# γ_equity = [np.random.triangular(0.2,0.3,0.4) for i in range(I)]
# β_equity = 0.7
#
# d_bank_1 = [np.random.triangular(0.5,1,1.5) for j in range(J)]
#
# T = 100
# P = [] ; D = [] ; Y_R = [] ; Y_E = []
# for t in range(T):
#     model = pre_equity(p_bank, μ_equity, σ_equity, d_bank_1, f_bank, Y_equity, w_equity, n_equity, δ_equity, β_equity, γ_equity, E)
#
#     d_bank_2 = [np.random.triangular(0.5,1,1.5) for j in range(J)]
#     f_bank = d_bank_2
#
#     w_equity = model.w_equity
#     new_e_Y = model.new_e_Y
#     d_bank = [np.random.triangular(0.5,1,1.5) for j in range(J)]
#     p_bank = model.p_bank
#     μ_equity = model.μ_equity
#     σ_equity = model.σ_equity
#
#     model2 = post_equity(p_bank, d_bank_2, new_e_Y, w_equity, n_equity, E)
#
#     d_bank_1 = d_bank_2
#     Y_equity = model2.Y_equity
#     w_equity = model.w_equity
#     n_equity = model2.n_equity
#
#     P.append(np.mean(p_bank))
#     D.append(np.mean(d_bank))
#     Y_R.append(np.sum(Y_equity))
#     Y_E.append(np.sum(model.new_e_Y))
#
# import matplotlib.pyplot as plt
# T = list(range(T))
# fig, ax = plt.subplots()
#
# color = 'tab:red'
# ax.plot(T, P, color = color, label = 'Price')
# color = 'tab:blue'
# ax.plot(T, D, color = color, label = 'Return')
# ax.legend()
# ax.set_xlabel('Time')
# ax.set_title('Price and Returns')
#
# fig.tight_layout()
# plt.show()
#
# print(np.corrcoef(P,D))

# fig, ax = plt.subplots()
#
# color = 'tab:red'
# ax.plot(T, Y_R, color = color, label = 'Realized')
# color = 'tab:blue'
# ax.plot(T, Y_E, color = color, label = 'Expected')
# ax.legend()
# ax.set_xlabel('Time')
# ax.set_title('Income')
#
# fig.tight_layout()
# plt.show()
