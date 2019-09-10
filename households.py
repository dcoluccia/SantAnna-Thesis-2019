import numpy as np
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω
#------------------------------------------------------------------------------------------------------------------------
class households:
    """
    Evaluate each household's income and get the resulting demand for goods that is fed into
    the goods market.
    What this does:
        1 - Evaluate the income from work and profit shares of each household
        2 - Evaluate the demand for good

    Inputs - Lists
    --------------------
        d_firm  :   Dividends (gross returns) of firms
                    [J*1]
        Ω_emp   :   Employed sets of firms
                    [J*H]
        Ω_firm  :   Client sets of firms
                    [J*H]
        σ_firm  :   Shares of firm ownership
                    [H*J]

    Inputs - Parameters
    --------------------
        w         :   wage (outcome of job market)
        b_h       :   social security wage
        τ_house   :   tax rate (tba)

    Output
    --------------------
        Y_D :   Demand vector to each firm
        G   :   Government Budget

    @author: ColucciaDM
    """

    def __init__(self, G, d_firm, p_firm, Ω_emp, Ω_firm, σ_firm, w, b_h, τ_house):
        #   INPUT - lists
        self.G = G
        self.d_firm = d_firm
        self.p_firm = p_firm
        self.Ω_emp = Ω_emp
        self.Ω_firm = Ω_firm
        self.σ_firm = σ_firm
        #   INPUT - params
        self.w = w
        self.b_h = b_h
        self.τ_house = τ_house
        #   OUTPUT
        self.Y_D = []
        self.U = []
        self.YY = []

        self.social_income = 0
        self.K = len(d_firm)
        self.H = 0
        for k in range(self.K):
            self.H += len(self.Ω_firm[k])
        #   WorkFlow
        self.retrieve_unemployed()
        self.compute_social_income()
        self.compute_wealth()
        self.compute_Y_D()
        self.update_government_budget()

    def retrieve_unemployed(self):
        'Populate one list of employed workers'
        emp = []
        for k in range(self.K):
            emp.extend(self.Ω_emp[k])
        self.U = emp

    def compute_social_income(self):
        'Compute total amount of resources due to social income'
        G = self.G
        unemployed = len(self.U)
        social_income = min(G, unemployed * self.b_h * self.w)

        self.G -= social_income
        self.social_income = social_income

    def compute_income(self, h):
        'Compute the work and financial income for h'
        social_income = self.social_income

        w = 0
        if h in self.U:
            w = social_income / len(self.U)
        else:
            w = self.w

        I = 0
        for k in range(self.K):
            i = self.σ_firm[h][k] * self.d_firm[k]
            I += i

        W = w + I
        return W

    def compute_wealth(self):
        'Get the wealth of each household'
        WW = []
        for h in range(self.H):
            ww = self.compute_income(h)
            WW.append(ww)

        self.YY = WW

    def compute_Y_D(self):
        'Compute the demand for consumption good for firms'
        DD = []
        for k in range(self.K):
            client_set = self.Ω_firm[k]
            dd = 0
            for h in client_set:
                dd += self.YY[h] * (1-self.τ_house) / self.p_firm[k]
            DD.append(dd)

        self.Y_D = DD

    def update_government_budget(self):
        'Get the budget set of government from income taxation'
        GG = 0
        for h in range(self.H):
            gg = self.τ_house * self.YY[h]
            GG += gg

        self.G = GG

#------------------------------------------------------------------------------------------------------------------------
# I = 100
# J = 10
# K = 50
# H = 1000
#
# d_firm = [0.1 for j in range(J)]
#
# Ω_emp = []
# emp = list(range(H))
# for k in range(K):
#     workers_k = np.random.choice(emp, int((H*0.5) / K), replace = False).tolist()
#     for worker_k in workers_k:
#         emp.remove(worker_k)
#     Ω_emp.append(workers_k)
#
# Ω_firm = []
# cons = list(range(H))
# for k in range(K):
#     consumers = cons[k:k+10]
#     Ω_firm.append(consumers)
#
# σ_firm = [[1/I for k in range(K)] for i in range(I)]
# w = 1
# b_h = 0.3
# τ_house = 0.0
#
# model = households(d_firm, Ω_emp, Ω_firm, σ_firm, w, b_h, τ_house)
