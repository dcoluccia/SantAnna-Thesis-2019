import numpy as np
# α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω, Ω, Δ
#------------------------------------------------------------------------------------------------------------------------
class firm_transitions:
    """
    Following the closure of markets, firms have a probability to switch bank
    that is governed by the interest rate that banks charge.
    What this does:
        1 - Compute within-bank average interest rate
        2 - Evaluate firm-level probability of transition
        3 - Compute transitions
        4 - Run for all firms

    Inputs - Lists
    --------------------
        r_bank      :   interest rate on loans
                        [J*K]
        Ω_bank      :   list of clients of banks
                        [J*K]

        μ_bank      :    old MA expected return of firm loans
                        [J*K]
        σ_bank      :    old MA expected variances of firm loans
                        [J*K]
        rr_bank     :   realized old interest rate on loans
                        [J*K]

    Inputs - Parameters
    --------------------
        υ_1         :   sensitivity to interest rate mark-ups

    Output
    --------------------
        Ω_bank      :   new list of clients of banks
                        [J*K]
        r_bank      :   interest rate on loans
                        [J*K]
        rr_bank     :   realized old interest rate on loans
                        [J*K]
        μ_bank      :    old MA expected return of firm loans
                        [J*K]
        σ_bank      :    old MA expected variances of firm loans
                        [J*K]

    @author: ColucciaDM
    """

    def __init__(self, r_bank, Ω_bank, μ_bank, σ_bank, rr_bank, υ_1):
        #   INPUT - Lists
        self.r_bank = r_bank
        self.Ω_bank = Ω_bank
        self.μ_bank = μ_bank
        self.σ_bank = σ_bank
        self.rr_bank = rr_bank

        self.J = len(self.r_bank)
        self.K = 0
        for j in range(self.J):
            self.K += len(self.Ω_bank[j])
        self.average_r = []
        #   INPUT - Parameters
        self.υ_1 = υ_1
        #   OUTPUT

        #   WorkFlow
        self.compute_average()
        self.update_all()

    def compute_average(self):
        'Compute within-bank average interest rate'
        average_r = []
        for j in range(self.J):
            if len(self.r_bank[j]) > 0:
                average_rr = np.mean(self.r_bank[j])
            else:
                average_rr = 0
            average_r.append(average_rr)

        self.average_r = average_r

    def transition_k(self, k):
        'Compute the (possibly) new bank for firm k'
        bank = 0
        for j in range(self.J):
            if k in self.Ω_bank[j]:
                bank = j
                index = self.Ω_bank[bank].index(k)
        r = self.r_bank[bank][index] ; old_j = bank

        active_banks = []
        for j in range(self.J):
            if self.average_r[j] < r:
                active_banks.append(j)

        if len(active_banks) > 0:
            chosen_j = np.random.choice(active_banks, 1).tolist()[0] ; pick_j = chosen_j
            ratio = (self.average_r[chosen_j] - r) / r
            ψ = max(1 - np.exp(self.υ_1 * ratio),0)
            θ = np.random.binomial(1, ψ)
        else:
            θ = 0 ; pick_j = old_j

        return θ, pick_j, old_j

    def update_k(self, k):
        'Update all lists if k transitions'
        θ, new_j, old_j = self.transition_k(k) ; r = self.average_r[new_j]

        if θ == 1:
            index_old_k = self.Ω_bank[old_j].index(k)
            #   Add to new
            self.Ω_bank[new_j].append(k)
            self.r_bank[new_j].append(r)
            self.rr_bank[new_j].append(r)
            self.μ_bank[new_j].append(self.μ_bank[old_j][index_old_k])
            self.σ_bank[new_j].append(self.σ_bank[old_j][index_old_k])
            #   Remove from old
            self.Ω_bank[old_j].remove(k)
            self.r_bank[old_j].remove(self.r_bank[old_j][index_old_k])
            self.rr_bank[old_j].remove(self.rr_bank[old_j][index_old_k])
            self.μ_bank[old_j].remove(self.μ_bank[old_j][index_old_k])
            self.σ_bank[old_j].remove(self.σ_bank[old_j][index_old_k])

    def update_all(self):
        'Update all transitions'
        for k in range(self.K):
            self.update_k(k)
#------------------------------------------------------------------------------------------------------------------------
class household_transitions:
    """
    Following the closure of markets, households can change the firm supplying them
    with the consumption good.
    What this does:
        1 - Evaluate active firms (those charging lower price)
        2 - Pick one and eventually change
        3 - Update

    Inputs - Lists
    --------------------
        Ω_firm      :   Client sets of firms
                        [K*H]
        p_firm      :   Price of consumption good
                        (K*1)

    Inputs - Parameters
    --------------------
        υ_2         :   sensitivity to price mark-ups

    Output
    --------------------
        Ω_firm      :   Client sets of firms
                        [K*H]

    @author: ColucciaDM
    """

    def __init__(self, Ω_firm, p_firm, υ_2):
        #   INPUT - Lists
        self.Ω_firm = Ω_firm
        self.p_firm = p_firm

        self.K = len(self.p_firm)
        self.H = 0
        for k in range(self.K):
            self.H += len(self.Ω_firm[k])
        #   INPUT - Params
        self.υ_2 = υ_2
        #   OUTPUT

        #   WorkFlow

    def transition_h(self, h):
        'Compute the (possibly) new firm for household h'
        firm = 0
        for k in range(self.K):
            if h in self.Ω_firm[k]:
                firm = k
        old_k = firm

        p = self.p_firm[firm]
        active_firms = []
        for k in range(self.K):
            if self.p_firm[k] < p:
                active_firms.append(k)

        if len(active_firms) > 0:
            chosen_k = np.random.choice(active_firms, 1) ; pick_k = chosen_k
            ratio = (self.p_firm[chosen_k] - p) / p
            ψ = 1 - np.exp(self.υ_2 * ratio)
            θ = np.random.binomial(1, ψ)
        else:
            θ = 0
            pick_k = old_k

        return θ, pick_k, old_k

    def update_h(self, h):
        'Update all lists if h transitions'
        θ, new_k, old_k = self.transition_h(h)

        if θ == 1:
            self.Ω_firm[old_k].remove(h)
            self.Ω_firm[new_k].append(h)

    def update_all(self):
        'Update all transitions'
        for h in range(self.H):
            self.update_h(h)
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
# μ_bank = [[0.1 for k in range(len(Ω_bank[j]))] for j in range(J)]
# σ_bank = [[0.5 for k in range(len(Ω_bank[j]))] for j in range(J)]
# r_bank = [[np.random.triangular(1,2,3) for k in range(len(Ω_bank[j]))] for j in range(J)]
# rr_bank = [[np.random.triangular(1,2,3) for k in range(len(Ω_bank[j]))] for j in range(J)]
# υ_1 = 0.1
#
# transitions_k = firm_transitions(r_bank, Ω_bank, μ_bank, σ_bank, rr_bank, υ_1)
#
# Ω_firm = []
# cons = list(range(H))
# for k in range(K):
#     consumers = cons[k:k+10]
#     Ω_firm.append(consumers)
#
# p_firm = [0.1 for k in range(K)]
# υ_2 = 0.1
#
# transitions_h = household_transitions(Ω_firm, p_firm, υ_2)
