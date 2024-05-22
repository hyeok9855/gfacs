from functools import cached_property
from itertools import combinations
import time
from typing import cast

import numba as nb
import numpy as np
import torch
from torch.distributions import Categorical


CAPACITY = 1


@nb.njit()
def calc_fullness_weighted_nbins(s: np.ndarray, demand: np.ndarray):
    """
    Modification of the fitness function proposed in the following paper:
        Falkenauer, Emanuel, and Alain Delchambre. "A genetic algorithm for bin packing and line balancing." ICRA. 1992.
    """
    ret = np.zeros(len(s),)
    n, m =  s.shape
    for i in range(n):
        f = 0
        sub_f = 0
        is_bin = 0
        for j in range(1, m):
            if s[i, j] != 0: # not dummy node
                sub_f += demand[s[i, j]]
                is_bin = 1
            else:
                if is_bin > 0:
                    f += 1 / (sub_f / CAPACITY)**2
                sub_f = 0
                is_bin = 0
        ret[i] = f  # number of bins, each bin has weight 1 / fullness, where fullness = (occupancy / CAPACITY)^2
    return ret


class ACO():
    def __init__(
        self,  # 0: depot
        demand,  # (n,)
        capacity=CAPACITY,
        n_ants=20,
        heuristic=None,
        pheromone=None,
        decay=0.9,
        alpha=1,
        beta=1,
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        shift_delta=True,
        use_local_search=False,
        device='cpu',
    ):
        self.problem_size = len(demand)
        self.demand = demand
        self.capacity = capacity

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites if n_elites is not None else n_ants // 10
        self.shift_delta = shift_delta

        self.use_local_search = use_local_search
        self.device = device

        self.heuristic = self.demand.unsqueeze(0).repeat(len(demand), 1) if heuristic is None else heuristic
        self.heuristic[:, 0] = 1e-10
        self.pheromone = torch.ones(len(demand), len(demand), device=device) * (1 / len(demand)) if pheromone is None else pheromone

        self.best_cost = np.inf
        self.best_sol = None

    @cached_property
    def demand_numpy(self):
        return self.demand.cpu().numpy()

    def sample(self, invtemp=1.0):
        sols, log_probs = self.gen_sol(invtemp=invtemp, require_prob=True)
        costs = self.gen_sol_costs(sols)
        return costs, log_probs, sols

    @torch.no_grad()
    def run(self, n_iterations):
        start = time.time()
        for _ in range(n_iterations):
            sols = cast(torch.Tensor, self.gen_sol(require_prob=False))
            costs = self.gen_sol_costs(sols)
            _sols = sols.clone()

            if self.use_local_search:
                sols, costs = self.local_search(sols, inference=True)

            best_cost, best_idx = costs.min(dim=0)
            if  best_cost < self.best_cost:
                self.best_cost = best_cost
                self.best_sol = sols[:, best_idx]

            self.update_pheromone(sols, costs)
        end = time.time()

        # Pairwise Jaccard similarity between sols
        edge_sets = []
        _sols = _sols.T.cpu().numpy()
        for _p in _sols:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.best_cost, diversity, end - start
       
    @torch.no_grad()
    def update_pheromone(self, sols, costs):
        '''
        Args:
            sols: torch tensor with shape (max_len, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        assert self.best_sol is not None

        sols = sols.clone()[:-1, :]  # 0-start, 0-end
        sol_gb = self.best_sol[:-1]
        deltas = 1 / costs
        delta_gb = 1 / self.best_cost
        if self.shift_delta:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * (sols.shape[0] - 1))

            deltas = (deltas + shifter).clamp(min=1e-10)
            delta_gb += shifter

        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_delta, best_idx = deltas.max(dim=0)
            best_sol = sols[:, best_idx]
            self.pheromone[best_sol, torch.roll(best_sol, shifts=1)] += best_delta
            self.pheromone[torch.roll(best_sol, shifts=1), best_sol] += best_delta

        elif self.rank_based:
            # Rank-based pheromone update
            elite_indices = torch.argsort(deltas, descending=True)[:self.n_elites]
            elite_sols = sols[:, elite_indices]
            elite_deltas = deltas[elite_indices]
            if self.best_cost < costs.min():
                diff_length = elite_sols.size(0) - sol_gb.size(0)
                if diff_length > 0:
                    sol_gb = torch.cat([sol_gb, torch.zeros(diff_length, device=self.device)])
                elif diff_length < 0:
                    elite_sols = torch.cat([elite_sols, torch.zeros((-diff_length, self.n_elites), device=self.device)], dim=0)

                elite_sols = torch.cat([sol_gb.unsqueeze(1), elite_sols[:, :-1]], dim=1)
                elite_deltas = torch.cat([torch.tensor([delta_gb], device=self.device), elite_deltas[:-1]])

            rank_denom = (self.n_elites * (self.n_elites + 1)) / 2
            for i in range(self.n_elites):
                sol = elite_sols[:, i]
                delta = elite_deltas[i] * (self.n_elites - i) / rank_denom
                self.pheromone[sol, torch.roll(sol, shifts=1)] += delta
                self.pheromone[torch.roll(sol, shifts=1), sol] += delta

        else:
            for i in range(self.n_ants):
                sol = sols[:, i]
                delta = deltas[i]
                self.pheromone[sol, torch.roll(sol, shifts=1)] += delta
                self.pheromone[torch.roll(sol, shifts=1), sol] += delta

        if self.maxmin:
            _max = delta_gb * (1 - self.decay)
            p_dec = 0.05 ** (1 / self.problem_size)
            _min = _max * (1 - p_dec) / (0.5 * self.problem_size - 1) / p_dec
            self.pheromone = torch.clamp(self.pheromone, min=_min, max=_max)
            # check convergence
            if (self.pheromone[sol_gb, torch.roll(sol_gb, shifts=1)] >= _max * 0.99).all():
                self.pheromone = 0.5 * self.pheromone + 0.5 * _max

        self.pheromone[self.pheromone < 1e-10] = 1e-10

    @torch.no_grad()
    def gen_sol_costs(self, sols: torch.Tensor):
        # sols.shape: (max_seq_len, n_ants)
        u = sols.T.cpu().numpy()
        # diff = u[:, 1:] - u[:, :-1]
        # n_bins = (diff != 0).sum(1) - self.problem_size + 1
        weighted_n_bins = calc_fullness_weighted_nbins(u, self.demand_numpy)
        return torch.tensor(weighted_n_bins, device=self.device)

    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None, n_ants=None):
        n_ants = n_ants or self.n_ants
        actions = torch.zeros((n_ants,), dtype=torch.long, device=self.device)
        sols_list = [actions]
        log_probs_list = []

        visit_mask = torch.ones(size=(n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions, n_ants)
        used_capacity = torch.zeros(size=(n_ants,), device=self.device)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity, n_ants)

        done = self.check_done(visit_mask, actions)
        i = 0
        while not done:
            guiding_node = sols[i + 1] if (sols is not None) and (i + 1 < sols.size(0)) else None
            actions, log_probs = self.pick_move(actions, visit_mask, capacity_mask, require_prob, invtemp, guiding_node)

            sols_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions, n_ants)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity, n_ants)
            done = self.check_done(visit_mask, actions)

            i += 1

        if require_prob:
            return torch.stack(sols_list), torch.stack(log_probs_list)
        else:
            return torch.stack(sols_list)

    def pick_move(self, prev, visit_mask, capacity_mask, require_prob, invtemp=1.0, guiding_node=None):
        if guiding_node is not None and not require_prob:
            return guiding_node, None

        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
        mask = visit_mask * capacity_mask
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * mask  # shape: (n_ants, p_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(dist)
        actions = dist.sample() if guiding_node is None else guiding_node  # shape: (n_ants,)
        log_prob = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_prob

    def update_visit_mask(self, visit_mask, actions, n_ants):
        visit_mask[torch.arange(n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1  # depot can be revisited with one exception
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0  # one exception is here
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity, n_ants):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]

        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size)  # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(n_ants, 1)  # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0

        # depot can be visited even if the capacity is exceeded (due to the floating point error)
        capacity_mask[capacity_mask.sum(1) == 0, 0] = 1
        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()

    ### Destroy & Repair local search
    def get_symmetric_sols(self, sols, n_symmetry):
        """
        Args:
            sols: (max_len, n_ants)
        Return:
            symm_sols: (max_len, n_ants, n_symmetry)
        """
        max_len, n_ants = sols.size()
        sols_np = sols.cpu().numpy().T

        symm_sols = np.repeat(sols_np[:, np.newaxis, :], n_symmetry, axis=1)
        # (n_ants, n_symmetry, max_len)

        # Get symmetric solutions by flip-and-shift operation
        n_tail_zeros = ((sols_np[:, 1:] - sols_np[:, :-1]) == 0).sum(1)
        for i in range(n_ants):
            seq_len = max_len - n_tail_zeros[i]
            symm_sols_single = symm_sols[i, :, :seq_len]
            zero_indices = np.argwhere(symm_sols_single[0] == 0).squeeze()
            flip_idx = np.sort(np.random.choice(zero_indices, size=(n_symmetry - 1, 2)), axis=1)
            shift_idx = np.random.choice(zero_indices, size=n_symmetry - 1)
            for j in range(n_symmetry - 1):  # the last symmetry is the original solution
                # Flip
                f_11, f_12 = flip_idx[j]
                symm_sols_single[j, f_11:f_12 + 1] = symm_sols_single[j, f_11:f_12 + 1][::-1]
                # Shift
                shift = shift_idx[j]
                shift = (f_11 + f_12 - shift) if f_11 < shift < f_12 else shift
                # symm_sols_single[j, 1:] = np.roll(symm_sols_single[j, 1:], -shift)
                symm_sols_single[j, 1:] = np.concatenate([symm_sols_single[j, 1:][shift:], symm_sols_single[j, 1:][:shift]])
            symm_sols[i, :, :seq_len] = symm_sols_single
        symm_sols = symm_sols.transpose(2, 0, 1)
        return torch.tensor(symm_sols, device=self.device)

    @torch.no_grad()
    def local_search(self, sols: torch.Tensor, inference=False):
        """
        Destroy & Repair local search, considering symmetry.
        Args:
            sols: (max_len, n_ants)
        """
        N_ANTS = sols.size(1)
        N_ROUNDS = 2  # Do not increase this value too much
        INVTEMP_MIN = 1.0
        INVTEMP_MAX = 1.0
        DESTROY_RATE_MAX = 0.5
        DESTROY_RATE_MIN = 0.5
        N_SYMMETRY = 10 if inference else 4  # maximum number of symmetric solutions to consider
        N_REPAIR = 5  # number of repairs each round
        ROUND_BUDGET = N_SYMMETRY * N_REPAIR
        TOPK = ROUND_BUDGET // 10
        assert TOPK > 0

        best_sols = sols.clone()
        best_costs = torch.ones(N_ANTS) * 1e10

        new_sols = sols.clone()
        for n in range(N_ROUNDS):
            invtemp = INVTEMP_MIN + (INVTEMP_MAX - INVTEMP_MIN) * n / max(1, N_ROUNDS - 1)
            destroy_rate = DESTROY_RATE_MAX - (DESTROY_RATE_MAX - DESTROY_RATE_MIN) * n / max(1, N_ROUNDS - 1)
            protected_len = int(sols.shape[0] * (1 - destroy_rate))
            _n_ants = new_sols.size(1)  # N_ANTS (* TOPK)
            _n_repair = max(1, (ROUND_BUDGET * N_ANTS) // (N_SYMMETRY * _n_ants))
            # _n_repair * N_SYMMETRY * _n_ants = ROUND_BUDGET * N_ANTS

            symm_sols = self.get_symmetric_sols(new_sols, N_SYMMETRY)
            # (max_len, N_ANTS (* TOPK), N_SYMMETRY)

            # Slicing is the most simple way to destroy the solutions, but there could be more sophisticated ways.
            destroyed_sols = symm_sols[:protected_len].unsqueeze(3).expand(-1, -1, -1, _n_repair)
            # (protected_len, N_ANTS (* TOPK), N_SYMMETRY, _n_repair)

            destroyed_sols = destroyed_sols.reshape(protected_len, _n_ants * N_SYMMETRY * _n_repair)
            new_sols = cast(
                torch.Tensor,
                self.gen_sol(
                    invtemp=invtemp, require_prob=False, sols=destroyed_sols, n_ants=_n_ants * N_SYMMETRY * _n_repair
                ),
            )
            new_max_seq_len = new_sols.size(0)
            # (max_seq_len, N_ANTS (* TOPK) * N_SYMMETRY * _n_repair)
            new_costs = self.gen_sol_costs(new_sols)
            # (N_ANTS (* TOPK) * N_SYMMETRY * _n_repair)

            new_sols = new_sols.view(new_max_seq_len, N_ANTS, -1)
            # (max_seq_len, N_ANTS, N_SYMMETRY * _n_repair)
            new_costs = new_costs.view(N_ANTS, -1)
            # (max_seq_len, N_ANTS, (TOPK *) N_SYMMETRY * _n_repair)

            best_idx = new_costs.argmin(dim=1)
            best_sols = new_sols[:, torch.arange(N_ANTS), best_idx]
            best_costs = new_costs[torch.arange(N_ANTS), best_idx]

            # Top-10% selection each ants
            topk_indices = torch.argsort(new_costs, dim=1)[:, :TOPK]
            new_sols = new_sols.gather(2, topk_indices.unsqueeze(0).expand(new_max_seq_len, -1, -1)).view(new_max_seq_len, -1)
            # (max_seq_len, N_ANTS * TOPK)

        return best_sols, best_costs
