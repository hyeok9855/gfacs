from itertools import combinations
import time
from typing import cast

import torch
from torch.distributions import Categorical


class ACO():
    def __init__(
        self,
        due_time: torch.Tensor,  # (n,)
        weights: torch.Tensor,  # (n,)
        processing_time: torch.Tensor, # (n,)
        n_ants=20,
        heuristic: torch.Tensor | None = None,
        pheromone: torch.Tensor | None = None,
        decay=0.9,
        alpha=1,
        beta=1,
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        shift_delta=True,
        use_local_search=False,
        device='cpu'
    ):
        self.problem_size = len(due_time)
        self.due_time = due_time
        self.weights = weights
        self.processing_time = processing_time
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10
        self.shift_delta = shift_delta

        self.use_local_search = use_local_search
        self.device = device

        # A Weighted Population Update Rule for PACO Applied to the Single Machine Total Weighted Tardiness Problem
        # perfer jobs with smaller due time, [n + 1, n + 1], includes dummy node 0
        if heuristic is None:
            dummy_due_time = torch.cat([torch.tensor([1], device=device), self.due_time])
            self.heuristic = (1 / dummy_due_time).repeat(self.problem_size + 1, 1)
        else:
            self.heuristic = heuristic.to(device)

        self.pheromone = pheromone if pheromone is not None else torch.ones_like(self.heuristic)
        self.pheromone = self.pheromone.to(self.device)

        self.best_cost = 1e10
        self.best_sol = None
    
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
                sols, costs = self.local_search(sols, costs, inference=True)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.best_cost:
                self.best_sol = sols[:, best_idx]
                self.best_cost = best_cost

            self.update_pheromone(sols, costs)
        end = time.time()

        # Pairwise Jaccard similarity between paths
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
            sols: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        assert self.best_sol is not None

        sols = sols.clone()
        sol_gb = self.best_sol
        deltas = 1.0 / costs
        delta_gb = 1.0 / self.best_cost

        if self.shift_delta:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (n_ants_for_update * sols.size(0))

            deltas = (deltas + shifter).clamp(min=1e-10)
            delta_gb += shifter

        self.pheromone = self.pheromone * self.decay 

        if self.elitist:
            best_delta, best_idx = deltas.max(dim=0)
            best_sol = sols[:, best_idx]
            self.pheromone[best_sol[:-1], torch.roll(best_sol, shifts=-1)[:-1]] += best_delta

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
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += delta

        else:
            for i in range(self.n_ants):
                sol = sols[:, i]
                delta = deltas[i]
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += delta

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
    def gen_sol_costs(self, sols):
        sols = (sols - 1).T  # due to the dummy node 0, (n_ants, problem_size)
        ants_time = self.processing_time[sols]  # corresponding processing time (n_ants, problem_size)
        ants_presum_time = torch.stack(  # presum (total) time (n_ants, problem_size)
            [ants_time[:, :i].sum(dim=1) for i in range(1, self.problem_size + 1)]
        ).T
        ants_due_time = self.due_time[sols]  # (n_ants, problem_size)
        ants_weights = self.weights[sols]  # (n_ants, problem_size)
        diff = ants_presum_time - ants_due_time
        diff[diff < 0] = 0
        ants_weighted_tardiness = (ants_weights * diff).sum(dim=1)
        return ants_weighted_tardiness  # (n_ants,)
        
    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None, n_ants=None):
        '''
        Tour contruction for all ants
        Returns:
            sols: torch tensor with shape (problem_size, n_ants), sols[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        n_ants = n_ants or self.n_ants

        start = torch.zeros(size=(n_ants,), dtype=torch.long, device=self.device)
        
        visit_mask = torch.ones(size=(n_ants, self.problem_size + 1), device=self.device)
        visit_mask[:, 0] = 0  # exlude the dummy node (starting node) 0
                
        sols_list = []  # sols_list[i] is the ith action (tensor) for all ants
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions

        prev = start
        for i in range(self.problem_size):
            guiding_node = sols[i] if (sols is not None) and (i < sols.size(0)) else None
            actions, log_probs = self.pick_move(prev, visit_mask, require_prob, invtemp, guiding_node)
            sols_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            prev = actions
            visit_mask[torch.arange(n_ants), actions] = 0

        if require_prob:
            return torch.stack(sols_list), torch.stack(log_probs_list)
        else:
            return torch.stack(sols_list)

    def pick_move(self, prev, mask, require_prob, invtemp=1.0, guiding_node=None):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        if guiding_node is not None and not require_prob:
            return guiding_node, None

        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * mask  # shape: (n_ants, p_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(dist)
        actions = dist.sample() if guiding_node is None else guiding_node  # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_probs

    ### Destroy & Repair local search
    def get_symmetric_sols(self, sols):
        """
        Args:
            sols: (max_len, n_ants)
        Return:
            symm_sols: (max_len, n_ants, n_symmetry)
        """
        return sols.unsqueeze(2)  # No symmetry in SMTWTP solution

    @torch.no_grad()
    def local_search(self, sols: torch.Tensor, objs: torch.Tensor, inference=False):
        """
        Destroy & Repair local search, considering symmetry.
        Args:
            sols: (max_len, n_ants)
            objs: (n_ants,)
        """
        N_ANTS = sols.size(1)
        N_ROUNDS = 2
        INVTEMP_MIN = 1.0
        INVTEMP_MAX = 1.0
        DESTROY_RATE_MAX = 0.5
        DESTROY_RATE_MIN = 0.5
        N_SYMMETRY = 1  # maximum number of symmetric solutions to consider
        N_REPAIR = 50 if inference else 20  # number of repairs each round
        ROUND_BUDGET = N_SYMMETRY * N_REPAIR
        TOPK = ROUND_BUDGET // 10
        assert TOPK > 0

        topk_sols = sols.unsqueeze(2).expand(-1, -1, TOPK)
        # (max_seq_len, N_ANTS, TOPK)
        topk_objs = objs.unsqueeze(1).expand(-1, TOPK)
        # (N_ANTS, TOPK)
        for n in range(N_ROUNDS):
            new_sols = topk_sols.clone().view(-1, N_ANTS * TOPK)
            # (max_seq_len, N_ANTS * TOPK)
            new_objs = topk_objs.clone().view(-1)
            # (N_ANTS * TOPK)

            invtemp = INVTEMP_MIN + (INVTEMP_MAX - INVTEMP_MIN) * n / max(1, N_ROUNDS - 1)
            destroy_rate = DESTROY_RATE_MAX - (DESTROY_RATE_MAX - DESTROY_RATE_MIN) * n / max(1, N_ROUNDS - 1)
            protected_len = int(topk_sols.shape[0] * (1 - destroy_rate))
            _n_ants = new_sols.size(1)  # N_ANTS * TOPK
            _n_repair = max(1, ROUND_BUDGET // (N_SYMMETRY * TOPK))
            # _n_repair * N_SYMMETRY * _n_ants = ROUND_BUDGET * N_ANTS
            # => _n_repair * N_SYMMETRY * TOPK = ROUND_BUDGET

            symm_sols = self.get_symmetric_sols(new_sols)
            # (max_seq_len, N_ANTS * TOPK, N_SYMMETRY)

            # Slicing is the most simple way to destroy the solutions, but there could be more sophisticated ways.
            destroyed_sols = symm_sols[:protected_len].unsqueeze(3).expand(-1, -1, -1, _n_repair)
            # (protected_len, N_ANTS * TOPK, N_SYMMETRY, _n_repair)

            destroyed_sols = destroyed_sols.reshape(protected_len, _n_ants * N_SYMMETRY * _n_repair)
            new_sols = cast(
                torch.Tensor,
                self.gen_sol(
                    invtemp=invtemp, require_prob=False, sols=destroyed_sols, n_ants=_n_ants * N_SYMMETRY * _n_repair
                )
            )
            # (max_seq_len, N_ANTS * TOPK * N_SYMMETRY * _n_repair)
            new_objs = self.gen_sol_costs(new_sols)
            # (N_ANTS * TOPK * N_SYMMETRY * _n_repair)

            new_max_seq_len = new_sols.size(0)
            new_sols = new_sols.view(new_max_seq_len, N_ANTS, -1)
            # (max_seq_len, N_ANTS, TOPK * N_SYMMETRY * _n_repair)

            # zero padding if necessary
            diff_length = topk_sols.size(0) - new_max_seq_len
            if diff_length > 0:
                size = (diff_length, *new_sols.size()[1:])
                new_sols = torch.cat([new_sols, torch.zeros(size, device=self.device, dtype=torch.int64)], dim=0)
            elif diff_length < 0:
                size = (-diff_length, *topk_sols.size()[1:])
                topk_sols = torch.cat([topk_sols, torch.zeros(size, device=self.device, dtype=torch.int64)], dim=0)

            new_sols = torch.cat([topk_sols, new_sols], dim=2)
            # (max_seq_len, N_ANTS, TOPK * (N_SYMMETRY * _n_repair + 1))

            new_objs = new_objs.view(N_ANTS, -1)
            # (max_seq_len, N_ANTS, TOPK * N_SYMMETRY * _n_repair)
            new_objs = torch.cat([topk_objs, new_objs], dim=1)
            # (N_ANTS, TOPK * (N_SYMMETRY * _n_repair + 1))

            # Top-K selection each ants
            topk_indices = torch.argsort(new_objs, dim=1)[:, :TOPK]
            topk_sols = new_sols.gather(2, topk_indices.unsqueeze(0).expand(new_max_seq_len, -1, -1))
            # (max_seq_len, N_ANTS, TOPK)
            topk_objs = new_objs.gather(1, topk_indices)
            # (N_ANTS, TOPK)

        best_idx = topk_objs.argmin(dim=1)
        best_sols = topk_sols[:, torch.arange(N_ANTS), best_idx]
        best_objs = topk_objs[torch.arange(N_ANTS), best_idx]
        return best_sols, best_objs
