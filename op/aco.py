from itertools import combinations
import time
from typing import cast

import torch
from torch.distributions import Categorical


class ACO():
    def __init__(
        self,
        distances,
        prizes,
        max_len,
        n_ants=20, 
        heuristic=None,
        k_sparse=None,
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
        self.problem_size = len(prizes)
        self.distances = distances
        self.prizes = prizes
        self.Q = 1 / prizes.sum()  # sum of prizes as a normalization factor
        self.max_len = max_len

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10
        self.shift_delta = shift_delta

        self.use_local_search = use_local_search
        self.device = device

        if heuristic is None:
            assert k_sparse is not None
            self.heuristic = self.simple_heuristic(distances, k_sparse)
        else:
            self.heuristic = heuristic.to(device)
        self.add_dummy_node()  # add dummy node to the graph

        self.pheromone = pheromone if pheromone is not None else torch.ones_like(self.distances)
        self.pheromone = self.pheromone.to(self.device)

        self.best_obj = 0
        self.best_sol = None

    def add_dummy_node(self):
        '''
        One has to sparsify the graph first before adding dummy node
        distance: 
                [[1e9 , x   , x   , 0  ],
                [x   , 1e9 , x   , 0  ],
                [x   , x   , 1e9 , 0  ],
                [1e10, 1e10, 1e10, 0  ]]
        pheromone: [1]
        heuristic: [>0]
        prizes: [x,x,...,0]
        '''
        self.prizes = torch.cat((self.prizes, torch.tensor([1e-10], device=self.device)))
        self.distances = torch.cat((self.distances, 1e10 * torch.ones(size=(1, self.problem_size), device=self.device)), dim=0)  # cannot reach other nodes from dummy node
        self.distances = torch.cat((self.distances, 1e-10 + torch.zeros(size=(self.problem_size + 1, 1), device=self.device)), dim=1)

        self.heuristic = torch.cat((self.heuristic, torch.zeros(size=(1, self.problem_size), device=self.device)), dim=0)  # cannot reach other nodes from dummy node
        self.heuristic = torch.cat((self.heuristic, torch.ones(size=(self.problem_size + 1, 1), device=self.device)), dim=1)

        self.distances[self.distances == 1e-10] = 0
        self.prizes[-1] = 0

    @torch.no_grad()
    def simple_heuristic(self, distances, k_sparse):
        '''
        Sparsify the OP graph to obtain the heuristic information 
        used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(distances), device=self.device), repeats=k_sparse
        )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = distances[edge_index_u, edge_index_v]
        heuristic = self.prizes.unsqueeze(0) / sparse_distances
        return heuristic

    def sample(self, invtemp=1.0):
        sols, log_probs = self.gen_sol(invtemp=invtemp, require_prob=True)
        objs = self.gen_sol_objs(sols)
        return objs, log_probs, sols

    @torch.no_grad()
    def run(self, n_iterations):
        start = time.time()
        for _ in range(n_iterations):
            sols = cast(torch.Tensor, self.gen_sol(require_prob=False))
            objs = self.gen_sol_objs(sols)
            _sols = sols.clone()

            if self.use_local_search:
                sols, objs = self.local_search(sols, objs, inference=True)

            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.best_obj:
                self.best_obj = best_obj
                self.best_sol = sols[:, best_idx]

            self.update_pheromone(sols, objs)
        end = time.time()

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _sols = _sols.T.cpu().numpy()  # type: ignore
        for _p in _sols:
            _p = _p[_p != self.problem_size]
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.best_obj, diversity, end - start

    @torch.no_grad()
    def update_pheromone(self, sols: torch.Tensor, objs: torch.Tensor):
        # sols.shape: (max_len, n_ants)
        assert self.best_sol is not None

        sol_gb = self.best_sol
        deltas = self.Q * objs
        delta_gb = self.Q * self.best_obj
        if self.shift_delta:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * sols.size(0))

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
            if self.best_obj > objs.max():
                diff_length = elite_sols.size(0) - sol_gb.size(0)
                if diff_length > 0:
                    sol_gb = torch.cat([sol_gb, torch.zeros(diff_length, device=self.device)])
                elif diff_length < 0:
                    elite_sols = torch.cat([elite_sols, torch.zeros((-diff_length, self.n_elites), device=self.device)], dim=0)

                elite_sols = torch.cat([sol_gb.unsqueeze(1), elite_sols[:, :-1]], dim=1)  # type: ignore
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
            _max = delta_gb / (1 - self.decay)
            p_dec = 0.05 ** (1 / self.problem_size)
            _min = _max * (1 - p_dec) / (0.5 * self.problem_size - 1) / p_dec
            self.pheromone = torch.clamp(self.pheromone, min=_min, max=_max)
            # check convergence
            if (self.pheromone[sol_gb, torch.roll(sol_gb, shifts=1)] >= _max * 0.99).all():
                self.pheromone = 0.5 * self.pheromone + 0.5 * _max

        self.pheromone[self.pheromone < 1e-10] = 1e-10

    @torch.no_grad()
    def gen_sol_objs(self, solutions):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        objs = self.prizes[solutions.T].sum(dim=1)
        return objs

    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None, n_ants=None):
        '''
        Solution contruction for all ants
        '''
        n_ants = n_ants or self.n_ants
        solutions = []
        log_probs_list = []

        solutions = [torch.zeros(size=(n_ants,), device=self.device, dtype=torch.int64)]
        mask = torch.ones(size=(n_ants, self.problem_size + 1), device=self.device)
        done = torch.zeros(size=(n_ants,), device=self.device)
        travel_dis = torch.zeros(size=(n_ants,), device=self.device)
        cur_node = torch.zeros(size=(n_ants,), dtype=torch.int64, device=self.device)
        
        mask = self.update_mask(travel_dis, cur_node, mask, n_ants)
        done = self.check_done(mask)
        # construction
        i = 0
        while not done:
            guiding_node = sols[i + 1] if (sols is not None) and (i + 1 < sols.size(0)) else None
            next_node, log_prob = self.pick_node(mask, cur_node, require_prob, invtemp=invtemp, guiding_node=guiding_node)

            # update solution and log_probs
            solutions.append(next_node) 
            log_probs_list.append(log_prob)
            # update travel_dis, cur_node and mask
            travel_dis += self.distances[cur_node, next_node]
            cur_node = next_node
            if require_prob:
                mask = mask.clone()
            mask = self.update_mask(travel_dis, cur_node, mask, n_ants)
            # check done
            done = self.check_done(mask)

            i += 1

        if require_prob:
            return torch.stack(solutions), torch.stack(log_probs_list)  # (max_len, n_ants)
        else:
            return torch.stack(solutions)

    def pick_node(self, mask, cur_node, require_prob, invtemp=1.0, guiding_node=None):
        if guiding_node is not None and not require_prob:
            return guiding_node, None

        pheromone = self.pheromone[cur_node]  # shape: (n_ants, p_size + 1)
        heuristic = self.heuristic[cur_node]  # shape: (n_ants, p_size + 1)
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * mask
        # set the prob of dummy node to 1 if dist is all 0
        dist[(dist==0).all(dim=1), -1] = 1
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(dist)
        item = dist.sample() if guiding_node is None else guiding_node
        log_prob = dist.log_prob(item) if require_prob else None
        return item, log_prob  # (n_ants,)

    def update_mask(self, travel_dis, cur_node, mask, n_ants):
        '''
        Args:
            travel_dis: (n_ants,)
            cur_node: (n_ants,)
            mask: (n_ants, n + 1)
        '''
        mask[torch.arange(n_ants), cur_node] = 0

        dist_mask = (travel_dis.unsqueeze(1) + self.distances[cur_node] + self.distances[:, 0].unsqueeze(0)) <= self.max_len
        mask = mask * dist_mask

        mask[:, -1] = 0 # mask the dummy node for all ants
        go2dummy = (mask[:, :-1] == 0).all(dim=1) # unmask the dummy node for these ants
        mask[go2dummy, -1] = 1
        return mask

    def check_done(self, mask):
        # is all masked ?
        return (mask[:, :-1] == 0).all()

    ### Destroy & Repair local search
    def get_symmetric_sols(self, sols):
        """
        Args:
            sols: (max_len, n_ants)
        Return:
            symm_sols: (max_len, n_ants, n_symmetry)
        """
        # convert tail self.n to 0
        flipped_sols = sols.clone()
        flipped_sols[flipped_sols == self.problem_size] = 0

        n_tail_zeros = flipped_sols.size(0) - 1 - flipped_sols.count_nonzero(0)
        flipped_sols = flipped_sols.flip((0,))  # naive flipping cause the leading 0, so we need to roll it.
        for i in range(flipped_sols.size(1)):
            flipped_sols[:, i] = torch.roll(flipped_sols[:, i], shifts=1 - n_tail_zeros[i].item(), dims=0)

        flipped_sols[1:][flipped_sols[1:] == 0] = self.problem_size
        return torch.stack([sols, flipped_sols], dim=2)

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
        N_SYMMETRY = 2  # maximum number of symmetric solutions to consider
        N_REPAIR = 25 if inference else 10  # number of repairs each round
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
            new_objs = self.gen_sol_objs(new_sols)
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
            topk_indices = torch.argsort(new_objs, dim=1, descending=True)[:, :TOPK]
            topk_sols = new_sols.gather(2, topk_indices.unsqueeze(0).expand(new_max_seq_len, -1, -1))
            # (max_seq_len, N_ANTS, TOPK)
            topk_objs = new_objs.gather(1, topk_indices)
            # (N_ANTS, TOPK)

        best_idx = topk_objs.argmax(dim=1)
        best_sols = topk_sols[:, torch.arange(N_ANTS), best_idx]
        best_objs = topk_objs[torch.arange(N_ANTS), best_idx]
        return best_sols, best_objs
