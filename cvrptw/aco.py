from functools import cached_property, partial
import itertools
import time
from typing import cast

import numpy as np
import torch
from torch.distributions import Categorical

from pyvrp_local_search import pyvrp_batched_local_search


CAPACITY = 1.0 # The input demands shall be normalized


def get_subroutes(route, end_with_zero = True):
    x = torch.nonzero(route == 0).flatten()
    subroutes = []
    for i, j in zip(x, x[1:]):
        if j - i > 1:
            if end_with_zero:
                j = j + 1
            subroutes.append(route[i: j])
    return subroutes


def merge_subroutes(subroutes, length, device):
    route = torch.zeros(length, dtype = torch.long, device=device)
    i = 0
    for r in subroutes:
        if len(r) > 2:
            if isinstance(r, list):
                r = torch.tensor(r[:-1])
            else:
                r = r[:-1].clone().detach()
            route[i: i + len(r)] = r
            i += len(r)
    return route


class ACO():
    def __init__(
        self,  # 0: depot
        distances: torch.Tensor, # (n, n)
        demands: torch.Tensor,   # (n, )
        windows: torch.Tensor,  # (n, 2)
        capacity=CAPACITY,
        service_time=0.0,
        positions: torch.Tensor | None = None,
        n_ants=20, 
        heuristic=None,
        pheromone=None,
        decay=0.9,
        alpha=1,
        beta=1,
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        smoothing=False,
        smoothing_thres=5,
        smoothing_delta=0.5,
        shift_delta=True,
        use_local_search = False,
        local_search_params = None,
        device='cpu',
    ):
        
        self.problem_size = len(distances)
        self.distances = distances
        self.capacity = capacity
        self.demands = demands
        self.windows = windows
        self.service_time = service_time
        self.positions = positions
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10
        self.smoothing = smoothing
        self.smoothing_cnt = 0
        self.smoothing_thres = smoothing_thres
        self.smoothing_delta = smoothing_delta
        self.shift_delta = shift_delta

        self.use_local_search = use_local_search
        self.local_search_params = local_search_params or {}
        self.device = device

        assert positions is not None if use_local_search else True
        
        self.heuristic = 1 / distances if heuristic is None else heuristic

        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
        else:
            self.pheromone = pheromone

        self.shortest_path = None
        self.lowest_cost = float('inf')

    def sample(self, invtemp=1.0):
        paths, log_probs = self.gen_path(require_prob=True, invtemp=invtemp)  # type: ignore
        costs = self.gen_path_costs(paths)
        return costs, log_probs, paths

    @cached_property
    @torch.no_grad()
    def heuristic_dist(self):
        heu = self.heuristic.detach().cpu().numpy()  # type: ignore
        return (1 / (heu/heu.max(-1, keepdims=True) + 1e-2))

    @torch.no_grad()
    def run(self, n_iterations):
        start = time.time()
        for _ in range(n_iterations):
            paths = cast(torch.Tensor, self.gen_path(require_prob=False))
            _paths = paths.clone()

            if self.use_local_search:
                paths = self.local_search(paths, inference=True)
            costs = self.gen_path_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx].clone()
                self.lowest_cost = best_cost.item()

            self.update_pheromone(paths, costs)
        end = time.time()

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _paths = _paths.T.cpu().numpy()
        for _p in _paths:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in itertools.combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.lowest_cost, diversity, end - start

    @torch.no_grad()
    def update_pheromone(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        paths = paths[:-1, :]  # 0-start, 0-end
        path_gb = self.shortest_path.clone()[:-1]  # type: ignore
        deltas = 1.0 / costs
        delta_gb = 1.0 / self.lowest_cost
        if self.shift_delta:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * paths.size(0))

            deltas = (deltas + shifter).clamp(min=1e-10)
            delta_gb += shifter

        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_delta, best_idx = deltas.max(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += best_delta
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += best_delta

        elif self.rank_based:
            # Rank-based pheromone update
            elite_indices = torch.argsort(deltas, descending=True)[:self.n_elites]
            elite_paths = paths[:, elite_indices]
            elite_deltas = deltas[elite_indices]
            if self.lowest_cost < costs.min():
                # Zero-padding to the shorter path
                diff_length = elite_paths.size(0) - path_gb.size(0)
                if diff_length > 0:
                    path_gb = torch.nn.functional.pad(path_gb, (0, diff_length))
                elif diff_length < 0:
                    elite_paths = torch.nn.functional.pad(elite_paths, (0, 0, 0, -diff_length))

                elite_paths = torch.cat([path_gb.unsqueeze(1), elite_paths[:, :-1]], dim=1)
                elite_deltas = torch.cat([torch.tensor([delta_gb], device=self.device), elite_deltas[:-1]])

            rank_denom = (self.n_elites * (self.n_elites + 1)) / 2
            for i in range(self.n_elites):
                path = elite_paths[:, i]
                delta = elite_deltas[i] * (self.n_elites - i) / rank_denom
                self.pheromone[path, torch.roll(path, shifts=1)] += delta
                self.pheromone[torch.roll(path, shifts=1), path] += delta

        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                delta = deltas[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += delta
                self.pheromone[torch.roll(path, shifts=1), path] += delta

        if self.maxmin:
            _max = delta_gb * (1 - self.decay)
            p_dec = 0.05 ** (1 / self.problem_size)
            _min = _max * (1 - p_dec) / (0.5 * self.problem_size - 1) / p_dec
            self.pheromone = torch.clamp(self.pheromone, min=_min, max=_max)
            # check convergence
            if (self.pheromone[path_gb, torch.roll(path_gb, shifts=1)] >= _max * 0.99).all():
                self.pheromone = 0.5 * self.pheromone + 0.5 * _max

        else:  # maxmin has its own smoothing
            # smoothing the pheromone if the lowest cost has not been updated for a while
            if self.smoothing:
                self.smoothing_cnt = max(0, self.smoothing_cnt + (1 if self.lowest_cost < costs.min() else -1))
                if self.smoothing_cnt >= self.smoothing_thres:
                    self.pheromone = self.smoothing_delta * self.pheromone + (self.smoothing_delta) * torch.ones_like(self.pheromone)
                    self.smoothing_cnt = 0

        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_numpy_path_costs(self, paths):
        '''
        Args:
            paths: numpy ndarray with shape (n_ants, problem_size), note the shape
        Returns:
            Lengths of paths: numpy ndarray with shape (n_ants,)
        '''
        u = paths
        v = np.roll(u, shift=1, axis=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return np.sum(self.distances_cpu[u, v], axis=1)

    def gen_path(self, require_prob=False, invtemp=1.0, paths=None):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)

        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        time_window_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        cur_time = torch.zeros(size=(self.n_ants,), device=self.device)

        prob_mat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        prev = actions

        paths_list = [actions]  # paths_list[i] is the ith move (tensor) for all ants
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        done = self.check_done(visit_mask, actions)

        ##################################################
        # given paths
        i = 0
        feasible_idx = torch.arange(self.n_ants, device=self.device) if paths is not None else None
        ##################################################
        while not done:
            selected = paths[i + 1] if paths is not None else None
            actions, log_probs = self.pick_move(prob_mat[prev], visit_mask, capacity_mask, time_window_mask, require_prob, invtemp, selected)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            cur_time, time_window_mask = self.update_time_window_mask(actions, cur_time, prev)

            ##################################################
            # NLS may generate infeasible solutions
            if paths is not None:
                infeasible_idx = (torch.where(capacity_mask.sum(-1) == 0)[0]) | (torch.where(time_window_mask.sum(-1) == 0)[0])

                # remove infeasible ants
                if len(infeasible_idx) > 0:
                    is_feasible = capacity_mask.sum(-1) > 0 and time_window_mask.sum(-1) > 0
                    feasible_idx = feasible_idx[is_feasible]  # type: ignore

                    actions = actions[is_feasible]
                    visit_mask = visit_mask[is_feasible]
                    used_capacity = used_capacity[is_feasible]
                    cur_time = cur_time[is_feasible]
                    capacity_mask = capacity_mask[is_feasible]
                    time_window_mask = time_window_mask[is_feasible]

                    paths_list = [p[is_feasible] for p in paths_list]
                    if require_prob:
                        log_probs_list = [l_p[is_feasible] for l_p in log_probs_list]
                    if paths is not None:
                        paths = paths[:, is_feasible]

                    self.n_ants -= len(infeasible_idx)
            ##################################################

            done = self.check_done(visit_mask, actions)
            prev = actions
            i += 1

        if require_prob:
            if paths is not None:
                return torch.stack(paths_list), torch.stack(log_probs_list), feasible_idx  # type: ignore
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)

    def pick_move(self, dist, visit_mask, capacity_mask, time_window_mask, require_prob, invtemp=1.0, guiding_node=None):
        if guiding_node is not None and not require_prob:
            return guiding_node, None

        dist = (dist ** invtemp) * visit_mask * capacity_mask * time_window_mask  # shape: (n_ants, p_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(dist)
        actions = dist.sample() if guiding_node is None else guiding_node  # shape: (n_ants,)
        log_prob = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_prob

    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demands[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demands.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat + 1e-10] = 0

        return used_capacity, capacity_mask
    
    def update_time_window_mask(self, cur_nodes, cur_time, prev_nodes):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            cur_time: shape (n_ants, )
            time_window_mask: shape (n_ants, p_size)
        Returns:
            ant_cur_time: updated time
            time_window_mask: updated mask
        '''
        time_window_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)

        # update cur_time
        cur_time = cur_time + self.distances[prev_nodes, cur_nodes]
        cur_time = torch.max(cur_time, self.windows[cur_nodes, 0])
        cur_time = cur_time + self.service_time

        # depot -> cur_time = 0
        cur_time[cur_nodes == 0] = 0

        # update time_window_mask
        cur_time_repeat = cur_time.unsqueeze(-1).repeat(1, self.problem_size)
        window_repeat = self.windows.unsqueeze(0).repeat(self.n_ants, 1, 1)
        arrive_time = cur_time_repeat + self.distances[cur_nodes, :]

        # Arrive within the time window
        time_window_mask[(arrive_time > window_repeat[:, :, 1] + 1e-10)] = 0

        arrive_time_in_depot = torch.max(arrive_time, window_repeat[:, :, 0])
        arrive_time_in_depot += self.service_time + self.distances[:, 0].unsqueeze(0).repeat(self.n_ants, 1)
        arrive_time_in_depot[:, 0] = 0

        # Come to depot before the end of the time window
        time_window_mask[(arrive_time_in_depot > window_repeat[:, [0], 1].repeat(1, self.problem_size) + 1e-10)] = 0

        return cur_time, time_window_mask

    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()

    @cached_property
    @torch.no_grad()
    def positions_cpu(self):
        return self.positions.cpu().numpy() if self.positions is not None else None

    @cached_property
    @torch.no_grad()
    def distances_cpu(self):
        return self.distances.cpu().numpy()

    @cached_property
    @torch.no_grad()
    def demands_cpu(self):
        return self.demands.cpu().numpy()

    @cached_property
    @torch.no_grad()
    def windows_cpu(self):
        return self.windows.cpu().numpy()

    @torch.no_grad()
    def local_search(self, paths, inference=False, T_nls=1) -> torch.Tensor:
        paths_np = paths.T.cpu().numpy()
        partial_func = partial(
            pyvrp_batched_local_search,
            positions=self.positions_cpu,  # type: ignore
            demands=self.demands_cpu,
            windows=self.windows_cpu,
            neighbourhood_params=self.local_search_params.get("neighbourhood_params"),
            max_trials=self.local_search_params.get("max_trials", 10),
            inference=inference,
            seed=self.local_search_params.get("seed", 0),
            n_cpus=min(self.n_ants, self.local_search_params.get("n_cpus", 1)),
        )

        ce_params = self.local_search_params.get("cost_evaluator_params")
        heu_ce_params = None
        if ce_params is not None:
            heu_ce_params = ce_params.copy()
            if "load_penalty" in heu_ce_params:
                heu_ce_params["load_penalty"] /= 10
            if "tw_penalty" in heu_ce_params:
                heu_ce_params["tw_penalty"] /= 10

        best_paths = partial_func(
            paths=paths_np, distances=self.distances_cpu, cost_evaluator_params=ce_params, allow_infeasible=False
        )
        best_costs = self.gen_numpy_path_costs(best_paths)
        new_paths = best_paths

        path_len = new_paths.shape[1]
        for _ in range(T_nls):
            perturbed_paths = partial_func(
                paths=new_paths,
                distances=self.heuristic_dist,
                cost_evaluator_params=heu_ce_params,
                allow_infeasible=True,
            )
            new_paths = partial_func(
                paths=perturbed_paths,
                distances=self.distances_cpu,
                cost_evaluator_params=ce_params,
                allow_infeasible=False,
            )
            new_costs = self.gen_numpy_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            improved_paths = new_paths[improved_indices]
            new_path_len = new_paths.shape[1]
            if path_len > new_path_len:
                improved_paths = np.pad(improved_paths, ((0, 0), (0, path_len - new_path_len)), constant_values=0)
            elif new_path_len > path_len:
                best_paths = np.pad(best_paths, ((0, 0), (0, new_path_len - path_len)), constant_values=0)
                path_len = new_path_len

            best_paths[improved_indices] = improved_paths
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.tensor(best_paths.T.astype(np.int64), device=self.device)
        return best_paths
