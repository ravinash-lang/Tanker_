"""Strong anytime solver for Algo Ranabhoomi Tanker Run.

The solver builds several feasible starting solutions and then applies:
- 2-opt inside routes
- best/first improving single-village relocation
- cross-route 1-1 swaps
- cross-route tail exchanges (2-opt*)
- occasional route re-seeding

Everything is integer-only and uses the competition's distance matrix.
"""

from time import perf_counter
from math import atan2

from adapter import Solver, route_load
from data import distance_matrix
from adapters.starter import StarterSolver

SAFETY = 0.18


def route_cost(d, r):
    if not r:
        return 0
    c = d[0][r[0]]
    for a, b in zip(r, r[1:]):
        c += d[a][b]
    return c + d[r[-1]][0]


def total_cost(d, routes):
    return sum(route_cost(d, r) for r in routes if r)


def clean(routes):
    return [r for r in routes if r]


def clone(routes):
    return [r[:] for r in routes]


def nearest_construct(instance, d, seed_mode=0):
    n = instance.size
    unvisited = set(range(1, n + 1))
    routes = []
    K = instance.fleet

    while unvisited:
        route = []
        load = 0
        pos = 0
        while True:
            fits = [v for v in unvisited if load + instance.demand[v] <= instance.capacity]
            if not fits:
                break
            if seed_mode == 0:
                v = min(fits, key=lambda x: (d[pos][x], x))
            elif seed_mode == 1:
                # Start/end-point diversity: prefer a far point when opening a route,
                # then return to nearest-neighbour behavior inside it.
                if not route:
                    v = max(fits, key=lambda x: (d[0][x], x))
                else:
                    v = min(fits, key=lambda x: (d[pos][x], x))
            elif seed_mode == 2:
                # Slight demand-aware tie breaking without sacrificing geometric greed.
                v = min(fits, key=lambda x: (d[pos][x], -instance.demand[x], x))
            else:
                v = min(fits, key=lambda x: (d[pos][x] - 2 * instance.demand[x], x))
            route.append(v)
            load += instance.demand[v]
            pos = v
            unvisited.remove(v)
        routes.append(route)
        if len(routes) > K:
            return None
    return routes


def sweep_construct(instance, d, variant=0):
    n = instance.size
    depx, depy = instance.coords[0]
    ids = list(range(1, n + 1))
    if variant == 0:
        ids.sort(key=lambda v: (atan2(instance.coords[v][1] - depy,
                                     instance.coords[v][0] - depx), d[0][v], v))
    elif variant == 1:
        ids.sort(key=lambda v: (atan2(instance.coords[v][1] - depy,
                                     instance.coords[v][0] - depx), -d[0][v], v))
    else:
        ids.sort(key=lambda v: (-atan2(instance.coords[v][1] - depy,
                                       instance.coords[v][0] - depx), d[0][v], v))

    routes = []
    loads = []
    for v in ids:
        # Best-fit into the route whose end is closest, otherwise open a route.
        candidates = [i for i in range(len(routes))
                      if loads[i] + instance.demand[v] <= instance.capacity]
        if candidates:
            i = min(candidates, key=lambda i: (d[routes[i][-1]][v],
                                                -loads[i], i))
            routes[i].append(v)
            loads[i] += instance.demand[v]
        else:
            routes.append([v])
            loads.append(instance.demand[v])
    return routes if len(routes) <= instance.fleet else None


def savings_construct(instance, d):
    # Clarke-Wright-style merging, starting with one route per village.
    routes = [[v] for v in range(1, instance.size + 1)]
    loads = [instance.demand[v] for v in range(1, instance.size + 1)]
    target = instance.fleet

    while len(routes) > target:
        best = None
        best_s = -10**18
        m = len(routes)
        for a in range(m):
            ra = routes[a]
            for b in range(a + 1, m):
                rb = routes[b]
                if loads[a] + loads[b] > instance.capacity:
                    continue
                # Four endpoint orientations; maximize the edge saved.
                for rev_a in (False, True):
                    ea = ra[0] if rev_a else ra[-1]
                    for rev_b in (False, True):
                        sb = rb[-1] if rev_b else rb[0]
                        saving = d[0][ea] + d[0][sb] - d[ea][sb]
                        if saving > best_s:
                            best_s = saving
                            best = (a, b, rev_a, rev_b)
        if best is None:
            return None
        a, b, rev_a, rev_b = best
        ra = routes[a][:]
        rb = routes[b][:]
        if rev_a:
            ra.reverse()
        if rev_b:
            rb.reverse()
        routes[a] = ra + rb
        loads[a] += loads[b]
        del routes[b]
        del loads[b]
    return routes


def two_opt(d, r, deadline):
    n = len(r)
    if n < 4:
        return False
    changed = False
    while perf_counter() < deadline:
        improved = False
        for i in range(n - 1):
            a = r[i - 1] if i else 0
            ai = r[i]
            for j in range(i + 1, n):
                b = r[j + 1] if j + 1 < n else 0
                if d[a][r[j]] + d[ai][b] < d[a][ai] + d[r[j]][b]:
                    r[i:j + 1] = reversed(r[i:j + 1])
                    improved = True
                    changed = True
                    break
            if improved or perf_counter() >= deadline:
                break
        if not improved:
            break
    return changed


def relocate_pass(instance, d, routes, loads, deadline):
    # Best improving relocation. This is the baseline's key move, but scans all
    # insertion positions and takes the strongest move available in the pass.
    best = None
    best_delta = 0
    R = len(routes)
    for a in range(R):
        ra = routes[a]
        if not ra:
            continue
        for i, v in enumerate(ra):
            if perf_counter() >= deadline:
                return False
            p = ra[i - 1] if i else 0
            q = ra[i + 1] if i + 1 < len(ra) else 0
            remove_delta = d[p][q] - d[p][v] - d[v][q]
            for b in range(R):
                if a == b or loads[b] + instance.demand[v] > instance.capacity:
                    continue
                rb = routes[b]
                for j in range(len(rb) + 1):
                    x = rb[j - 1] if j else 0
                    y = rb[j] if j < len(rb) else 0
                    add_delta = d[x][v] + d[v][y] - d[x][y]
                    delta = remove_delta + add_delta
                    if delta < best_delta:
                        best_delta = delta
                        best = (a, i, b, j, v)
    if best is None:
        return False
    a, i, b, j, v = best
    routes[a].pop(i)
    if a < b:
        j -= 0  # insertion route index unchanged; only route contents changed
    routes[b].insert(j, v)
    loads[a] -= instance.demand[v]
    loads[b] += instance.demand[v]
    return True


def swap_pass(instance, d, routes, loads, deadline):
    best = None
    best_delta = 0
    R = len(routes)
    for a in range(R):
        ra = routes[a]
        for b in range(a + 1, R):
            rb = routes[b]
            for i, va in enumerate(ra):
                if perf_counter() >= deadline:
                    return False
                for j, vb in enumerate(rb):
                    if loads[a] - instance.demand[va] + instance.demand[vb] > instance.capacity:
                        continue
                    if loads[b] - instance.demand[vb] + instance.demand[va] > instance.capacity:
                        continue
                    pa = ra[i - 1] if i else 0
                    na = ra[i + 1] if i + 1 < len(ra) else 0
                    pb = rb[j - 1] if j else 0
                    nb = rb[j + 1] if j + 1 < len(rb) else 0
                    old = d[pa][va] + d[va][na] + d[pb][vb] + d[vb][nb]
                    new = d[pa][vb] + d[vb][na] + d[pb][va] + d[va][nb]
                    delta = new - old
                    if delta < best_delta:
                        best_delta = delta
                        best = (a, i, b, j)
    if best is None:
        return False
    a, i, b, j = best
    va, vb = routes[a][i], routes[b][j]
    routes[a][i], routes[b][j] = vb, va
    loads[a] += instance.demand[vb] - instance.demand[va]
    loads[b] += instance.demand[va] - instance.demand[vb]
    return True


def tail_exchange_pass(instance, d, routes, loads, deadline):
    # 2-opt*: exchange suffixes of two routes. It is especially useful when a
    # good geometric route has the wrong capacity composition.
    best = None
    best_delta = 0
    R = len(routes)
    for a in range(R):
        ra = routes[a]
        if not ra:
            continue
        prefix_a = [0]
        for v in ra:
            prefix_a.append(prefix_a[-1] + instance.demand[v])
        for b in range(a + 1, R):
            rb = routes[b]
            if not rb:
                continue
            prefix_b = [0]
            for v in rb:
                prefix_b.append(prefix_b[-1] + instance.demand[v])
            for i in range(len(ra)):
                if perf_counter() >= deadline:
                    return False
                # Cut after ra[i]; suffix starts i+1.
                x = ra[i]
                xn = ra[i + 1] if i + 1 < len(ra) else 0
                pa = prefix_a[i + 1]
                for j in range(len(rb)):
                    y = rb[j]
                    yn = rb[j + 1] if j + 1 < len(rb) else 0
                    pb = prefix_b[j + 1]
                    new_a_load = pa + (prefix_b[-1] - pb)
                    new_b_load = pb + (prefix_a[-1] - pa)
                    if new_a_load > instance.capacity or new_b_load > instance.capacity:
                        continue
                    delta = d[x][yn] + d[y][xn] - d[x][xn] - d[y][yn]
                    if delta < best_delta:
                        best_delta = delta
                        best = (a, i, b, j)
    if best is None:
        return False
    a, i, b, j = best
    ra, rb = routes[a], routes[b]
    routes[a] = ra[:i + 1] + rb[j + 1:]
    routes[b] = rb[:j + 1] + ra[i + 1:]
    loads[a] = route_load(instance, routes[a])
    loads[b] = route_load(instance, routes[b])
    return True


def improve(instance, d, routes, deadline):
    routes[:] = clean(routes)
    loads = [route_load(instance, r) for r in routes]
    while perf_counter() < deadline:
        changed = False
        # Cheap intra-route improvement first.
        for r in routes:
            if perf_counter() >= deadline:
                break
            if two_opt(d, r, deadline):
                changed = True

        if perf_counter() >= deadline:
            break
        if relocate_pass(instance, d, routes, loads, deadline):
            routes[:] = clean(routes)
            loads[:] = [route_load(instance, r) for r in routes]
            changed = True
            continue

        if perf_counter() >= deadline:
            break
        if swap_pass(instance, d, routes, loads, deadline):
            changed = True
            continue

        if perf_counter() >= deadline:
            break
        if tail_exchange_pass(instance, d, routes, loads, deadline):
            changed = True
            continue

        if not changed:
            break
    routes[:] = clean(routes)
    return routes


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        start = perf_counter()
        # We intentionally leave a small margin for the framework to record the
        # final candidate and avoid overrunning the 6-second wall-clock budget.
        deadline = start + 6.0 - SAFETY
        d = distance_matrix(instance)

        candidates = []
        # Published starter is always a strong feasibility anchor.
        candidates.append(StarterSolver().solve(instance, submit_candidate)["routes"])

        # Diverse deterministic constructions.
        for mode in range(1, 4):
            if perf_counter() >= deadline:
                break
            r = nearest_construct(instance, d, mode)
            if r is not None:
                candidates.append(r)
        for mode in range(3):
            if perf_counter() >= deadline:
                break
            r = sweep_construct(instance, d, mode)
            if r is not None:
                candidates.append(r)
        if perf_counter() < deadline:
            r = savings_construct(instance, d)
            if r is not None:
                candidates.append(r)

        best_routes = None
        best_cost = 10**30

        # Submit every feasible starting solution and retain the best.
        for routes in candidates:
            if perf_counter() >= deadline:
                break
            routes = clean(clone(routes))
            if len(routes) > instance.fleet:
                continue
            c = total_cost(d, routes)
            if c < best_cost:
                best_cost = c
                best_routes = clone(routes)
                submit_candidate({"routes": clone(routes)})

        if best_routes is None:
            best_routes = clean(StarterSolver().solve(instance, submit_candidate)["routes"])
            best_cost = total_cost(d, best_routes)
            submit_candidate({"routes": clone(best_routes)})

        # Spend most of the remaining budget on local search from the best start.
        # Restart from other good constructions when they are competitive.
        seeds = [best_routes]
        for r in candidates:
            if perf_counter() >= deadline:
                break
            rr = clean(clone(r))
            if len(rr) <= instance.fleet:
                seeds.append(rr)

        idx = 0
        while perf_counter() < deadline and idx < len(seeds):
            base = clone(seeds[idx])
            idx += 1
            local_deadline = min(deadline, perf_counter() + 0.95)
            improve(instance, d, base, local_deadline)
            c = total_cost(d, base)
            if c < best_cost:
                best_cost = c
                best_routes = clone(base)
                submit_candidate({"routes": clone(best_routes)})

        # Intensify the best solution until the safety margin expires.
        while perf_counter() < deadline:
            base = clone(best_routes)
            local_deadline = min(deadline, perf_counter() + 0.55)
            before = total_cost(d, base)
            improve(instance, d, base, local_deadline)
            after = total_cost(d, base)
            if after < best_cost:
                best_cost = after
                best_routes = clone(base)
                submit_candidate({"routes": clone(best_routes)})
            elif after >= before:
                break

        return {"routes": clean(best_routes)}
