/* Grainbridge engine: a line-by-line JavaScript port of citysim Stage 1
 * (citysim/layers/s1_real_economy/{rules,real_economy,addons}.py).
 * tests/test_twin_parity.py runs this file under node and checks it against
 * the Python model, so the town on screen follows the same rules as the report.
 * Works in the browser (window.CityEngine) and in node (module.exports).
 */
(function (root) {
  "use strict";
  const EPS = 1e-12;
  const SECTORS = ["farming", "processing", "distribution", "maintenance"];
  const CAPITAL_TYPES = ["factory", "land_improvement", "tools_processing", "tools_distribution"];
  const RELIEVES = { land_improvement: "farming", tools_processing: "processing", tools_distribution: "distribution" };

  // Python >= 3.12 sum() over floats uses Neumaier compensated summation; match it exactly.
  function psum(xs) {
    let s = 0.0, c = 0.0;
    for (const x of xs) {
      const t = s + x;
      if (Math.abs(s) >= Math.abs(x)) c += (s - t) + x; else c += (x - t) + s;
      s = t;
    }
    return (c && isFinite(c)) ? s + c : s;
  }
  const vals = (o) => SECTORS.map((k) => o[k]);

  // ---------------------------------------------------------------- technology
  function makeTech(o) {
    const t = Object.assign({}, o);
    t.a_x = t.a * Math.pow(1.0 + t.land_improvement, t.land_elasticity);
    t.v_p_eff = t.v_p * Math.pow(1.0 + t.tools_processing, -t.tool_elasticity);
    t.v_d_eff = t.v_d * Math.pow(1.0 + t.tools_distribution, -t.tool_elasticity);
    t.total_capacity = t.capacity * psum(t.conditions);
    const sorted = t.conditions.slice().sort((x, y) => y - x);
    let acc = 0.0;
    t.cum = sorted.map((x) => { acc += x; return t.capacity * acc; });
    t.sorted = sorted;
    return t;
  }
  function withCapital(t, kind, delta) {
    const o = Object.assign({}, t);
    if (kind === "factory") o.conditions = t.conditions.concat([delta]);
    else o[kind] = t[kind] + delta;
    return makeTech(o);
  }

  // ---------------------------------------------------------------- farming
  function farmOutput(farmers, a_x, limit, beta) {
    if (farmers <= 0) return 0.0;
    if (limit === null || farmers <= limit) return a_x * farmers;
    return a_x * Math.pow(limit, beta) * Math.pow(farmers, 1.0 - beta);
  }
  function farmersRequired(raw, a_x, limit, beta) {
    if (raw <= 0) return 0.0;
    if (a_x <= 0) return Infinity;
    if (limit === null || raw <= a_x * limit) return raw / a_x;
    if (beta >= 1.0) return Infinity;
    return limit * Math.pow(raw / (a_x * limit), 1.0 / (1.0 - beta));
  }

  // ---------------------------------------------------------------- factories
  function bisectLeft(arr, x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function operatingFactories(output, t) {
    if (output <= EPS || t.conditions.length === 0) return [0, 0.0];
    const k = bisectLeft(t.cum, output * (1 - 1e-12));
    if (k >= t.cum.length) return [t.cum.length, t.cum[t.cum.length - 1]];
    return [k + 1, t.cum[k]];
  }
  function capacityOfBest(k, t) { return t.capacity * psum(t.sorted.slice(0, k)); }

  // ---------------------------------------------------------------- labor
  function maintenanceRequired(t) {
    return (t.F_m * t.conditions.length
      + t.land_maintenance * t.land_improvement
      + t.tool_maintenance * (t.tools_processing + t.tools_distribution));
  }
  function productionLabor(output, t) {
    const farmers = farmersRequired(output / t.e, t.a_x, t.land_limit, t.land_exponent);
    const k = operatingFactories(output, t)[0];
    return [farmers, t.F_p * k + t.v_p_eff * output];
  }
  function laborRequirements(output, t) {
    const [f, p] = productionLabor(output, t);
    return { farming: f, processing: p, distribution: t.v_d_eff * output, maintenance: maintenanceRequired(t) };
  }
  const totalLabor = (output, t) => psum(vals(laborRequirements(output, t)));

  function allocatePriority(req, workers, order) {
    let left = workers; const alloc = {};
    for (const s of order) { const give = Math.min(req[s], Math.max(left, 0.0)); alloc[s] = give; left -= give; }
    return alloc;
  }
  function potentialOutput(workers, t, iters = 60) {
    const cap = t.total_capacity;
    if (totalLabor(0.0, t) > workers || cap <= 0) return 0.0;
    if (totalLabor(cap, t) <= workers) return cap;
    let lo = 0.0, hi = cap;
    for (let i = 0; i < iters; i++) { const mid = 0.5 * (lo + hi); if (totalLabor(mid, t) <= workers) lo = mid; else hi = mid; }
    return lo;
  }
  function maxExtra(base, limit, budget, t, iters = 60) {
    if (limit <= EPS || budget <= EPS) return [0.0, 0.0];
    const [f0, p0] = productionLabor(base, t);
    const extra = (x) => { const [f, p] = productionLabor(base + x, t); return (f - f0) + (p - p0); };
    if (extra(limit) <= budget) return [limit, extra(limit)];
    let lo = 0.0, hi = limit;
    for (let i = 0; i < iters; i++) { const mid = 0.5 * (lo + hi); if (extra(mid) <= budget) lo = mid; else hi = mid; }
    return [lo, extra(lo)];
  }
  function makePlan(workers, t, order, rule, restockDemand) {
    const need = workers * t.c;
    let core = Math.min(need, t.total_capacity);
    if (rule === "max_feasible") core = Math.min(core, potentialOutput(workers, t));
    let required = laborRequirements(core, t);
    const totalReq = psum(vals(required));
    if (totalReq > workers * (1 + 1e-12)) {
      const alloc = allocatePriority(required, workers, order);
      return { workers, need, core, restock: 0.0, required, alloc, surplus: 0.0,
               shortfall: Math.max(totalReq - workers, 0.0), k: operatingFactories(core, t)[0], restockLabor: 0.0 };
    }
    let alloc = Object.assign({}, required);
    let surplus = workers - totalReq;
    const [restock, rl] = maxExtra(core, Math.min(restockDemand, t.total_capacity - core), surplus, t);
    if (restock > 0) {
      const [f, p] = productionLabor(core + restock, t);
      alloc.farming = f; alloc.processing = p;
      required = Object.assign({}, alloc);
      surplus = workers - psum(vals(alloc));
    }
    return { workers, need, core, restock, required, alloc, surplus: Math.max(surplus, 0.0), shortfall: 0.0,
             k: operatingFactories(core + restock, t)[0], restockLabor: rl };
  }

  // ---------------------------------------------------------------- investment
  function factoriesWanted(need, totalCap, capacity, threshold) {
    const gap = need / threshold - totalCap;
    return gap <= 0 ? 0.0 : Math.floor(gap / capacity) + 1;
  }
  function netSaving(ctx, kind) {
    const units = ctx.labor / ctx.costs[kind];
    return totalLabor(ctx.output, ctx.tech) - totalLabor(ctx.output, withCapital(ctx.tech, kind, units));
  }
  function utilization(ctx) { const cap = ctx.tech.total_capacity; return cap <= 0 ? Infinity : ctx.need / cap; }
  const INVESTMENT_RULES = {
    bottleneck(ctx) {
      if (ctx.inProgress || utilization(ctx) >= ctx.threshold) return "factory";
      const ranked = Object.keys(RELIEVES).slice().sort((x, y) => -(ctx.sectorLabor[RELIEVES[x]] || 0) + (ctx.sectorLabor[RELIEVES[y]] || 0));
      for (const kind of ranked) if (netSaving(ctx, kind) > EPS) return kind;
      return null;
    },
    best_return(ctx) {
      if (ctx.inProgress || utilization(ctx) >= ctx.threshold) return "factory";
      let best = null, bv = -Infinity;
      for (const k of Object.keys(RELIEVES)) { const v = netSaving(ctx, k); if (v > bv) { bv = v; best = k; } }
      return bv > EPS ? best : null;
    },
  };

  function nextPopulation(n, cons, c, growth, mu) {
    if (cons >= c * (1 - 1e-9)) return n * (1.0 + growth);
    return n * Math.max(0.0, 1.0 - mu * (1.0 - cons / c));
  }
  function unmaintainedShare(labor, req) {
    if (req <= EPS) return 0.0;
    return Math.min(1.0, Math.max(0.0, 1.0 - labor / req));
  }
  function analyticalNmin(p, a) {
    const d = 1.0 - p.c / (p.e * a) - p.v_p * p.c - p.v_d * p.c;
    return d <= 0 ? Infinity : (p.F_p + p.F_m) / d;
  }

  // ---------------------------------------------------------------- defaults
  const DEFAULTS = {
    a: 5, c: 1, e: 1, F_p: 50, v_p: 0.1, v_d: 0.1, F_m: 30, delta: 0.05, capacity: 1000,
    N0: 150, initial_factories: 1, scrap_condition: 0.01,
    land_elasticity: 0.25, tool_elasticity: 0.25, land_maintenance: 1.0, tool_maintenance: 1.0,
    priority: SECTORS.slice(), shortage_rule: "priority",
    investment: { enabled: true, s: 0.3, rule: "bottleneck", threshold: 0.9, factory_cost: 1000, land_cost: 200, tool_cost: 200 },
    demography: { enabled: false, n: 0.01, mu: 0.1 },
    storage: { enabled: false, spoilage: 0.1, target_ticks: 1.0, initial_stock: 0.0 },
    land_limits: { enabled: false, limit: 500, beta: 0.5 },
    shocks: [],
  };
  function merge(base, over) {
    const out = JSON.parse(JSON.stringify(base));
    for (const k of Object.keys(over || {})) {
      if (over[k] && typeof over[k] === "object" && !Array.isArray(over[k]) && typeof out[k] === "object") out[k] = Object.assign(out[k], over[k]);
      else out[k] = over[k];
    }
    return out;
  }

  // ---------------------------------------------------------------- simulation
  class Town {
    constructor(params) {
      this.p = merge(DEFAULTS, params);
      this.tick = 0;
      this.N = this.p.N0;
      this.conditions = Array(this.p.initial_factories).fill(1.0);
      // stocks (the books): holder -> item -> amount
      this.stock = { farm: {}, factory: {}, distribution: {}, households: {}, warehouse: {} };
      this.stock.factory.factory = this.p.initial_factories;
      if (this.p.storage.enabled && this.p.storage.initial_stock) this.stock.warehouse.final = this.p.storage.initial_stock;
      this.prevPotential = null;
      this.booksOk = true;
    }
    get(h, i) { return this.stock[h][i] || 0.0; }
    // flows: "production" adds, outflow flows remove, "transfer" moves. Every change goes through here.
    post(flow, item, from, to, amt) {
      if (amt === 0) return;
      if (amt < 0) throw new Error("negative posting " + flow + " " + item);
      if (from) this.stock[from][item] = this.get(from, item) - amt;
      if (to) this.stock[to][item] = this.get(to, item) + amt;
      const f = this.flows[item] || (this.flows[item] = {});
      f[flow] = (f[flow] || 0) + amt;
    }
    totals() {
      const t = {};
      for (const h of Object.keys(this.stock)) for (const [i, v] of Object.entries(this.stock[h])) t[i] = (t[i] || 0) + v;
      return t;
    }
    tech() {
      const p = this.p;
      return makeTech({
        c: p.c, e: p.e, a: p.a, v_p: p.v_p, v_d: p.v_d, F_p: p.F_p, F_m: p.F_m, capacity: p.capacity,
        land_improvement: this.get("farm", "land_improvement"),
        tools_processing: this.get("factory", "tools_processing"),
        tools_distribution: this.get("distribution", "tools_distribution"),
        land_elasticity: p.land_elasticity, tool_elasticity: p.tool_elasticity,
        land_maintenance: p.land_maintenance, tool_maintenance: p.tool_maintenance,
        conditions: this.conditions.slice(),
        land_limit: p.land_limits.enabled ? p.land_limits.limit : null,
        land_exponent: p.land_limits.enabled ? p.land_limits.beta : 0.5,
      });
    }
    storageTarget() { return this.p.storage.target_ticks * this.N * this.p.c; }

    step() {
      const p = this.p;
      this.tick += 1;
      this.flows = {};
      const opening = this.totals();
      const s = { tick: this.tick, N0: this.N, events: [] };

      // Phase 1: plan, then weather
      const tech = this.tech();
      const restock = p.storage.enabled
        ? Math.max(0.0, this.storageTarget() - this.get("warehouse", "final") * (1 - p.storage.spoilage)) : 0.0;
      const plan = makePlan(this.N, tech, p.priority, p.shortage_rule, restock);
      if (this.prevPotential === null) this.prevPotential = potentialOutput(plan.workers, tech);
      let mult = 1.0;
      for (const sh of p.shocks) if (sh.start <= this.tick && this.tick < sh.start + sh.duration) mult *= sh.multiplier;

      // Phase 2: farming, then processing
      const farmers = plan.alloc.farming;
      const raw = farmOutput(farmers, tech.a_x, tech.land_limit, tech.land_exponent) * mult;
      this.post("production", "raw", null, "farm", raw);
      this.post("transfer", "raw", "farm", "factory", raw);
      const labor = plan.alloc.processing;
      let k = plan.k;
      if (tech.F_p > 0) k = Math.min(k, Math.floor(labor / tech.F_p + 1e-9));
      let final = 0.0;
      if (k > 0) {
        const variable = Math.max(labor - tech.F_p * k, 0.0);
        const byLabor = tech.v_p_eff > 0 ? variable / tech.v_p_eff : Infinity;
        final = Math.min(raw * tech.e, byLabor, capacityOfBest(k, tech), plan.core + plan.restock);
      }
      this.post("use", "raw", "factory", null, Math.min(final / tech.e, this.get("factory", "raw")));
      this.post("production", "final", null, "factory", final);

      // Phase 3: maintenance and depreciation, then storage spoilage
      const u = unmaintainedShare(plan.alloc.maintenance, plan.required.maintenance);
      const loss = p.delta * u;
      let scrapped = 0;
      if (loss > 0) {
        const kept = []; let lost = 0.0;
        for (const cond of this.conditions) {
          const after = cond * (1 - loss);
          if (after < p.scrap_condition) { lost += cond; scrapped += 1; } else { lost += cond - after; kept.push(after); }
        }
        this.conditions = kept;
        this.post("depreciation", "factory", "factory", null, Math.min(lost, this.get("factory", "factory")));
        for (const [item, holder] of [["land_improvement", "farm"], ["tools_processing", "factory"], ["tools_distribution", "distribution"]])
          this.post("depreciation", item, holder, null, this.get(holder, item) * loss);
      }
      let spoiled = 0.0;
      // UI only (the Python model cannot switch storage off mid-run): a granary that is
      // switched off is emptied, its contents written off as waste.
      if (!p.storage.enabled && this.get("warehouse", "final") > 0) this.post("waste", "final", "warehouse", null, this.get("warehouse", "final"));
      if (p.storage.enabled) {
        spoiled = this.get("warehouse", "final") * p.storage.spoilage;
        this.post("spoilage", "final", "warehouse", null, spoiled);
      }

      // Phase 5: distribution and consumption
      const distCap = tech.v_d_eff > 0 ? plan.alloc.distribution / tech.v_d_eff : Infinity;
      const wanted = Math.min(plan.need, distCap);
      const fromFactory = Math.min(this.get("factory", "final"), wanted);
      this.post("transfer", "final", "factory", "households", fromFactory);
      let delivered = fromFactory, drawn = 0.0, added = 0.0;
      if (p.storage.enabled && wanted - delivered > EPS) {
        drawn = Math.min(wanted - delivered, this.get("warehouse", "final"));
        this.post("transfer", "final", "warehouse", "households", drawn);
        delivered += drawn;
      }
      this.post("consumption", "final", "households", null, this.get("households", "final"));
      if (p.storage.enabled) {
        added = this.get("factory", "final");
        this.post("transfer", "final", "factory", "warehouse", added);
      }
      const wastedFinal = this.get("factory", "final"), wastedRaw = this.get("factory", "raw");
      this.post("waste", "final", "factory", null, wastedFinal);
      this.post("waste", "raw", "factory", null, wastedRaw);
      const cons = plan.workers > 0 ? delivered / plan.workers : 0.0;

      // Phase 6: investment
      const spent = { factory: 0.0, land_improvement: 0.0, tools_processing: 0.0, tools_distribution: 0.0 };
      let newFactories = 0;
      const inv = p.investment;
      if (inv.enabled) {
        let budget = inv.s * plan.surplus;
        const costs = { factory: inv.factory_cost, land_improvement: inv.land_cost, tools_processing: inv.tool_cost, tools_distribution: inv.tool_cost };
        for (let it = 0; it < CAPITAL_TYPES.length; it++) {
          if (budget <= EPS) break;
          const t2 = this.tech();
          const wip = this.get("factory", "factory_wip");
          const ctx = { labor: budget, output: plan.core, need: plan.need, tech: t2, costs, threshold: inv.threshold,
                        inProgress: wip > EPS,
                        sectorLabor: { farming: plan.alloc.farming, processing: t2.v_p_eff * plan.core, distribution: plan.alloc.distribution } };
          const choice = INVESTMENT_RULES[inv.rule](ctx);
          if (choice === null || (choice !== "factory" && spent[choice] > 0)) break;
          let units = budget / costs[choice];
          if (choice === "factory") {
            const want = factoriesWanted(plan.need, t2.total_capacity, t2.capacity, inv.threshold);
            units = Math.min(units, Math.max(want, 1.0) - wip);
            this.post("production", "factory_wip", null, "factory", units);
            const done = Math.trunc(this.get("factory", "factory_wip") + 1e-9);
            if (done) {
              this.post("use", "factory_wip", "factory", null, Math.min(done, this.get("factory", "factory_wip")));
              this.post("production", "factory", null, "factory", done);
              for (let i = 0; i < done; i++) this.conditions.push(1.0);
              newFactories += done;
            }
          } else {
            const holder = { land_improvement: "farm", tools_processing: "factory", tools_distribution: "distribution" }[choice];
            this.post("production", choice, null, holder, units);
          }
          const lab = units * costs[choice];
          spent[choice] += lab;
          budget -= lab;
        }
      }
      const investment = psum(CAPITAL_TYPES.map((k2) => spent[k2]));

      // Phase 8: demography
      if (p.demography.enabled) this.N = nextPopulation(this.N, cons, p.c, p.demography.n, p.demography.mu);

      // Phase 9: check the books (goods conservation for every item)
      const closing = this.totals();
      const items = new Set([...Object.keys(opening), ...Object.keys(closing), ...Object.keys(this.flows)]);
      let ok = true;
      for (const item of items) {
        const f = this.flows[item] || {};
        const expect = (opening[item] || 0) + (f.production || 0)
          - (f.use || 0) - (f.consumption || 0) - (f.depreciation || 0) - (f.spoilage || 0) - (f.waste || 0);
        const got = closing[item] || 0;
        if (Math.abs(got - expect) > 1e-9 * Math.max(1, Math.abs(got), Math.abs(expect))) ok = false;
        for (const h of Object.keys(this.stock)) if ((this.stock[h][item] || 0) < -1e-9 * Math.max(1, Math.abs(expect))) ok = false;
      }
      this.booksOk = this.booksOk && ok;

      const techEnd = this.tech();
      const potential = potentialOutput(this.N, techEnd);
      const growth = this.prevPotential > 0 ? potential / this.prevPotential - 1.0 : 0.0;
      this.prevPotential = potential;

      return Object.assign(s, {
        N: this.N, plan, alloc: plan.alloc, required: plan.required, surplus: plan.surplus, shortfall: plan.shortfall,
        need: plan.need, harvestMultiplier: mult, raw, final, delivered, cons, wastedRaw, wastedFinal,
        unmaintained: u, scrapped, newFactories, conditions: this.conditions.slice(), factoriesRunning: k,
        wip: this.get("factory", "factory_wip"),
        land_improvement: this.get("farm", "land_improvement"),
        tools_processing: this.get("factory", "tools_processing"),
        tools_distribution: this.get("distribution", "tools_distribution"),
        effective_a: farmers > 0 ? raw / farmers : tech.a_x * mult,
        a_x: techEnd.a_x, capacity: techEnd.total_capacity,
        storage: this.get("warehouse", "final"), storageTarget: p.storage.enabled ? this.storageTarget() : 0,
        drawn, spoiled, added, restock: plan.restock, restockLabor: plan.restockLabor,
        spent, investment, leisure: plan.surplus - investment,
        potential, growth, booksOk: ok,
        nMinToday: analyticalNminTech(techEnd, p),
      });
    }
  }

  // Smallest town that keeps everything running with today's capital (fixed overhead / labor left per person).
  function analyticalNminTech(t, p) {
    const d = 1.0 - t.c / (t.e * t.a_x) - t.v_p_eff * t.c - t.v_d_eff * t.c;
    const fixed = t.F_p + maintenanceRequired(t);
    return d <= 0 ? Infinity : fixed / d;
  }

  // Workers needed to feed everyone (N*c, capped by factory capacity) and keep all capital repaired.
  function handsToFeedEveryone(town) {
    const t = town.tech(), core = Math.min(town.N * t.c, t.total_capacity);
    return psum(vals(laborRequirements(core, t)));
  }
  const api = { Town, DEFAULTS, analyticalNmin, potentialOutput, makeTech, makePlan, psum, handsToFeedEveryone };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CityEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
