# Stages

| Stage | Status | Report |
|---|---|---|
| 1 · Real economy baseline (planner, no money) | **Done** (2026-10-02) | `reports/stage_1.html` |
| 2 · Barter, money and prices | **Done** (2026-10-02) | `reports/stage_2.html` |
| 3 · Credit and banking | **Done** (2026-10-02) | `reports/stage_3.html` |
| 4 · Many firms, competition and specialization | **Done** (2026-10-02) | `reports/stage_4.html` |
| 5 · Government and public goods | **Done** (2026-10-02) | `reports/stage_5.html` |
| 6 · Central bank and monetary policy | **Done** (2026-10-02) | `reports/stage_6.html` |
| 7 · Financial markets and risk sharing | **Done** (2026-10-02) | `reports/stage_7.html` |
| 8 · Trade between cities | **Done** (2026-10-02) | `reports/stage_8.html` |
| 9 · Technology and long-run growth | **Done** (2026-10-02) | `reports/stage_9.html` |
| 10 · The modern economy and comparing systems | **Done** (2026-10-03) | `reports/stage_10.html` |

---

## Stage 1 · Real economy baseline

### What was built

**Core engine (`citysim/core`)**
- `World`: tick, agent groups, per-layer state, per-tick blackboard, seeded RNG.
- A double-entry `Ledger` with flow-tagged postings and direction checks.
- The 9-phase `Scheduler`.
- The `Layer` plugin interface: params, hooks, invariants, metrics.
- Engine invariants checked every tick, with tick and posting trace on failure.
- A metric `Recorder` writing tidy (tick, metric, group, value) tables.
- A runner for single runs, grid sweeps, multi-seed runs and comparisons.

**Config (`citysim/config.py`)**: pydantic-validated YAML scenarios with `inherits:` deep-merge. Parameters can be addressed by symbol (`a`, `s`, `N0`, …), by name, or as `layer.param`.

**Stage 1 layers (`citysim/layers/s1_real_economy`)**. Each can be switched on or off in the scenario:

| Layer | Mechanism |
|---|---|
| `real_economy` | Four sectors; central planner (labor requirement, priority shortage rule, surplus); farming → processing; maintenance and depreciation; distribution and consumption |
| `investment` | Share s of surplus labor builds factories, land improvements, or processing/distribution tools (swappable `bottleneck` / `best_return` rule) |
| `demography` | Grow at n when fed; shrink by 0.1 × the shortfall share otherwise |
| `storage` | One tick of consumption kept in store, 10% spoilage, restocked from surplus labor |
| `land_limits` | a·(land_limit/farmers)^β above 500 farmers |
| `harvest_shocks` | Scheduled multipliers on realised harvests |

**CLI**:
- `python -m citysim run <scenario.yaml>`
- `python -m citysim sweep <scenario.yaml> --param a=1:10:0.5`
- `python -m citysim compare <a.yaml> <b.yaml>`
- `python -m citysim experiments stage1` (regenerates the report)

**Grainbridge town twin** (`twin/`, build a local copy with `python -m citysim twin` → `reports/grainbridge.html`): an animated view of the Stage 1 economy for non-specialists.
- Little figures walk to the fields, the factory floor, the delivery carts, the repair crew, building sites or the park; wheat and bread carts move along the roads.
- Factories crack and stop smoking when unrepaired, houses go dark when hungry, the granary fills and empties.
- A plain-language narrator describes each year, and a town diary records the turning points.
- Eight one-click stories (healthy village, too few hands, knife's edge, boomtown, drought, no building, strict vs generous planner) plus live rule controls and a drought button.
- `twin/engine.js` is a port of the Stage 1 rules; `tests/test_twin_parity.py` checks it against the Python model tick by tick in 9 scenarios, to 1e-9.

**Tests**: 85 passing in ~6 s, including 16 twin-parity cases and the edge cases found in review:
- unit tests for every rule function and the ledger;
- invariant tests that tamper with stocks, labor and money and expect the run to stop;
- analytical tests (N_min, a_min, stationary state at N_min, the below-N_min steady state, zero growth at s = 0, plateau);
- every scenario run with invariants on, plus layer-toggle and config tests;
- golden regression files for the four Stage 1 scenarios (`tests/golden/`, regenerate deliberately with `python -m tests.make_golden`).

### Key results

All numbers are from `reports/stage_1.html`. Across the five experiments: **1,873 runs, 170,350 ticks, zero invariant violations, ~30 s total.**

**1. N_min.** The smallest town with no labor shortfall in the planner's books is **133.33333**, against the analytical 80/0.6 = 133.33333 (difference 2.5e-7, the bisection tolerance; the spec required within one person).
- Caveat raised in review: that test is the formula rearranged, so the match confirms the planner's accounting, not an emergent result.
- The outcome-based threshold is the smallest town that keeps everyone fully fed for T ticks:

| Horizon T | Smallest town fully fed |
|---|---|
| 20 ticks | 83.3 |
| 100 ticks | 111.4 |
| 300 ticks | 126.4 |
| 1,000 ticks | 131.3 |

- It approaches N_min only in the long run, because the factory's spare capacity lets an undersized town eat well while its capital decays.
- At N_min the economy is exactly stationary: surplus 0, potential output constant.
- Above N_min, the surplus share at tick 1 is 0.067 (N = 150), 0.33 (N = 300) and 0.40 (N = 400). It keeps rising as capital accumulates, e.g. N = 150 goes from 0.067 to 0.128 by tick 600.
- Below N_min with a fixed population, the priority planner starves maintenance. The factory decays until the freed labor can maintain it. Consumption per person then settles at exactly (N − 80)/(0.4 N): 0.833 at N = 120, 0.5 at N = 100.

**2. Minimum farm productivity.** Simulated a_min matches c/(e(1 − v_p c − v_d c − 80/N)) to within 4e-7 at every population tested:

| N | 150 | 200 | 300 | 400 | 1000 |
|---|---|---|---|---|---|
| a_min | 3.75 | 2.50 | 1.875 | 1.667 | 1.389 |

- No productivity rescues N ≤ 100.
- As N → ∞, a_min tends to 1.25.
- A 36 × 37 viability grid matches the analytical boundary.

**3. Investment share.**
- s = 0 gives **exactly zero growth** (max |growth| = 0.0 at every tick for N0 = 150 and 300), even though a surplus of 6.7% / 33% of labor exists.
- Growth rises with s. Average potential-output growth per tick over the first 50 ticks (N0 = 150):

| s | 0 | 0.1 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| growth per tick | 0 | 0.051% | 0.128% | 0.179% | 0.223% |

- Over 300 ticks the curve flattens above s ≈ 0.35, because capital reaches its optimum: one more unit saves less labor than its upkeep costs. Higher s only gets there sooner (tick 279 at s = 0.35, tick 123 at s = 0.8).
- With demography on, s = 0 pins the city at **1,000 people**, the capacity of its one factory. Any s ≥ 0.1 keeps it growing (7.5–8.0k by tick 400).

**4. Long run (demography + land limits).** Population grows at 1%/tick and then hits a wall. Leisure per person falls from its peak of 0.76 to 0, and nothing is left to invest. This is a Malthusian plateau.

| Variant | Plateau | Growth < 0.1% from tick | Farm labor share at end | Effective a at end |
|---|---|---|---|---|
| Land limits + land improvements (default) | ~4.3 M (peak 5.0 M) | 1,065 | 82% | 1.21 (peak 19.8) |
| Land limits, no land improvements | ~10.2 k | 450 | 81% | 1.23 |
| No land limits | none: still 1%/tick at 23 M | — | 1.2% | 80 |

- At the plateau, effective farm productivity is driven down to the viability floor of experiment 2.
- Drift over the last 200 ticks is −0.6% (default) and −0.3% (no land improvements).

**5. Harvest shock (−30% for 3 ticks, unanticipated):**

| Case | Min consumption/person | Consumption lost | Investment lost (20 ticks) | Population |
|---|---|---|---|---|
| No storage, fixed population | 0.70 | 135 units | 0 | — |
| Storage, fixed population | **1.00** | 0 | 12.5 labor-ticks | — |
| No storage, demography | 0.70 | 228.5 units | 125 | −8.7%, back to pre-shock level after 12 ticks, ends 11.4% below the no-shock path |
| Storage, demography | **1.00** | 0 | 21 | unaffected |

- Without storage, consumption recovers the tick after the shock ends. With storage the store is refilled 5 ticks after the shock starts (fixed population).
- Storage is insurance with a premium: keeping the store topped up against 10% spoilage cuts steady-state investment from 4.17 to 2.51 workers (−40%) at N = 150.
- A one-tick store fully absorbs a 30% shortfall lasting up to **6 ticks**, because the planner restocks during the shock. It starts failing at 7 ticks (min consumption 0.94).

### Review (Claude review agent, 2026-10-02) and fixes

A separate review agent checked the code, the numbers and the town page. Confirmed issues, all fixed:
- **A conversion rate e ≠ 1 could crash a run** (a negative waste posting caused by rounding). Fixed in both the Python model and the JS engine, and now tested.
- **The invariants were largely true by construction.** Added physical checks:
  - each harvest equals what the assigned farmers can grow;
  - bread baked fits the wheat used, the labor and the capacity;
  - bread eaten fits the carters and the need;
  - every unit of capital built is paid for in labor;
  - population changes only through the demography rule.
- **The N_min benchmark is an accounting identity.** Reframed above, with outcome-based thresholds added.
- **"Consumption lost" in experiment 5 counted people who were never born.** It now counts bread missing from people actually alive (228.5 units); the population gap is reported separately.
- **The parity test let NaN mismatches through.** Fixed, and parity now also covers e ≠ 1, c ≠ 1, a custom priority order, overlapping droughts and a town dying out.
- **Town page bugs:**
  - compare mode stalled when town A died;
  - droughts could stack while the narration said 30%;
  - a story's state survived a manual restart;
  - the strict planner's narration missed rationing ("short of hands" now also counts a cut bread target);
  - the hands panel mixed in granary restocking.

### Surprises

1. **N_min is not a collapse threshold once population can move.** With demography on, the default priority planner survives from N0 ≈ **83.3**, far below N_min = 133.3. It feeds everyone first and lets maintenance slip, so population keeps growing at 1% on the factory's spare capacity and passes N_min before the capital is gone: it borrows from its capital stock. The `max_feasible` planner, which protects capital and rations food, collapses for every N0 below **133.33**, exactly N_min. The same planner logic that hides a shortage also makes the system resilient here; in other settings it would hide a slow collapse.
2. **Priority rationing hides shortages.** A small labor shortfall never shows up in consumption, only in capital decay. Below N_min with fixed population, consumption stays at 1.0 for ~150 ticks while the factory decays from 1,000 to ~120 capacity.
3. **Below ~85 people with fixed population, investment loops.** Investment repeatedly rebuilds a factory the city can never crew, which shows as the jagged part of experiment 1's chart.
4. **Capital saturates.** With upkeep costs and a fixed population, there is an optimal capital stock, and s beyond ~0.35 only reaches it faster. Long-run growth then needs population growth or new technology.
5. **The plateau is set by how land improvements compare with land limits.** With the default elasticity 0.25 < β = 0.5, the default plateau (4.3 M) is ~400× the no-improvement plateau (10 k). Its scale depends heavily on parameters I chose (D-08) and should not be read literally.
6. **Building one factory per tick throttled growth.** An early version could add at most one factory per tick, and around 350 k people that limit capped growth. Investment now builds as many factories as needed (D-10).

### Open questions

- Should Stage 1's growth be bang-bang, as the spec states, or smoother, e.g. growth proportional to the surplus? The "slowing" in experiment 4 currently shows in leisure and surplus, not in the population growth rate.
- The capital functional forms, costs and upkeep (D-08) and the starvation rate μ = 0.1 (D-14) are my choices. Should any be calibrated differently before they become the Stage 2 benchmark?
- Should the default shortage rule stay `priority` (spec), given that it makes N_min a soft threshold under demography?

### Proposed next step: Stage 2 (barter, money, prices)

What Stage 2 should test, motivated by Stage 1:
- **Problem visible in Stage 1:** the planner needs perfect knowledge of a, v_p, v_d, capacity and the shock to allocate labor. Its two shortage rules give very different survival thresholds (83 vs 133), and it misallocates whenever its information is wrong (shocks).
- **Proposal:** replace the planner with households choosing occupations by expected income and sector firms setting prices and wages.
  1. Barter first, measuring failed trades.
  2. Then a money token, with the Godley–Lavoie transaction and balance-sheet matrices produced every tick. The ledger and balance-sheet invariant are already in place.
- **Benchmark:** compare against the Stage 1 planner on output, welfare and time to settle.
- **Decisions needed from you first:** price and wage adjustment rules, how households choose occupations, the initial money stock, and whether firms or a market own capital and decide investment.

I'll write a detailed design for your approval before building.

---

## Stage 2 · Barter, money and prices

Started on the user's instruction to build the full civilization simulation without waiting for per-stage approval. Design choices are in DECISIONS.md (D-22 to D-30).

### What was built

**`market` layer (`citysim/layers/s2_money_prices/`)**: a base economy that replaces the planner. It provides `economy`, so demography, land limits and harvest shocks work unchanged. Technology, capital and depreciation are exactly Stage 1's.
- **Firms:** one per sector by default (`firms_per_sector` lets Stage 4 add more):
  - the farm sells wheat to the factory;
  - the factory sells bread wholesale to the distributor;
  - the distributor sells bread to households;
  - the repair shop sells maintenance to every capital owner.
- **Prices:** normal-cost pricing, price = (1 + markup) × unit cost at the volume that feeds the town. The markup rises when stocks are short, falls when they overflow, and drifts back to normal.
- **Wages and jobs:**
  - wages rise with vacancies and fall when an occupation's unemployment exceeds 5%;
  - people drift toward occupations with better expected pay and open jobs;
  - wages are paid from the firm's cash after sales (pro rata if short).
- **Households:** they spend 90% of income and 10% of savings, buy their bread need if they can afford it, and want at most 1.5 loaves each.
- **Profits:** paid back as dividends, or set aside for building (irrigation, tools, factories) by Stage 1's investment rule. Firms pay out cash beyond a two-tick buffer.
- **Money:** a fixed cash stock created at the start. **Barter mode:** a trade needs a double coincidence of wants, found by random search.

**Other pieces:**
- `citysim/core/sfc.py`: Godley–Lavoie transaction-flow and balance-sheet matrices from the ledger.
- New invariants:
  - household money in the agent arrays equals the ledger;
  - no firm hires beyond its occupation's pool;
  - every money transfer is classified, so the transaction-flow matrix is complete;
  - market physics: harvests equal what the farmers can grow, bread fits the wheat, crews and capacity, deliveries fit the carters, and capital built equals the builders paid.
- `python -m citysim experiments stage2` produces `reports/stage_2.html`.
- 15 new tests and a golden file for `stage2_market`. Stage 1 golden outputs are unchanged.

### Key results

110 runs, 40,260 ticks, zero invariant violations. The matrix rows and columns close to within 2.5e-14.

**1. Barter vs money.** With five kinds of goods, a random meeting is a double coincidence 1 time in 16, and every link of the supply chain compounds the loss.

| Meetings per trader per tick | 16 or fewer | 32 | 48 | 64 | Money |
|---|---|---|---|---|---|
| Bread per person | 0 | 0.48 | 0.96 | 0.99 | 1.04 |

So barter needs roughly 50 meetings per trader per tick before the town can feed itself.

**2. Planner vs market** (fixed population, 400 ticks, both starting from the planner's allocation):

| Town size | Planner: bread / leisure share | Market: bread / unemployment |
|---|---|---|
| 120 | 0.83 / 0 | 0.71 / 4% |
| 130 | 1.00 / 0 | 0.86 / 4% |
| 150 | 1.00 / 12% | 1.18 / 3% |
| 200 | 1.00 / 28% | 1.50 (satiated) / 11% |
| 400 | 1.00 / 51% | 1.50 / 37%, prices fall towards zero |

- **Near the survival line** the market does worse: about 4–5% of people are always between jobs, and the planner uses every hand.
- **In a mid-sized town** the market does better, because firms turn spare hands into extra bread.
- **In a large town** demand is satiated, so spare hands become unemployment rather than leisure, and wages and prices deflate without end.
- **Time to settle:** 60–320 ticks, while the planner is immediate.

**3. Money.**
- **Neutral when prices start in line with it:** money from 100 to 10,000 gives prices from 0.147 to 14.7, with the same velocity (0.232) and the same real outcomes.
- **Scarce when prices start at the usual level:** the normal stock is about 875.
  - At 150–250 the economy freezes. Firms run out of cash for inputs, all the money ends up idle with households who have nothing to buy, and prices fall towards zero. This is a liquidity trap.
  - At 400–600 there is a deflationary dip, then recovery.
- **A growing town with fixed money** grows 16,000-fold while prices fall by a factor of about 18,000 (×5.6e-5): secular deflation.

**4. Price speed.** Slow markups settle smoothly; fast ones oscillate.

| Price speed | 0.005 | 0.05 | 0.4 |
|---|---|---|---|
| Price volatility | 1.2% | 8% | 14% |
| Settles | at tick 251 | at tick 317 | never within 400 ticks |

**5. Harvest shock** (−30% for 3 ticks):
- **Planner:** bread falls to 0.70 per person; 135 loaves are lost; it recovers in 3 ticks.
- **Market:** shop stocks cushion the blow, so bread only falls to 0.99, with 31 loaves lost and an 11% price spike. But the ripples through the supply chain take 97 ticks to die out.

### Surprises

1. **The market collapsed four times while I built it.** Each collapse exposed a real economic mechanism, now documented in D-23 to D-28:
   - wages paid up front plus all cash parked with households produced a cash trap;
   - inventory-based prices with no anchor gave runaway inflation;
   - retained profits meant workers couldn't buy back their output (Kalecki);
   - job-switchers spread evenly and drained the factory, the sector with the big fixed crew;
   - investment hiring distorted the job signals.
2. **Credit is the missing piece.** The liquidity trap (scarce money, a cold start, or a big drought with population dynamics) all fail the same way: a firm with no cash can't buy inputs, and nobody can sell to it on account.
3. **Satiation breaks the market.** Without other goods to want, spare hands become unemployment and deflation. The planner turns them into leisure.

### Open questions

- Should households want something besides bread (leisure, or a second good) so that labor supply responds to wages? That would remove the satiation deflation.
- With several firms per sector, factories are indivisible, so a factory firm with no factory sits idle. Stage 4 will handle entry and exit properly.

### Next step: Stage 3 (credit and banking)

The liquidity trap above is exactly what banks solve: a bank can lend a firm the cash for wages and inputs, creating deposits as it does.

---

## Stage 3 · Credit and banking

### What was built

**`banking` layer (`citysim/layers/s3_banking/`)**:
- **Deposits and loans.** One bank takes everyone's cash as reserves and gives deposits in return; payments become deposit transfers. Loans create deposits and repayments destroy them, so the money supply (deposits plus cash outside the bank) is endogenous.
- **Two kinds of lending:**
  - working capital, when a firm's till can't cover wages, inputs or repairs;
  - building loans, when a project saves enough wages to beat the loan rate + repayment + 1% per tick.
- **Limits:**
  - the borrower: half its collateral plus a tick of revenue, and no more than 15% of revenue (actual or expected) in debt service;
  - the bank: equity ≥ 8% of loans.
- **Repayment:** interest at 1% per tick plus 5% of principal; credit is revolving, so cash beyond one tick of costs repays early.
- **Distress:**
  - a borrower 10% of its debt in arrears defaults, the bank seizes up to 40% of the debt from its deposits and writes off the rest, and the borrower is cut off for 20 ticks;
  - a firm in debt pays no dividends out of borrowed money.
- **The bank's own finances:** it passes half its loan interest on to depositors and pays half its profit as dividends when its capital is above 15%.
- **Runs and failures:** households run when bank capital falls below 4% and redeposit when it recovers. A run the reserves can't meet, or negative equity, triggers resolution: depositors are bailed in to restore the requirement, and lending is frozen for 10 ticks.
- **Minsky option:** `procyclicality` loosens lending limits while loans perform, collapses them after defaults, and makes firms bolder, so they build factories before they are full and accept lower returns.

**Engine:** a new ledger item kind, `claim` (deposits, loans), which is negative for the debtor. Two new invariants:
- claim signs: no overdrafts, and the bank never holds loans as a liability;
- *money explained*: each tick, the change in the money supply must equal the sum of the bank's money-creating and money-destroying postings.

The market now pays in whatever money item is active and asks credit providers when its till is short. Without a bank, the Stage 2 golden output is unchanged.

### Key results

33 runs, 13,900 ticks, zero invariant violations; the transaction-flow matrix closes to within 1.6e-14.

**1. Does credit rescue Stage 2's failures?** (bread per person over the last 100 of 300 ticks)

| Case | Market only | Market + bank |
|---|---|---|
| Scarce money, M0 = 300 | 0 | **1.00** |
| Scarce money, M0 = 200 | 0 | **0.88** (about 70% for 200 ticks, then full) |
| Scarce money, M0 = 150 | 0 | 0 (189 in defaults, 1 bank failure) |
| Cold start (equal split) | 0 | 0 |
| Famine with demography | town dies | town dies |

- **Credit fixes a shortage of money:** loans create what was missing.
- **It can't fix the cold start:** no loan conjures 50 factory workers at once.
- **It can't fix the famine:** that pushes the town below its survival line.

**2. Credit limits vs growth** (firms keep only 5% of profit for building).
- **Fixed town of 150:** nobody borrows, because projects don't beat the hurdle, so limits change nothing.
- **Growing town:** consumption over the last 100 ticks rises with credit and then saturates.

| Credit scale | 0 | 0.25 | 0.5 or more |
|---|---|---|---|
| Bread per person | 1.16 | 1.37 | 1.47 |

**3. Minsky** (growing town, 500 ticks):

| Procyclicality | Credit peak | Default waves | Bank failures | Deposits destroyed |
|---|---|---|---|---|
| 0 | 31 | 3 | 0 | 0 |
| 0.02 | 134 | 5 | 3 | 55 |
| 0.05 | 267 | 7 | 6 | 193 |
| 0.1 | 286 | 9 | 8 | 280 |
| 0.2 | 294 | 11 | 11 | 369 |

**4. Bank failure after a harvest shock** (−50% for 5 ticks, credit-heavy growing town, 70 of defaults):
- **Thin capital** (2%): a 4-tick run, one failure, and depositors lose 30 in a haircut.
- **Normal (8%) or thick (16%) capital:** the same losses are absorbed with no run.
- Without the shock, no bank fails.

### Surprises

1. **A careless bank is worse than no bank.** The first version made a growing town collapse. Each failure exposed a real banking mechanism, now fixed and recorded in D-33 to D-37:
   - paying dividends out of starting capital;
   - lending against the very project being financed (self-referential collateral);
   - debt-financed dividends;
   - no debt-service cap;
   - an insolvent bank left open until a run drained it.
2. **A calm town barely uses the bank.** With retained profits, firms self-finance. Credit matters for shortages of money, for growth, and when lending standards loosen.
3. **Credit doesn't fix satiation deflation.** In a growing town with fixed outside money, half the people end up idle once bread needs are met, wages fall, and every nominal quantity, including bank capital, shrinks toward zero. Stages 4–6 add new goods, government and a central bank, the institutions meant to address this.

### Next step: Stage 4

Many firms, entry and exit, new sectors (tools and services, which give households something to want beyond bread), owners, and agent-mode households with inequality measures.

---

## Stage 4 · Many firms, competition and specialization

### What was built

All in `citysim/layers/s4_competition/`.

**`agent_market`** is the base layer: a subclass of Stage 2's market, so every Stage 2–3 rule is inherited and their golden outputs are unchanged.
- **Households are individual agents** in numpy arrays, one per person by default. Each has an occupation, an employer or none, savings, this tick's income, a hunger flag, and a retraining countdown.
- **Labor market with job search.** Firms lay off down to their labor demand. Unemployed people of the right occupation fill vacancies, best-paying firms first, with 80% matching efficiency.
- **Occupation switching.** Mostly the unemployed reconsider (20% per tick, versus 1% of the employed). They move only if the new occupation's expected pay, net of the time lost retraining (3 ticks, weighed over a 50-tick horizon), beats staying.
- **Owners.** Each firm's dividends are shared by 20 owner households (`owners_per_firm`; `ownership: public` pays everyone per head). Owners' shares pass to heirs when they die.
- **Market power.** A firm's normal markup is 10% + 50% × its sector's Herfindahl index of sales: 10% under full competition, 60% for a monopoly.
- **Births and deaths** add agents and remove the hungriest, whose savings pass to heirs.

**`services`:** a labor-only sector (0.5 workers per unit plus a 2-person fixed crew per firm). Households spend on services whatever they planned but didn't spend on bread, but only once they are fed: necessities come first.

**`entry_exit`:**
- **Entry:** a household with enough savings founds a firm where the sector's smoothed margin exceeds 20% (5% chance per tick), paying in 4 ticks of a starting crew's wages as equity. Factory entry is closed, because the 1,000 labor-tick plant is a barrier to entry.
- **Exit:** a firm leaves after 12 loss-making ticks with an empty till. The acquirer takes over its plant, stocks, workers and customers; its debts default and its cash goes to its owners.

**Engine and metrics:**
- Income and wealth Gini, the top-10% wealth share, the share of people short of bread, markups, concentration and firm counts per sector, entries and exits.
- New invariants: nobody works for a firm that doesn't exist, no firm has more employees than it hired, and services sold never exceed what the staff could serve.
- Physics checks now use the workforce recorded at production time.

**The base scenario includes Stage 3's bank.** Without credit, any disruption (a factory closing, a wave of job switching) could recreate the liquidity trap.

### Key results

96 runs and 39,900 ticks, each setting averaged over 3 seeds; zero invariant violations.

- **Scale:** 100,000 agents and 90 firms run 1,000 ticks in about 2.7 minutes with every invariant checked. Target met.
- **1. Monopoly vs competition** (bread chain with 1, 2, 4 and 8 firms per sector, market power on):

| Firms per sector | 1 | 8 |
|---|---|---|
| Retail markup | 0.95 | 0.33 |
| Real wage | 0.50 | 0.76 |
| Bread per person | 0.67 | 0.95 |
| Income Gini | 0.47 | 0.34 |
| Wealth Gini | 0.88 | 0.74 |

With market power switched off, a monopolist's real wage is 0.83: the concentration markup is what hurts.

- **2. Fixed costs:** as the services fixed crew rises from 0.5 to 20, the number of surviving services firms falls from 5.0 to 3.0, services output falls from 0.62 to 0.43 per person, and concentration rises from 0.20 to 0.33.
- **3. New sectors:**
  - **Services emerge on their own.** Started with none, the first firm appears around tick 10–48, once fed households have money left over.
  - **They cure Stages 2–3's satiation trap:**

| Economy | Unemployment | Price level at the end | Bread per person |
|---|---|---|---|
| Bread only | 42% | ≈ 0 (deflation) | 0.75 |
| Bread + services | 3% | 0.88 (stable) | 0.98 |

- **4. Ownership drives inequality:**

| Ownership | Wealth Gini | Short of bread | Bread per person |
|---|---|---|---|
| 1 owner per firm | 0.95 | 98% | 0.71 |
| 5 owners | 0.89 | 84% | 0.80 |
| 20 owners (base) | 0.67 | 60% | 0.98 |
| 60 owners | 0.65 | 58% | 1.01 |
| Public | 0.07 | 4% | 1.34 |

  Concentrated owners buy services while workers go short of bread.
- **5. Structural unemployment:** with retraining of 10–30 ticks and low job switching, unemployment is 31–36%. It is about 3.5% when retraining takes 0–3 ticks.

### Surprises

1. **Concentrated ownership alone can collapse a market.** With one owner per firm and no bank, the owners can't spend their dividends (they can only eat so much bread), so money stops circulating. Workers can't buy back what they produce, and demand collapses (Kalecki). Spreading ownership, adding services and adding credit each soften this.
2. **Necessities first matters.** When hungry households still bought services, labor drained out of bread into services and the bread chain collapsed.
3. **A factory closing without a buyer for its workforce is a catastrophe.** Firing half the bread workforce at once triggered the liquidity trap. Acquirers taking over the workforce fixed it, which is realistic.
4. **Inequality kills, given demography.** With population dynamics on, about 60% of people go short of bread even though the average is near 1. The poor starve, the town shrinks below the size its firms need, and it dies within about 125 ticks. That is the case for Stage 5's transfers.

### Open questions

- Factory entry is closed by assumption. Should entrants be able to borrow to build a plant?
- Wealthy owners hold savings with nowhere to invest them until Stage 7 adds financial assets.

### Next step: Stage 5 (government)

Taxes on income or profit, unemployment benefits and a minimum income as a safety net, public maintenance and infrastructure, and public debt. The experiments target the starvation spiral above and tax-funded versus private maintenance.

---

## Stage 5 · Government and public goods

### What was built

**`government` layer (`citysim/layers/s5_government/`)**, working with the Stage 2 market or the Stage 4 agent market, with or without a bank:
- **Taxes:** a flat income tax on households (default 10%) and a profit tax that firms pay before dividends (20%), through a new market hook.
- **Transfers:** unemployment benefits as a share of the average wage (40%), and an optional basic income per head.
- **Public works:** the government hires idle people for road building and road upkeep, up to 10% of the population. Public workers don't also collect benefits.
  - Roads cut every distributor's labor per loaf by (1 + roads)^−0.25: a public good no single firm would build.
  - Unmaintained roads decay.
- **Public maintenance (option):** the government pays the repair shops for everyone's capital upkeep, through a market hook.
- **Debt:** deficits are financed by selling bonds to the bank, which pays with new deposits. Bonds carry a capital risk weight, so they can crowd out loans. Bond interest is 0.5% per tick, there is a debt ceiling of 30 ticks of revenue, and surpluses repay debt.
- **Fiscal rule:** money beyond two ticks of spending first repays debt, then is rebated per head. A first version simply kept its surpluses, pulled money out of circulation, and caused 33% unemployment and deflation.
- **Waste:** a share of public payroll that produces nothing.
- **Invariants:** the government never spends money it doesn't have and never holds its own bonds; roads built never exceed what the useful public labor could build; bond sales, interest and redemptions are covered by the bank's money-explained check.

### Key results

93 runs, 27,180 ticks, 3 seeds per setting; zero invariant violations.

**Base effect** (600 people, 400 ticks, with vs without government):

| | No government | With government |
|---|---|---|
| Bread per person | 0.98 | 1.09 |
| Unemployment | 4% | 2% |
| Income Gini | 0.34 | 0.26 |
| Wealth Gini | 0.68 | 0.44 |

**1. Does a safety net stop Stage 4's starvation spiral?** (demography on, 300 ticks)

| Policy | Outcome |
|---|---|
| No government, taxes only, or 40% unemployment benefits | The town dies |
| Benefits 80% + income tax 20% | Survives at 1,556 people, bread 0.72 |
| Benefits 40% + basic income 20% | Grows to 2,545 people, bread 1.02, 13% unemployment |

Unemployment benefits miss the point: the hungry are mostly low-paid workers, not the unemployed.

**2. Tax-funded vs private maintenance:**
- **Calm times:** both work; public upkeep costs more in taxes (104 vs 88 per tick).
- **50% harvest shock:**

| Maintenance | Bread per person | Unemployment |
|---|---|---|
| Private | 0.56 | 33% |
| Public | 0.77 | 11% |

  Cash-strapped firms skip repairs and their capital decays; tax-funded upkeep keeps the factories working.

**3. Benefits in a harvest shock don't stabilize consumption** (lowest bread per person during the shock):

| Benefits | 0 | 40% | 80% |
|---|---|---|---|
| Lowest bread per person | 0.50 | 0.42 | 0.49 |

  A harvest failure is a *supply* shock: benefits add spending power but no bread, so they bid up prices and redistribute a smaller loaf. Automatic stabilizers are a tool for *demand* shocks.

**4. Roads:**

| Road target | 0 | 5 | 20 |
|---|---|---|---|
| Distribution labor share | 8.7% | 6.0% | 4.9% |
| Bread per person | 1.07 | 1.08 | 1.13 |
| Unemployment | 3.5% | 2.3% | 0.8% |

  The 10% public-employment cap limits roads to about 20.

**5. Deficits and crowding out:**
- **Modest deficits** (basic income 30%, taxes 10%) with a thin bank and bonds needing capital: private loans fall from 105 to 39. Crowding out.
- **Large deficits** (basic income 60%, taxes 3%):
  - when the bank can keep buying bonds, the economy destabilizes, with bread at 0.42–0.76 and some deficit-financed inflation (+0.44% per tick);
  - when it can't (thin capital and bonds needing capital), the government is forced into austerity and the economy stays healthy (bread 1.02).

**6. Waste:** at 30% waste, only 4 of the 20 target road units exist, and at 90% almost none. Wasted labor also starves upkeep, so the damage is more than proportional.

### Surprises

1. **A government that hoards its surpluses causes a recession.** A fiscal rule that returns excess money fixed it.
2. **Targeting matters more than generosity.** A small basic income saved a town that generous unemployment benefits couldn't, because the working poor were the ones starving.
3. **The famous automatic stabilizers do nothing against a harvest failure,** a supply shock. They need a demand shock to shine.

### Next step: Stage 6 (central bank)

A central bank that holds bank reserves, sets the policy rate with a Taylor rule, and acts as lender of last resort. Experiments: a Taylor rule vs a fixed money stock under supply (harvest) and demand (spending) shocks; the zero lower bound; moral hazard from bailouts.


## Stage 6 · Central bank and monetary policy

### What was built

**`central_bank` layer (`citysim/layers/s6_central_bank/`)**, on top of the Stage 5 economy:
- **Policy rate:** the bank lends at the policy rate plus 0.7%, and government bonds pay the policy rate. Households spend less when the rate is above neutral (0.3% per tick) and more when it is below: spending × (1 − 15 × (rate − neutral)), bounded to ×0.5–1.5. Every rule holds the rate at neutral for the first 50 ticks while prices settle.
- **Rules:**
  - **Taylor:** rate = neutral + 1.5 × (inflation − 0) − 0.05 × (unemployment − 4%), floored at 0 and smoothed. Inflation is measured on a price index of 70% bread and 30% services.
  - **Fixed rate.**
  - **Money growth (monetary targeting):** the money supply should follow M0 × (1 + g)^t. Above the path, the rate rises 0.1 points per 1% excess; below it, new money is paid per head.
- **Reserves:** the bank keeps 20% of deposits as cash. The rest sits in central bank bills, which count as capital but can't pay depositors.
- **Lender of last resort:** a solvent bank facing a run borrows new cash at the policy rate + 0.5%. Interest it can't pay accrues to the debt; it repays 20% per calm tick; the central bank is senior in a resolution.
- **Bailouts (option):** an insolvent bank is recapitalised with new money. If bailouts are expected (moral hazard), depositors never run and the bank barely tightens after defaults.
- **`demand_shocks` layer:** scheduled multipliers on household spending, the demand counterpart of harvest shocks.
- **Bank additions:**
  - a rumour-driven panic;
  - with a central bank present, resolution leaves the bank enough capital to lend again (no zombie bank).
- **Invariants:** the rate respects its floor; the central bank's claims are never negative; every unit of central bank money is an explicit creation covered by the bank's money-explained check.

**Review fixes.** A review agent checked Stages 6–7. In the first version:
- the fixed-money rule capped lending at the money path, which left zero room, so it shut off credit entirely;
- the Taylor rule had no warm-up, and the start-up price swings sent it to 8% per tick;
- emergency-loan interest was silently forgiven when reserves were empty;
- failure counts double-counted one panic.

All four are fixed, and the numbers below are from the fixed version.

### Key results

93 runs, 26,760 ticks, 3–5 seeds per setting; zero invariant violations.

**1. Taylor rule vs fixed money stock** (losses against the same seed without the shock, ticks 150–220):

| Shock | Rule | Bread lost | Peak unemployment |
|---|---|---|---|
| Demand (spending −40%, 20 ticks) | Taylor | 7,106 | 37% |
| | Fixed money stock | 6,314 | 31% |
| | Fixed rate | 7,410 | 32% |
| Supply (harvest −50%, 5 ticks) | Taylor | 16,262 | 43% |
| | Fixed money stock | 21,670 | 52% |
| | Fixed rate | 16,754 | 52% |

- **Demand shock:** the money rule does best. When spending collapses, money piles up unspent, so it is "above path" only briefly, and per-head top-ups replace what repayments destroy. The Taylor rule hits the zero floor at once and then tightens hard against the inflation of the recovery.
- **Supply shock:** the money rule is clearly the worst. The harvest failure raises prices, the money stock can't stretch to pay them, and the rule raises rates when money moves above its path. The Taylor rule beats a passive fixed rate.

**2. Inflation from printing money:**

| Money growth per tick | 0 | 0.25% | 0.5% | 1% |
|---|---|---|---|---|
| Inflation per tick | −0.1% | 0.3% | 0.4% | 0.5% |
| Price volatility | 0.8% | 1.3% | 1.9% | 4.0% |
| Bread per person | 1.10 | 1.06 | 0.89 | 0.90 |
| Real wage | 0.78 | 0.73 | 0.59 | 0.67 |
| Bank failures | 0 | 0 | 1.3 | 6.7 |

- Printing feeds inflation, makes prices noisy, cuts real wages and breaks banks.
- Beyond about 0.5% per tick, inflation stops keeping pace with money, because the economy itself is breaking down.

**3. Zero lower bound** (spending −50% for 30 ticks):

| Policy | Bread lost | Peak unemployment |
|---|---|---|
| Fixed rate | 9,662 | 37% |
| Taylor, floor at 0 | 10,050 | 40% |
| Taylor, no floor (down to −3%) | 6,260 | 34% |
| Taylor, floor at 0, neutral rate 1% | 4,544 | 37% |

- At the floor the rule can't deliver the −3% it asks for, and does no better than a fixed rate.
- Without the floor, losses fall 38%.
- A higher neutral rate gives room to cut and halves the losses.

**4. Lender of last resort** (a rumour: 40% of deposits withdrawn per tick for 6 ticks; 20% reserves):

| | Failure episodes | Ticks withdrawals blocked | Ticks credit frozen |
|---|---|---|---|
| No lender of last resort | 1 | 4 | 13 |
| Central bank lends in the run | 0 | 0 | 0 |

The emergency loan (243) is repaid once the panic is over.

**5. Bailouts and moral hazard** (credit boom, 4% capital requirement, 5 seeds):

| Policy | Average loans | Defaults | Failure episodes | Bread | Unemployment |
|---|---|---|---|---|---|
| Bail-in (depositors pay) | 90 | 949 | 14.4 | 0.80 | 17% |
| Surprise bailouts | 134 | 1,046 | 0.8 | 0.93 | 14% |
| Expected bailouts | 184 | 1,707 | 0 | 1.03 | 5% |

- **Costs:** bail-in costs 856 in deposit haircuts; each bailout policy costs about 1,080 of new money.
- **Moral hazard is real:** expecting rescue, the bank loosens most, lends twice as much and suffers 80% more defaults.
- **The real economy does better**, because repeated failures and credit freezes are what hurt bread and jobs. That is the policy dilemma in miniature.

### Surprises

1. **Under the Taylor rule, the zero lower bound bites within a tick of any big demand shock,** because the neutral rate is low.
2. **A fixed money stock is decent against demand shocks and bad against supply shocks.**
3. **Bailouts help the real economy while making the financial system riskier.**
4. **A resolved bank with no capital is a zombie.** Without working-capital credit, output fell to zero.

### Next step: Stage 7 (financial markets)

- Shares in firms traded by fundamental traders and trend followers, with leverage and margin rules.
- Harvest futures and insurance for farmers.
- Experiments: whether futures smooth farmer income; margin requirements against bubble size.

## Stage 7 · Financial markets and risk sharing

### What was built

**`stock_market` layer (`citysim/layers/s7_financial_markets/stock_market.py`):**
- **The index:** all firms' equity is one index of shares that receives every dividend, replacing founder ownership. At the listing (tick 20) the index is split so that, at fair value, shares are half of households' wealth.
- **Fundamental value:** average dividends per share ÷ (policy rate + 5% premium).
- **Households:**
  - 30% value traders;
  - 30% trend followers, who trust a trend less the further the price is from value and may borrow from the bank on margin;
  - the rest hold, with a little noise trading.
- **Trading:**
  - Orders need money and are matched with rationing.
  - Margin calls force sales first.
  - The price moves with the order imbalance, at most 10% per tick.
  - Leveraged buying is limited by the bank's capital.
  - A borrower left with debt and no shares defaults.

**`risk_sharing` layer (`risk_sharing.py`):**
- **Futures:** farms sell wheat forward, cash-settled after 4 ticks against the spot price. Speculating households take the other side.
- **Crop insurance:** paid when the harvest multiplier is below 1. The premium is (1 + 10% loading) × the insurer's learned loss rate. The insurer, owned by households, has its own capital, which it can exhaust.

**Supporting changes:**
- random weather (`harvest_shocks.volatility`);
- random demand swings (`demand_shocks.volatility`, AR(1));
- hooks so dividends follow shareholdings, and so per-person arrays survive births and deaths.

**Invariants:**
- shares are conserved, and the agent arrays match the ledger;
- no short selling or negative debt;
- every unit of margin debt is a ledger loan;
- the insurer never pays out money it doesn't have;
- margin loans, interest, repayments and defaults are part of the bank's money-explained check.

### Key results

155 runs, 59,900 ticks; medians across seeds where slumps would dominate a mean; zero invariant violations.

**1. Margin requirements and bubbles** (5 seeds; bubble = 95th percentile of price ÷ value):

| Margin requirement | 100% | 50% | 20% | 10% |
|---|---|---|---|---|
| Bubble size | 1.05 | 1.20 | 1.44 | 1.33 |
| Worst crash | 71% | 79% | 89% | 88% |
| Peak margin debt | 0 | 748 | 2,535 | 2,645 |
| Margin defaults | 0 | 14 | 616 | 625 |
| Bank failures | 2 | 0 | 5 | 5 |

Below 20%, the bank's capital, not the margin rule, limits borrowing.

**2. Trader mix:**

| Trend followers | 0% | 15% | 30% | 50% |
|---|---|---|---|---|
| Bubble size | 1.16 | 1.13 | 1.20 | 2.16 |
| Worst crash | 68% | 66% | 79% | 96% |

Value traders hold the price near fundamentals until trend followers dominate.

**3. Do futures smooth farmers' income?**

| Risk | Hedging | Income sd | Worst tick | Loss ticks |
|---|---|---|---|---|
| Random harvests | None | 8.0 | −7.2 | 2.6% |
| | Futures 100% | 9.2 | −10.5 | 5.1% |
| | Insurance 80% | 6.9 | −2.2 | 1.0% |
| Random demand | None | 6.4 | — | — |
| | Futures 50% | 5.5 | — | — |
| | Insurance 80% | — | — | — |

- **Harvest risk:** insurance helps, futures make income more volatile. Wheat prices here are cost-based and barely move with the weather, so farm income risk is about *quantity*, which price futures don't hedge, and the futures add their own price noise.
- **Demand risk:** futures help a little. Insurance never pays out, so it is pure cost.
- **Lesson:** match the hedge to the risk.

**4. Correlated disaster** (harvest −60% for 5 ticks):

| Insurer capital | Claims paid | Claims unpaid |
|---|---|---|
| 30 ticks of premiums | 23 | 191 |
| 300 ticks of premiums | 123 | 78 |

Every farm claims at once, so a thinly capitalised insurer fails exactly when it is needed.

**5. Who owns the firms:**

| Ownership | Bread | Unemployment (sd) | Wealth Gini |
|---|---|---|---|
| Founders | 1.09 | 5% (3%) | 0.36 |
| Everyone per head | 1.25 | 11% (11%) | 0.08 |
| Listed shares, no trading | 1.25 | 11% | 0.06 |
| Listed shares, traded | 1.08 | 13% | 0.28 |

- Spreading ownership raises bread consumption and flattens wealth, but employment becomes much more volatile: more households have money to switch between bread and services.
- Trading re-concentrates wealth.

### Surprises

1. **Futures don't help farmers whose risk is the weather** when prices don't move with the harvest.
2. **Without doubt, trend-following plus margin loans produced runaway prices.** Rising prices raised collateral, collateral allowed more borrowing, and the bank's interest income expanded its lending room. Bubbles end only when trend followers lose conviction far from value.
3. **The stock market's big real effect is ownership, not trading:** who receives dividends changes how the town spends.
4. **The economy is path-dependent.** A small change (an insurance premium) can tip one run into a long slump, which is why medians are reported.

### Next step: Stage 8 (trade between cities)

- Several cities with different land, skills and technology.
- Trade with transport costs, migration, and either separate currencies or a currency union.
- Experiments: gains from comparative advantage; floating rates vs a currency union under a regional shock.


## Stage 8 · Trade between cities

### What was built

**A region runner (`citysim/region.py`)** steps several complete city simulations (each a Stage 5 economy with its own ledger, layers and invariants) phase by phase in lockstep. A region scenario lists the cities with their own parameters (`scenarios/stage8_*.yaml`).

**A trade network (`citysim/layers/s8_trade/trade.py`):**
- **Importers and merchants:**
  - Each town gets a wheat importer and a bread importer, which sell in its local markets alongside the farms and bakeries.
  - After the local markets close, merchants buy from the other town's producers.
  - They only buy when the landed cost (supplier price, converted, ÷ (1 − transport loss)) plus a 2% markup beats the local price.
  - Shipments leave as `export` and arrive as `import`, minus an iceberg transport loss.
- **Money:**
  - **Floating:** a currency dealer holds money in every town and moves the rate towards the currency it is short of.
  - **Currency union:** payments move bank reserves between towns.
  - **Fiscal union (option):** moves a share of the gap in money per head from richer to poorer households.
- **People and capital:**
  - **Migration:** people move towards higher real income (wage × employment ÷ bread price), the unemployed first, taking their money.
  - **Capital flight:** households in the town with more unemployment send money abroad.
- **Embargoes:** cut trade between chosen towns for a period.

**Supporting changes:**
- `import` and `export` ledger flows, counted by goods and money conservation;
- a region invariant: every export arrives as an import, and under floating rates no money crosses a border;
- migration helpers on the agent market;
- buyers' orders split by price when importers are present.

### Key results

90 region runs, 48,660 city-ticks, 3 seeds; zero invariant violations.

The towns:
- **Grainbridge:** rich land (9 per farmer), clumsy bakers (0.35 workers per loaf).
- **Millford:** poor land (3.5), skilled bakers (0.04).

Results are from the version fixed after the review agent's audit (see Surprises).

**1. Comparative advantage** (floating currencies; bread + services per person, last 100 ticks):

| | Grainbridge | Millford | Region total | Millford farm labor | Shipped per tick |
|---|---|---|---|---|---|
| Autarky | 1.51 | 1.21 | 2.72 | 18% | 0 |
| Trade, transport 2% | 1.32 | 2.04 | 3.36 | 1% | 883 |
| Trade, transport 5% | 1.27 | 2.08 | 3.35 | 1% | 974 |
| Trade, transport 25% | 1.31 | 1.83 | 3.14 | 14% | 879 |
| Identical towns: autarky | 1.72 | 1.72 | 3.44 | — | 0 |
| Identical towns: trade | 1.68 | 1.70 | 3.38 | — | 73 |

- **Specialisation:** Millford almost stops farming, and Grainbridge doubles its farm labor (10% → 19%).
- **Gains:** the region's consumption rises 23%. The gains shrink as transport costs rise, and at 25% Millford keeps some of its own farms.
- **Distribution:** the gains are unequal. Millford gains a lot, while Grainbridge, the cheap-wheat supplier, ends up slightly worse off.
- **Control:** identical towns trade little and lose a little to transport.

**2. Money regimes:**

| | Grainbridge | Millford | Region | Shipped |
|---|---|---|---|---|
| Floating | 1.27 | 2.08 | 3.35 | 974 |
| Currency union | 1.33 | 1.70 | 3.03 | 526 |
| Union + fiscal transfers | 1.33 | 1.80 | 3.13 | 603 |

- **The union trades less and gains less.** Without an exchange rate, relative prices must adjust through sticky wages and money flows.
- **Bread falls below need in both union towns:** Grainbridge 0.95, Millford 0.86 (74% of Millford households short), even as services rise.

**3. A regional shock** (Grainbridge's harvest −60% for 10 ticks; bread lost against no shock):

| Regime | Grainbridge | Millford | Region | Grainbridge currency |
|---|---|---|---|---|
| Autarky | 32,620 | 0 | 32,620 | — |
| Floating | 29,844 | 39,806 | 69,650 | −23% |
| Currency union | 17,997 | 28,522 | 46,519 | — |
| Union + transfers | 21,440 | 31,694 | 53,134 | — |
| Union + capital flight | 14,883 | 14,984 | 29,867 | — |
| Floating + capital flight | 22,496 | 18,150 | 40,646 | −39% |

- **Specialisation brings exposure.** Millford grows almost no wheat of its own, so Grainbridge's harvest failure is Millford's too, whatever the money regime. The region loses more than twice as much as under autarky.
- **Floating rates don't isolate a supply shock** to the partner's essential input: the wheat simply isn't there.
- **The union cushions the stricken town,** whose loss falls 45%, but the region still loses 43% more than under autarky.

**4. Dependency on a single partner** (currency union; embargo between Grainbridge and Millford for 30 ticks):

| | Millford bread during embargo | Lowest | Unemployment |
|---|---|---|---|
| Two towns, no embargo | 0.91 | 0.83 | 7% |
| Two towns, embargo | 0.53 | 0.25 | 29% |
| Three towns, embargo | 0.88 | 0.74 | 9% |

A second supplier (Cornhill) turns a famine into a dip.

**5. Migration** (5% per unit of real-income gap beyond a 10% moving cost):

| Regime | Grainbridge pop. | Millford pop. | Real income (G / M) | Unemployment (G / M) | Region bread |
|---|---|---|---|---|---|
| Floating, no migration | 600 | 600 | 0.64 / 0.70 | 4% / 4% | 1.02 |
| Floating, migration | 1,186 | 14 | 0.75 / 0.24 | 11% / 20% | 1.00 |
| Union, no migration | 600 | 600 | 0.64 / 0.52 | 3% / 7% | 0.91 |
| Union, migration | 1,091 | 109 | 0.77 / 0.31 | 3% / 27% | 1.04 |

- **Migration empties Millford in both regimes.** Real income there falls as its people leave, and leaving raises unemployment for those who stay: the classic "brain drain" spiral.
- **The region gains bread only in the union,** where the towns were worse off to begin with.

### Surprises

1. **Gains from trade only appeared once merchants obeyed the arbitrage rule.** At first, importers bought whatever they expected to sell, and buyers split orders equally between sellers. Identical towns shipped 800 units a tick back and forth and lost 15% of their bread. With merchants buying only what beats the local price, and local firms losing only the orders an importer can fill, the control trades little.
2. **The first floating-rate design was wrong** (found by the review agent).
   - **How it worked:** a currency dealer with fixed money targets moved the rate by the gap.
   - **What went wrong:** the dealer lost money every swing, ran out of both currencies, and trade became rationed by its stock. Shipments didn't respond to transport costs.
   - **The fix:** the exchange rate now clears each tick's currency orders (D-78).
   - **Other bugs fixed in the same audit:** capital-flight money leaked to the other town's importer, and importers' costs were overstated when a payment was clipped.
3. **Specialisation is a bet on the partner:** the best years and the worst shocks both come from it.

### Next step: Stage 9 (technology and long-run growth)

- A research sector, learning by doing, and slow diffusion of technology between firms.
- Experiments: break the Stage 1 plateau; research share against long-run growth.


## Stage 9 · Technology and long-run growth

### What was built

**`technology` layer (`citysim/layers/s9_technology/`)**, working with the Stage 1 planner and with the market economy:
- **What technology does:** a technology level multiplies farm output per worker and divides the labor per loaf in baking and delivery. Fixed crews, maintenance and services don't improve.
- **Research:** a share of the population works on ideas. The frontier grows by 5% × research share per tick, divided by frontier^difficulty (ideas getting harder to find).
- **Diffusion:** users adopt a share of the gap to the frontier each tick. In the market, each firm has its own lognormal adoption speed, and entrants start 10% below their sector's average.
- **Learning by doing:** × (1 + cumulative output ÷ 20,000)^0.05.
- **Planner:** takes the researchers out of the workforce before planning production; everyone still eats.
- **Market:**
  - Research is a sixth occupation. A public institute hires in the labor market, and its wages are paid by the government, or by a levy on households without one.
  - The institute is excluded from entry and exit.
- **Invariants:** the frontier never falls; no firm uses more than the frontier; the planner's physics check uses the improved technology.

### Key results

60 runs, 52,200 ticks; zero invariant violations.

**1. Breaking the Stage 1 plateau** (planner, population growth, land limits, investment; 1,500 ticks):

| Case | Plateau at tick | Population at 1,500 | Technology in use |
|---|---|---|---|
| No technology (Stage 1) | 1,062 | 4.3 M | 1.0 |
| Learning by doing only | 1,317 | 65 M | 2.0 |
| Research 2% | none | 455 M (still +1%/tick) | 4.2 |
| Research 2% + learning by doing | none | 455 M | 8.7 |

**2. Research share and long-run growth** (planner):
- **Constant returns to research:**
  - 1% research delays the plateau to tick 1,306.
  - From 2% to 20%, population grows at the maximum rate (1% per tick) to the end.
  - At 35% the town starves by tick 201: researchers don't grow food.
- **Ideas get harder (difficulty 1):** 2% research only delays the plateau to tick 1,396, and 5% is needed for sustained growth.
- **Malthus at work:** in a subsistence economy, consumption per person stays at need (1.0) whatever the technology. Better technology buys more people, not better lives.

**3. Slow diffusion between firms** (market, 600 people, research 3%, 3 seeds):

| Diffusion per tick | 0 | 0.005 | 0.02 | 0.1 |
|---|---|---|---|---|
| Technology in use | 0.87 | 1.85 | 2.32 | 2.51 |
| Best/worst firm | 1.62 | 1.51 | 1.29 | 1.18 |
| Services per person | 0.53 | 0.77 | 0.83 | 0.87 |
| Real wage | 0.86 | 1.07 | 1.14 | 1.19 |
| Price of services ÷ loaf | 0.55 | 0.68 | 0.73 | 0.76 |

- Without diffusion, average technology falls below its starting level. This simply reflects the rule that entrants start 10% below their sector's average; it isn't a finding.
- Faster spread raises living standards and narrows the gap between firms.
- Services get relatively dearer as bread gets cheaper. This is Baumol's cost disease, and it follows from the modelling choice that technology doesn't speed up services (D-82). The size of the effect is the model's, the direction is assumed.

**4. Research in the market economy:**

| Research share | 0% | 1% | 3% | 6% | 12% |
|---|---|---|---|---|---|
| Services per person | 0.58 | 0.69 | 0.81 | 0.55 | 0 |
| Real wage | 0.86 | 0.96 | 1.18 | 1.19 | 2.3 (collapse) |
| Unemployment | 4% | 3% | 4% | 25% | 82% |

- An inverted U: 3% is best.
- At 6%, technical change outruns the economy's ability to move people and keep demand up, and unemployment reaches 25%.
- At 12%, productivity grows twentyfold in 400 ticks with no central bank. Bank credit can expand money, but it doesn't keep up. Prices fall ~0.5% per tick, firms with sticky wages and debts go bust, the farm sector dwindles to nobody, and the town collapses.

### Surprises

1. **Malthus is alive in the planner:** technology became people, not prosperity.
2. **The market economy can be killed by too much progress:** fast technical change, with no central bank to expand money, gives a deflationary collapse.
3. **The planner-vs-market contrast:** the same research raised living standards in the market town (services, real wages) but only population in the planner's subsistence economy.

### Open questions

- Would a central bank (Stage 6) prevent the high-research collapse by expanding money as productivity grows? Stage 10's modern economy includes one.
- The planner's growth is capped by the demography rule (1% per tick), so it can't show an inverted U in growth itself.

### Next step: Stage 10

Combine every layer into a modern economy calibrated to rich-country stylized facts. Compare central planning, free markets and mixed economies, and switch layers off one at a time.


## Stage 10 · The modern economy and comparing allocation systems

### What was built

- **`stage10_modern.yaml`: every layer at once.**
  - **Layers:** agent households and many firms, services, a bank, a central bank (Taylor rule), a government (taxes, benefits, roads), a stock market (10% trend followers, 50% margin), crop insurance, and research (3% of people) with slow diffusion.
  - **Calibration:** 1,000 people, and productive farms (40 per farmer), so few people farm.
- **`stage10_planned.yaml`: the planned alternative.** It has the same people, technology and research. The planner feeds everyone, invests 30% of free labor and puts 80% of the rest into services (new `planned_services` layer, Stage 1 planner).
- **A smooth necessity rule (`services.necessity_band`).** Since Stage 4, only fully fed households bought services. Bread shortfalls are shared across everyone, so a 1% shortfall made every household stop buying services in the same tick. Services jobs vanished overnight: up to 62% unemployment in single ticks. In the modern scenario, services spending now falls in proportion to the bread shortfall (zero at 30% short). Stages 4–9 keep the strict rule and their published results.
- **Experiments:** systems compared on output, growth, stability, shock resilience and inequality; then every layer switched off in turn.

### Key results

135 runs, 81,000 ticks, 5 seeds; zero invariant violations.

**1. Does it look like a rich economy?** (employment shares in the last 200 ticks)

| | Model | Rich economies today |
|---|---|---|
| Agriculture | 1.6% | 1–3% |
| Manufacturing, repair, construction | 19% | 10–20% |
| Services (incl. retail, public, research) | 69% | the large majority |
| Unemployed | 10% | 3–8% (missed) |
| Money held as bank deposits | 100% | most |

**2. Central planning, free markets and the mixed economy:**

| | Planning | Free market | Mixed |
|---|---|---|---|
| Output (bread + services per person) | 2.30 | 2.68 | 2.43 |
| Instability (sd of output) | 0.02 | 0.16 | 0.33 |
| Unemployment | 0% | 6% | 9% |
| Income inequality (Gini) | 0 | 0.24 | 0.09 |
| Technology at the end | 2.73× | 2.35× | 2.35× |
| Output lost to a harvest failure | 0.2 | 61 | 69 |
| Output lost to a spending slump | — | 48 | 29 |

- **The idealised planner is the steadiest and most equal, and adopts technology fastest.** It knows everything and has no incentive problems, so this is a best case for planning.
- **The markets produce more,** because households choose what to buy beyond bread. Markets also overeat bread (1.25–1.34), which the planner caps at need.
- **The mixed economy trades output for equality.** Its Gini is less than half the free market's, and it handles a spending slump with 40% smaller losses.
- **Its stability, however, is worse than the free market's here.** The government's fiscal rule returns surpluses at once and so cushions little, and the Taylor rule hits zero in every big slump.

**3. Which layer matters most?** (change when switched off, mixed economy)

| Layer switched off | Output | Unemployment | Inequality | Instability |
|---|---|---|---|---|
| Services | −1.35 | +63 pts | +0.33 | −0.16 |
| Banking (with central bank and stock market) | −0.90 | +8 pts | +0.14 | +0.45 |
| Technology | −0.46 | +1 pt | +0.01 | +0.20 |
| Insurance and futures | −0.11 | +0.5 pt | 0 | +0.05 |
| Stock market | −0.10 | +1 pt | +0.18 | +0.05 |
| Government | +0.06 | −2 pts | +0.17 | −0.07 |
| Central bank | +0.28 | 0 | −0.01 | −0.11 |
| Entry and exit | +0.39 | −2 pts | −0.01 | −0.18 |

- **Output:** services and credit are the economy's backbone; without them the modern town falls apart. Technology is the third pillar.
- **Equality:** the government and the stock market matter most. Dispersed shareholding and taxes plus benefits each cut the Gini by about 0.17.
- **Stability:** firm churn (entry and exit) and the central bank, as built here, add volatility. Entrants start small and unproductive, and the rate rule over-reacts to noisy prices.

### Surprises

1. **One rule from Stage 4 caused most of the volatility in Stages 4–10.** "Necessities first" was all-or-nothing, so a tiny shared bread shortfall stopped all services spending. Making it smooth cut unemployment from 11% to 8% and its swings by a third.
2. **Planning wins on stability and equality and loses on output,** even with a perfectly informed planner, because services follow what people want.
3. **A government whose budget balances quickly is not a stabiliser.** The ablation shows its value is in equality, not in smoothing the cycle.

### Open questions

- **Unemployment stays near 10%,** above rich-country levels. Wages adjust slowly and job matching is simple.
- **The planner is idealised.** A planner with information delays or slower adoption would be a fairer comparison.
- **Stabilisers:** a counter-cyclical fiscal rule (run deficits in slumps, surpluses in booms) is the obvious next experiment.

## Stage 11 · The modern town and the internet

### What was built

- **A pre-internet modern town** (`modern_town`): a stock-flow consistent sector model on quarterly ticks.
  - Farms, factories, retail, services, telecom and finance firms.
  - Workers and owners as households; a bank, a central bank (Taylor rule plus stress cuts), a city government with a fiscal rule, an insurer, a stock market and a university whose research drives the productivity trend.
  - Imports and exports: when local factories are sold out, machines and goods come from abroad, and the rest of the world later spends those dollars on the town's exports.
  - It starts in 1976. The first decade is burn-in, and from 1985 the town runs in ordinary business cycles (unemployment 3.5–7%, inflation 2–3%).
- **The breakthrough** (`internet`), from 1993 Q2:
  - Bass adoption by households and businesses; online shopping, internet access and online advertising take budget shares.
  - Startups: seed and venture rounds sized from owners' wealth and the new-economy mood; triage; "get big fast" hiring; network effects; IPO windows; acquisitions; survival mode; failure.
  - A new supply chain: three fibre carriers, built on bank credit and junk-rated debt, with legacy voice revenue, a bandwidth price war, bankruptcy and reorganisation. Corelink, a router and computer maker, sells to carriers, startups and every business that buys IT.
  - Valuations at revenue multiples that swell with the mood. The mood feeds on momentum, novelty (fading as the web goes mainstream), failures of listed firms, rate rises, the fall from the peak, and the "Fed put".
  - Productivity spillovers to the old economy, plus an IT capex boom and hangover.
  - Four scripted historical moments (`history: true`): the 1996 warning, the 1998 rescue cuts, the Y2K liquidity, and the 2003 promise to hold at 1%.
- **The digital twin**: `twin/web.html` ("The Wiring of Grainbridge"), a narrated documentary of one run.
  - Narration uses browser speech in a nature-documentary register. Every number spoken is read from the run.
  - Event-driven chapters, a bubble balloon, branching nerves, carriers with debt bars and bankruptcy stamps, tombstones on Silicon Lane, named winners and a flameout, vital signs with healthy ranges, what-if scenarios and a realism scorecard.

### Key results (5 seeds, `reports/stage_11.html`)

| Stylized fact | History | Model |
|---|---|---|
| Internet stocks peak | Mar 2000 | 2000 Q2 |
| Fall within a year | 75–95% | 88% |
| 1996–2000 startups gone by 2004 | ~50% | 72% ✗ |
| Startups that listed | 10–15% | 11% |
| Policy rate peak / trough | 6.5% / 1% | 6.2% / 1.0% |
| Unemployment low / after the bust | 3.9% / 6.3% | 3.6% / 5.2% |
| Fibre in use at the glut's worst | small fraction | 8% |
| Fall in bandwidth prices | 60–90% | 84% |
| Carriers bankrupt (of 3) | many, 2001–02 | 2.0, first in 2002 Q1 |
| Established shares in the bust | held up | −1% |
| Fibre build-out at peak, share of GDP | ~1.5% | 8.8% ✗ |
| Growth in the recovery 2002–03 | ~2.2% | 1.7% |
| Top portal's share, 2008 | winner-take-most | 47% |

- **After the reviewer's audit** the scorecard was tightened to 18 scored rows (12 match). The 2003 floor of 1% is marked as scripted and not scored. New timing rows expose three misses: cuts too slow in 2001–02 (4.1% at the end of 2002 vs 1.25%), hikes too small in 1999–2000 (+0.4 vs +1.75 points), and unemployment still 3.8% at the end of 2001 (vs 5.7%). The slump comes about two years late.
- **The internet leaves the town 16% richer by 2010,** despite the crash.
- **Leaning against the bubble** (rates to 8.2%) does not stop the mania (price ÷ sales still 88×), and it deepens the bust (unemployment 7.3%).
- **A central bank that never moves** lets the mania run two to three years longer, with hotter inflation (up to ~3.9%) and more startups (568 founded). Its bust is *gentler*: in this model, rising rates are what prick the bubble.

### Surprises

1. **The first versions had no recession at all.** The town ran at full employment with every sector short of capital, so a fall in demand only trimmed excess demand. Two changes created slack: firms do not hire staff they have no machines for, and they import what local factories cannot make. After that, the bust worked through investment (factory jobs −2.5 points), as in 2000–03.
2. **The fibre carriers did not go bankrupt until the price war became persistent.** While prices simply followed utilisation, traffic growth filled the glut within two years. The glut also needed junk spreads that soar when the market sours, and a solvency test lenders apply only once the refinancing window shuts.
3. **Novelty matters for timing.** Without "buzz" fading as adoption matures, the mood stayed euphoric a year after the index peaked, and a second wave of late-funded startups died in 2005–06.

### Open questions

- **Exits are too many** (72% vs ~50%), and survival mode rarely reaches profit.
- **The fibre build-out is about 6× too large relative to GDP.** The town is small and the overbuild is magnified for visibility.
- **The recession is milder and later than in 2001–03,** and Corelink keeps growing after 2001 (its structural IT orders rise with adoption), unlike Cisco.
