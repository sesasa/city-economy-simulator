# City Economy Simulator — Build Spec for Claude Code

Oct 2, 2026 · @Saeed Asadzadeh

## Purpose and working rules

Build a modular simulator of a closed city economy. It starts as four sectors exchanging real goods. Stage by stage, it grows into a model with money, banks, markets, government and finance.

Each stage adds one historical solution to the problem of allocating and distributing resources. The goal is to see what problem each solution fixed and what new failure it introduced.

Working rules for Claude Code:

1. Build only the stage requested. At the end of each stage, run its experiments, report the results and stop. Ask before starting the next stage.
2. Never break earlier stages. With later layers switched off, every earlier scenario must reproduce its earlier results exactly.
3. Every new mechanism is a layer that can be turned on or off in config.
4. Prefer simple, transparent rules over realism. Every behavioral rule is a documented, swappable function whose parameters live in config.
5. Stock-flow consistency is non-negotiable. Every unit of goods and money comes from somewhere and goes somewhere, and the engine checks this every tick.
6. When a design choice is ambiguous, pick the simplest option, record it in DECISIONS.md and continue.

Tech stack: Python 3.11+, numpy, pandas, pydantic for config, pytest, and plotly for HTML reports. A streamlit dashboard is optional from Stage 2 on. Avoid heavy agent-based frameworks such as Mesa unless there is a clear reason.

## Core architecture

The engine is a tick loop over a shared world state. Each economic mechanism is a layer that registers agents, accounts, metrics and invariants, and hooks into fixed phases of the tick.

### Project layout

```
citysim/
  core/
    world.py        # World state: agents, goods stocks, ledger, clock, seeded RNG
    ledger.py       # Double-entry ledger for goods and money
    phases.py       # Phase enum and scheduler
    layer.py        # Layer base class (plugin interface)
    invariants.py   # Conservation and sanity checks run every tick
    metrics.py      # Metric registry and time-series recorder
    runner.py       # Run, sweep, compare, multi-seed
  layers/
    s1_real_economy/
    s2_money_prices/
    ...             # one package per stage
  scenarios/        # YAML scenarios (can inherit from a base scenario)
  reports/          # generated HTML reports
  tests/
  DECISIONS.md      # design choices and why
  STAGES.md         # status and results of each stage
```

### Key abstractions

- **World** holds the tick counter, population, agents or sector aggregates, goods stocks, the ledger and a seeded RNG.
- **Agents** start as sector aggregates in Stage 1. Define a common interface so any aggregate can later be replaced by many individual agents (households, firms, banks) without changing the engine. Each sector has a `representation: aggregate | agents` switch.
- **Goods** start as raw material, final good and capital (factory capacity, tools, land improvements). Each holder has a stock of each good. New goods can be added by later layers.
- **Ledger** is double-entry. Every transaction posts a debit and a credit. Goods and money are tracked separately, and each agent's balance sheet is derived from the ledger.

### Layer interface

```python
class Layer:
    name: str
    requires: list[str]                      # layers this one depends on
    def setup(self, world, config): ...      # register agents, accounts, goods
    def hooks(self) -> dict[Phase, Callable]: ...
    def invariants(self) -> list[Callable]: ...
    def metrics(self) -> list[Metric]: ...
```

### Phase order within one tick

1. Exogenous shocks and policy (weather, technology, government decisions)
2. Production: farming first, then processing
3. Maintenance and depreciation
4. Income: wages, profits, interest, taxes, transfers
5. Distribution and consumption: goods move to consumers and purchases settle
6. Investment and capital formation
7. Financial settlement: loans, repayments, defaults, asset trades
8. Demography and labor: births, deaths, migration, job switching
9. Invariant checks and metric recording

### Invariants (checked every tick)

- **Goods conservation:** for every good, next stock = stock + production − use − consumption − depreciation.
- **Money conservation:** total money changes only through explicit creation or destruction events, such as a bank loan or central bank operation.
- **Balance sheets balance** for every agent and in aggregate.
- **No negative stocks** unless a layer explicitly allows them, as with debt.

A violation stops the run with the tick number and a trace of the transactions involved.

### Config, metrics and experiments

- Config uses pydantic models loaded from YAML. A scenario lists the enabled layers and their parameters.
- Every layer registers its metrics. The recorder writes a tidy table (tick, metric, group, value) to parquet or CSV.
- The runner supports single runs, parameter grids, multiple seeds and side-by-side scenario comparisons, each producing an HTML report.

### Scaling targets

Aggregate mode must be fast enough for large sweeps. Agent mode stores agents as numpy arrays rather than one Python object per agent. Target 100,000 agents for 1,000 ticks in a few minutes by Stage 4.

## Stage 1: Real economy baseline (no money)

A central planner assigns labor to four sectors, and the city collapses, stays stationary or grows depending on farm productivity, fixed overhead and the investment share. This stage reproduces the analytical model and serves as the benchmark for every later stage.

### Sectors

- **Farmers** turn labor (and land) into raw material.
- **Factory** turns raw material into the final good. It needs a fixed crew to run at all, plus labor per unit processed, up to a capacity limit.
- **Distributors** move the final good to consumers, with labor per unit distributed.
- **Maintenance** keeps capital working. Capital that is not maintained depreciates.

Everyone, in every sector, consumes the final good.

### Parameters

| Parameter | Meaning | Default |
| --- | --- | --- |
| `farm_output_per_worker` (a) | Raw units per farmer per tick | 5 |
| `consumption_per_person` (c) | Final units each person needs per tick | 1 |
| `conversion_rate` (e) | Final units per raw unit | 1 |
| `factory_fixed_crew` (F\_p) | Workers needed to run the factory at all | 50 |
| `factory_variable_labor` (v\_p) | Workers per final unit processed | 0.1 |
| `distribution_labor` (v\_d) | Workers per final unit distributed | 0.1 |
| `maintenance_fixed_crew` (F\_m) | Workers needed to maintain one factory | 30 |
| `depreciation_rate` (δ) | Share of unmaintained capital lost per tick | 0.05 |
| `factory_capacity` | Max final units per factory per tick | 1,000 |
| `land_limit` | Farmers before diminishing returns start | 500 |
| `land_exponent` (β) | Strength of diminishing returns | 0.5 |
| `investment_share` (s) | Share of surplus labor used to build capital | 0.3 |
| `storage_spoilage` | Share of stored final goods lost per tick | 0.1 |
| `population_growth` (n) | Growth per tick when fully fed | 0.01 |
| `initial_population` (N0) | Starting population | 150 |

### Rules per tick

1. **Labor requirement.** Target output is N × c. Required workers are farmers N·c/(e·a), factory F\_p + v\_p·N·c, distribution v\_d·N·c, and maintenance F\_m per factory.
2. **Shortage.** If required labor exceeds N, allocate by a configurable priority order (default: farming, factory, distribution, maintenance). Consumption then falls short of N × c, and unmaintained capital depreciates.
3. **Surplus.** Leftover labor is the surplus. A share s goes to investment. The rest is leisure, which is recorded as a welfare metric.
4. **Investment.** Investment labor builds capital. Each capital type has a cost in labor-ticks and an effect: land improvements raise a, tools lower v\_p or v\_d, and a new factory adds capacity but also needs its own F\_p and F\_m.
5. **Investment allocation.** By default, invest in the current bottleneck sector, meaning the one closest to its limit. Make this a swappable rule.
6. **Diminishing returns.** When farmers exceed `land_limit`, effective productivity is a × (land\_limit / farmers)^β.
7. **Demography (toggle).** If consumption per person is at least c, population grows at rate n. If it is below c, population shrinks in proportion to the shortfall. With the toggle off, population stays fixed.
8. **Storage (toggle).** Unconsumed final goods can be stored and lose `storage_spoilage` per tick.

### Analytical benchmark

The minimum viable population is the fixed overhead divided by the share of labor left after variable needs:

```latex
N_{min} = \frac{F_p + F_m}{1 - \frac{c}{e\,a} - v_p\,c - v_d\,c}
```

With the defaults this gives 80 / 0.6 ≈ 133.3 people. The simulation must match this within one person. Viability also requires the denominator to be positive, which sets a minimum farm productivity.

### Experiments

1. Sweep N0 from 50 to 400 with fixed population. Show collapse below N\_min, a stationary state at N\_min and a growing surplus above it.
2. Sweep a from 1 to 10. Find the minimum farm productivity for viability at several population sizes.
3. Sweep s from 0 to 0.8. Show that s = 0 gives zero growth even when a surplus exists, and plot growth against s.
4. Long run with demography and land limits on. Show growth slowing to a plateau.
5. Shock: cut harvests by 30% for 3 ticks, with and without storage. Measure the consumption drop and recovery time.

### Metrics

Population, labor share per sector, surplus share, consumption per person, leisure per person, capital stock by type, total output, growth rate, maintenance share of labor, and effective farm productivity.

### Done when

All invariants pass across every sweep. N\_min matches the formula. Zero investment gives zero growth. The long run reaches a plateau. The report `reports/stage_1.html` shows all five experiments.

## Roadmap of later stages

Each later stage adds one mechanism people invented to allocate and distribute resources, roughly in historical order. Each stage should be motivated by a problem visible in the previous stage's results, and should report the new failure modes it introduces. The details below are a starting point. Refine them with me before building each stage.

### Stage 2: Barter, money and prices

**Solves:** the planner needs perfect information. Markets let decentralized decisions coordinate through prices.

- Replace the planner with households that choose occupations by expected income, and sector firms that set prices and wages.
- Run barter mode first and measure failed trades, which shows the double-coincidence-of-wants problem. Then add a money token.
- The factory buys raw material at price p\_r and sells the final good at p\_f. Prices adjust toward target inventories, and wages adjust toward labor demand.
- From this stage on, generate the stock-flow consistent transaction matrix and balance-sheet matrix every tick, following the Godley–Lavoie approach.
- **New failures:** price oscillations, unemployment, inventory gluts, deflation when money is scarce.
- **Experiments:** compare planner and market allocation on output, welfare and time to settle; vary the money stock and measure velocity and price levels.

### Stage 3: Credit and banking

**Solves:** investment no longer has to wait for someone to save first.

- A bank takes deposits and makes loans. Loans create deposits, so money supply becomes endogenous.
- Investment is financed by loans. Add interest rates, collateral, default rules and a bank capital requirement.
- **New failures:** credit booms and busts, bank runs, debt deflation.
- **Experiments:** credit limits against growth; a Minsky-style cycle of rising leverage; bank failure after a harvest shock.

### Stage 4: Many firms, competition and specialization

**Solves:** one giant factory has no competitive pressure and no room for new kinds of work.

- Split the factory into many firms with entry and exit. Add new sectors such as tool-making and services.
- Firms have owners who receive profits. Add a labor market with job search.
- Switch households to agent mode and track inequality (Gini for income and wealth).
- **New failures:** monopoly pricing, concentration from increasing returns, inequality, structural unemployment.
- **Experiments:** monopoly versus competition; how many firms survive under different fixed costs; when new sectors emerge.

### Stage 5: Government and public goods

**Solves:** maintenance and infrastructure are underprovided when no single firm benefits enough to pay for them.

- Taxes on income, profits or sales. Public infrastructure and maintenance. Transfers such as unemployment benefits. Public debt.
- **New failures:** persistent deficits, crowding out, misallocated public spending.
- **Experiments:** tax-funded versus private maintenance; benefits as automatic stabilizers during a harvest shock.

### Stage 6: Central bank and monetary policy

**Solves:** private banking alone produces unstable money and credit.

- A central bank sets the policy rate, holds bank reserves and acts as lender of last resort. Implement an interest-rate rule such as a Taylor rule.
- **New failures:** inflation, the zero lower bound, moral hazard from bailouts.
- **Experiments:** a Taylor rule versus a fixed money stock under supply and demand shocks.

### Stage 7: Financial markets and risk sharing

**Solves:** savers and investors need ways to share risk and own productive capital without running it.

- Shares in firms, bonds, and a market with different trader types (fundamental traders and trend followers).
- Leverage and margin rules. Insurance and harvest futures for farmers.
- **New failures:** bubbles, fire sales, leverage spirals.
- **Experiments:** whether harvest futures smooth farmer income; margin requirements against bubble size.

### Stage 8: Trade between cities

**Solves:** one city cannot produce everything efficiently.

- Several cities with different land, skills and technology. Trade with transport costs, migration, and either separate currencies with exchange rates or a shared currency.
- **New failures:** trade imbalances, capital flight, dependency on a single partner.
- **Experiments:** gains from comparative advantage; floating exchange rates versus a currency union under a regional shock.

### Stage 9: Technology and long-run growth

**Solves:** the Stage 1 plateau, where land limits and maintenance costs stop growth.

- A research sector, learning by doing, and slow spread of new technology between firms. Optional resource and energy limits.
- **Experiments:** break the Stage 1 plateau; research share against long-run growth.

### Stage 10: The modern economy and comparing allocation systems

**Goal:** combine all layers into a modern scenario and compare whole systems.

- Calibrate loosely to stylized facts of today's rich economies, treated as approximate targets: agriculture around 1–3% of employment, manufacturing around 10–20%, services the large majority, and most money held as bank deposits.
- Compare central planning, free markets and mixed economies on output, growth, stability, resilience to shocks and inequality.
- Report which layer contributes most to each outcome by switching layers off one at a time.

## Outputs, tests and the first message

Every stage ends with a reproducible report, passing tests and a short written summary in STAGES.md.

### Outputs per stage

- `reports/stage_N.html` with charts of the key metrics and parameter sweeps. From Stage 2 on, include the transaction and balance-sheet matrices for a sample tick.
- Command line: `python -m citysim run <scenario.yaml>`, `python -m citysim sweep <scenario.yaml> --param a=1:10:0.5`, and `python -m citysim compare <a.yaml> <b.yaml>`.
- An optional dashboard from Stage 2 on, with sliders for the main parameters and live charts.
- A STAGES.md entry covering what was built, key results with numbers, surprises, open questions and a proposed next step.

### Tests

- Unit tests for every rule function.
- Invariant checks on every scenario and every sweep point.
- Regression tests: each stage's base scenario saves golden outputs at a fixed seed. Later stages, with their layers off, must reproduce them.
- Analytical tests wherever theory gives an answer, such as N\_min, steady states and simple spending multipliers.

### Definition of done for any stage

Tests pass, the report is generated, STAGES.md is updated, no invariant fails anywhere in the sweeps, and runtime targets are met.

### First message to give Claude Code

Save this document in the project root as `SPEC.md` (export it as Markdown), then send:

```
Read SPEC.md. Set up the project skeleton and the core architecture
(world, ledger, phases, Layer interface, invariants, metrics, runner).
Then implement Stage 1 only. Run the five Stage 1 experiments, generate
reports/stage_1.html, write the STAGES.md entry, and stop.
Ask me before starting Stage 2.
```

For each later stage, send: `Read SPEC.md and STAGES.md. Propose a detailed design for Stage N, wait for my approval, then build it.`
