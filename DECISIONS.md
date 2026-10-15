# Design decisions

Choices the spec left open, the option picked, and why. Rule of thumb (SPEC rule 6): the simplest option that keeps the mechanism transparent. Each entry names the config knob, if any, that lets you swap it.

## Project and engine

**D-01 Layout.** `citysim/` is the Python package (`core/`, `layers/`, `experiments/`, `config.py`, `cli.py`, `report.py`). `scenarios/`, `reports/`, `tests/`, `DECISIONS.md` and `STAGES.md` sit at the project root next to `SPEC.md`. The spec's tree shows them all under `citysim/`; keeping data and docs out of the importable package is the usual Python layout. `SPEC.md` is a copy of the build-spec file that was in the directory.

**D-02 Environment.** A project-local virtualenv (`.venv`, Python 3.13) with numpy, pandas, pydantic, pyyaml, pytest, plotly and pyarrow, so the user's base conda environment is untouched. Dependencies are listed in `pyproject.toml`.

**D-03 Agent representation.** Every sector and the household population is an `AgentGroup`. Stage 1 implements only `aggregate` (one record per group, with a float head count). The `representation` switch exists per sector in config, but `agents` raises `NotImplementedError`. The `ArrayGroup` placeholder stores attributes as numpy arrays, as the spec requires, ready for Stage 4. Population is continuous (float) in aggregate mode.

**D-04 Ledger conventions.** A holder's balance is its stock: debit raises it, credit lowers it. Goods enter through an external source account (`ext:production`) and leave through external sink accounts named after the flow (`ext:use`, `ext:consumption`, `ext:depreciation`, `ext:spoilage`, `ext:waste`, and for money later `ext:creation` and `ext:destruction`). Every posting carries a flow tag, and the ledger rejects postings whose direction doesn't match their tag. Layers never touch balances directly.

**D-05 Invariants.**
- Engine-level checks, run every tick:
  - double-entry identity: every item sums to 0 over all accounts;
  - goods conservation per good, computed from the opening snapshot plus the journal grouped by flow tag;
  - money conservation;
  - no negative stocks;
  - balance sheets.
- Layer-level checks: labor adds up to population, the factory condition list matches the ledger, and no household consumes more than N·c.
- Stage 1 has no financial claims. The balance-sheet check therefore reduces to "assets − liabilities = net worth" per agent, plus "internal money nets to outside money"; it becomes substantive in Stage 2.
- Tolerance is 1e-9, relative.
- A violation raises `InvariantViolation` with the tick, the failed checks and the postings involved.

**D-06 Layer communication.** Layers talk through three channels:
- `world.vars`: a per-tick blackboard, cleared each tick, documented in `real_economy.py`;
- explicit extension points on the base layer: `demand_providers`, `supply_sources`, `leftover_sinks`, `land_limit`;
- the ledger itself.

Hooks run in phase order, and within a phase in the order the layers are listed in the scenario. A layer must be listed after the layers it `requires`.

## Stage 1 economics

**D-07 Maintenance covers every existing factory, idle or not.** I tried maintaining only the factories in use. A newly built reserve factory then rotted before it was needed and had to be rebuilt, over and over. Maintaining every factory follows the spec literally (F_m per factory) and makes reserve capacity visibly costly. Factories whose condition falls below `scrap_condition` (default 0.01) are scrapped, so dead capital stops demanding a maintenance crew.

**D-08 How capital works.** The spec gives effects but no functional forms or costs. Choices:
- Land improvements: a_x = a·(1+L)^0.25.
- Tools: v_eff = v·(1+T)^−0.25, separately for processing and distribution.
- Upkeep: 1 maintenance worker per unit of land improvement or tools.
- Costs: 200 labor-ticks per unit; 1,000 labor-ticks per factory.
- A factory is the integer unit. Its *condition* in [0,1] scales its 1,000 capacity, and the ledger good `factory` is the sum of conditions.
- Construction accumulates as `factory_wip`.
- The land-improvement elasticity (0.25) is deliberately below the land-limit exponent β = 0.5, so land improvements can postpone the land wall but not remove it. With an elasticity ≥ β, farm labor per person would keep falling and there would be no plateau.

Config knobs: `real_economy.*_elasticity` / `*_maintenance` and `investment.*_cost`.

**D-09 Shortage rule.** The default is `priority` (spec): sectors are filled in full in order, farming → processing → distribution → maintenance. A small shortage therefore lands entirely on maintenance: everyone is still fed and capital quietly decays. The alternative `max_feasible` cuts the consumption target to what labor can sustain while keeping capital maintained. The two give the same N_min but behave very differently with demography (see STAGES.md).

Under `max_feasible`, when the town is too small to crew and repair the factory (30 < N < 80 with defaults), the planner bakes nothing and leaves the rest idle rather than run an unrepaired factory. That is deliberate: this planner never lets capital decay. The planner targets min(N·c, factory capacity).

Config knobs: `real_economy.shortage_rule`, `real_economy.priority`.

**D-10 Investment rule ("bottleneck", default).**
- Factory capacity is a hard limit. If need ÷ capacity ≥ 0.9, or a factory is half built, build whole factories until utilization is back below 0.9.
- Otherwise labor is the limit. Invest in the capital that relieves the sector using the most labor (farming → land improvement, processing → processing tools, distribution → distribution tools), but only if it saves more labor than its own upkeep.
- Budget left after a project goes to the rule's next pick. Labor that no project wants becomes leisure.
- Alternative: `best_return`, which picks the largest net labor saving.

Config knobs: `investment.rule`, `investment.bottleneck_threshold`.

The labor each sector "uses" for this ranking:
- **Processing:** variable labor only, because tools cannot shrink the fixed crew.
- **Farming:** all farm labor, including labor growing wheat for the granary, because land improvements raise every farmer's harvest.

The review agent noted this tilts the ranking toward farming. It is kept as a deliberate choice.

**D-11 Harvest shocks are unanticipated.** The planner assigns farmers for the normal harvest. The shock multiplies realised raw output only. Raw material the factory cannot process is wasted.

**D-12 No raw storage. Final goods are wasted unless the storage layer is on.** Unused raw material and final goods are posted as `waste` at the end of the tick.

**D-13 Storage policy.** When storage is on, the planner keeps a store of `storage_target_ticks` × N·c (default 1 tick). The rules:
- It plans extra output each tick to bring the store back to target after this tick's spoilage.
- Restocking comes after maintenance and before investment in labor priority.
- Restocking needs farm and factory labor but no distribution labor. Distribution labor is planned for N·c, so draws from the store are covered.
- Consumption is served from fresh output first, then from the store.
- The store starts empty.

**D-14 Demography.** Population is N' = N(1+n) if consumption per person ≥ c, otherwise N' = N(1 − μ·(1 − cons/c)), with μ = `starvation_rate` = 0.1. The spec says "shrinks in proportion to the shortfall" but gives no constant. With μ = 1, a 30% shortfall would kill 30% of people in a single tick. Growth is all-or-nothing, as in the spec.

**D-15 Phase placement of the planner.** Labor allocation is a policy decision, so it runs in Phase 1, before production. Depreciation and spoilage run in Phase 3, distribution and consumption in Phase 5, investment in Phase 6 (it uses capital *after* this tick's depreciation), demography in Phase 8. Labor is perfectly mobile between sectors every tick.

**D-16 Growth metric.** With a fixed population, output is pinned at N·c, so "growth" is measured on **potential output**: the largest final output the city could sustain with all its labor, its current capital fully maintained, and no investment, capped by factory capacity. `growth_rate` is its tick-on-tick change. With demography on, population growth is reported too.

**D-17 Viability.** A run is viable when there is no labor shortfall at any tick and consumption per person is ≥ c at every tick. Simulated N_min and a_min are found by bisection on this criterion, which is monotone.

**D-18 Determinism.** Stage 1 is deterministic: the seeded `numpy` RNG exists in `World`, but no Stage 1 rule draws from it. A test checks that different seeds give identical output.

**D-19 Reports.** Plotly is embedded in each HTML file, so reports work offline (~5 MB each). The colors are a validated colorblind-safe categorical palette in fixed order. The CLI's `run` writes the tidy metric table (tick, metric, group, value) as CSV next to the HTML. `core.metrics.write_frame` also writes parquet when the path ends in `.parquet`.

## Town twin

**D-20 Twin engine in JavaScript, held to the Python model.** The animated town (`twin/`) runs a JavaScript port of the Stage 1 rules so it can react live to buttons and sliders in a browser. It replays Python's float summation (Neumaier, as in Python ≥ 3.12's `sum`) so the two give the same numbers. A parity test runs both on 9 scenarios and fails if any metric differs by more than 1e-9 (relative, or relative to population for labor quantities). When a later stage changes Stage 1 behaviour, the twin must be updated or this test fails.

**D-21 Twin presentation.** Ticks are shown as years. Each figure stands for a round number of people (1, 2, 5, 10, ...) so at most ~90 figures are drawn. The "hands needed" panel compares population with this year's total required labor, which stays exact with many factories and crowded farmland. The page is single-file HTML plus `engine.js`; `python -m citysim twin` inlines the engine for offline use.

## Stage 2 · Barter, money and prices

**D-22 Scope approval.** The user asked for the full civilization simulation and to proceed on best guesses, consulting a review agent when in doubt. Stages are still built one at a time with experiments and reports, but without waiting for approval between them.

**D-23 The market is a separate base layer.** `market` and `real_economy` both provide `economy`. Generic layers (demography, land limits, harvest shocks) find the active base through `world.state["economy"]`. The planner-only layers (investment, storage) require `planner`. This keeps the Stage 1 golden outputs exact.

**D-24 Households.** In aggregate mode there is one household agent per occupation, with the numbers held in numpy arrays and their totals tied to the ledger's `households` account by an invariant. Agent mode (many households) arrives in Stage 4 on the same arrays.

**D-25 Pricing.** Normal-cost pricing: price = (1 + markup) × unit cost at the volume that feeds the town, at current wages and input prices. The markup moves with the inventory gap (or the order gap, for repairs) and drifts back to a normal 20%.
- Pure inventory-based prices had no anchor and ran away.
- Average-cost pricing on actual sales collapsed when sales fell, because the fixed crew was spread over almost nothing.

**D-26 Wages and jobs.**
- Wages rise with vacancies and fall only when an occupation's unemployment exceeds 5% (frictional).
- Job switchers choose occupations in proportion to jobs on offer × expected income³, with 10% reconsidering each tick.
- Builders are temporary work for the unemployed and do not count as vacancies.

**D-27 Payment timing.** Wages are owed when people are hired and paid from cash after sales, pro rata if a firm is short; unpaid wages are recorded as arrears. Paying up front let a firm with an empty till stop hiring for good.

**D-28 Profits and investment.**
- Each tick, a share s = 0.3 of last tick's profit is set aside for building; the rest is paid out as dividends.
- Building money not used is paid out the next tick, and cash beyond a two-tick buffer is paid out at 50% per tick.
- Without this, firms hoarded money and households could not buy back what they produced.
- The investment choice uses Stage 1's rules. Each firm builds the capital it owns, using unemployed people from its own occupation.

**D-29 Consumption.** Households spend 90% of income and 10% of savings. They first try to afford their need (c per person) and never buy more than 1.5c each.

**D-30 Barter.** Barter is modeled as search friction. With n kinds of goods, a random meeting is a double coincidence with probability 1/(n−1)², and after K meetings a desired trade happens with probability 1 − (1 − 1/(n−1)²)^K. That share applies to every exchange: hiring (wages in kind), wheat, wholesale bread, repairs and retail. Trades are still recorded in a unit of account, but nothing works as money: the double coincidence is the only thing that differs.

**D-31 Starting state.** The market starts from the planner's allocation, with prices set at cost plus a 10% markup. Money is sized so firms hold two ticks of costs and households one tick of income, scaled to `initial_money` when that is given. `prices_match_money: false` starts prices at the usual level whatever the money stock, for the money-scarcity experiment. `initial_allocation: equal` is the "start from scratch" experiment.

## Stage 3 · Credit and banking

**D-32 Money and claims.** With a bank, everyone's cash becomes deposits and the bank holds it as reserves. Deposits and loans are `claim` items: positive for the holder, negative for the issuer. The money supply is deposits held outside the bank plus cash households hoard after a run. Outside cash only moves between households and the bank.

**D-33 Bank income.** Depositors get half of each tick's loan interest; the bank keeps the spread. Dividends are half of the tick's profit, and only while bank capital exceeds 15% of loans. Earlier versions paid a fixed deposit rate and paid dividends out of starting capital, and bled the bank dry.

**D-34 Borrower limits.** The limit is the smaller of:
- 0.5 × collateral (capital at replacement cost plus stocks) + 1 tick of revenue;
- what 15% of revenue can service at loan rate + repayment.

Revenue here is the larger of recent average revenue and expected sales × price, so a stalled firm can restart. The whole limit is then multiplied by `credit_scale` and by (1 + optimism). Lending against collateral alone let a firm's limit grow with the very project it borrowed for.

**D-35 Repayment.** Interest of 1% per tick plus 5% of principal, and revolving: cash beyond one tick of running costs repays early, so working capital self-liquidates (a monetary circuit). A firm with debt pays dividends only from cash net of debt and the next payment due.

**D-36 Default.** A borrower in arrears by more than 10% of its debt defaults. The bank seizes deposits up to 40% of the debt, writes off the rest, and cuts the borrower off for 20 ticks. The sector firm keeps operating: firms can't exit until Stage 4.

**D-37 Runs and resolution.** Households run when bank capital falls below 4% of loans, withdrawing 30% of deposits per tick as cash, and return 10% per tick once capital is at least 6%. The bank is resolved if it can't meet a run or its equity turns negative: depositors are bailed in pro rata so that equity covers the capital requirement, and lending is frozen for 10 ticks. There is no deposit insurance or lender of last resort yet; those arrive in Stages 5 and 6.

**D-38 Minsky.** Optimism rises by `procyclicality` per tick while there are no defaults and falls by procyclicality × 20 × the default rate, capped between 0 and 3. Borrower limits scale by (1 + optimism). Firms' boldness (optimism ÷ 3) lowers both the factory-building threshold and the hurdle rate for borrowed projects.

## Stage 4 · Many firms, competition and specialization

**D-39 Agent market as a subclass.** `agent_market` extends `market`, inheriting every Stage 2–3 rule, and provides `economy`, `market` and `agents`, so Stage 3's bank works with it unchanged. The household arrays keep Stage 2's names (`size`, `sector`, `cash`, `income`, `hunger`), plus `employer` and `retrain`.

**D-40 Who gets paid.** Wages go to the firm's employees; builders' wages go to idle, non-retraining people of the firm's occupation; dividends and liquidation cash go to the firm's owners, equally, or everyone per head under public ownership.

**D-41 Ownership is spread.** Each firm has 20 owner households by default. One owner per firm made the economy collapse through owners' inability to spend, which is kept as an experiment. A founder who enters is among the new firm's owners.

**D-42 Market power.** Normal markup = 0.1 + 0.5 × the Herfindahl index of the sector's sales. Markups still move with inventories and drift back to that normal level, as in Stage 2.

**D-43 Necessities first.** Households spend on services only once they got all the bread they need.

**D-44 Rational job switching.** A switcher moves only if expected income in the target occupation × (1 − retraining ticks ÷ 50) exceeds its current occupation's. Without this, long retraining produced endless futile churn and collapse.

**D-45 Exit is an acquisition.** A closing firm's plant, stocks, workers and expected customers go to the sector's largest firm at zero price. Its debts default and its cash goes to its owners. Firing everyone instead triggered liquidity collapses.

**D-46 Entry.**
- When a sector's smoothed margin exceeds 20%, there is a 5% chance per tick that a household able to pay in 4 ticks of a starting crew's wages founds a firm.
- An empty services sector counts as attractive.
- Factory entry is off, because a plant costs 1,000 labor-ticks: a barrier to entry.
- Entrants expect a fair share of the sector's sales.

**D-47 Randomness.** Stage 4 is the first stage to use the seeded RNG: for shuffling, matching, job switching, births and deaths, and owner choice. The same seed gives the same history (tested). Experiments average 3 seeds.

**D-48 Empty town.** If everyone dies, their savings stay on the households account as an estate; the household-money invariant is skipped once there are no agents.

## Stage 5 · Government and public goods

**D-49 Hooks, not rewrites.** The government acts through market extension points:
- `profit_tax`: called before dividends;
- `maintenance_payer`: the government pays repair shops for everyone's upkeep;
- `road_factor`: scales distribution labor in the technology;
- `money_holders`: the government's deposits count in the money supply.

Each is a no-op without a government, so Stage 2–4 golden outputs are unchanged.

**D-50 Order within the income phase.** Market wages and dividends (with profit tax) come first. Then the government, in order: public works (it hires idle people and pays their wages), income tax on all income received this tick, unemployment benefits to those still without work, and the basic income.

**D-51 Debt.** Bonds are claims sold only to the bank, which pays with new deposits. They carry a capital risk weight (default 1), so they compete with loans for bank capital. Interest is 0.5% per tick. The debt ceiling is 30 ticks of average tax revenue; above it the government can only spend what it has. Without a bank the government must balance its budget.

**D-52 Fiscal rule.** Money beyond 2 ticks of spending first repays debt, then 50% of the excess per tick is rebated per head. Without it, surpluses drained money from circulation.

**D-53 Spending priority when funds are short.** Bond interest first, then benefits, public maintenance, road upkeep and road building.

**D-54 Waste.** A share `waste` of public road workers' labor builds and maintains nothing. They are still paid.


## Stage 6

**D-55 How the policy rate reaches the economy.** Through three channels:
- the bank lends at policy + 0.7% (0.3% + 0.7% = the old fixed 1%);
- government bonds pay the policy rate;
- households' spending propensities are scaled by 1 − 15 × (rate − neutral), with the multiplier bounded to [0.5, 1.5] and each propensity capped at 1.

The hurdle for borrowed investment follows the loan rate. Without a central bank everything is exactly as before (goldens unchanged).

**D-56 Taylor rule settings.**
- Coefficients: 1.5 on inflation; 0.05 on unemployment above u* = 4% (the measured normal level of the Stage 5 economy); smoothing 0.5.
- Inflation is a smoothed per-tick change in a 70/30 bread/services price index.
- The target is 0%, because the base economy drifts slightly deflationary.

**D-57 Fixed money stock (monetary targeting).** The money supply should follow M0 × (1 + g)^t.
- **Above the path,** the rate rises by 0.1 per log point of excess.
- **Below it,** the shortfall is paid out as new money per head, in FINANCE after the bank settles.
- **Why not cap lending at the path:** a first version did that. Money sat exactly on the path, so the cap left zero room and credit stopped completely (found by the review agent).

**D-58 Fractional reserves.** The starting money was all cash, so the bank began with 100% reserves and no run could drain it. A reserve ratio (base 20%) swaps the excess for central bank bills: safe and counted as capital, but unusable in a run. These bills pay no interest.

**D-59 Lender of last resort, Bagehot style.**
- It lends only to a solvent bank, at the policy rate + 0.5%.
- The loan is repaid 20% per calm tick, out of reserves.
- It is senior in a resolution: depositors are haircut so that the central bank is paid in full.
- The bank settles interest and repayment *before* its own solvency check (`bank_obligations` hook). Otherwise interest pushed a just-resolved bank below zero every tick.

**D-60 Resolution capital.** A failed or bailed-out bank is left with capital equal to the requirement on max(risk assets, 10% of deposits). The previous target, the requirement on risk assets, was zero for a bank with no loans: a zombie that could never lend again. With working-capital credit gone, output fell to zero. The insolvency check has a 1e-9 tolerance, so rounding no longer triggers repeated "failures".

**D-61 Moral hazard.** It is modelled as two behaviours when bailouts are expected (`moral_hazard` m):
- depositors never run;
- the bank's panic sensitivity to defaults is scaled by (1 − m).

Surprise bailouts (m = 0) keep runs and normal caution.

**D-62 Panic.** A scheduled rumour makes depositors withdraw `panic_withdrawal` (40%) per tick whatever the capital ratio. This is the clean test of a solvent-but-illiquid bank.

**D-63 Demand shocks.** A scheduled multiplier on households' spending propensities (`spending_multiplier`), reset each tick by the market and scaled by later layers (demand shocks, the interest-rate channel).

**D-64 Policy warm-up.** All rules hold the rate at neutral for the first 50 ticks. The start-up price transient (prices find their level in the first 10–20 ticks) otherwise drove the Taylor rule to 8% per tick.

**D-65 Emergency-loan interest** that reserves can't pay is added to the debt. The bank counts it as a cost before paying dividends.

**D-66 Stage 6 resolution rules apply only with a central bank.** The restart capital and the insolvency tolerance apply only when a central bank layer exists, so Stages 3–5 keep their published behaviour.

## Stage 7

**D-67 One index instead of per-firm shares.** All firms' equity is one index receiving all dividends.
- Per-firm stocks would add entry and exit bookkeeping without changing the bubble and leverage mechanics.
- Shares are a ledger claim (+ for households, − for an "equity" issuer account).
- Trades between households move money inside the households account.

**D-68 Listing split.** At tick 20 the index is split so that, at fair value, shares equal `base_share/(1 − base_share)` × households' money.
- **Why:** trades need money. If the market's fair value is many times the money stock, the price is pinned by the cash constraint, not by value (found by the review agent).
- **Fair value:** average dividends ÷ (policy rate + 5% premium).

**D-69 Trader rules.** Each trader type sets a desired share of wealth θ and trades 20% of the gap to it per tick.
- **Value traders:** θ = 0.5 + 1 × ln(V/P), clipped to [0, 1].
- **Trend followers:** θ = 0.5 + 100 × conviction × trend, clipped to [0, 1/margin].
  - conviction = 1 / (1 + (ln(P/V)/0.5)²);
  - trend = EMA(0.2) of log returns.
- **Why conviction is needed:** without it, leveraged trend-following explodes, because collateral rises with the price.

**D-70 Price formation and rationing.**
- **Price rule:** P × exp(0.1 × (B − S)/(B + S + 1% of shares)): bounded, and moved by imbalance relative to volume.
- **Rationing:** forced sales fill first.
- **Funding:** buy orders the bank can't fund are not placed.

**D-71 Margin.**
- **Margin calls:** shares are sold so that debt ≤ (1 − m) × value.
- **Interest:** unpaid interest compounds as a loan.
- **Default:** a borrower with no shares left defaults on what cash can't cover.

**D-72 Futures settle against the farm's own wheat price after 4 ticks.** If the farm can't pay, the contract is dropped: a counterparty loss to the speculators. Insurance pays claims even if a premium was short.

**D-73 Medians across seeds** for the stock market and futures experiments. Slumps make dividends, and so fair value, collapse, which would dominate maxima and means. Tail measures are also reported as means.


## Stage 8

**D-74 Cities are separate simulations stepped in lockstep.** Each keeps its own ledger and invariants. Cross-border flows use `export` and `import` flows, and a region check matches them. This needed no change to any city layer beyond an import seller list.

**D-75 Importers are trading houses, not firms.**
- They are not in the market's firm list (no labor, wages, entry or exit). They sell in the wheat and bread wholesale markets and restock after retail closes.
- Expectations: when importers are present, buyers record orders by price (choosiness 4), and local firms lose only the orders an importer can fill from stock. Equal splitting made local bakeries cut output by a third because of an importer with empty shelves.

**D-76 Arbitrage rule.** Merchants import only if landed cost × 1.02 is below the average local price. Without it, identical towns cross-hauled.

**D-77 Importer finance.**
- **Price:** never below landed cost.
- **Working capital:** owners keep about two ticks of purchases in the business, topping it up from household money (20% of the shortfall per tick) and taking out the excess.
- **Why:** an importer that could sell below cost and could not be refinanced ran dry and starved the town it supplied.

**D-78 Floating rate: a clearing market.**
- **Each tick:**
  1. Importers place currency orders.
  2. Orders to buy a currency are filled as far as orders to sell it allow, plus up to 20% of a dealer's float.
  3. The currency in excess demand appreciates by exp(0.1 × (demand − supply) ÷ (demand + supply)).
  4. Conversions carry a 0.5% spread.
- **Why it replaced the first design:** that design had a dealer whose rate followed fixed inventory targets. It lost money on every swing and ran dry (review finding).
- **What it means:** with no capital flows, trade must nearly balance each tick.

**D-79 Union payments move reserves.** The payer's deposits are destroyed at its bank and the cash reserves are exported; the receiving bank imports the reserves and credits deposits. A town that runs out of reserves can't pay.

**D-80 Migration and capital flight are simple share rules.**
- **Migration:** 5% × (relative real-income gap − 10%) of the town per tick.
- **Capital flight:** 20% × unemployment gap of households' money per tick; 5% per calm tick comes back.

**D-81 Fiscal union.** Moves 10% of the gap in money per head between towns' households each tick. It is a stand-in for federal taxes and transfers.


## Stage 9

**D-82 What technology improves.** Farm output per worker, and variable labor per loaf in baking and delivery. Fixed crews, maintenance and services stay as they are: services are personal and labor-bound, which gives Baumol's cost disease.

**D-83 Research production function.** The frontier grows by 5% × (researchers ÷ population) ÷ frontier^difficulty per tick, with constant returns to the research share by default.

**D-84 Researchers in the planner** are subtracted from the workforce before planning (`make_plan(eaters=N)`). The plan records the population it feeds (`Plan.people`), so the physics and consumption checks still use the whole population.

**D-85 Research in the market is a public institute.**
- It is a sixth occupation and hires through the normal labor market at market wages.
- The government funds its wage bill (borrowing if needed), or households pay a levy without a government.
- It is excluded from entry and exit, and its first researchers leave their old jobs.

**D-86 Diffusion is heterogeneous.** Each firm's adoption speed is lognormal (sd 0.5) around the mean, drawn from the scenario seed. Entrants start 10% below their sector's average adoption.

**D-87 Learning by doing** is firm-specific in the market (each firm's own cumulative output) and economy-wide under the planner.


## Stage 10

**D-88 Services necessity band.**
- **Default (0):** the Stage 4 rule, where only fully fed households buy services.
- **Modern scenario (0.3):** services spending is scaled by 1 − hunger ÷ 0.3.
- **Why:** under the strict rule, a shared 1% bread shortfall stopped all services at once. The band applies only to the modern scenario, so earlier goldens are unchanged.

**D-89 Modern calibration.**
- Farm output 40 per farmer.
- Baking: 0.06 workers per loaf plus a crew of 30 per factory. Delivery: 0.06 per loaf. Maintenance crew: 20.
- 1,000 people; 10% trend followers (fewer than the Stage 7 bubble experiments).

**D-90 Planned services.** The planner puts 80% of the labor left after investment into services, at 0.5 workers per unit (the market's rate), so that both systems can produce services.

**D-91 Comparison measures.** Output is bread + services per person. The planner caps bread at need, and leisure isn't counted. Shock losses compare a seed against itself without the shock over 80 ticks. In unstable runs, chaotic divergence adds to these.

**D-92 Government debt repayment speed (`repay_speed`).** A new parameter, default 1 (unchanged). Tested at 0.05 for the modern economy; it didn't help, so it is left at 1.


## Stage 11

**D-93 Narration voice.** Browser speech synthesis, preferring British English voices, with a documentary-style script. No cloning or imitation of any real narrator, and the page says so.

**D-94 Start and burn-in.** The town starts in 1976 at a solved equilibrium. The twin and the "steady state" facts start in 1985, after the initial transient.

**D-95 Hiring is capped by capital** ("no staff without machines"), and **unmet demand for machines and goods is imported** (paid into the rest of the world's deposits). Foreigners recycle 8% of those deposits a quarter into exports, from spare capacity only. GDP = C + I + G + X − M. Without these, the economy was permanently supply-constrained and demand shocks did not move employment.

**D-96 New-economy investment counts in GDP.** Carriers' fibre and routers and startups' servers are booked as investment. Business IT is bought from the router maker when it exists.

**D-97 Carriers.**
- Appetite 0.25 / 1.0 / 1.6 and legacy weights 2 / 1 / 1, so the incumbent is cautious and keeps a cash reserve.
- Junk spread 2% plus up to 8% as the mood sours. The bandwidth price falls by (utilisation ÷ 0.35)^0.15 per quarter while idle and recovers 2% a quarter. Legacy voice is exposed to that price at power 0.3.
- Bankruptcy follows two quarters of debt above both market value and 8× annual operating profit, or interest cover below 1, while the mood is below −0.5. 60% of the debt is written off. Carriers emerge after 4 quarters and can build again, from cash or rationed credit.

**D-98 Mood.**
- Buzz fades as adoption passes half its ceiling (novelty 2).
- Peak memory 0.8 a quarter.
- Failures of listed firms count fully; private failures count 0.2.
- Venture money scales with exp(mood), with the mood floored at −1 for that purpose.

**D-99 Scripted history** (`internet.history`, default on): the Dec 1996 warning (mood −0.12); autumn 1998 (+1.5% risk premium, three 0.25% cuts); Y2K (liquidity, mood +0.3); and 2003 Q1–2004 Q2 (rate capped at 1% if jobs are scarce or inflation is below target).

**D-100 Realism scorecard.**
- 16 stylized facts with explicit ranges, evaluated on the mean of 5 seeds.
- Misses are shown, not hidden.
- The twin replays the seed with the most matches.
