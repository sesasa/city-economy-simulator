// Used by tests/test_twin_parity.py: node parity_runner.js <params.json> <ticks>  -> JSON rows on stdout
const { Town } = require("./engine.js");
const fs = require("fs");
const params = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ticks = parseInt(process.argv[3], 10);
const town = new Town(params);
const rows = [];
for (let t = 0; t < ticks; t++) {
  const s = town.step();
  if (!s.booksOk) { console.error("books do not balance at tick " + s.tick); process.exit(2); }
  rows.push({
    tick: s.tick, population: s.N, consumption_per_person: s.cons, surplus_share: s.surplus / (s.plan.workers || 1),
    labor_shortfall: s.shortfall, total_output: s.final, raw_output: s.raw, potential_output: s.potential,
    growth_rate: s.growth, investment_labor: s.investment, factories: s.conditions.length,
    "capital/factory": s.conditions.reduce((x, y) => x + y, 0), "capital/land_improvement": s.land_improvement,
    "capital/tools_processing": s.tools_processing, "capital/tools_distribution": s.tools_distribution,
    "capital/factory_wip": s.wip, unmaintained_share: s.unmaintained, effective_farm_productivity: s.effective_a,
    "labor_share/farming": s.alloc.farming / (s.plan.workers || 1), "labor_share/maintenance": s.alloc.maintenance / (s.plan.workers || 1),
    storage_stock: s.storage,
  });
}
process.stdout.write(JSON.stringify(rows));
