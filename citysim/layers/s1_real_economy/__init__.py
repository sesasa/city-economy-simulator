"""Stage 1: real economy baseline (central planner, no money)."""
from .addons import Demography, HarvestShocks, Investment, LandLimits, Storage
from .real_economy import RealEconomy

STAGE1_LAYERS = [RealEconomy, Investment, Demography, Storage, LandLimits, HarvestShocks]
