"""Layer registry: scenario layer names -> Layer classes. One package per stage."""
from __future__ import annotations

from ..core.layer import Layer
from .s1_real_economy import STAGE1_LAYERS
from .s2_money_prices import STAGE2_LAYERS
from .s3_banking import STAGE3_LAYERS
from .s4_competition import STAGE4_LAYERS
from .s5_government import STAGE5_LAYERS
from .s6_central_bank import STAGE6_LAYERS
from .s7_financial_markets import STAGE7_LAYERS
from .s9_technology import STAGE9_LAYERS
from .s10_modern import STAGE10_LAYERS
from .s11_modern_town import STAGE11_LAYERS

REGISTRY: dict[str, type[Layer]] = {cls.name: cls for cls in [*STAGE1_LAYERS, *STAGE2_LAYERS, *STAGE3_LAYERS, *STAGE4_LAYERS, *STAGE5_LAYERS, *STAGE6_LAYERS, *STAGE7_LAYERS, *STAGE9_LAYERS, *STAGE10_LAYERS, *STAGE11_LAYERS]}
