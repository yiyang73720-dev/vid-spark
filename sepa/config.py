"""Tunable thresholds for the SEPA engine.

Defaults follow Mark Minervini's published rules (Trade Like a Stock Market
Wizard, ch. 5 "Trend Template" and ch. 10-11 on VCP / pivot buying). Every
number the engine compares against lives here so a screen is reproducible and
can be overridden with ``--config my.json``.
"""

from dataclasses import asdict, dataclass, field, fields


@dataclass
class TrendConfig:
    ma_short: int = 50
    ma_mid: int = 150
    ma_long: int = 200
    # Criterion 3: the 200-day MA must be higher than it was one "month"
    # (21 trading days) ago. Minervini prefers 4-5 months of rise.
    month_bars: int = 21
    min_ma_long_up_months: int = 1
    preferred_ma_long_up_months: int = 4
    week52_bars: int = 252
    min_above_low: float = 0.30  # criterion 6: >= 30% above the 52-week low
    max_below_high: float = 0.25  # criterion 7: within 25% of the 52-week high
    min_rs_rating: float = 70  # criterion 8: RS rating >= 70 (prefer 80-90+)
    # 52-week extremes from intraday High/Low (True) or from closes (False).
    use_intraday_extremes: bool = True


@dataclass
class StageConfig:
    ma: int = 150  # Weinstein's 30-week moving average
    slope_bars: int = 25  # ~5 weeks, used to call the MA rising / falling
    flat_band: float = 0.01  # |MA change over slope_bars| below this = flat
    prior_bars: int = 126  # window before the slope window, to tell Stage 1 from 3
    prior_advance: float = 0.15  # MA gain over prior_bars that marks a finished advance


@dataclass
class RSConfig:
    # IBD-style weighted performance: 40% last quarter, 20% each earlier quarter.
    weights: list = field(default_factory=lambda: [[63, 0.4], [126, 0.2], [189, 0.2], [252, 0.2]])
    # Below this many ranked stocks a percentile rating is not meaningful and
    # criterion 8 is reported as unknown instead of pass/fail.
    min_universe: int = 20
    rs_line_bars: int = 252


@dataclass
class FundamentalsConfig:
    min_eps_growth: float = 0.20  # latest quarter EPS YoY
    preferred_eps_growth: float = 0.25
    min_revenue_growth: float = 0.0  # revenue must at least grow while accelerating
    yoy_min_days: int = 330  # period ends this far apart count as the same quarter a year ago
    yoy_max_days: int = 400


@dataclass
class VCPConfig:
    max_base_bars: int = 325  # 65 weeks
    min_base_bars: int = 15  # 3 weeks
    min_prior_advance: float = 0.25  # the base must follow an advance of >= 25%
    prior_advance_bars: int = 126
    # Swing detection threshold: a reversal of this fraction confirms a swing.
    # It scales with the stock's own volatility (median true range in the base)
    # and is clamped to [zigzag_min, zigzag_max].
    zigzag_atr_mult: float = 1.5
    zigzag_min: float = 0.025
    zigzag_max: float = 0.06
    min_contractions: int = 2
    max_contractions: int = 6
    max_first_depth: float = 0.50  # corrections deeper than this rarely set up
    max_final_depth: float = 0.10  # the last contraction should be tight
    depth_tolerance: float = 0.01  # slack when checking each contraction is smaller
    max_pivot_below_high: float = 0.15  # pivot must sit near the top of the base
    require_volume_contraction: bool = True
    volume_avg_bars: int = 50
    dry_up_ratio: float = 0.5  # a day below this x 50-day avg volume counts as dry-up
    breakout_volume_ratio: float = 1.4  # breakout day volume >= 140% of 50-day avg
    max_chase: float = 0.05  # buy range: pivot .. pivot + 5%
    near_pivot: float = 0.03  # within 3% below the pivot = ready
    max_stop: float = 0.10  # stop at the final contraction low should be <= 10% away
    breakout_scan_bars: int = 15  # look this far back for the base a breakout came from


@dataclass
class Config:
    trend: TrendConfig = field(default_factory=TrendConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    rs: RSConfig = field(default_factory=RSConfig)
    fundamentals: FundamentalsConfig = field(default_factory=FundamentalsConfig)
    vcp: VCPConfig = field(default_factory=VCPConfig)

    @classmethod
    def from_dict(cls, data):
        """Build a Config, overriding defaults with the sections present in ``data``."""
        cfg = cls()
        for section in fields(cls):
            overrides = (data or {}).get(section.name) or {}
            current = getattr(cfg, section.name)
            known = {f.name for f in fields(current)}
            unknown = set(overrides) - known
            if unknown:
                raise ValueError(f"unknown {section.name} settings: {sorted(unknown)}")
            for key, value in overrides.items():
                setattr(current, key, value)
        return cfg

    def to_dict(self):
        return asdict(self)
