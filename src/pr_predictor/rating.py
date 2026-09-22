"""The headline PR-accountability rating: one of four levels per company.

The levels run from `pr_high` (commitments kept, and visibly reported on) to
`pr_low` (commitments missed or quietly dropped). The dashboard shows all four
and reveals which one applies.

PLACEHOLDER. `rate` does not yet read the record -- every company is `pr_med_low`.
The response says so (`basis: "placeholder"`) and the dashboard shows it,
because a rating of a named company that does not follow from its record is
exactly the unsupported claim this product exists to avoid. Replace `rate` with
a function of the profile, and change `basis` to name the method.
"""

from __future__ import annotations

from typing import Dict, List

# Best to worst. The dashboard lays the images out in this order.
LEVELS: List[Dict[str, str]] = [
    {"level": "pr_high", "label": "High"},
    {"level": "pr_med_high", "label": "Medium-high"},
    {"level": "pr_med_low", "label": "Medium-low"},
    {"level": "pr_low", "label": "Low"},
]


def rate(profile: Dict[str, object]) -> Dict[str, object]:
    """The rating for a company, given its `outcomes.accountability_profile`."""
    return {
        "level": "pr_med_low",
        "basis": "placeholder",
        "note": "Hard-coded for every company; not yet derived from the record.",
        "scale": LEVELS,
    }
