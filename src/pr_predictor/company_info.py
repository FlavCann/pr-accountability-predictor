"""Hand-written facts about each covered company, for its profile page.

Curated, not extracted: nothing here comes from a transcript or a model, and
none of it is a claim the pipeline makes. Add a company with its description
and, optionally, a logo under `static/img/companies/`.
"""

from __future__ import annotations

from typing import Dict, Optional

ABOUT: Dict[str, Dict[str, str]] = {
    "Barry Callebaut": {
        "description": (
            "Barry Callebaut is a global manufacturer of chocolate and cocoa "
            "products, supplying food companies, confectionery brands, bakeries, "
            "and professional chefs. Headquartered in Switzerland, it operates "
            "across the cocoa and chocolate value chain, from sourcing and "
            "processing cocoa beans to producing finished chocolate ingredients. "
            "The company is also active in sustainability initiatives focused on "
            "responsible cocoa sourcing, farmer livelihoods, and reducing "
            "environmental impact."
        ),
        "logo": "img/companies/barry-callebaut.png",
    },
}


def about(company: str) -> Optional[Dict[str, str]]:
    """The company's description and logo path, or None if not written yet."""
    return ABOUT.get(company)
