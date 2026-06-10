from __future__ import annotations

from functools import lru_cache

from app.rag.store import VectorStore

# Seed institutional knowledge the planner retrieves to ground its narrative:
# promo playbooks, anomaly post-mortems, inventory policy. In production these would
# be your team's real forecast reports; here a small representative set.

_DOCS: list[tuple[str, dict]] = [
    (
        "Promo playbook: Active promotions on Rossmann stores lift daily sales 20-40% "
        "on average, with the largest effect on the first two days. When a promo flag "
        "is set in the forecast window, bias the reorder point toward the upper CI.",
        {"type": "playbook", "topic": "promotion"},
    ),
    (
        "Post-mortem 2014-12: Christmas week (Dec 24-26) saw sales collapse as stores "
        "closed or ran reduced hours. State holidays of type b/c are strong negative "
        "drivers; never staff inventory to the mean across a holiday window.",
        {"type": "postmortem", "topic": "holiday"},
    ),
    (
        "Inventory policy: Target service level 95%. Reorder point = forecast mean over "
        "lead time + safety stock, where safety stock covers the gap between mean and "
        "upper CI. Wide confidence intervals on high-value stores warrant human review.",
        {"type": "policy", "topic": "inventory"},
    ),
    (
        "Weather note: Cold snaps modestly raise drugstore footfall (seasonal health "
        "products); heatwaves shift mix but have small net effect on total daily sales.",
        {"type": "playbook", "topic": "weather"},
    ),
    (
        "Seasonality: Rossmann daily sales show strong weekly seasonality — Mondays and "
        "Saturdays peak, Sundays are mostly closed. Use season_length=7 for daily models.",
        {"type": "playbook", "topic": "seasonality"},
    ),
]


@lru_cache(maxsize=1)
def get_knowledge_base() -> VectorStore:
    store = VectorStore()
    store.add([d for d, _ in _DOCS], [m for _, m in _DOCS])
    return store
