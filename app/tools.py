# ruff: noqa
import chromadb

_collection = None

SEED_DATA = [
    ("Cylinder head bolt torque spec: tighten in three stages — 30 Nm, 60 Nm, then 90 Nm in a cross pattern. Always use new bolts after removal; TTY bolts are single-use.",
     "Dave Kowalski (Senior Engine Assembly Tech)", "specification"),
    ("Hydraulic press not reaching full tonnage: first check the hydraulic fluid level and look for leaks at the cylinder seals. Nine times out of ten it is air in the system — crack the bleed screw at the top of the cylinder while someone cycles the press slowly.",
     "Maria Santos (Maintenance Lead)", "troubleshooting"),
    ("Pre-shift CNC machine checklist: (1) check coolant level and concentration, (2) inspect tooling for wear or chipping, (3) verify zero-point offsets against the setup sheet, (4) run air-cut on the first part, (5) check first-off against the control plan before running production.",
     "Tom Patel (CNC Supervisor)", "process"),
    ("Oil pressure drop after warm-up: most likely worn main bearings or a failing oil pump pick-up screen clogged with sludge. Before condemning the pump, drop the sump and inspect the screen. Also check the pressure relief valve — it can stick open.",
     "Dave Kowalski (Senior Engine Assembly Tech)", "troubleshooting"),
    ("When you receive a customer NCR, log it in the CAR system within 24 hours. Do an immediate containment — sort all suspect stock, quarantine it, and slap a red hold tag on the pallet. Do NOT ship anything from that batch until quality signs off. Customer wants an 8D? Give them an initial response within 48 hours even if it is just containment confirmed.",
     "Linda Okafor (Quality Manager)", "process"),
    ("Gauge calibration: every gauge in the production area must have a current sticker. Frequency is in the calibration plan but as a rule — micrometers and verniers every 6 months, CMM probes every 3 months, pressure gauges annually. If a gauge is found out of calibration, quarantine all parts measured since the last valid cal date and do a risk assessment.",
     "Linda Okafor (Quality Manager)", "safety"),
    ("Weld spatter on the mating face is one of the most common reasons for a leak reject on the engine cover. Fix: check wire stick-out (should be 10–12 mm), reduce voltage by 0.5 V increments, and make sure the operator is not dragging the torch too slowly. Anti-spatter spray on the fixture helps but is not a root-cause fix.",
     "Raj Mehta (Welding Engineer)", "troubleshooting"),
    ("PPAP submission for a new supplier: you need the full 18 elements unless the customer grants a waiver. The ones that trip people up are the process flow diagram (must match the control plan exactly) and the MSA studies (need Gage R&R under 10% for critical dimensions). Start PPAP at least 8 weeks before SOP.",
     "Linda Okafor (Quality Manager)", "process"),
    ("Safety rule for press operations: never reach into the die area while the press is energised, even if it has stopped. Always lock out / tag out before clearing a jam. Two people have been hurt on presses here in the last 10 years — both times someone reached in without LOTO.",
     "Maria Santos (Maintenance Lead)", "safety"),
    ("Engine block bore inspection: use the dial bore gauge, not the micrometer — you need to check for taper and ovality, not just diameter. Measure at three depths (top, middle, bottom) and in two planes (thrust and non-thrust). Maximum taper 0.005 mm, maximum ovality 0.003 mm per our control plan.",
     "Dave Kowalski (Senior Engine Assembly Tech)", "specification"),
]


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.EphemeralClient()
        _collection = client.get_or_create_collection(
            name="workforce_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        if _collection.count() == 0:
            _collection.add(
                documents=[d[0] for d in SEED_DATA],
                metadatas=[{"source": d[1], "category": d[2]} for d in SEED_DATA],
                ids=[f"seed_{i}" for i in range(len(SEED_DATA))],
            )
    return _collection


def store_knowledge(content: str, source: str, category: str) -> dict:
    """Store a piece of expert knowledge in the knowledge base.

    Args:
        content: The knowledge content to store (one clear, self-contained fact or procedure).
        source: Who or what this came from, e.g. 'John Smith (Test Engineer)' or 'SOP-4421'.
        category: One of: process, troubleshooting, specification, safety, general.

    Returns:
        dict with status and total entry count.
    """
    collection = _get_collection()
    doc_id = f"entry_{collection.count()}"
    collection.add(
        documents=[content],
        metadatas=[{"source": source, "category": category}],
        ids=[doc_id],
    )
    return {"status": "stored", "id": doc_id, "total_entries": collection.count()}


def search_knowledge(query: str, n_results: int) -> dict:
    """Search the knowledge base for entries relevant to the query.

    Args:
        query: The question or topic to search for.
        n_results: Number of results to return (1 to 5).

    Returns:
        dict with matching entries, each with content, source, and category.
    """
    collection = _get_collection()
    n = min(max(n_results, 1), 5, collection.count())
    results = collection.query(query_texts=[query], n_results=n)
    entries = [
        {
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "category": results["metadatas"][0][i].get("category", "general"),
        }
        for i in range(len(results["documents"][0]))
    ]
    return {"status": "success", "results": entries, "total_in_kb": collection.count()}


def list_sources() -> dict:
    """List all sources and categories currently in the knowledge base.

    Returns:
        dict with each unique source, their categories, and total entry count.
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return {"status": "empty", "total": 0, "sources": []}
    all_data = collection.get(include=["metadatas"])
    sources: dict[str, set] = {}
    for meta in all_data["metadatas"]:
        src = meta.get("source", "unknown")
        cat = meta.get("category", "general")
        sources.setdefault(src, set()).add(cat)
    return {
        "status": "success",
        "total": total,
        "sources": [{"name": k, "categories": sorted(v)} for k, v in sources.items()],
    }
