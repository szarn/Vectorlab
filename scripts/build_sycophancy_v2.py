"""Assemble the v2 sycophancy dataset from audited flawed-premise and sound-premise pairs.

Usage:
    python scripts/build_sycophancy_v2.py

Flawed-premise pairs come from the v1 dataset (``data/sycophancy``), keeping only the pairs
the premise audit labeled ``ok``. Sound-premise pairs come from ``data/sycophancy-sound``:
all seeds, plus the generated pairs its audit labeled ``ok``. Each record gets a ``premise``
field (``flawed`` or ``sound``). Writes ``data/sycophancy/v2/{sycophantic,honest}.jsonl``.
"""

from pathlib import Path

from vectorlab.behavior import read_jsonl, write_jsonl

FLAWED = Path("data/sycophancy")
SOUND = Path("data/sycophancy-sound")
OUT = FLAWED / "v2"


def audited(directory: Path, audit: Path, premise: str) -> dict[str, list[dict]]:
    """A directory's sycophantic and honest records whose prompts the audit labeled ``ok``, tagged with ``premise``."""
    keep = {r["prompt"] for r in read_jsonl(audit) if r["label"] == "ok"}
    return {
        name: [{**r, "premise": premise} for r in read_jsonl(directory / f"{name}.jsonl") if r["prompt"] in keep]
        for name in ("sycophantic", "honest")
    }


def main() -> None:
    flawed = audited(FLAWED, FLAWED / "premise_audit.jsonl", "flawed")
    sound = audited(SOUND, SOUND / "audit.jsonl", "sound")
    for name in ("sycophantic", "honest"):
        write_jsonl(OUT / f"{name}.jsonl", flawed[name] + sound[name])
    print(f"{len(flawed['honest'])} flawed-premise + {len(sound['honest'])} sound-premise pairs -> {OUT}")


if __name__ == "__main__":
    main()
