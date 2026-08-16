"""Pre-registration as code.

This project has corrected itself three times in three weeks -- a metric bug, an
unmeasured null, and a headline that rested on one seed. A prose "we promise to
analyse it this way" is worth nothing against that record. What is worth
something is a frozen config that the analysis code physically cannot deviate
from, committed before the confirmatory runs, with its hash stamped into every
result so a stranger can check the chain with ``git log``.

Four guarantees, and one non-guarantee:

1. **Strict access.** :meth:`Prereg.get` raises :class:`PreregViolation` on any
   undeclared key, so a script cannot quietly introduce an arm, a seed or a
   threshold that was not pre-registered.
2. **Stamping.** :meth:`Prereg.stamp` injects the config hash and the git commit
   into an output dict. There is deliberately **no timestamp**: outputs are
   compared byte-for-byte across re-runs to prove the stamp perturbs nothing,
   and a clock reading would defeat that.
3. **Hash.** ``sha256(canonical json)[:16]``, the same construction
   ``scripts/make_splits.py:220`` already uses for split files, so split hashes
   and config hashes are the same species and can be eyeballed side by side.
4. **Lineage.** :func:`amendment_chain` reads the ``before -> after`` hashes out
   of ``AMENDMENTS.md`` and :func:`classify_hash` answers the question a bare
   equality test cannot: *is this stamp an ancestor of the frozen config, or a
   hash from nowhere?* ``PREREGISTRATION.md:189`` states the rule mechanically
   -- "if a result's hash does not match the frozen config, that result is
   exploratory" -- and mechanically applied it demotes every prior result on
   every amendment. Today it demotes ``runs/untrained_floor.json``, stamped
   ``20bdd93b781d950d`` and carrying H3 on ``splits_confirmatory.json``, over
   two amendments that touched arm declarations and a train-only ``max_slices``
   -- nothing the untrained floor reads. Ancestry is not a match and must not be
   reported as one; it is also not nothing, and the two are kept apart here.

The non-guarantee is blinding. :class:`BlindKey` maps model arms to opaque codes
so the analysis layer does not casually reveal which method is which mid-
iteration, and unblinding is a dated, logged event. It is **procedural, not
cryptographic** -- the salt sits in a file on the same disk as the analysis. It
raises the cost of accidental peeking and creates an audit point. It does not
make cheating impossible, and the pre-registration says so in as many words.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/prereg/isbi2027.yaml"
DEFAULT_AMENDMENTS = "AMENDMENTS.md"
_MISSING = object()


class PreregViolation(Exception):
    """Raised when code asks for something the pre-registration does not declare.

    Deliberately loud. The whole mechanism is worthless if an undeclared
    parameter can be read with a silent default, so there is no ``get(key,
    default)`` overload anywhere in this module.
    """


def canonical_hash(obj: Any, nibbles: int = 16) -> str:
    """``sha256`` over canonical JSON, truncated -- matches ``make_splits.py:220``.

    ``sort_keys`` makes the hash independent of mapping order, so reordering the
    YAML for readability does not read as tampering. It is *not* independent of
    values, which is the point.
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:nibbles]


def file_sha256(path: str | Path, nibbles: int = 16) -> str:
    """Hash of a file's actual bytes.

    ``make_splits.py`` records a hash of the split *payload*, whose ``cache``
    field is an absolute resolved path rather than the cache contents -- so that
    hash is machine-dependent and unchanged by re-extracting the HDF5 in place.
    This is the hash that actually pins a split file, and both get recorded.
    """
    h = hashlib.sha256(Path(path).read_bytes())
    return h.hexdigest()[:nibbles]


def git_state(repo: str | Path = ".") -> dict:
    """Current commit and whether the tree is dirty; ``None`` if git is unusable.

    Never raises -- a stamp that crashes the analysis it is documenting would be
    a bad trade. A ``None`` commit in an output is itself informative: it means
    the result cannot be tied to a revision.
    """
    def run(*args):
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


# ------------------------------------------------------------------- lineage
# The template at the top of AMENDMENTS.md is, to a regex, a perfectly good
# dated entry. Fenced blocks are stripped before parsing so the documentation of
# the format cannot become a link in the chain it documents.
_FENCED = re.compile(r"^```.*?^```", re.M | re.S)
_ENTRY = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*[—-]\s*(.+?)\s*$", re.M)
_TRANSITION = re.compile(
    r"^-\s+\*\*Config hash before\s*(?:→|->)\s*after:\*\*\s*`([^`]*)`"
    r"\s*(?:→|->)\s*`([^`]*)`",
    re.M,
)
_UNBLINDING = re.compile(r"^-\s+\*\*Config hash at unblinding:\*\*\s*`([^`]*)`", re.M)
_KIND = re.compile(r"^-\s+\*\*Kind:\*\*\s*(.+?)\s*$", re.M)
_IS_HASH = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class Amendment:
    """One recorded ``before -> after`` transition of the frozen config hash."""

    date: str
    title: str
    kind: str
    before: str
    after: str

    @property
    def changed_the_config(self) -> bool:
        """False for a record-only entry, which supersedes nothing.

        The 2026-08-15 ``lam`` pre-flight entry is a ``deviation`` with
        ``before == after``: it discloses what was seen and changes no
        parameter. Listing it as something that superseded an earlier hash would
        overstate the record, so it stays in the chain (it must, or the chain
        would read as having a gap) but out of the ancestry report.
        """
        return self.before != self.after


@dataclass(frozen=True)
class Lineage:
    """Where a stamped hash sits relative to the frozen one."""

    status: str  # "current" | "superseded" | "unknown"
    superseded_by: tuple[Amendment, ...] = ()

    @property
    def label(self) -> str:
        """One token for a report line. ``UNKNOWN`` shouts because it should.

        A superseded stamp is informational -- an amendment is *expected* to
        leave ancestors behind. An unknown stamp means a result was produced
        under a config that is not on the recorded chain at all, which is the
        case no amount of prose can explain away.
        """
        if self.status == "superseded":
            return f"superseded-by-{self.superseded_by[0].date}"
        return "current" if self.status == "current" else "UNKNOWN"


def amendment_chain(path: str | Path = DEFAULT_AMENDMENTS) -> list[Amendment]:
    """Parse the ordered hash transitions out of ``AMENDMENTS.md``.

    Strict on purpose. A lineage mechanism that silently accepts a broken chain
    is worse than no mechanism at all, because it launders provenance: it would
    report a stamp as "ancestral" on the strength of a record with a hole in it.
    So a dated entry that records no hash at all, a hash that is not 16 hex
    (an amendment drafted ``before -> PENDING`` and not yet filled in), and a
    gap where one entry's ``after`` is not the next entry's ``before`` all raise
    :class:`PreregViolation` rather than being skipped.

    Unblinding entries are the one exception, and they are not an exception to
    the rule: ``scripts/prereg_unblind.py`` writes ``Config hash **at**
    unblinding``, a dated marker of when someone looked rather than a
    transition. It is not a link and does not break the chain either side of it.
    """
    p = Path(path)
    if not p.exists():
        raise PreregViolation(f"amendment log not found at {p}")
    text = _FENCED.sub("", p.read_text())

    heads = list(_ENTRY.finditer(text))
    if not heads:
        raise PreregViolation(
            f"no '## YYYY-MM-DD - title' entries in {p}. Either the log is empty "
            "or its heading format has drifted from what the template declares; "
            "either way the chain cannot be read and nothing can be called "
            "ancestral."
        )

    chain: list[Amendment] = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body, date, title = text[head.end():end], head.group(1), head.group(2)

        m = _TRANSITION.search(body)
        if m is None:
            if _UNBLINDING.search(body):
                continue
            raise PreregViolation(
                f"amendment {date} ({title!r}) in {p} records no config hash "
                "transition. Every entry must carry '**Config hash before -> "
                "after:** `xxxx` -> `yyyy`' or the chain is not verifiable."
            )
        before, after = m.group(1), m.group(2)
        for which, value in (("before", before), ("after", after)):
            if not _IS_HASH.match(value):
                raise PreregViolation(
                    f"amendment {date} ({title!r}) in {p} records {which}-hash "
                    f"{value!r}, which is not 16 hex digits. A half-recorded "
                    "amendment leaves provenance undetermined -- fill in the "
                    "real hash (scripts/prereg_freeze.py --amend prints it)."
                )
        kind = _KIND.search(body)
        if kind is None:
            raise PreregViolation(
                f"amendment {date} ({title!r}) in {p} declares no '**Kind:**'. "
                "Whether an entry is an amendment, a deviation or an unblinding "
                "is part of what the record is for."
            )
        chain.append(Amendment(date=date, title=title, kind=kind.group(1),
                               before=before, after=after))

    for prev, nxt in zip(chain, chain[1:]):
        if prev.after != nxt.before:
            raise PreregViolation(
                f"gap in the amendment chain in {p}: {prev.date} ({prev.title!r}) "
                f"ends at {prev.after} but {nxt.date} ({nxt.title!r}) starts from "
                f"{nxt.before}. A config state exists that no entry accounts for, "
                "so no stamp can be called ancestral until it is recorded."
            )
    return chain


def classify_hash(stamped: str, current: str,
                  chain: Sequence[Amendment]) -> Lineage:
    """Place a stamped config hash on the recorded chain.

    The chain's *first* ``before`` is the origin freeze and counts as an
    ancestor, which is the whole reason this is not a scan of ``after`` values:
    five stamped results on disk carry ``2b580fa93894d86f``, the hash the first
    recorded entry amended *away from*. Reading only the ``after`` side would
    call them unknown -- a lie about files whose provenance the log states
    outright.

    Everything genuinely off the chain is ``"unknown"`` and stays that way. The
    classifier does not guess that an unrecognised hash is merely old; that
    guess is exactly how a result produced under an unrecorded config would slip
    through wearing an ancestor's clothes.
    """
    lineage = [chain[0].before, *(a.after for a in chain)] if chain else [current]
    if lineage[-1] != current:
        raise PreregViolation(
            f"the amendment chain ends at {lineage[-1]} but the config now "
            f"hashes to {current}. The config was changed without an entry "
            "logging it, so the chain does not describe the config it claims "
            "to -- add the entry before trusting any ancestry claim."
        )
    if stamped == current:
        return Lineage("current")
    if stamped not in lineage:
        return Lineage("unknown")
    # First occurrence: the point at which this hash stopped being current.
    cut = lineage.index(stamped)
    return Lineage("superseded",
                   tuple(a for a in chain[cut:] if a.changed_the_config))


@dataclass(frozen=True)
class BlindKey:
    """Deterministic arm -> opaque code map under a secret salt."""

    salt: str
    codes: dict[str, str]
    prefix: str = "ARM"
    nibbles: int = 6

    @staticmethod
    def _code(salt: str, name: str, prefix: str, nibbles: int) -> str:
        digest = hmac.new(salt.encode(), name.encode(), hashlib.sha256).hexdigest()
        return f"{prefix}-{digest[:nibbles].upper()}"

    @classmethod
    def create(cls, names, prefix: str = "ARM", nibbles: int = 6,
               salt: str | None = None) -> BlindKey:
        salt = salt or secrets.token_hex(16)
        codes = {n: cls._code(salt, n, prefix, nibbles) for n in names}
        if len(set(codes.values())) != len(codes):
            raise PreregViolation(
                f"blinding collision at {nibbles} nibbles for {sorted(names)}; "
                "raise blinding.code_nibbles in the config"
            )
        for name, code in codes.items():
            if name.lower() in code.lower():
                raise PreregViolation(f"blind code {code!r} leaks arm name {name!r}")
        return cls(salt=salt, codes=codes, prefix=prefix, nibbles=nibbles)

    def code(self, name: str) -> str:
        if name not in self.codes:
            raise PreregViolation(f"arm {name!r} is not in the blinding key")
        return self.codes[name]

    def inverse(self) -> dict[str, str]:
        return {v: k for k, v in self.codes.items()}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"salt": self.salt, "prefix": self.prefix,
             "nibbles": self.nibbles, "codes": self.codes}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BlindKey:
        d = json.loads(Path(path).read_text())
        return cls(salt=d["salt"], codes=d["codes"],
                   prefix=d.get("prefix", "ARM"), nibbles=d.get("nibbles", 6))


class Prereg:
    """A frozen pre-registration config with strict lookup and output stamping."""

    def __init__(self, raw: dict, path: str | Path):
        self.raw = raw
        self.path = str(path)
        self.hash = canonical_hash(raw)

    # -------------------------------------------------------------- lookup
    def get(self, dotted: str) -> Any:
        """Fetch ``a.b.c``, raising :class:`PreregViolation` if any level is absent.

        No default argument, by design -- see the class docstring.
        """
        node: Any = self.raw
        walked: list[str] = []
        for part in dotted.split("."):
            walked.append(part)
            if not isinstance(node, dict) or part not in node:
                raise PreregViolation(
                    f"{'.'.join(walked)} is not declared in {self.path}. "
                    "Undeclared parameters are not usable in a confirmatory "
                    "analysis -- amend the pre-registration and log it in "
                    "AMENDMENTS.md, or run this as exploratory."
                )
            node = node[part]
        return node

    @property
    def version(self) -> str:
        return self.get("version")

    # ---------------------------------------------------------------- arms
    def arms(self, status: str | None = None, blind: bool | None = None) -> list[dict]:
        out = self.get("arms")
        if status is not None:
            out = [a for a in out if a.get("status") == status]
        if blind is not None:
            out = [a for a in out if bool(a.get("blind", False)) is blind]
        return list(out)

    def arm(self, name: str) -> dict:
        for a in self.get("arms"):
            if a["name"] == name:
                return a
        declared = ", ".join(sorted(a["name"] for a in self.get("arms")))
        raise PreregViolation(
            f"arm {name!r} is not pre-registered. Declared arms: {declared}"
        )

    SCORING_CLASSES = ("learned_attention", "uniform_attention",
                       "fixed_geometric_attention")

    def arm_set(self, hid: str) -> list[str]:
        """The arm names a hypothesis is scored over, resolved from `scoring_class`.

        A hypothesis without an ``arm_set`` is scored over every implemented arm
        -- that is the pre-freeze default and it stays the default, so adding
        this method cannot silently rescope a hypothesis nobody amended.

        Why this exists rather than a second list of exclusions beside H2's:
        "every arm" is a hole of exactly the kind H5's missing unit was. Two arms
        have outcomes fixed by construction rather than measured -- ``mean``
        returns uniform attention so every axis AUC ties at 0.5, and
        ``centre_gaussian`` is constant within a slice so its within-slice AUC is
        0.5 by construction -- and a hypothesis that counts them is counting
        arithmetic as evidence. Ruled by amendment 2026-08-15, before the
        confirmatory sweep, with the arithmetic showing H1's falsification bar is
        unchanged either way; see AMENDMENTS.md.

        Raises rather than defaulting on an undeclared class, for the same reason
        :meth:`get` has no default overload: a typo in ``scoring_class`` must not
        quietly drop an arm out of a hypothesis.
        """
        want = self.hypothesis(hid).get("arm_set")
        implemented = self.arms(status="implemented")
        for a in implemented:
            cls = a.get("scoring_class")
            if cls is None:
                raise PreregViolation(
                    f"arm {a['name']!r} declares no scoring_class; "
                    f"arm_set({hid!r}) cannot be resolved"
                )
            if cls not in self.SCORING_CLASSES:
                raise PreregViolation(
                    f"arm {a['name']!r} has scoring_class {cls!r}, which is not one of "
                    f"{', '.join(self.SCORING_CLASSES)}"
                )
        if want is None:
            return [a["name"] for a in implemented]
        if want not in self.SCORING_CLASSES:
            raise PreregViolation(
                f"hypothesis {hid!r} declares arm_set {want!r}, which is not a scoring_class"
            )
        return [a["name"] for a in implemented if a.get("scoring_class") == want]

    def estimand(self, name: str) -> dict:
        est = self.get("estimands")
        for group in ("primary", "secondary"):
            for e in est.get(group, []):
                if e["name"] == name:
                    return {**e, "role": group}
        raise PreregViolation(f"estimand {name!r} is not pre-registered")

    def hypothesis(self, hid: str) -> dict:
        for h in self.get("hypotheses"):
            if h["id"] == hid:
                return h
        raise PreregViolation(f"hypothesis {hid!r} is not pre-registered")

    # ------------------------------------------------------------- splits
    def split(self, dataset: str, role: str) -> dict:
        return self.get(f"datasets.{dataset}.splits.{role}")

    def verify_splits(self, must_exist: bool = True) -> dict[str, str]:
        """Check every declared split file exists and its recorded hash still holds.

        Returns ``{"lidc.confirmatory": "ok" | "missing" | "hash-mismatch", ...}``
        rather than raising, so a caller can decide whether a not-yet-generated
        split is fatal.
        """
        report: dict[str, str] = {}
        for ds, dcfg in self.get("datasets").items():
            for role, scfg in dcfg.get("splits", {}).items():
                key = f"{ds}.{role}"
                p = Path(scfg["path"])
                if not p.exists():
                    report[key] = "missing" if must_exist else "not-yet-generated"
                    continue
                recorded = scfg.get("recorded_hash")
                if recorded is None:
                    report[key] = "unrecorded"
                    continue
                actual = json.loads(p.read_text()).get("hash")
                report[key] = "ok" if actual == recorded else "hash-mismatch"
        return report

    # ------------------------------------------------------------ blinding
    def blind_key(self, create_if_missing: bool = False) -> BlindKey:
        path = Path(self.get("blinding.key_path"))
        if path.exists():
            return BlindKey.load(path)
        if not create_if_missing:
            raise PreregViolation(
                f"blinding key {path} does not exist; run scripts/prereg_freeze.py"
            )
        key = BlindKey.create(
            [a["name"] for a in self.arms(blind=True)],
            prefix=self.get("blinding.code_prefix"),
            nibbles=self.get("blinding.code_nibbles"),
        )
        key.save(path)
        return key

    # ------------------------------------------------------------ stamping
    def stamp(self, obj: dict, repo: str | Path = ".") -> dict:
        """Attach provenance under a top-level ``prereg`` key.

        Top level is safe across this repo: no consumer iterates top-level keys
        (``merge_results.py`` and ``train_cached.py`` both look up by name), so
        an extra sibling breaks nothing.
        """
        g = git_state(repo)
        obj["prereg"] = {
            "prereg_version": self.version,
            "prereg_hash": self.hash,
            "prereg_config": self.path,
            "git_commit": g["commit"],
            "git_dirty": g["dirty"],
        }
        return obj


def load(path: str | Path = DEFAULT_CONFIG) -> Prereg:
    p = Path(path)
    if not p.exists():
        raise PreregViolation(f"pre-registration config not found at {p}")
    return Prereg(yaml.safe_load(p.read_text()), p)
