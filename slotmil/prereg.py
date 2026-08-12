"""Pre-registration as code.

This project has corrected itself three times in three weeks -- a metric bug, an
unmeasured null, and a headline that rested on one seed. A prose "we promise to
analyse it this way" is worth nothing against that record. What is worth
something is a frozen config that the analysis code physically cannot deviate
from, committed before the confirmatory runs, with its hash stamped into every
result so a stranger can check the chain with ``git log``.

Three guarantees, and one non-guarantee:

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
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/prereg/isbi2027.yaml"
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
