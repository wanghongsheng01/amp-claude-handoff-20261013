"""Hash-bound train/dev loader; no default path and no test-role support."""
import csv
import hashlib
from pathlib import Path

ALPHABET = frozenset("ARNDCEQGHILKMFPSTWYV")
CLASS = {"FREE": "0", "AMD": "1"}


def read_split(path, *, role, expected_sha256, expected_rows):
    if role not in ("train", "development"):
        raise ValueError("only train and reused development data may be read")
    path = Path(path).resolve()
    if any("test" in part.lower() for part in path.parts):
        raise ValueError("test path prohibited")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("split hash differs from prelocked input")
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    if len(rows) != expected_rows:
        raise ValueError("split size mismatch")
    seen = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("ragged row")
        s, terminal = row["sequence"], row["terminal_state"]
        if not 1 <= len(s) <= 48 or not set(s) <= ALPHABET or CLASS.get(terminal) != row["class"]:
            raise ValueError("invalid canonical sequence or terminal class")
        sha = hashlib.sha256(s.encode("ascii")).hexdigest()
        molecule = hashlib.sha256(("AMP_MOLECULE_V1\n" + s + "\n" + terminal).encode("ascii")).hexdigest()
        if (row["sequence_sha256"] != sha or row["molecule_sha256"] != molecule
                or row["length"] != str(len(s)) or not row["component_id"] or molecule in seen):
            raise ValueError("invalid identity, length, component, or duplicate")
        for field in ("split", "partition", "role"):
            allowed = {"train"} if role == "train" else {"val", "validation", "development"}
            if field in row and row[field].lower() not in allowed:
                raise ValueError("row split conflicts with input role")
        seen.add(molecule)
    return {"role": role, "path": str(path), "sha256": actual_sha256, "rows": rows}


def assert_split_isolation(train, development):
    if train["role"] != "train" or development["role"] != "development":
        raise ValueError("train/dev role mismatch")
    for field in ("sequence", "molecule_sha256", "component_id"):
        if {r[field] for r in train["rows"]} & {r[field] for r in development["rows"]}:
            raise ValueError("train/development overlap: " + field)
    return True
