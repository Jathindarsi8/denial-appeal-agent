"""
Day 14: a real authorization check.

`check_prior_authorization` was a substring search on the claim text:

    if "prior authorization" in text or "pa-" in text:
        return "A prior authorization reference appears in the claim documentation."

It searched the notes the model had already read and reported back what was in
them. It could not verify anything, and it could not fail. Day 13 built a
guardrail requiring it to be called before any decision on an
`authorization_missing` claim, and wrote up the model's refusal to call it as
the model not treating verification as necessary.

The model was right. There was no verification. Skipping a lookup that returns
a fact you already hold is correct behaviour, and the rule was forcing a
ceremony.

This is what the tool should have been. An authorization system of record,
separate from the claim, that the claim's own text cannot influence. Now the
check can return things the notes never could:

  the referenced authorization does not exist at all
  it exists but belongs to a different member
  it exists but the date of service falls outside its window
  it exists but authorises a different procedure than the one billed
  it exists and was later revoked

Every one of those changes the answer, and none is visible in the claim text.

Synthetic data, same as everything else here. A real deployment queries the
payer's authorization system; the shape of the answer is the same.

    python authorizations.py seed        create and populate the table
    python authorizations.py list        what is on file
    python authorizations.py PA-77104    check one
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date
from typing import Optional

from store import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS authorizations (
    auth_number      TEXT PRIMARY KEY,
    patient_id       TEXT NOT NULL,
    procedure_code   TEXT NOT NULL,
    provider_npi     TEXT NOT NULL,
    valid_from       TEXT NOT NULL,
    valid_to         TEXT NOT NULL,
    status           TEXT NOT NULL,   -- approved / revoked / expired
    authorized_units INTEGER NOT NULL DEFAULT 1,
    note             TEXT
);
"""

# Written so the interesting cases are the ones a substring match cannot reach.
SEED = [
    # Clean: exists, approved, in window, matches the billed procedure.
    ("PA-77104", "SYNTH-005", "29827", "1234567890",
     "2026-05-15", "2026-08-15", "approved", 1,
     "the straightforward case: the appeal is justified"),

    # Exists, approved, but authorises a DIFFERENT procedure than the one
    # billed. Policy AU-07 treats a partial match as no authorization. The
    # claim notes say an authorization exists and they are telling the truth.
    ("PA-88213", "SYNTH-004", "64483", "1234567890",
     "2026-05-01", "2026-09-01", "approved", 1,
     "authorises a different procedure than the claim billed"),

    # Exists but was revoked after issue. Nothing in the claim notes would
    # ever say this.
    ("PA-55010", "SYNTH-007", "27447", "9876543210",
     "2026-04-01", "2026-07-01", "revoked", 1,
     "revoked after issue, notes still reference it"),

    # Exists, approved, but the window closed before the service.
    ("PA-61200", "SYNTH-008", "43239", "1234567890",
     "2026-01-10", "2026-03-10", "approved", 1,
     "expired before the date of service"),

    # Deliberately absent from this list: PA-99999. A claim referencing it
    # gets "does not exist", where the stub said "an authorization is
    # referenced" because the string was in the notes.
]

AUTH_PATTERN = re.compile(r"\b(PA[-\s]?\d{4,6})\b", re.IGNORECASE)


@dataclass
class AuthRecord:
    auth_number: str
    patient_id: str
    procedure_code: str
    provider_npi: str
    valid_from: str
    valid_to: str
    status: str
    authorized_units: int
    note: Optional[str] = None


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def seed() -> None:
    init()
    with connect() as conn:
        for row in SEED:
            conn.execute(
                """INSERT OR REPLACE INTO authorizations
                   (auth_number, patient_id, procedure_code, provider_npi,
                    valid_from, valid_to, status, authorized_units, note)
                   VALUES (?,?,?,?,?,?,?,?,?)""", row)
        conn.commit()
    print(f"seeded {len(SEED)} authorizations")


def lookup(auth_number: str) -> Optional[AuthRecord]:
    init()
    normalised = auth_number.upper().replace(" ", "-")
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM authorizations WHERE UPPER(auth_number) = ?",
            (normalised,)).fetchone()
    if row is None:
        return None
    return AuthRecord(**{k: row[k] for k in row.keys()})


def extract_reference(text: str) -> Optional[str]:
    m = AUTH_PATTERN.search(text or "")
    if not m:
        return None
    return m.group(1).upper().replace(" ", "-")


def verify(denial) -> str:
    """The tool result. Says what was checked, so a human reading the trace can
    see which facts were established and which were not."""
    referenced = extract_reference(denial.documentation_summary)

    if not referenced:
        return (
            "No authorization number appears in the claim documentation, and "
            "no authorization is on file for this claim. Nothing to verify."
        )

    record = lookup(referenced)

    if record is None:
        return (
            f"{referenced} is referenced in the claim documentation but DOES "
            f"NOT EXIST in the authorization system of record. The reference "
            f"in the notes is not evidence that an authorization was issued. "
            f"Treat this as no authorization."
        )

    findings: list[str] = [
        f"{referenced} exists. Status: {record.status}. "
        f"Authorises procedure {record.procedure_code} for member "
        f"{record.patient_id}, valid {record.valid_from} to {record.valid_to}."
    ]
    problems: list[str] = []

    if record.status != "approved":
        problems.append(
            f"the authorization is {record.status}, not approved"
        )

    if record.patient_id != denial.patient_id:
        problems.append(
            f"it belongs to member {record.patient_id}, but this claim is for "
            f"{denial.patient_id}"
        )

    dos = getattr(denial, "date_of_service", None)
    if dos:
        if not (record.valid_from <= dos <= record.valid_to):
            problems.append(
                f"the date of service {dos} falls outside the authorised "
                f"window {record.valid_from} to {record.valid_to}"
            )
    else:
        findings.append(
            "The claim record carries no date of service, so the authorised "
            "window could not be checked against it."
        )

    billed = getattr(denial, "procedure_code", None)
    if billed:
        if billed != record.procedure_code:
            problems.append(
                f"it authorises procedure {record.procedure_code}, but "
                f"procedure {billed} was billed. Policy AU-07 treats a partial "
                f"match as no authorization"
            )
    else:
        findings.append(
            "The claim record carries no procedure code, so the authorised "
            "procedure could not be checked against what was billed."
        )

    if problems:
        return (
            "\n".join(findings)
            + "\n\nPROBLEMS FOUND:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nThis authorization does not support the claim as submitted."
        )

    return (
        "\n".join(findings)
        + "\n\nNo discrepancies found against the fields available on this "
          "claim. The authorization supports the claim as submitted."
    )


# ------------------------------------------------------------------ cli

def show() -> None:
    init()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM authorizations ORDER BY auth_number").fetchall()
    if not rows:
        print("no authorizations on file. Run: python authorizations.py seed")
        return
    print(f"{len(rows)} authorization(s)\n")
    for r in rows:
        print(f"  {r['auth_number']}  {r['status']:<9} "
              f"member {r['patient_id']}  proc {r['procedure_code']}  "
              f"{r['valid_from']} to {r['valid_to']}")
        if r["note"]:
            print(f"    {r['note']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "seed":
        seed()
    elif cmd == "list":
        show()
    else:
        rec = lookup(cmd)
        if rec is None:
            print(f"{cmd} does not exist in the authorization system")
        else:
            print(f"{rec.auth_number}  {rec.status}")
            print(f"  member    {rec.patient_id}")
            print(f"  procedure {rec.procedure_code}")
            print(f"  valid     {rec.valid_from} to {rec.valid_to}")
            if rec.note:
                print(f"  note      {rec.note}")
