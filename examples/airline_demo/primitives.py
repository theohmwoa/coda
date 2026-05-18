"""Airline retrieval primitives — typed objects the agent calls from code.

These match the "memory as tools, implemented per workflow" shape from the
design conversation: an airline's memory isn't a generic vector search — it's
`bookings.get(id)`, `flights.find(date=...)`, `policies.lookup(name)`. Each
primitive's docstring is its contract; the implementation IS the schema.

The intent is for the agent to compose these across shapes:
- direct lookup (`bookings.get`, `passengers.get`, `flights.get`)
- tabular query (`bookings.find`, `flights.find`, `tickets.find`)
- document-shaped retrieval (`policies.lookup` returns the full section
  tree; `policies.get_section` extracts one section by path)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .mock_data import (
    BOOKINGS,
    Booking,
    FLIGHTS,
    Flight,
    PASSENGERS,
    POLICIES,
    Passenger,
    TICKETS,
    Ticket,
)


def _booking_dict(b: Booking) -> dict[str, Any]:
    return asdict(b)


def _flight_dict(f: Flight) -> dict[str, Any]:
    return asdict(f)


def _passenger_dict(p: Passenger) -> dict[str, Any]:
    return asdict(p)


def _ticket_dict(t: Ticket) -> dict[str, Any]:
    return asdict(t)


# --- Passengers --------------------------------------------------------------


class Passengers:
    """Customer records — direct lookup by id, search by email or name."""

    def get(self, id: str) -> dict[str, Any]:
        """Fetch the full passenger record by id.

        Returns the passenger as a dict with keys: id, name, email, tier,
        join_year, ytd_spend_usd, home_airport. Raises KeyError if the id
        is unknown.
        """
        p = PASSENGERS.get(id)
        if p is None:
            raise KeyError(f"unknown passenger: {id!r}")
        return _passenger_dict(p)

    def find(self, email: str | None = None, name_contains: str | None = None) -> list[dict[str, Any]]:
        """Find passengers by email (exact) or name substring."""
        out: list[Passenger] = []
        for p in PASSENGERS.values():
            if email is not None and p.email == email:
                out.append(p)
                continue
            if name_contains is not None and name_contains.lower() in p.name.lower():
                out.append(p)
        return [_passenger_dict(p) for p in out]


# --- Bookings ----------------------------------------------------------------


class Bookings:
    """Booking records — direct lookup by id, tabular search by criteria."""

    def get(self, id: str) -> dict[str, Any]:
        """Fetch the full booking record by id.

        Returns dict with keys: id, passenger_id, flight_ids, date, fare_usd,
        fare_class, status, refundable. Raises KeyError if unknown.
        """
        b = BOOKINGS.get(id)
        if b is None:
            raise KeyError(f"unknown booking: {id!r}")
        return _booking_dict(b)

    def find(
        self,
        *,
        passenger_id: str | None = None,
        flight_id: str | None = None,
        date: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find bookings matching all of the given filters (AND).

        Any argument left as None is unconstrained. Returns the matching
        bookings as a list; empty list if none match.
        """
        out = []
        for b in BOOKINGS.values():
            if passenger_id is not None and b.passenger_id != passenger_id:
                continue
            if flight_id is not None and flight_id not in b.flight_ids:
                continue
            if date is not None and b.date != date:
                continue
            if status is not None and b.status != status:
                continue
            out.append(_booking_dict(b))
        return out


# --- Flights -----------------------------------------------------------------


class Flights:
    """Flight operational data — direct lookup and tabular search."""

    def get(self, flight_id: str, date: str) -> dict[str, Any]:
        """Fetch the flight record for a specific (flight_id, date) pair.

        Returns dict with keys: id, date, origin, destination, operator,
        aircraft_tail, scheduled_depart, status, cancellation_reason.
        Raises KeyError if the flight isn't found on that date.
        """
        f = FLIGHTS.get((flight_id, date))
        if f is None:
            raise KeyError(f"no flight {flight_id} on {date}")
        return _flight_dict(f)

    def find(
        self,
        *,
        date: str | None = None,
        operator: str | None = None,
        status: str | None = None,
        origin: str | None = None,
        destination: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find flights matching filters (AND across non-None args)."""
        out = []
        for f in FLIGHTS.values():
            if date is not None and f.date != date:
                continue
            if operator is not None and f.operator != operator:
                continue
            if status is not None and f.status != status:
                continue
            if origin is not None and f.origin != origin:
                continue
            if destination is not None and f.destination != destination:
                continue
            out.append(_flight_dict(f))
        return out


# --- Tickets (relational lookup over support history) ------------------------


class Tickets:
    """Support-ticket history — graph-shaped retrieval (per-passenger threads)."""

    def find(
        self,
        *,
        passenger_id: str | None = None,
        category: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """List tickets matching filters, newest first.

        `since` is an ISO date; tickets opened on or after that date are kept.
        """
        out: list[Ticket] = []
        for t in TICKETS:
            if passenger_id is not None and t.passenger_id != passenger_id:
                continue
            if category is not None and t.category != category:
                continue
            if since is not None and t.opened_at < since:
                continue
            out.append(t)
        out.sort(key=lambda t: t.opened_at, reverse=True)
        return [_ticket_dict(t) for t in out]


# --- Policies (document-shaped retrieval) -----------------------------------


class Policies:
    """Long-form policy documents — by name or by named section."""

    def list_available(self) -> list[str]:
        """Return the names of every policy document available."""
        return sorted(POLICIES.keys())

    def lookup(self, name: str) -> str:
        """Fetch the full text of policy `name`.

        This is expensive in token terms — a typical policy is ~1.5K tokens.
        Prefer get_section() when you know which section you need; use this
        only when you need the structure or are unsure where the answer lives.
        """
        if name not in POLICIES:
            raise KeyError(f"unknown policy: {name!r}. Available: {sorted(POLICIES)}")
        return POLICIES[name]

    def get_section(self, name: str, section_number: str) -> str:
        """Extract one section by its leading number from a policy.

        Example: get_section("involuntary_cancellation", "3") returns the
        whole of Section 3 — TIER-SPECIFIC ENHANCEMENTS. Useful when you've
        already walked the structure once and know which slice you need.
        """
        body = self.lookup(name)
        # Find lines beginning with "Section N" (or sub-numbered children).
        lines = body.splitlines()
        out: list[str] = []
        capturing = False
        prefix = f"Section {section_number}"
        # Allow "Section 3.X" to be captured as part of "Section 3" too.
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("Section "):
                if stripped.startswith(prefix + " ") or stripped == prefix:
                    capturing = True
                    out.append(line)
                    continue
                # Different top-level section — stop if we were capturing.
                if capturing and not stripped.startswith(f"Section {section_number}."):
                    if not stripped.startswith(f"{section_number}."):
                        break
                if capturing:
                    out.append(line)
                continue
            if capturing:
                out.append(line)
        if not out:
            raise KeyError(f"section {section_number} not found in {name}")
        return "\n".join(out).rstrip()
