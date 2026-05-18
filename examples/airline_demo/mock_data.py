"""Mock airline data — small but realistic enough to exercise ctx.

Five tables (passengers, bookings, flights, policies, tickets) shaped roughly
like what a real CRM + ops + legal-docs system would expose. Sized so the
"raw view" of a customer escalation crosses ~5K tokens (enough to make the
ctx-vs-globals distinction matter on a budget set at 12K-15K tokens for the
demo). Policies are deliberately long because they're the document-shape
data; tickets are deliberately many because they're the relational one.

Nothing here is dynamic or randomized at import time — the data is fixed so
demo runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Passengers --------------------------------------------------------------


@dataclass
class Passenger:
    id: str
    name: str
    email: str
    tier: str  # "blue" | "silver" | "gold" | "platinum"
    join_year: int
    ytd_spend_usd: int
    home_airport: str


PASSENGERS: dict[str, Passenger] = {
    "P-1001": Passenger("P-1001", "Mira Chen", "mira.chen@example.com", "platinum", 2018, 84_300, "SFO"),
    "P-1002": Passenger("P-1002", "Theo Park", "theo.park@example.com", "gold", 2020, 31_200, "ORD"),
    "P-1003": Passenger("P-1003", "Aanya Rao", "aanya.rao@example.com", "silver", 2022, 8_400, "DEL"),
    "P-1004": Passenger("P-1004", "Liam Walsh", "liam.walsh@example.com", "blue", 2024, 1_900, "DUB"),
    "P-1005": Passenger("P-1005", "Maya Okonkwo", "maya.okonkwo@example.com", "gold", 2019, 27_800, "LHR"),
}


# --- Flights -----------------------------------------------------------------


@dataclass
class Flight:
    id: str            # e.g. "UA1234"
    date: str          # ISO date
    origin: str
    destination: str
    operator: str
    aircraft_tail: str
    scheduled_depart: str
    status: str        # "scheduled" | "cancelled" | "delayed" | "departed"
    cancellation_reason: str | None = None


FLIGHTS: dict[tuple[str, str], Flight] = {
    ("UA1234", "2026-05-19"): Flight(
        "UA1234", "2026-05-19", "SFO", "ORD", "united", "N12345",
        "08:15", "cancelled", "weather + air-traffic-control",
    ),
    ("AA9876", "2026-05-19"): Flight(
        "AA9876", "2026-05-19", "ORD", "LHR", "american", "N67890",
        "18:30", "scheduled",
    ),
    ("UA0451", "2026-05-19"): Flight(
        "UA0451", "2026-05-19", "SFO", "ORD", "united", "N99999",
        "11:00", "scheduled",
    ),
    ("BA0287", "2026-05-19"): Flight(
        "BA0287", "2026-05-19", "ORD", "LHR", "british_airways", "N11111",
        "20:00", "scheduled",
    ),
    ("UA2200", "2026-05-19"): Flight(
        "UA2200", "2026-05-19", "DEL", "FRA", "united", "N22222",
        "01:45", "scheduled",
    ),
}


# --- Bookings ----------------------------------------------------------------


@dataclass
class Booking:
    id: str
    passenger_id: str
    flight_ids: list[str]   # ordered list of flight ids (legs)
    date: str
    fare_usd: int
    fare_class: str         # "Y" | "J" | "F"
    status: str             # "confirmed" | "checked_in" | "boarded" | "completed"
    refundable: bool


BOOKINGS: dict[str, Booking] = {
    "B-77001": Booking("B-77001", "P-1001", ["UA1234", "AA9876"], "2026-05-19", 4_280, "J", "checked_in", False),
    "B-77002": Booking("B-77002", "P-1002", ["UA1234"], "2026-05-19", 590, "Y", "confirmed", False),
    "B-77003": Booking("B-77003", "P-1003", ["UA1234", "BA0287"], "2026-05-19", 1_120, "Y", "confirmed", False),
    "B-77004": Booking("B-77004", "P-1004", ["UA0451"], "2026-05-19", 410, "Y", "confirmed", False),
    "B-77005": Booking("B-77005", "P-1005", ["UA2200"], "2026-05-19", 920, "Y", "confirmed", True),
}


# --- Tickets (prior support interactions) ------------------------------------


@dataclass
class Ticket:
    id: str
    passenger_id: str
    opened_at: str       # ISO date
    channel: str         # "phone" | "email" | "chat"
    category: str        # "schedule_change" | "refund" | "baggage" | "complaint" | "compliment"
    summary: str
    resolution: str
    compensation_usd: int


TICKETS: list[Ticket] = [
    Ticket("T-2001", "P-1001", "2026-04-08", "phone", "schedule_change",
           "Schedule change moved her business connection to a redeye",
           "Rebooked on direct flight; goodwill upgrade", 0),
    Ticket("T-2002", "P-1001", "2026-03-12", "email", "baggage",
           "Lost suitcase on LHR-SFO routing",
           "Bag located in 26h; $300 inconvenience credit issued", 300),
    Ticket("T-2003", "P-1001", "2026-02-21", "phone", "compliment",
           "Praised crew on UA0451; specific FA mentioned by name",
           "Compliment logged; crew notified", 0),
    Ticket("T-2004", "P-1001", "2025-11-04", "phone", "refund",
           "IRROPS cancellation cascade — missed wedding",
           "Full refund + 30K bonus miles + apology letter from VP", 0),
    Ticket("T-2005", "P-1001", "2025-08-30", "chat", "schedule_change",
           "Equipment swap dropped lie-flat seat she paid for",
           "Refund of fare-class differential ($650)", 650),
    Ticket("T-2006", "P-1001", "2025-07-19", "email", "complaint",
           "Frustration about app booking flow for multi-city trips",
           "Forwarded to product; standard apology", 0),
    Ticket("T-2007", "P-1002", "2026-03-30", "phone", "refund",
           "Cancelled trip after family emergency",
           "Refund per documented bereavement policy", 0),
    Ticket("T-2008", "P-1003", "2026-01-08", "email", "baggage",
           "Bag delayed by 4h, not lost",
           "Standard delivery; no comp issued (under policy threshold)", 0),
]


# --- Policies (the long, structured documents) ------------------------------


POLICIES: dict[str, str] = {
    "involuntary_cancellation": (
        "INVOLUNTARY CANCELLATION POLICY\n"
        "=====================================\n\n"
        "Section 1 — SCOPE\n"
        "This policy applies whenever a confirmed itinerary is cancelled by the "
        "carrier prior to scheduled departure, for any cause that is not "
        "attributable to the passenger. Causes include: equipment unavailability, "
        "crew availability, ATC programs, weather (when carrier-cancelled rather "
        "than passenger-elected), and any operational decision made by the "
        "carrier's operations center.\n\n"
        "Section 2 — STANDARD REMEDIES\n"
        "2.1 The carrier will, at the passenger's option, do ONE of the following:\n"
        "    (a) re-protect the passenger on the next available flight operated "
        "        by the carrier or a partner carrier, at no additional charge, "
        "        in the originally-booked cabin or higher;\n"
        "    (b) issue a full refund to the original form of payment, processed "
        "        within 7 business days;\n"
        "    (c) issue a non-refundable travel credit equal to 110% of fare paid, "
        "        valid for 12 months from issuance.\n\n"
        "2.2 If the cancellation results in an overnight stay away from the "
        "passenger's origin or final destination, the carrier will provide hotel "
        "accommodation and ground transportation, OR reimburse documented "
        "expenses up to USD 250 per night.\n\n"
        "Section 3 — TIER-SPECIFIC ENHANCEMENTS\n"
        "3.1 Platinum tier passengers are entitled to:\n"
        "    - Priority re-protection (must be offered the next available\n"
        "      flight in their booked cabin within 4 hours of original\n"
        "      departure, regardless of standard load-factor rules);\n"
        "    - Compensation equal to 25% of the affected segment's fare, in\n"
        "      addition to any refund or rebooking elected;\n"
        "    - Lounge access at the affected airport(s) regardless of the\n"
        "      passenger's cabin on the new itinerary.\n"
        "3.2 Gold tier passengers are entitled to:\n"
        "    - Priority re-protection (8-hour window);\n"
        "    - Compensation equal to 15% of the affected segment's fare;\n"
        "    - Lounge access where the carrier operates one at the affected\n"
        "      airport.\n"
        "3.3 Silver tier: priority re-protection (12-hour window). No additional\n"
        "    monetary compensation beyond standard remedies in Section 2.\n"
        "3.4 Blue tier: standard remedies in Section 2 only.\n\n"
        "Section 4 — CONNECTING ITINERARIES\n"
        "4.1 When a cancelled segment is part of a multi-segment itinerary,\n"
        "    the carrier is responsible for re-protecting the passenger on ALL\n"
        "    downstream segments to the final ticketed destination, even where\n"
        "    those segments are operated by partner carriers.\n"
        "4.2 If a downstream segment will be missed due to the upstream\n"
        "    cancellation, that segment must be cancelled by the carrier (not\n"
        "    the passenger) so it does not count as a no-show for the\n"
        "    passenger's record.\n"
        "4.3 Where the passenger's final destination cannot be reached the same\n"
        "    operational day, Section 2.2 applies in full.\n\n"
        "Section 5 — EXCLUSIONS\n"
        "5.1 This policy does NOT apply when cancellation is attributable to\n"
        "    the passenger (no-show, undocumented travel, denial of carriage\n"
        "    under the conditions of carriage Section 9).\n"
        "5.2 This policy does NOT apply to operational delays of less than 3\n"
        "    hours; see the Schedule Change Policy for those cases.\n\n"
        "Section 6 — INTERACTION WITH OTHER POLICIES\n"
        "Where this policy and the Schedule Change Policy could both apply,\n"
        "this policy controls if and only if the carrier explicitly designates\n"
        "the event as a cancellation in the operational record."
    ),
    "schedule_change": (
        "SCHEDULE CHANGE POLICY\n"
        "==========================\n\n"
        "Section 1 — SCOPE\n"
        "This policy governs all carrier-initiated changes to a confirmed\n"
        "itinerary that DO NOT rise to a cancellation. Typical cases include\n"
        "departure-time adjustments, equipment substitutions, and routing\n"
        "changes within the same operational day.\n\n"
        "Section 2 — DE MINIMIS CHANGES\n"
        "2.1 A change of less than 60 minutes in departure time, with no change\n"
        "    in routing or operating carrier, does not entitle the passenger\n"
        "    to alternative remedies; the new itinerary becomes the confirmed\n"
        "    itinerary automatically.\n\n"
        "Section 3 — SUBSTANTIVE CHANGES\n"
        "3.1 A change exceeding 60 minutes, OR any change of routing, OR a\n"
        "    change of operating carrier, entitles the passenger to either:\n"
        "    (a) accept the new itinerary;\n"
        "    (b) elect a refund to original form of payment;\n"
        "    (c) elect re-protection on any other itinerary the carrier can\n"
        "        offer within +/- 7 days at no additional charge.\n\n"
        "Section 4 — EQUIPMENT SUBSTITUTIONS\n"
        "4.1 If an equipment substitution removes a fare-class amenity the\n"
        "    passenger paid for (e.g., lie-flat seating in business class on\n"
        "    a longhaul segment), the passenger is entitled to the fare-class\n"
        "    differential as a refund.\n"
        "4.2 Where the substitution affects seat assignment but not amenity,\n"
        "    the carrier will re-seat the passenger as close as possible to\n"
        "    the original assignment; no compensation is owed."
    ),
    "bereavement": (
        "BEREAVEMENT POLICY\n"
        "==========================\n\n"
        "Documented passing of an immediate family member entitles a passenger\n"
        "to: (a) a full refund of any non-refundable fare; (b) waiver of any\n"
        "change fees on related travel; (c) re-protection without fare-class\n"
        "downgrade. Documentation must be provided within 30 days of the\n"
        "incident; the carrier may accept a death certificate, funeral home\n"
        "letterhead, or attending physician's note."
    ),
}
