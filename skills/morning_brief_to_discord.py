"""Triage recent Gmail inbox messages and post a morning brief to Discord."""


def morning_brief_to_discord(hours: int = 24, max_results: int = 25) -> dict:
    """Triage recent Gmail inbox messages and post a morning brief to Discord.

    Pulls inbox messages from the last `hours` hours, runs the triage sub-agent
    on each, formats a markdown brief (high-priority, needs-reply, worth-a-look,
    low-priority count), and posts it to the announcements Discord channel.

    Returns a dict with the brief text, the parsed triage list, and the Discord
    post response.
    """
    import re
    from collections import Counter

    raw = gmail.search_emails(query=f"newer_than:{hours}h in:inbox", maxResults=max_results)

    # Parse the string response into structured entries
    emails = []
    if isinstance(raw, str):
        for block in re.split(r"\n\s*\n", raw.strip()):
            d = {}
            for line in block.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    d[k.strip().lower()] = v.strip()
            if d.get("id"):
                emails.append({
                    "id": d["id"],
                    "subject": d.get("subject", ""),
                    "from": d.get("from", ""),
                    "date": d.get("date", ""),
                })
    elif isinstance(raw, list):
        emails = raw

    # Triage each
    triaged = []
    for e in emails:
        t = triage_email(email=e)
        triaged.append({
            **e,
            "importance": t.importance,
            "category": t.category,
            "needs_reply": t.needs_reply,
            "one_line": t.one_line,
        })

    high = [t for t in triaged if t["importance"] == "high"]
    normal = [t for t in triaged if t["importance"] == "normal"]
    low = [t for t in triaged if t["importance"] == "low"]
    needs_reply = [t for t in triaged if t["needs_reply"]]
    cat_counts = Counter(t["category"] for t in triaged)

    def short_from(s: str) -> str:
        m = re.match(r"^(.*?)\s*<", s)
        return (m.group(1).strip() if m else s).strip('"')

    # Date header from first message if present
    from datetime import datetime
    header_date = datetime.now().strftime("%a %d %b %Y")

    lines = [f"**☀️ Morning Brief — {header_date}**"]
    lines.append(
        f"Inbox last {hours}h: **{len(triaged)} messages** · "
        f"{len(high)} high · {len(normal)} normal · {len(low)} low"
    )
    if cat_counts:
        lines.append("Mix: " + ", ".join(f"{k}: {v}" for k, v in cat_counts.most_common()))
    lines.append("")

    if high:
        lines.append("**🔴 High priority**")
        for t in high:
            lines.append(f"• *{short_from(t['from'])}* — {t['one_line']}")
        lines.append("")

    if needs_reply:
        lines.append("**✉️ Needs a reply**")
        for t in needs_reply:
            lines.append(f"• *{short_from(t['from'])}* — {t['subject']}")
        lines.append("")
    else:
        lines.append("**✉️ Needs a reply:** none — you're clear ✅")
        lines.append("")

    if normal:
        lines.append("**🟡 Worth a look**")
        for t in normal:
            lines.append(f"• *{short_from(t['from'])}* — {t['one_line']}")
        lines.append("")

    lines.append(f"_Plus {len(low)} low-priority items (newsletters, promos, automated notices)._")
    brief = "\n".join(lines)

    resp = post_to_discord(message=brief)
    return {"brief": brief, "triaged": triaged, "discord": resp}
