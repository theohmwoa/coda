"""Build a morning brief from recent Gmail inbox messages and optionally post to Discord."""


def gmail_morning_brief(hours: int = 24, max_results: int = 25, post: bool = True) -> str:
    """Build a morning brief from recent Gmail inbox messages and optionally post to Discord.

    Searches the inbox for messages newer than `hours`, triages each one, then
    formats a markdown brief grouping high/medium-importance items and summarizing
    the rest. If `post` is True, sends the brief to the Discord announcements channel.
    Returns the brief string either way.
    """
    from collections import Counter
    from datetime import datetime

    # Gmail search uses day-granularity tokens; convert hours -> days (min 1)
    days = max(1, round(hours / 24))
    raw = gmail.search_emails(query=f"newer_than:{days}d in:inbox", maxResults=max_results)

    # Parse the text response into dicts
    blocks = [b.strip() for b in str(raw).split("\n\n") if b.strip()]
    emails = []
    for b in blocks:
        d = {}
        for line in b.splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                d[k.strip().lower()] = v.strip()
        if d:
            emails.append(d)

    # Triage each one
    triaged = []
    for e in emails:
        t = triage_email(e)
        triaged.append({
            "email": e,
            "importance": t.importance,
            "category": t.category,
            "needs_reply": t.needs_reply,
            "one_line": t.one_line,
        })

    high = [t for t in triaged if t["importance"] == "high"]
    med  = [t for t in triaged if t["importance"] == "medium"]
    needs_reply = [t for t in triaged if t["needs_reply"]]
    cats = Counter(t["category"] for t in triaged)

    today = datetime.now().strftime("%a %d %b %Y")
    lines = [
        f"**☀️ Morning Brief — {today}**",
        f"_Scanned {len(triaged)} inbox messages from the last {hours}h_",
        "",
    ]

    def sender_of(t):
        f = t["email"].get("from", "")
        return f.split("<")[0].strip() or f

    if high:
        lines.append("**🔴 Needs attention**")
        for t in high:
            lines.append(f"• **{sender_of(t)}** — {t['one_line']}")
        lines.append("")

    if med:
        lines.append("**🟡 Worth a look**")
        for t in med:
            lines.append(f"• **{sender_of(t)}** — {t['one_line']}")
        lines.append("")

    lines.append("**📊 The rest**")
    lines.append(
        f"• {cats.get('promotional', 0)} promotional · "
        f"{cats.get('newsletter', 0)} newsletters · "
        f"{cats.get('notification', 0)} notifications"
    )
    if needs_reply:
        lines.append(f"• ✉️ {len(needs_reply)} message(s) flagged as needing a reply")

    brief = "\n".join(lines)

    if post:
        post_to_discord(message=brief)

    return brief
