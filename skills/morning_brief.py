"""Fetch recent Gmail, triage with the triage_email sub-agent, post a"""


def morning_brief(hours_back: int = 24, max_results: int = 25) -> str:
    """Fetch recent Gmail, triage with the triage_email sub-agent, post a
    Discord summary, and return the brief text.

    Assumes the runtime exposes:
      - `gmail` MCP server with search_emails / read_email
      - `triage_email` sub-agent returning an object with .importance,
        .needs_reply, .one_line, .category
      - `post_to_discord(message=...)` inline tool
    """
    import re
    from datetime import date as _date

    # 1. Search
    search_result = gmail.search_emails(
        query=f"newer_than:{hours_back}h category:primary",
        maxResults=max_results,
    )

    entries = re.findall(
        r"ID:\s*(\S+)\s*\nSubject:\s*(.*?)\nFrom:\s*(.*?)\nDate:\s*(.*?)(?=\n\n|\Z)",
        search_result,
        re.DOTALL,
    )

    # 2. Read each + triage
    triaged = []
    for msg_id, subject, sender, _date_str in entries:
        full = gmail.read_email(messageId=msg_id)
        parts = full.split("\n\n", 1)
        body = parts[1] if len(parts) > 1 else ""
        snippet = body.strip()[:200]
        msg = {
            "id": msg_id,
            "subject": subject.strip(),
            "from": sender.strip(),
            "snippet": snippet,
        }
        verdict = triage_email(email={
            "from": msg["from"],
            "subject": msg["subject"],
            "snippet": msg["snippet"],
        })
        triaged.append((msg, verdict))

    # 3. Group
    critical = [(m, v) for m, v in triaged if v.importance == "critical"]
    high = [(m, v) for m, v in triaged
            if (v.importance == "high" or v.needs_reply) and v.importance != "critical"]
    normal = [(m, v) for m, v in triaged if v.importance == "normal"]
    low = [(m, v) for m, v in triaged if v.importance == "low"]

    lines = [f"**📬 Morning brief — {_date.today().isoformat()}**"]
    lines.append(f"**🔴 Critical:** {len(critical)}")
    for m, v in critical:
        lines.append(f"- {m['from']}: {v.one_line}")
    lines.append(f"**🟠 High (needs reply):** {len(high)}")
    for m, v in high:
        lines.append(f"- {m['from']}: {v.one_line}")
    lines.append(f"**🟢 Normal:** {len(normal)}  • **⚪ Low:** {len(low)}")

    brief = "\n".join(lines)
    if len(brief) > 1800:
        brief = brief[:1797] + "..."

    # 4. Post to Discord
    post_to_discord(message=brief)

    return brief
