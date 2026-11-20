"""Alert hooks that reach a person: a webhook posts each event as JSON to a URL (Slack, Discord,
ntfy, a server of your own), and `only` passes on just the severities you care about. Standard
library only. A post blocks the step it is made in, at most `timeout` seconds; events are rare,
so that is cheap, and a post that fails becomes the watch's warning, never the run's end."""

import json
import math
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Mapping
from typing import Any

from rlvitals.monitor import Event, Severity
from rlvitals.watch import Hook, format_event

Body = Callable[[Event], dict[str, Any]]


def number(value: float) -> float | str:
    """A number JSON can carry: NaN and the infinities aren't JSON, so they go as "nan", "inf"
    and "-inf"."""
    return value if math.isfinite(value) else str(value)


def event_body(event: Event) -> dict[str, Any]:
    """Everything about the event, and its line under "text" for a person to read."""
    return {
        "text": format_event(event),
        "step": event.step,
        "detector": event.detector,
        "metric": event.metric,
        "severity": event.severity,
        "details": {key: number(value) for key, value in event.details.items()},
    }


def slack_body(event: Event) -> dict[str, Any]:
    """What a Slack incoming webhook takes: the line, as "text"."""
    return {"text": format_event(event)}


def discord_body(event: Event) -> dict[str, Any]:
    """What a Discord webhook takes: the line, as "content"."""
    return {"content": format_event(event)}


def webhook(
    url: str,
    *,
    body: Body = event_body,
    headers: Mapping[str, str] | None = None,
    timeout: float = 5.0,
) -> Hook:
    """A hook that POSTs body(event) as JSON to `url`. Only http and https URLs are taken. A
    webhook URL is often its own password, so the hook never names it: not in its repr, not in
    its errors. A body that isn't strict JSON (a NaN in it), an answer of 400 or more, or no
    answer within `timeout` raises."""
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError("a webhook URL must start with http:// or https://")
    sent = {"Content-Type": "application/json", **(headers or {})}

    def post(event: Event) -> None:
        data = json.dumps(body(event), allow_nan=False).encode()
        request = urllib.request.Request(url, data=data, headers=sent, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            answer.read()

    return post


def only(severities: Collection[Severity], hook: Hook) -> Hook:
    """`hook`, seeing only events of these severities: only({"HALT"}, webhook(...))."""
    wanted = frozenset(severities)

    def filtered(event: Event) -> None:
        if event.severity in wanted:
            hook(event)

    return filtered
