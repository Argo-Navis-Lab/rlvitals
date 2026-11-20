import json
import math
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from rlvitals.hooks import discord_body, event_body, only, slack_body, webhook
from rlvitals.monitor import Event
from rlvitals.watch import Watch

SPIKE = Event(301, "spike", "kl", "WARN", {"value": 5.0, "median": 1.0, "mad": 0.25, "z": 10.79})
HALT = Event(290, "divergence", "grad_norm", "HALT", {"value": float("inf")})
LINE = "rlvitals: step 301: WARN, spike in kl (value 5, median 1, mad 0.25, z 10.79)"


class Inbox(ThreadingHTTPServer):
    """A web server on this machine that keeps every POST it gets and answers `status`."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.posts: list[tuple[str, dict[str, str], Any]] = []
        self.status = 200

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


def refuse(constant: str) -> None:
    raise ValueError(f"{constant} is not JSON")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        assert isinstance(self.server, Inbox)
        length = int(self.headers["Content-Length"])
        self.server.posts.append(
            (
                self.path,
                dict(self.headers),
                json.loads(self.rfile.read(length), parse_constant=refuse),
            )
        )
        self.send_response(self.server.status)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the test output quiet


@pytest.fixture
def inbox() -> Iterator[Inbox]:
    server = Inbox()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_the_event_body_holds_the_line_and_every_field() -> None:
    assert event_body(SPIKE) == {
        "text": LINE,
        "step": 301,
        "detector": "spike",
        "metric": "kl",
        "severity": "WARN",
        "details": {"value": 5.0, "median": 1.0, "mad": 0.25, "z": 10.79},
    }
    assert slack_body(SPIKE) == {"text": LINE}
    assert discord_body(SPIKE) == {"content": LINE}


def test_nan_and_the_infinities_go_as_words() -> None:
    event = Event(7, "divergence", "loss", "HALT", {"value": math.nan, "low": -math.inf})
    assert event_body(event)["details"] == {"value": "nan", "low": "-inf"}
    assert event_body(HALT)["details"] == {"value": "inf"}


def test_a_body_with_a_nan_is_refused_before_it_is_sent(inbox: Inbox) -> None:
    hook = webhook(inbox.url, body=lambda event: {"value": math.nan})
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        hook(SPIKE)
    assert inbox.posts == []


def test_a_webhook_posts_the_event_as_json(inbox: Inbox) -> None:
    hook = webhook(inbox.url + "/alerts")
    hook(SPIKE)
    hook(HALT)
    [(path, headers, body), (_, _, halt)] = inbox.posts
    assert path == "/alerts"
    assert headers["Content-Type"] == "application/json"
    assert body == event_body(SPIKE)
    assert halt["details"] == {"value": "inf"}


def test_a_webhook_sends_the_body_and_headers_it_is_given(inbox: Inbox) -> None:
    hook = webhook(inbox.url, body=slack_body, headers={"Authorization": "Bearer t0ken"})
    hook(SPIKE)
    [(_, headers, body)] = inbox.posts
    assert headers["Authorization"] == "Bearer t0ken"
    assert body == {"text": LINE}


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "example.com"])
def test_only_http_and_https_urls_are_taken(url: str) -> None:
    with pytest.raises(ValueError, match="must start with http:// or https://"):
        webhook(url)


def test_a_refused_post_is_a_warning_that_never_names_the_url(inbox: Inbox) -> None:
    inbox.status = 500
    seen: list[Event] = []
    watch = Watch(hooks=[webhook(inbox.url + "/hooks/s3cr3t"), seen.append])
    with pytest.warns(RuntimeWarning, match="HTTPError 500") as warned:
        watch.step(0, {"loss": 0.5, "grad_norm": float("nan")})
    assert len(warned.list) == 1
    assert "s3cr3t" not in str(warned.list[0].message)
    assert [(event.step, event.severity) for event in seen] == [(0, "HALT")]
    assert len(inbox.posts) == 1


def test_only_passes_on_the_severities_it_is_given() -> None:
    seen: list[Event] = []
    hook = only({"HALT", "SNAPSHOT"}, seen.append)
    hook(SPIKE)
    hook(HALT)
    assert seen == [HALT]
