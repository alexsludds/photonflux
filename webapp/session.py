"""Live schematic session shared between the browser and notebook clients.

The editor is the *owner* of the schematic (it lives in the browser's
localStorage); the server is deliberately stateless about documents. This
module adds one small piece of shared state so other tools — a Jupyter
notebook via ``photonflux.nb``, primarily — can see and edit what is on the
canvas *right now*:

* the browser debounce-pushes its active tab here on every edit,
* a notebook reads it back, or pushes an edited document,
* both sides subscribe to change notifications (``/api/schematic/events``,
  server-sent events) so neither ever holds a stale copy.

The mirror is intentionally in-memory only: it is a live view, not storage.
On a dev auto-reload restart the browser re-seeds it as soon as its
EventSource reconnects.

A *document* here is what the browser mirrors for its active tab::

    {"title":     str,
     "schematic": {"instances": ..., "wires": ..., "probes": ..., ...},
     "analysis":  {...} | None,      # the analysis pane, run-config included
     "selection": {...} | None}      # what the user has selected on canvas

``rev`` increases on every accepted write; ``source`` names the writer
("browser" / "notebook" / ...) so clients can ignore their own echoes.
"""
from __future__ import annotations

import threading


class SchematicSession:
    """Thread-safe holder for the live document + change notification."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._rev = 0
        self._doc: dict | None = None
        self._source = ""

    def get(self) -> dict:
        with self._cond:
            return {"ok": True, "rev": self._rev, "source": self._source,
                    "doc": self._doc}

    def put(self, doc: dict, source: str,
            base_rev: int | None = None) -> dict:
        """Replace the document, bumping ``rev`` and waking event streams.

        With ``base_rev`` set, the write only lands if the mirror is still at
        that revision — the optimistic-concurrency check notebook
        read-modify-write edits use so they can't silently clobber a canvas
        edit that raced them. The browser pushes without it (its edits are
        the user's latest intent, they always win)."""
        with self._cond:
            if base_rev is not None and int(base_rev) != self._rev:
                return {"ok": False, "conflict": True, "rev": self._rev,
                        "error": f"rev {base_rev} is stale "
                                 f"(mirror is at {self._rev})"}
            self._rev += 1
            self._doc = doc
            self._source = source
            self._cond.notify_all()
            return {"ok": True, "rev": self._rev}

    def wait_change(self, since: int, timeout: float) -> tuple[int, str]:
        """Block until ``rev != since`` or ``timeout`` elapses; return the
        current ``(rev, source)`` either way. Event streams loop on this."""
        with self._cond:
            if self._rev == since:
                self._cond.wait(timeout)
            return self._rev, self._source


SESSION = SchematicSession()
