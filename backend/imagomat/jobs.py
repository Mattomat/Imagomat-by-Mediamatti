"""Persistente, abbrechbare und wiederaufnehmbare Job-Queue.

Jeder Job ist eine Python-Funktion ``fn(ctx, **params)``. Sie meldet Fortschritt über
``ctx.progress`` und prüft ``ctx.cancelled``. Da jeder Verarbeitungsschritt pro Bild in der
Tabelle ``steps`` markiert wird, setzt ein neu gestarteter Job dort fort, wo der alte
aufgehört hat.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .db import Database

log = logging.getLogger(__name__)

JobFn = Callable[..., Any]
_REGISTRY: dict[str, JobFn] = {}


def job(kind: str) -> Callable[[JobFn], JobFn]:
    def deco(fn: JobFn) -> JobFn:
        _REGISTRY[kind] = fn
        return fn

    return deco


class Cancelled(Exception):
    pass


@dataclass
class JobContext:
    db: Database
    job_id: int
    listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _finish: threading.Event = field(default_factory=threading.Event)
    _progress: int = 0
    _total: int = 0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    @property
    def finish_requested(self) -> bool:
        """"Jetzt fertigstellen": langlaufende Jobs brechen die Sammelphase ab und nutzen das Bisherige."""
        return self._finish.is_set()

    def set_total(self, total: int) -> None:
        self._total = total
        self.db.update_job(self.job_id, total=total)
        self._emit()

    def progress(self, done: int | None = None, message: str | None = None, advance: int = 0) -> None:
        if done is not None:
            self._progress = done
        self._progress += advance
        fields: dict[str, Any] = {"progress": self._progress}
        if message is not None:
            fields["message"] = message
        self.db.update_job(self.job_id, **fields)
        self._emit(message)

    def _emit(self, message: str | None = None) -> None:
        evt = {"job_id": self.job_id, "progress": self._progress, "total": self._total, "message": message}
        for fn in list(self.listeners):
            try:
                fn(evt)
            except Exception:  # Listener dürfen den Job nie abbrechen
                log.exception("job listener failed")


LANES = ("shoot", "learn")
LEARN_KINDS = {"train_profile", "learn_people", "calibrate_culling", "feedback", "import_profile", "learn_look",
               "learn_template"}


def lane_of(kind: str) -> str:
    return "learn" if kind in LEARN_KINDS else "shoot"


class JobManager:
    """Führt Jobs nacheinander in einem Hintergrund-Thread aus."""

    def __init__(self, db: Database, recover: bool = False):
        self.db = db
        # Zwei Spuren: Shoots (Analyse, Culling, Export ...) und Lernen (Stil, Personen, Kalibrierung).
        # Ein stundenlanges Stil-Training blockiert so nie das Verarbeiten eines neuen Shoots.
        self._queues: dict[str, list[int]] = {lane: [] for lane in LANES}
        self._cv = threading.Condition()
        self._contexts: dict[int, JobContext] = {}
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self._threads: dict[str, threading.Thread] = {}
        self._stop = False
        # Beim Start der App: unterbrochene Jobs als pausiert markieren. Nicht im Terminal-Aufruf,
        # sonst würden laufende Jobs der geöffneten App fälschlich als pausiert gelten.
        if recover:
            with db.tx() as c:
                c.execute("UPDATE jobs SET status='paused' WHERE status IN ('running','queued')")

    def start(self) -> None:
        for lane in LANES:
            if lane not in self._threads:
                t = threading.Thread(target=self._loop, args=(lane,), name=f"imagomat-jobs-{lane}", daemon=True)
                self._threads[lane] = t
                t.start()

    def _enqueue(self, job_id: int, kind: str) -> None:
        with self._cv:
            self._queues[lane_of(kind)].append(job_id)
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for ctx in self._contexts.values():
            ctx._cancel.set()

    def active(self) -> bool:
        """Läuft gerade ein Shoot-Auftrag (Analyse, Entwickeln, Export) oder wartet einer?"""
        if self._queues.get("shoot"):
            return True
        for jid in list(self._contexts):
            j = self.db.job(jid)
            if j and lane_of(j["kind"]) == "shoot":
                return True
        return False

    def submit(self, kind: str, shoot_id: int | None = None, **params: Any) -> int:
        if kind not in _REGISTRY:
            raise KeyError(f"unbekannter Job-Typ: {kind}")
        job_id = self.db.create_job(kind, shoot_id, params)
        self._enqueue(job_id, kind)
        return job_id

    def resume(self, job_id: int) -> None:
        j = self.db.job(job_id)
        if not j or j["status"] not in ("paused", "failed", "cancelled"):
            return
        self.db.update_job(job_id, status="queued", error=None)
        self._enqueue(job_id, j["kind"])

    def cancel(self, job_id: int) -> None:
        ctx = self._contexts.get(job_id)
        if ctx:
            ctx._cancel.set()
        else:
            with self._cv:
                for q in self._queues.values():
                    if job_id in q:
                        q.remove(job_id)
            self.db.update_job(job_id, status="cancelled")

    def finish(self, job_id: int) -> bool:
        """Laufenden Job bitten, mit dem bisher Gesammelten abzuschliessen. False, wenn er nicht läuft."""
        ctx = self._contexts.get(job_id)
        if ctx is None:
            return False
        ctx._finish.set()
        return True

    def run_sync(self, job_id: int) -> dict[str, Any] | None:
        """Führt einen Job im aufrufenden Thread aus (CLI, Tests)."""
        self._run(job_id)
        return self.db.job(job_id)

    def _loop(self, lane: str = "shoot") -> None:
        while True:
            with self._cv:
                while not self._queues[lane] and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job_id = self._queues[lane].pop(0)
            self._run(job_id)

    def _run(self, job_id: int) -> None:
        j = self.db.job(job_id)
        if j is None:
            return
        fn = _REGISTRY[j["kind"]]
        ctx = JobContext(self.db, job_id, listeners=self.listeners)
        self._contexts[job_id] = ctx
        self.db.update_job(job_id, status="running", error=None)
        try:
            params = json.loads(j["params"] or "{}")
            if j["shoot_id"] is not None:
                params.setdefault("shoot_id", j["shoot_id"])
            fn(ctx, **params)
            self.db.update_job(job_id, status="done", message="fertig")
        except Cancelled:
            self.db.update_job(job_id, status="cancelled", message="abgebrochen")
        except Exception as e:  # noqa: BLE001
            log.exception("Job %s (%s) fehlgeschlagen", job_id, j["kind"])
            self.db.update_job(job_id, status="failed", error=f"{e}\n{traceback.format_exc()}")
        finally:
            self._contexts.pop(job_id, None)
            ctx._emit("status")
