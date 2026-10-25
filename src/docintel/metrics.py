"""In-process metrics with Prometheus text exposition (no external dependency)."""
import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = defaultdict(float)
        self.sums: dict[tuple, float] = defaultdict(float)
        self.counts: dict[tuple, int] = defaultdict(int)

    @staticmethod
    def _key(name: str, labels: dict) -> tuple:
        return (name, tuple(sorted(labels.items())))

    def inc(self, name: str, value: float = 1, **labels) -> None:
        with self._lock:
            self.counters[self._key(name, labels)] += value

    def observe(self, name: str, seconds: float, **labels) -> None:
        with self._lock:
            k = self._key(name, labels)
            self.sums[k] += seconds
            self.counts[k] += 1

    def observe_llm(self, kind: str, outcome: str, seconds: float) -> None:
        self.inc("docintel_llm_calls_total", kind=kind, outcome=outcome)
        if outcome == "ok":
            self.observe("docintel_llm_seconds", seconds, kind=kind)

    def snapshot(self) -> dict:
        with self._lock:
            def fmt(k):
                return k[0] + ("{" + ",".join(f"{a}={b}" for a, b in k[1]) + "}" if k[1] else "")
            out = {fmt(k): v for k, v in self.counters.items()}
            for k, s in self.sums.items():
                out[fmt(k) + "_avg"] = round(s / self.counts[k], 3)
            return out

    def prometheus(self) -> str:
        lines = []
        with self._lock:
            def lab(labels):
                return "{" + ",".join(f'{a}="{b}"' for a, b in labels) + "}" if labels else ""
            for (name, labels), v in sorted(self.counters.items()):
                lines.append(f"{name}{lab(labels)} {v:g}")
            for (name, labels), s in sorted(self.sums.items()):
                lines.append(f"{name}_sum{lab(labels)} {s:.4f}")
                lines.append(f"{name}_count{lab(labels)} {self.counts[(name, labels)]}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.sums.clear()
            self.counts.clear()


METRICS = Metrics()
