"""
GPU Scheduler Client for NVIDIA Grace Blackwell GB10 Cluster (Port 9090 v3/v4 Protocol).

Strictly complies with `gpu_scheduler_NVIDIA_GB10_128GB.py`:
- Queries `GET /status` to inspect current GPU lease and `mem_available_mib`.
- Acquires lease via `POST /acquire` (`{"who": ..., "timeout": ...}`) before running CUDA jobs.
- Releases lease via `POST /release` (`{"token": ...}`) in a guaranteed `finally` block.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class GPUSchedulerLease:
    """
    Context manager that acquires and releases the NVIDIA Grace Blackwell GB10 GPU Scheduler lock on port 9090.
    Automatically falls back to a no-op if running on a workstation without an active scheduler.
    """

    def __init__(
        self,
        who: str = "physpace_forge",
        timeout_sec: float = 600.0,
        scheduler_url: Optional[str] = None,
    ) -> None:
        self.who = who
        self.timeout_sec = timeout_sec
        port = os.environ.get("GB10_SCHEDULER_PORT", "9090")
        self.base_url = scheduler_url or f"http://127.0.0.1:{port}"
        self.token: Optional[str] = None
        self.lease_info: Dict[str, Any] = {}

    def get_status(self) -> Dict[str, Any]:
        """Query `GET /status` from the local GPU scheduler."""
        req = urllib.request.Request(f"{self.base_url}/status", method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def __enter__(self) -> "GPUSchedulerLease":
        try:
            status = self.get_status()
            print(
                f"  [Scheduler] Connected to {self.base_url} | "
                f"Current Owner: {status.get('current')} | "
                f"MemAvailable: {status.get('mem_available_mib')} MiB"
            )
        except Exception as exc:
            print(f"  [Scheduler] Local scheduler not reachable ({exc}); proceeding in direct mode.")
            return self

        payload = json.dumps({"who": self.who, "timeout": self.timeout_sec}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/acquire",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec + 10.0) as resp:
            self.lease_info = json.loads(resp.read().decode("utf-8"))
            self.token = self.lease_info.get("token")
            print(
                f"  [Scheduler] LEASE ACQUIRED by '{self.who}' | "
                f"Token: {self.token} | "
                f"MemAvailable: {self.lease_info.get('mem_available_mib', 'N/A')} MiB"
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.token:
            return
        try:
            payload = json.dumps({"token": self.token}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/release",
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                print(
                    f"  [Scheduler] LEASE RELEASED for token {self.token} | "
                    f"Status: {res.get('ok', True)}"
                )
        except Exception as exc:
            print(f"  [Scheduler] Warning during release of token {self.token}: {exc}")
        finally:
            self.token = None
