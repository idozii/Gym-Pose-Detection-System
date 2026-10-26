"""Read-only six-model inference deployment preflight (no landmarks transmitted).

Use with an owner-controlled HTTPS FastAPI host. Never uploads webcam frames,
publisher media, source features, or model bundles.
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from gym_pose.model_registry import KNOWN_TASKS

EXPECTED = tuple(f"{exercise}/{target}" for exercise, target in KNOWN_TASKS)


def canonical_api_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    localhost = host in {"localhost", "127.0.0.1", "::1"}
    if (
        not host or (parsed.scheme != "https" and
                     not (parsed.scheme == "http" and localhost))
        or parsed.username or parsed.password or parsed.path
        or parsed.query or parsed.fragment
    ):
        raise ValueError("API must be an HTTPS origin (HTTP localhost is permitted)")
    return url


def check_health_payload(payload: dict) -> dict:
    """Validate the exact public API contract, never printed model internals."""
    models = payload.get("models") if isinstance(payload, dict) else None
    statuses = models if isinstance(models, dict) else {}
    missing = [name for name in EXPECTED if statuses.get(name) != "loaded_experimental_only"]
    declared_ready = (
        payload.get("research_only") is True
        and payload.get("pose_estimator_source_parity_validated") is False
        and payload.get("ready_for_six_detector_demo") is True
        and payload.get("loaded_model_count") == len(EXPECTED)
        and payload.get("expected_model_count") == len(EXPECTED)
        and payload.get("status") == "research_api_ready"
    ) if isinstance(payload, dict) else False
    ready = declared_ready and not missing and set(statuses) == set(EXPECTED)
    return {
        "ready": bool(ready),
        "loaded": len(EXPECTED) - len(missing),
        "expected": len(EXPECTED),
        "unavailable": missing,
        "source_parity_validated": False,
        "research_only": True,
    }


def read_json(url: str, *, timeout: float) -> dict:
    req = Request(url, headers={"Accept": "application/json", "Cache-Control": "no-store"})
    with urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Unexpected server response")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True, help="HTTPS API origin (no /health suffix)")
    parser.add_argument("--timeout", type=float, default=8.0, help="Seconds per request")
    args = parser.parse_args(argv)
    try:
        url = canonical_api_url(args.api_url)
        if not 0 < args.timeout <= 30:
            raise ValueError("Timeout must be within (0, 30] seconds")
        assessment = check_health_payload(read_json(url + "/health", timeout=args.timeout))
        if assessment["ready"]:
            probe = read_json(url + "/ready", timeout=args.timeout)
            if probe.get("status") != "ready" or probe.get("loaded_model_count") != 6:
                assessment["ready"] = False
                assessment["reason"] = "Readiness endpoint disagrees with /health"
        print(json.dumps(assessment, indent=2))
        return 0 if assessment["ready"] else 1
    except (ValueError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({"ready": False, "error": type(exc).__name__,
                          "message": "API health/readiness check failed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
