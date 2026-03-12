"""
KaggleClient — the bridge between AutoResearch and Kaggle Kernels.

Every interaction with Kaggle goes through here. No agent touches
the Kaggle SDK directly — they all use this wrapper.

Design principles:
    - Plain English errors. "Kernel timed out" not "HTTPError 408"
    - Async polling so N kernels run in parallel
    - Retry with exponential backoff on transient failures
    - Never silently swallow errors

Plain English note (shown to user when things go wrong):
    Kaggle's API is occasionally slow or returns errors even when
    everything is fine on your end. AutoResearch retries automatically.
    If it fails 3 times in a row, it will tell you exactly what happened.
"""

import asyncio
import json
import logging
import os
import subprocess

# Force UTF-8 I/O on Windows so generated notebook code (with emojis/unicode)
# doesn't crash when written to disk.
os.environ.setdefault("PYTHONUTF8", "1")
import tempfile
import time
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import nbformat
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def _make_slug(text: str) -> str:
    """Convert a title to a Kaggle-compatible slug (lowercase, hyphens, max 50 chars)."""
    import re as _re
    slug = text.lower()
    slug = _re.sub(r"[\s_]+", "-", slug)
    slug = _re.sub(r"[^a-z0-9-]", "", slug)
    slug = _re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:50]


# ── Kernel status ─────────────────────────────────────────────────────────────

# Kaggle SDK v2+ returns status as an integer in some versions
KAGGLE_STATUS_MAP = {
    0: "queued",
    1: "running",
    2: "complete",
    3: "error",
    4: "cancel",
    5: "cancelack",
}


def _parse_kernel_status(status) -> str:
    """Normalize Kaggle kernel status to a lowercase string regardless of type."""
    if isinstance(status, int):
        return KAGGLE_STATUS_MAP.get(status, "unknown")
    if hasattr(status, "value"):   # enum object
        return str(status.value).lower()
    return str(status).lower()


class KernelStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETE  = "complete"
    ERROR     = "error"
    CANCELLED = "cancelled"
    UNKNOWN   = "unknown"

    @classmethod
    def from_kaggle(cls, raw) -> "KernelStatus":
        """Map Kaggle's status (int, enum, or string) to our enum."""
        mapping = {
            "queue":     cls.QUEUED,
            "queued":    cls.QUEUED,
            "running":   cls.RUNNING,
            "complete":  cls.COMPLETE,
            "error":     cls.ERROR,
            "cancelled": cls.CANCELLED,
            "cancel":    cls.CANCELLED,
            "cancelack": cls.CANCELLED,
        }
        return mapping.get(_parse_kernel_status(raw), cls.UNKNOWN)


@dataclass
class KernelPushResult:
    """What we get back after pushing a kernel."""
    kernel_slug: str       # e.g. "username/autoresearch-xgboost-abc123"
    kernel_url: str        # Direct link the researcher can visit
    version: int


@dataclass
class KernelPollResult:
    """What we get back when polling a kernel's status."""
    kernel_slug: str
    status: KernelStatus
    runtime_seconds: Optional[float]
    error_message: Optional[str]    # Only set on ERROR status
    log_tail: Optional[str]         # Last 50 lines of output


@dataclass
class KernelOutput:
    """Files produced by a completed kernel."""
    kernel_slug: str
    files: dict[str, bytes]         # filename → file contents
    metrics_json: Optional[dict]    # Parsed metrics.json if present
    stdout_log: str


# ── Exceptions ────────────────────────────────────────────────────────────────

class KaggleAuthError(Exception):
    """Credentials are wrong or missing."""
    pass

class KernelPushError(Exception):
    """Failed to push a kernel to Kaggle."""
    pass

class KernelTimeoutError(Exception):
    """Kernel exceeded max_runtime_hours."""
    pass

class KernelExecutionError(Exception):
    """Kernel ran but finished with an error."""
    def __init__(self, message: str, log_tail: str = ""):
        super().__init__(message)
        self.log_tail = log_tail


# ── Client ────────────────────────────────────────────────────────────────────

class KaggleClient:
    """
    Wraps the Kaggle Python SDK with:
      - Plain English error messages
      - Async parallel polling
      - Auto-retry on transient failures
      - Mentor-voice progress updates

    Usage:
        client = KaggleClient(
            username="your-kaggle-username",
            key="your-kaggle-key",
        )
        result = await client.push_and_wait(notebook, dataset_sources=["titanic/titanic"])
    """

    # How often to check if a kernel is done
    POLL_INTERVAL_SECONDS = 30

    def __init__(
        self,
        username: str,
        key: str,
        max_runtime_hours: float = 4.0,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        verbose: bool = True,
    ):
        self.username          = username
        self.max_runtime_hours = max_runtime_hours
        self.enable_gpu        = enable_gpu
        self.enable_internet   = enable_internet
        self.verbose           = verbose

        # Set credentials as env vars (kaggle SDK reads from here)
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"]      = key

        # Import here so missing kaggle package gives a clear error
        try:
            import kaggle
            self._api = kaggle.KaggleApi()
            self._api.authenticate()
        except ImportError:
            raise ImportError(
                "The 'kaggle' package is not installed. "
                "Run: pip install kaggle"
            )
        except Exception as e:
            raise KaggleAuthError(
                f"Kaggle authentication failed. "
                f"Check your username and API key in config.yaml.\n"
                f"You can get your key at: https://www.kaggle.com/settings → API\n"
                f"Technical detail: {e}"
            )

    # ── Push ──────────────────────────────────────────────────────────────────

    def push_kernel(
        self,
        notebook: nbformat.NotebookNode,
        kernel_slug_suffix: str,          # e.g. "xgboost-baseline"
        dataset_sources: list[str],        # e.g. ["username/titanic"]
        competition_sources: list[str],    # e.g. ["titanic"]
        force_cpu: bool = False,
        title: Optional[str] = None,
    ) -> KernelPushResult:
        """
        Push a notebook to Kaggle and start running it.

        The kernel slug will be: {username}/autoresearch-{suffix}
        This is what we use to poll and fetch output later.
        """
        # Kaggle requires title == slug-part-of-id (it derives the slug from the title)
        slug   = _make_slug(f"autoresearch-{kernel_slug_suffix}")
        title_ = slug  # title must equal the slug, not a human-readable string

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write notebook — open with explicit UTF-8 so Windows doesn't use charmap
            nb_path = tmppath / f"{slug}.ipynb"
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

            # Write kernel metadata
            use_gpu = self.enable_gpu and not force_cpu
            metadata = {
                "id":                  f"{self.username}/{slug}",
                "title":               title_,
                "code_file":           f"{slug}.ipynb",
                "language":            "python",
                "kernel_type":         "notebook",
                "is_private":          True,
                "enable_gpu":          use_gpu,
                "enable_tpu":          False,
                "enable_internet":     self.enable_internet,
                "dataset_sources":     dataset_sources,
                "competition_sources": competition_sources,
                "kernel_sources":      [],
            }
            meta_path = tmppath / "kernel-metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            if self.verbose:
                gpu_note = "GPU enabled" if use_gpu else "CPU only (saves your quota)"
                print(f"  → Pushing kernel '{slug}' to Kaggle ({gpu_note})...")

            try:
                self._push_with_retry(str(tmppath))
            except Exception as e:
                raise KernelPushError(
                    f"Failed to push the '{kernel_slug_suffix}' notebook to Kaggle.\n"
                    f"This is often a temporary Kaggle API issue — AutoResearch will retry.\n"
                    f"Technical detail: {e}"
                )

        kernel_url = f"https://www.kaggle.com/code/{self.username}/{slug}"

        if self.verbose:
            print(f"  ✓ Kernel pushed. You can watch it run at:\n    {kernel_url}")

        return KernelPushResult(
            kernel_slug=f"{self.username}/{slug}",
            kernel_url=kernel_url,
            version=1,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _push_with_retry(self, folder_path: str) -> None:
        """Push with automatic retry on transient failures."""
        self._api.kernels_push(folder_path)

    # ── Poll ──────────────────────────────────────────────────────────────────

    def poll_kernel(self, kernel_slug: str) -> KernelPollResult:
        """Check the current status of a kernel via the Kaggle CLI."""
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "status", kernel_slug],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (result.stdout + result.stderr).lower()

            if "complete" in output or "success" in output:
                status = KernelStatus.COMPLETE
            elif "running" in output:
                status = KernelStatus.RUNNING
            elif "error" in output or "fail" in output:
                status = KernelStatus.ERROR
            elif "cancel" in output:
                status = KernelStatus.CANCELLED
            else:
                status = KernelStatus.QUEUED

            error_msg = result.stderr.strip() if status == KernelStatus.ERROR else None
            return KernelPollResult(
                kernel_slug=kernel_slug,
                status=status,
                runtime_seconds=None,
                error_message=error_msg,
                log_tail=None,
            )
        except Exception as e:
            logger.warning(f"Poll failed for {kernel_slug}: {e} — retrying")
            return KernelPollResult(
                kernel_slug=kernel_slug,
                status=KernelStatus.UNKNOWN,
                runtime_seconds=None,
                error_message=str(e),
                log_tail=None,
            )

    async def wait_for_kernel(
        self,
        kernel_slug: str,
        method_name: str = "experiment",
    ) -> KernelPollResult:
        """
        Async: poll a kernel until it finishes or times out.

        Prints mentor-voice progress so the researcher knows what's happening.
        """
        max_polls   = int((self.max_runtime_hours * 3600) / self.POLL_INTERVAL_SECONDS)
        start_time  = time.time()
        last_status = None

        for poll_num in range(max_polls):
            result = self.poll_kernel(kernel_slug)

            # Print update only when status changes
            if result.status != last_status:
                elapsed = (time.time() - start_time) / 60
                self._print_status_update(method_name, result, elapsed)
                last_status = result.status

            if result.status == KernelStatus.COMPLETE:
                if self.verbose:
                    elapsed = (time.time() - start_time) / 60
                    print(f"  ✅ '{method_name}' finished in {elapsed:.1f} min")
                return result

            if result.status == KernelStatus.ERROR:
                log_tail = self._fetch_log_tail(kernel_slug)
                raise KernelExecutionError(
                    f"The '{method_name}' experiment failed on Kaggle.\n"
                    f"Error: {result.error_message or 'Unknown error'}\n"
                    f"AutoResearch will read the error and try to fix it automatically.",
                    log_tail=log_tail or "",
                )

            if result.status == KernelStatus.CANCELLED:
                raise KernelExecutionError(
                    f"The '{method_name}' kernel was cancelled on Kaggle. "
                    f"This sometimes happens due to Kaggle maintenance. "
                    f"AutoResearch will re-push it automatically."
                )

            # Wait before next poll
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

        raise KernelTimeoutError(
            f"The '{method_name}' experiment exceeded your {self.max_runtime_hours}hr limit.\n"
            f"This can happen with very large datasets or complex deep learning models.\n"
            f"Try reducing max_runtime_hours in config.yaml, or use a simpler method."
        )

    async def wait_for_all_kernels(
        self,
        kernels: list[tuple[str, str]],  # [(slug, method_name), ...]
    ) -> list[KernelPollResult]:
        """
        Async: wait for all N kernels in parallel.
        Returns results in the same order as input.
        """
        if self.verbose:
            print(f"\n  ⏳ Running {len(kernels)} experiments in parallel on Kaggle...")
            print(f"     (They run simultaneously — total wait is the slowest one, not the sum)\n")

        tasks = [
            self.wait_for_kernel(slug, name)
            for slug, name in kernels
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes from failures
        successes = []
        for (slug, name), result in zip(kernels, results):
            if isinstance(result, Exception):
                logger.error(f"Kernel {name} failed: {result}")
                successes.append(None)  # None signals failure to caller
            else:
                successes.append(result)

        return successes

    # ── Fetch output ──────────────────────────────────────────────────────────

    def fetch_output(self, kernel_slug: str) -> KernelOutput:
        """
        Download all output files from a completed kernel via the Kaggle CLI.

        Uses: kaggle kernels output <slug> -p <path>
        Avoids the Python SDK's connection pool and DNS issues.

        AutoResearch expects the kernel to have written:
          - /kaggle/working/eda_output.json  (EDA stage)
          - /kaggle/working/metrics.json     (execution stage)
          - /kaggle/working/model.*          (execution stage)
          - /kaggle/working/predictions.csv  (optional)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            result = subprocess.run(
                ["kaggle", "kernels", "output", kernel_slug, "-p", str(tmppath)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Couldn't download output from Kaggle for kernel '{kernel_slug}'.\n"
                    f"Make sure the kernel completed successfully.\n"
                    f"Technical detail: {result.stderr.strip()}"
                )

            files = {}
            for f in tmppath.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix == ".zip":
                    with zipfile.ZipFile(f) as zf:
                        for name in zf.namelist():
                            files[name] = zf.read(name)
                else:
                    files[f.name] = f.read_bytes()

            if not files:
                all_found = [str(p.relative_to(tmppath)) for p in tmppath.rglob("*") if p.is_file()]
                raise RuntimeError(
                    f"No output files found for kernel '{kernel_slug}'.\n"
                    f"CLI output: {result.stdout.strip()}\n"
                    f"Files in download dir: {all_found}"
                )

        # Parse metrics.json if present
        metrics = None
        if "metrics.json" in files:
            try:
                metrics = json.loads(files["metrics.json"].decode())
            except Exception:
                logger.warning(f"Could not parse metrics.json from {kernel_slug}")

        # Get stdout log if present
        stdout = files.get("stdout.txt", b"").decode(errors="replace")

        return KernelOutput(
            kernel_slug=kernel_slug,
            files=files,
            metrics_json=metrics,
            stdout_log=stdout,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_log_tail(self, kernel_slug: str, lines: int = 50) -> Optional[str]:
        """Fetch the last N lines of kernel output for error diagnosis."""
        try:
            output = self.fetch_output(kernel_slug)
            log    = output.stdout_log
            if log:
                return "\n".join(log.splitlines()[-lines:])
        except Exception:
            pass
        return None

    def _print_status_update(
        self,
        method_name: str,
        result: KernelPollResult,
        elapsed_min: float,
    ) -> None:
        """Print mentor-voice status updates."""
        if not self.verbose:
            return

        status_messages = {
            KernelStatus.QUEUED:  f"  ⏳ '{method_name}' is waiting in Kaggle's queue...",
            KernelStatus.RUNNING: f"  🔄 '{method_name}' is running on Kaggle ({elapsed_min:.0f} min elapsed)...",
            KernelStatus.UNKNOWN: f"  ❓ '{method_name}' status unclear — still checking...",
        }
        msg = status_messages.get(result.status)
        if msg:
            print(msg)

    def verify_competition_access(self, competition_slug: str) -> bool:
        """
        Check if the user has accepted the competition rules.

        Kaggle requires manual rule acceptance on the website before
        the API can download competition data.
        """
        try:
            files = self._api.competition_list_files(competition_slug)
            return len(files) > 0
        except Exception as e:
            if "403" in str(e) or "rules" in str(e).lower():
                print(
                    f"\n  ⚠️  You need to accept the competition rules first.\n"
                    f"     Visit: https://www.kaggle.com/c/{competition_slug}/rules\n"
                    f"     Click 'I Understand and Accept', then re-run AutoResearch.\n"
                )
                return False
            raise
