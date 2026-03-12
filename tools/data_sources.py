"""
Data Sources — connectors for each data source type.

Handles the difference between:
  - Kaggle datasets (mounted at /kaggle/input/ automatically)
  - HuggingFace datasets (download in kernel via datasets library)
  - Google Drive (rclone/gdown into kernel)
  - Manual CSV uploads (staged via Kaggle dataset API)

All sources are normalized into Kaggle kernel metadata format:
  dataset_sources:     list of "username/dataset-name"
  competition_sources: list of "competition-slug"
  kernel_setup_code:   extra code cells to prepend to the notebook
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedDataSource:
    """
    A data source fully resolved into Kaggle kernel parameters.

    `kernel_setup_cells` are injected at the top of every generated
    notebook when needed (e.g. HuggingFace download, Drive fetch).
    """
    dataset_sources:     list[str] = field(default_factory=list)
    competition_sources: list[str] = field(default_factory=list)
    kernel_setup_cells:  list[str] = field(default_factory=list)
    local_path:          Optional[str] = None   # For CSV uploads staged locally


class DataSourceResolver:
    """
    Resolves a DataSource config into kernel-ready parameters.

    Usage:
        resolver = DataSourceResolver(kaggle_client=client)
        resolved = resolver.resolve(data_source)
        # Pass resolved.dataset_sources to kaggle_client.push_kernel()
    """

    def __init__(self, kaggle_client=None):
        self.kaggle = kaggle_client

    def resolve(self, data_source) -> ResolvedDataSource:
        """Route to the appropriate resolver based on source type."""
        resolvers = {
            "kaggle":      self._resolve_kaggle,
            "huggingface": self._resolve_huggingface,
            "gdrive":      self._resolve_gdrive,
            "csv":         self._resolve_csv,
        }
        fn = resolvers.get(data_source.type)
        if fn is None:
            raise ValueError(
                f"Unknown data source type: '{data_source.type}'. "
                f"Supported: kaggle, huggingface, gdrive, csv"
            )
        return fn(data_source)

    def _resolve_kaggle(self, ds) -> ResolvedDataSource:
        """
        Kaggle source: either a competition or a dataset.

        Competition slug: "titanic" → competition_sources=["titanic"]
        Dataset slug:     "username/dataset" → dataset_sources=["username/dataset"]
        """
        ident = ds.identifier.strip().strip("/")

        # Detect competition vs dataset
        # Competitions: single word or "competition/slug"
        # Datasets: "username/dataset-name"
        if "/" in ident and not ident.startswith("competition/"):
            # Dataset format: username/dataset-name
            logger.info(f"Resolved Kaggle dataset: {ident}")
            return ResolvedDataSource(dataset_sources=[ident])
        else:
            # Competition format
            slug = ident.replace("competition/", "")
            if self.kaggle:
                ok = self.kaggle.verify_competition_access(slug)
                if not ok:
                    raise PermissionError(
                        f"Can't access competition '{slug}'. "
                        f"Accept the rules at: https://www.kaggle.com/c/{slug}/rules"
                    )
            logger.info(f"Resolved Kaggle competition: {slug}")
            return ResolvedDataSource(competition_sources=[slug])

    def _resolve_huggingface(self, ds) -> ResolvedDataSource:
        """
        HuggingFace dataset: download inside the kernel using datasets library.

        The kernel needs internet enabled (which KaggleClient sets by default).
        We inject a setup cell that downloads and saves the data as CSV.
        """
        dataset_name = ds.identifier.strip()
        hf_token     = os.environ.get("HF_TOKEN", "")

        token_line = f"os.environ['HF_TOKEN'] = '{hf_token}'" if hf_token else ""

        setup_cell = f"""\
# ── HuggingFace Dataset Setup ─────────────────────────────────────────────────
# Downloading '{dataset_name}' from HuggingFace
import os
{token_line}

try:
    from datasets import load_dataset
    print("Downloading {dataset_name} from HuggingFace...")
    hf_dataset = load_dataset("{dataset_name}")

    # Save as CSV so the rest of the notebook can use standard pandas
    split = 'train' if 'train' in hf_dataset else list(hf_dataset.keys())[0]
    df_hf = hf_dataset[split].to_pandas()
    df_hf.to_csv('/kaggle/working/hf_data.csv', index=False)
    print(f"Saved {{len(df_hf):,}} rows to /kaggle/working/hf_data.csv")
    DATA_PATH = '/kaggle/working/hf_data.csv'
except Exception as e:
    print(f"HuggingFace download failed: {{e}}")
    print("If this is a gated dataset, add your HF_TOKEN to config.yaml")
    raise
"""
        logger.info(f"Resolved HuggingFace dataset: {dataset_name}")
        return ResolvedDataSource(kernel_setup_cells=[setup_cell])

    def _resolve_gdrive(self, ds) -> ResolvedDataSource:
        """
        Google Drive: download using gdown inside the kernel.

        Accepts full share URLs or file IDs.
        e.g. "https://drive.google.com/file/d/FILE_ID/view"
        """
        url = ds.identifier.strip()

        # Extract file ID from full URL if needed
        if "drive.google.com" in url:
            import re
            match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
            file_id = match.group(1) if match else url
        else:
            file_id = url

        setup_cell = f"""\
# ── Google Drive Setup ────────────────────────────────────────────────────────
import subprocess, os

print("Downloading from Google Drive...")
subprocess.run(["pip", "install", "gdown", "-q"], check=True)

import gdown
output_path = '/kaggle/working/gdrive_data.csv'
gdown.download(id='{file_id}', output=output_path, quiet=False)

if os.path.exists(output_path):
    print(f"Downloaded to {{output_path}}")
    DATA_PATH = output_path
else:
    raise FileNotFoundError(
        "Google Drive download failed. Make sure the file is publicly shared "
        "(Anyone with the link can view) and the ID is correct."
    )
"""
        logger.info(f"Resolved Google Drive file: {file_id}")
        return ResolvedDataSource(kernel_setup_cells=[setup_cell])

    def _resolve_csv(self, ds) -> ResolvedDataSource:
        """
        Local CSV: must be uploaded as a Kaggle dataset first.

        If the identifier is a local file path, we upload it as a
        private Kaggle dataset and return its slug.
        If it's already a Kaggle dataset slug, we use it directly.
        """
        ident = ds.identifier.strip()

        # Already a Kaggle dataset slug (username/dataset-name)
        if "/" in ident and not os.path.exists(ident):
            logger.info(f"CSV source treated as Kaggle dataset: {ident}")
            return ResolvedDataSource(dataset_sources=[ident])

        # Local file path — needs to be uploaded
        if os.path.exists(ident):
            if self.kaggle is None:
                raise RuntimeError(
                    "Cannot upload CSV without a KaggleClient. "
                    "Pass kaggle_client to DataSourceResolver."
                )
            slug = self._upload_csv_as_dataset(ident)
            return ResolvedDataSource(dataset_sources=[slug])

        raise FileNotFoundError(
            f"CSV file not found: '{ident}'. "
            f"Provide either a local file path or a Kaggle dataset slug."
        )

    def _upload_csv_as_dataset(self, csv_path: str) -> str:
        """
        Upload a local CSV as a private Kaggle dataset.
        Returns the dataset slug (username/dataset-name).
        """
        import json as _json

        csv_path = Path(csv_path)
        dataset_name = f"autoresearch-{csv_path.stem.lower().replace(' ', '-')[:30]}"
        slug = f"{self.kaggle.username}/{dataset_name}"

        print(f"\n  📤 Uploading '{csv_path.name}' as Kaggle dataset...")
        print(f"     This lets the kernel access your CSV at /kaggle/input/")

        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            shutil.copy(csv_path, Path(tmpdir) / csv_path.name)

            metadata = {
                "title":          dataset_name,
                "id":             slug,
                "licenses":       [{"name": "CC0-1.0"}],
                "isPrivate":      True,
            }
            (Path(tmpdir) / "dataset-metadata.json").write_text(
                _json.dumps(metadata, indent=2)
            )

            try:
                self.kaggle._api.dataset_create_new(
                    folder=tmpdir,
                    public=False,
                    quiet=False,
                )
                print(f"  ✓ Dataset uploaded: {slug}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    # Upload new version instead
                    self.kaggle._api.dataset_create_version(
                        folder=tmpdir,
                        version_notes="Updated by AutoResearch",
                        quiet=False,
                    )
                    print(f"  ✓ Dataset updated: {slug}")
                else:
                    raise RuntimeError(
                        f"Failed to upload CSV to Kaggle: {e}\n"
                        f"Try uploading manually at: https://www.kaggle.com/datasets/new"
                    )

        return slug
