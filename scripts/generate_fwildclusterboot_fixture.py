"""Generate the one-off Docker-backed fwildclusterboot reference fixture."""

import argparse
import subprocess
import tempfile
from pathlib import Path

from es_index_explorer.question_analysis.errors import GateFailureError
from es_index_explorer.question_analysis.statistical_oracle import (
    CONTRACT_FILE,
    FIXTURE_ROOT,
    INPUT_FILE,
    ORACLE_ROOT,
    OUTPUT_FILE,
    PROVENANCE_FILE,
    WEIGHTS_FILE,
    OracleProvenance,
    build_f6_linear_fixture,
    fixture_csv_bytes,
)
from es_index_explorer.question_analysis.storage import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_file,
)

DEFAULT_IMAGE = "simplemode-fwildclusterboot:0.14.3"
DOCKER_PLATFORM = "linux/amd64"
FWILDCLUSTERBOOT_SOURCE_SHA256 = (
    "ea84ef950fbc76af12f0cddc05106744e548b6f092b976df288c86cf025893ca"
)
FWILDCLUSTERBOOT_REMOTE_SHA = "336bb574eba169ac0183317f01d0564791d8122f"


def build_parser() -> argparse.ArgumentParser:
    """Build the fixture-generator argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate the frozen fwildclusterboot F6 linear reference."
    )
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    return parser


def generate_r_reference(
    analysis_root: Path,
    *,
    image: str = DEFAULT_IMAGE,
) -> OracleProvenance:
    """Generate the one-off Docker-backed R fixture and its provenance."""
    fixture, contract = build_f6_linear_fixture(analysis_root)
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(INPUT_FILE, fixture_csv_bytes(fixture))
    atomic_write_bytes(CONTRACT_FILE, canonical_json_bytes(contract))
    with tempfile.TemporaryDirectory(prefix="simplemode-r-oracle-") as temporary:
        output_dir = Path(temporary)
        command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            DOCKER_PLATFORM,
            "--mount",
            f"type=bind,src={FIXTURE_ROOT},dst=/input,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/output",
            image,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = (
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr
                else "unknown error"
            )
            raise GateFailureError(f"R oracle container failed: {message}")
        for filename, target in (
            ("f6-linear-r-output.json", OUTPUT_FILE),
            ("f6-linear-rademacher-weights.csv", WEIGHTS_FILE),
        ):
            source = output_dir / filename
            if not source.is_file():
                raise GateFailureError(f"R oracle did not produce {filename}")
            atomic_write_bytes(target, source.read_bytes())
    inspect = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0 or not inspect.stdout.strip():
        raise GateFailureError("Cannot inspect the R oracle image identity")
    provenance = OracleProvenance(
        docker_platform=DOCKER_PLATFORM,
        docker_image=image,
        docker_image_id=inspect.stdout.strip(),
        dockerfile_sha256=sha256_file(ORACLE_ROOT / "Dockerfile"),
        dockerignore_sha256=sha256_file(ORACLE_ROOT / ".dockerignore"),
        renv_lock_sha256=sha256_file(ORACLE_ROOT / "renv.lock"),
        runner_sha256=sha256_file(ORACLE_ROOT / "run_reference.R"),
        input_sha256=sha256_file(INPUT_FILE),
        contract_sha256=sha256_file(CONTRACT_FILE),
        output_sha256=sha256_file(OUTPUT_FILE),
        weights_sha256=sha256_file(WEIGHTS_FILE),
        source_analysis_root=analysis_root.name,
        fwildclusterboot_source_sha256=FWILDCLUSTERBOOT_SOURCE_SHA256,
        fwildclusterboot_remote_sha=FWILDCLUSTERBOOT_REMOTE_SHA,
    )
    atomic_write_bytes(PROVENANCE_FILE, canonical_json_bytes(provenance))
    return provenance


def main() -> int:
    """Generate the fixture and print its immutable image identity."""
    arguments = build_parser().parse_args()
    provenance = generate_r_reference(
        arguments.analysis_root,
        image=arguments.image,
    )
    print(provenance.docker_image_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
