"""Write arm_libero.json for a student published before this release (export.json + adapter.pt).

    python scripts/convert_student_export.py --export <dir or export.json> --output <dir>/arm_libero.json

The output (format arm-libero-student-v1, arm_opsd/metadata.py) keeps what loading and evaluating the
student needs (base model revision, sampler steps, LoRA rank/alpha/targets, updates, EMA settings) and the
training method/experiment sections; it drops the machine section (cluster paths), resource identities,
checksums and the contract hash. The adapter is used byte for byte: place it next to the output as
`--weights` (default student_adapter.pt). Reads and writes local files only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_opsd import metadata as student_metadata  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="export.json, or the directory holding it")
    parser.add_argument("--output", required=True, help=f"the {student_metadata.FILENAME} to write")
    parser.add_argument("--weights", default=student_metadata.WEIGHTS,
                        help="file name of the adapter next to the output (the published adapter.pt, renamed)")
    args = parser.parse_args()
    source = Path(args.export)
    source = source / "export.json" if source.is_dir() else source
    meta = student_metadata.from_legacy_export(json.loads(source.read_text()), weights=args.weights)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("format", "kind", "method", "variant", "updates", "sampling", "base_model",
                                           "weights")}, indent=2))


if __name__ == "__main__":
    main()
