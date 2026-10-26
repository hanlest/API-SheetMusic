"""Exit 0 if CUDA can run a tiny kernel; else exit 1. Prints one status line."""

import sys
import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.cuda")


def main() -> None:
    if not torch.cuda.is_available():
        print("FAIL;no_cuda;")
        sys.exit(1)
    name = torch.cuda.get_device_name(0).replace(";", "/")
    major, minor = torch.cuda.get_device_capability(0)
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        reason = exc.__class__.__name__.replace(";", "/")
        print(f"FAIL;{major}.{minor};{name};{reason}")
        sys.exit(1)
    print(f"OK;{major}.{minor};{name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
