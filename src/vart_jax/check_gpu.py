from __future__ import annotations

import sys

import jax


def main() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")
    if backend != "gpu" or not any(device.platform == "gpu" for device in devices):
        raise SystemExit("GPU-backed JAX is required but was not detected.")


if __name__ == "__main__":
    main()
