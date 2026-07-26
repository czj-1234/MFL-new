from __future__ import annotations


def rdp_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """Return an RDP-accounted epsilon for the Gaussian client-update baseline.

    The experiment report must state the exact mechanism and assumptions. This
    helper uses Opacus' RDP accountant and is intended for the client-update
    clipping + Gaussian-noise baseline in the revision suite.
    """
    try:
        from opacus.accountants import RDPAccountant
    except ImportError as exc:
        raise ImportError(
            "Install opacus (listed in the revision branch requirements) to report DP epsilon."
        ) from exc
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0,1].")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1).")
    accountant = RDPAccountant()
    for _ in range(int(steps)):
        accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
    return float(accountant.get_epsilon(delta=float(delta)))
