from __future__ import annotations

import os
from typing import Optional


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return None


def _parse_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return None


def apply_s3_overrides() -> None:
    try:
        from clearml import config as clearml_config
    except Exception:
        return

    overrides = {}

    def _set_override(path: str, value: object) -> None:
        parts = path.split(".")
        node = overrides
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    endpoint = _first_env("AWS_ENDPOINT_URL")
    if endpoint:
        endpoint = endpoint.strip()
        if "://" in endpoint:
            scheme, host = endpoint.split("://", 1)
            if host:
                _set_override("sdk.aws.s3.host", host)
            if scheme.lower() == "http":
                _set_override("sdk.aws.s3.secure", False)
                _set_override("sdk.aws.s3.verify", False)
            elif scheme.lower() == "https":
                _set_override("sdk.aws.s3.secure", True)

    host = _first_env("CLEARML__sdk__aws__s3__host", "TRAINS__sdk__aws__s3__host")
    if host:
        _set_override("sdk.aws.s3.host", host.strip())

    secure = _parse_bool(_first_env("CLEARML__sdk__aws__s3__secure", "TRAINS__sdk__aws__s3__secure"))
    if secure is not None:
        _set_override("sdk.aws.s3.secure", secure)

    verify = _parse_bool(_first_env("CLEARML__sdk__aws__s3__verify", "TRAINS__sdk__aws__s3__verify"))
    if verify is not None:
        _set_override("sdk.aws.s3.verify", verify)

    region = _first_env(
        "CLEARML__sdk__aws__s3__region",
        "TRAINS__sdk__aws__s3__region",
        "AWS_DEFAULT_REGION",
    )
    if region:
        _set_override("sdk.aws.s3.region", region.strip())

    use_chain = _parse_bool(
        _first_env(
            "CLEARML__sdk__aws__s3__use_credentials_chain",
            "TRAINS__sdk__aws__s3__use_credentials_chain",
        )
    )
    if use_chain is not None:
        _set_override("sdk.aws.s3.use_credentials_chain", use_chain)

    if overrides:
        clearml_config.ConfigWrapper.set_overrides(overrides)
