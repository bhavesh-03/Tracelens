"""Tests for tracelens.config — config loading, defaults, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tracelens.config import TraceLensConfig, load_config


class TestDefaults:
    """Loading config when no file exists should return all defaults."""

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.judge_model == "gemini/gemini-2.5-flash"
        assert cfg.max_claims_per_step == 20
        assert cfg.attribution_threshold == 0.7
        assert cfg.db_path == "tracelens.db"
        assert cfg.temperature == 0.1
        assert cfg.max_concurrent_verifications == 1
        assert cfg.verification_timeout_s == 60

    def test_cost_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.costs.input_per_million == 0.075
        assert cfg.costs.output_per_million == 0.30


class TestLoadFromFile:
    """Loading config from an actual TOML file."""

    def test_custom_values(self, tmp_path: Path) -> None:
        toml_content = """
[tracelens]
judge_model = "openai/gpt-4o-mini"
max_claims_per_step = 10
attribution_threshold = 0.5
db_path = "custom.db"
temperature = 0.3

[costs]
input_per_million = 0.15
output_per_million = 0.60
"""
        config_path = tmp_path / "tracelens.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)
        assert cfg.judge_model == "openai/gpt-4o-mini"
        assert cfg.max_claims_per_step == 10
        assert cfg.attribution_threshold == 0.5
        assert cfg.db_path == "custom.db"
        assert cfg.temperature == 0.3
        assert cfg.costs.input_per_million == 0.15
        assert cfg.costs.output_per_million == 0.60

    def test_partial_override_uses_defaults_for_rest(self, tmp_path: Path) -> None:
        toml_content = """
[tracelens]
judge_model = "openai/gpt-4o-mini"
"""
        config_path = tmp_path / "tracelens.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)
        assert cfg.judge_model == "openai/gpt-4o-mini"
        # Everything else is default:
        assert cfg.max_claims_per_step == 20
        assert cfg.attribution_threshold == 0.7


class TestValidation:
    """Config validation should reject obviously wrong values."""

    def test_negative_claims_raises(self) -> None:
        cfg = TraceLensConfig(max_claims_per_step=0)
        with pytest.raises(ValueError, match="max_claims_per_step"):
            cfg.validate()

    def test_threshold_out_of_range_raises(self) -> None:
        cfg = TraceLensConfig(attribution_threshold=0.0)
        with pytest.raises(ValueError, match="attribution_threshold"):
            cfg.validate()

        cfg2 = TraceLensConfig(attribution_threshold=1.5)
        with pytest.raises(ValueError, match="attribution_threshold"):
            cfg2.validate()

    def test_valid_config_passes(self) -> None:
        cfg = TraceLensConfig()
        cfg.validate()  # Should not raise
