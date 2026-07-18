"""Versioned post-hoc status decoders for the frozen M5A classifier."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.router_types import RouterStatus
from langmani.language.text_classifier import BIN_LABELS, OBJECT_LABELS, STATUS_LABELS

DECODER_GRID_SCHEMA = "langmani-m5a-posthoc-decoder-grid-v0"
DECODER_CONFIG_SCHEMA = "langmani-m5a-posthoc-decoder-config-v0"
DECODER_DECISION_SCHEMA = "langmani-m5a-posthoc-decoder-decision-v0"

ROUTE_THRESHOLDS_V0 = tuple(value / 100.0 for value in range(40, 100, 5))
ROUTE_MARGIN_THRESHOLDS_V0 = tuple(value / 100.0 for value in range(-20, 55, 5))
OBJECT_CONFIDENCE_THRESHOLDS_V0 = ROUTE_THRESHOLDS_V0
BIN_CONFIDENCE_THRESHOLDS_V0 = ROUTE_THRESHOLDS_V0
TEMPERATURE_MODES_V0 = ("identity", "validation_temperature")


class ClassifierDecoderError(ValueError):
    """Raised when a decoder or fixed-grid contract is malformed."""


class DecoderCandidate(StrEnum):
    BASELINE_FOUR_WAY_ARGMAX_V0 = "BaselineFourWayArgmaxV0"
    AGGREGATED_REJECT_THRESHOLD_V0 = "AggregatedRejectThresholdV0"
    HIERARCHICAL_STATUS_DECODER_V0 = "HierarchicalStatusDecoderV0"
    CONSERVATIVE_ROUTE_DECODER_V0 = "ConservativeRouteDecoderV0"


def _finite_probability(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ClassifierDecoderError(f"{name} must be finite and lie in [0,1]")
    return float(value)


def _optional_finite(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ClassifierDecoderError(f"{name} must be finite and lie in [{minimum},{maximum}]")
    return float(value)


@dataclass(frozen=True, slots=True)
class DecoderThresholdGridV0:
    """The exact finite grid locked before any validation scoring."""

    route_thresholds: tuple[float, ...] = ROUTE_THRESHOLDS_V0
    route_margin_thresholds: tuple[float, ...] = ROUTE_MARGIN_THRESHOLDS_V0
    object_confidence_thresholds: tuple[float, ...] = OBJECT_CONFIDENCE_THRESHOLDS_V0
    bin_confidence_thresholds: tuple[float, ...] = BIN_CONFIDENCE_THRESHOLDS_V0
    temperature_modes: tuple[str, ...] = TEMPERATURE_MODES_V0
    grid_fingerprint: str = ""
    schema_version: str = DECODER_GRID_SCHEMA

    def __post_init__(self) -> None:
        expected = (
            ROUTE_THRESHOLDS_V0,
            ROUTE_MARGIN_THRESHOLDS_V0,
            OBJECT_CONFIDENCE_THRESHOLDS_V0,
            BIN_CONFIDENCE_THRESHOLDS_V0,
            TEMPERATURE_MODES_V0,
        )
        actual = (
            self.route_thresholds,
            self.route_margin_thresholds,
            self.object_confidence_thresholds,
            self.bin_confidence_thresholds,
            self.temperature_modes,
        )
        if actual != expected:
            raise ClassifierDecoderError("the M5A.1 fixed decoder grid cannot expand or change")
        if self.schema_version != DECODER_GRID_SCHEMA:
            raise ClassifierDecoderError("unknown decoder-grid schema")
        expected_fingerprint = f"sha256:{sha256_hex(self.identity_dict())}"
        if self.grid_fingerprint and self.grid_fingerprint != expected_fingerprint:
            raise ClassifierDecoderError("decoder grid fingerprint differs from its contents")
        object.__setattr__(self, "grid_fingerprint", expected_fingerprint)

    def identity_dict(self) -> dict[str, object]:
        return {
            "route_thresholds": list(self.route_thresholds),
            "route_margin_thresholds": list(self.route_margin_thresholds),
            "object_confidence_thresholds": list(self.object_confidence_thresholds),
            "bin_confidence_thresholds": list(self.bin_confidence_thresholds),
            "temperature_modes": list(self.temperature_modes),
            "candidates": [value.value for value in DecoderCandidate],
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "grid_fingerprint": self.grid_fingerprint}


@dataclass(frozen=True, slots=True)
class DecoderConfigurationV0:
    candidate: DecoderCandidate
    temperature_mode: str
    status_temperature: float
    route_threshold: float | None = None
    route_margin_threshold: float | None = None
    object_confidence_threshold: float | None = None
    bin_confidence_threshold: float | None = None
    configuration_fingerprint: str = ""
    schema_version: str = DECODER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DecoderCandidate):
            raise ClassifierDecoderError("candidate must be a DecoderCandidate")
        if self.temperature_mode not in TEMPERATURE_MODES_V0:
            raise ClassifierDecoderError("temperature mode is outside the locked comparison")
        if (
            isinstance(self.status_temperature, bool)
            or not isinstance(self.status_temperature, int | float)
            or not math.isfinite(float(self.status_temperature))
            or float(self.status_temperature) <= 0.0
        ):
            raise ClassifierDecoderError("status temperature must be finite and positive")
        object.__setattr__(self, "status_temperature", float(self.status_temperature))
        required = self.candidate is not DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0
        if required:
            object.__setattr__(
                self,
                "route_threshold",
                _optional_finite(
                    self.route_threshold,
                    name="route_threshold",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
            object.__setattr__(
                self,
                "route_margin_threshold",
                _optional_finite(
                    self.route_margin_threshold,
                    name="route_margin_threshold",
                    minimum=-1.0,
                    maximum=1.0,
                ),
            )
        elif self.route_threshold is not None or self.route_margin_threshold is not None:
            raise ClassifierDecoderError("baseline argmax cannot carry binary thresholds")
        confidence_required = self.candidate in {
            DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0,
            DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
        }
        if confidence_required:
            object.__setattr__(
                self,
                "object_confidence_threshold",
                _optional_finite(
                    self.object_confidence_threshold,
                    name="object_confidence_threshold",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
            object.__setattr__(
                self,
                "bin_confidence_threshold",
                _optional_finite(
                    self.bin_confidence_threshold,
                    name="bin_confidence_threshold",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        elif (
            self.object_confidence_threshold is not None
            or self.bin_confidence_threshold is not None
        ):
            raise ClassifierDecoderError("this decoder cannot carry object/bin thresholds")
        if self.schema_version != DECODER_CONFIG_SCHEMA:
            raise ClassifierDecoderError("unknown decoder-configuration schema")
        expected = f"sha256:{sha256_hex(self.identity_dict())}"
        if self.configuration_fingerprint and self.configuration_fingerprint != expected:
            raise ClassifierDecoderError("decoder configuration fingerprint differs")
        object.__setattr__(self, "configuration_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.value,
            "temperature_mode": self.temperature_mode,
            "status_temperature": self.status_temperature,
            "route_threshold": self.route_threshold,
            "route_margin_threshold": self.route_margin_threshold,
            "object_confidence_threshold": self.object_confidence_threshold,
            "bin_confidence_threshold": self.bin_confidence_threshold,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "configuration_fingerprint": self.configuration_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class DecoderDecisionV0:
    status: RouterStatus
    target_object_id: str | None
    target_bin_id: str | None
    configuration_fingerprint: str
    p_route: float
    p_reject: float
    route_margin: float
    object_confidence: float
    bin_confidence: float
    schema_version: str = DECODER_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.status, RouterStatus):
            raise ClassifierDecoderError("decoder status must be a RouterStatus")
        for name in ("p_route", "p_reject", "object_confidence", "bin_confidence"):
            object.__setattr__(self, name, _finite_probability(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_margin",
            _optional_finite(self.route_margin, name="route_margin", minimum=-1.0, maximum=1.0),
        )
        if not self.configuration_fingerprint.startswith("sha256:"):
            raise ClassifierDecoderError("decision requires its decoder configuration fingerprint")
        if self.status is RouterStatus.ROUTE:
            if (
                self.target_object_id not in OBJECT_LABELS[:3]
                or self.target_bin_id not in BIN_LABELS[:2]
            ):
                raise ClassifierDecoderError("routed decoder decisions require one valid TaskSpec")
        elif self.target_object_id is not None or self.target_bin_id is not None:
            raise ClassifierDecoderError("rejected decoder decisions cannot contain a TaskSpec")
        if self.schema_version != DECODER_DECISION_SCHEMA:
            raise ClassifierDecoderError("unknown decoder-decision schema")

    @property
    def task_spec(self) -> TaskSpec | None:
        if self.status is not RouterStatus.ROUTE:
            return None
        assert self.target_object_id is not None
        assert self.target_bin_id is not None
        return TaskSpec.from_mapping(
            {
                "target_object_id": self.target_object_id,
                "target_bin_id": self.target_bin_id,
                "instruction_template_id": "canonical_v0",
            }
        )

    def to_dict(self) -> dict[str, object]:
        task_spec = self.task_spec
        return {
            "status": self.status.value,
            "target_object_id": self.target_object_id,
            "target_bin_id": self.target_bin_id,
            "task_spec": None if task_spec is None else task_spec.to_dict(),
            "task_id": None if task_spec is None else stable_task_id(task_spec),
            "configuration_fingerprint": self.configuration_fingerprint,
            "p_route": self.p_route,
            "p_reject": self.p_reject,
            "route_margin": self.route_margin,
            "object_confidence": self.object_confidence,
            "bin_confidence": self.bin_confidence,
            "schema_version": self.schema_version,
        }


def fixed_decoder_grid_v0() -> DecoderThresholdGridV0:
    return DecoderThresholdGridV0()


def iter_decoder_configurations(
    grid: DecoderThresholdGridV0,
    *,
    temperature_values: Mapping[str, float],
) -> Iterator[DecoderConfigurationV0]:
    """Yield the complete predeclared candidate space in canonical order."""

    if tuple(temperature_values) != grid.temperature_modes:
        raise ClassifierDecoderError("temperature values must follow the locked mode order")
    for mode in grid.temperature_modes:
        temperature = temperature_values[mode]
        yield DecoderConfigurationV0(
            candidate=DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0,
            temperature_mode=mode,
            status_temperature=temperature,
        )
        for route_threshold, route_margin_threshold in product(
            grid.route_thresholds, grid.route_margin_thresholds
        ):
            yield DecoderConfigurationV0(
                candidate=DecoderCandidate.AGGREGATED_REJECT_THRESHOLD_V0,
                temperature_mode=mode,
                status_temperature=temperature,
                route_threshold=route_threshold,
                route_margin_threshold=route_margin_threshold,
            )
        for candidate in (
            DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0,
            DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
        ):
            for values in product(
                grid.route_thresholds,
                grid.route_margin_thresholds,
                grid.object_confidence_thresholds,
                grid.bin_confidence_thresholds,
            ):
                yield DecoderConfigurationV0(
                    candidate=candidate,
                    temperature_mode=mode,
                    status_temperature=temperature,
                    route_threshold=values[0],
                    route_margin_threshold=values[1],
                    object_confidence_threshold=values[2],
                    bin_confidence_threshold=values[3],
                )


def _probability_vector(values: Sequence[float], *, expected: int, label: str) -> tuple[float, ...]:
    result = tuple(
        _finite_probability(value, name=f"{label}[{index}]") for index, value in enumerate(values)
    )
    if len(result) != expected or not math.isclose(sum(result), 1.0, abs_tol=1e-5):
        raise ClassifierDecoderError(
            f"{label} must contain {expected} probabilities summing to one"
        )
    return result


def decode_classifier_probabilities(
    *,
    status_probabilities: Sequence[float],
    object_probabilities: Sequence[float],
    bin_probabilities: Sequence[float],
    configuration: DecoderConfigurationV0,
) -> DecoderDecisionV0:
    """Decode one example without lexical features or expected-label inputs."""

    status = _probability_vector(status_probabilities, expected=4, label="status_probabilities")
    objects = _probability_vector(object_probabilities, expected=4, label="object_probabilities")
    bins = _probability_vector(bin_probabilities, expected=3, label="bin_probabilities")
    p_route = status[0]
    p_reject = sum(status[1:])
    route_margin = p_route - p_reject
    object_index = max(range(len(objects)), key=lambda index: (objects[index], -index))
    bin_index = max(range(len(bins)), key=lambda index: (bins[index], -index))
    object_id = OBJECT_LABELS[object_index]
    bin_id = BIN_LABELS[bin_index]
    valid_pair = object_id != "none" and bin_id != "none"
    if configuration.candidate is DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0:
        route = max(range(len(status)), key=lambda index: (status[index], -index)) == 0
    else:
        assert configuration.route_threshold is not None
        assert configuration.route_margin_threshold is not None
        route = (
            p_route >= configuration.route_threshold
            and route_margin >= configuration.route_margin_threshold
        )
    if configuration.candidate in {
        DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0,
        DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
    }:
        assert configuration.object_confidence_threshold is not None
        assert configuration.bin_confidence_threshold is not None
        route = (
            route
            and objects[object_index] >= configuration.object_confidence_threshold
            and bins[bin_index] >= configuration.bin_confidence_threshold
        )
    route = route and valid_pair
    if route:
        decision_status = RouterStatus.ROUTE
        target_object_id: str | None = object_id
        target_bin_id: str | None = bin_id
    else:
        rejection_index = max(range(1, len(status)), key=lambda index: (status[index], -index))
        decision_status = RouterStatus(STATUS_LABELS[rejection_index])
        target_object_id = None
        target_bin_id = None
    return DecoderDecisionV0(
        status=decision_status,
        target_object_id=target_object_id,
        target_bin_id=target_bin_id,
        configuration_fingerprint=configuration.configuration_fingerprint,
        p_route=p_route,
        p_reject=p_reject,
        route_margin=route_margin,
        object_confidence=objects[object_index],
        bin_confidence=bins[bin_index],
    )


__all__ = [
    "BIN_CONFIDENCE_THRESHOLDS_V0",
    "DECODER_CONFIG_SCHEMA",
    "DECODER_DECISION_SCHEMA",
    "DECODER_GRID_SCHEMA",
    "OBJECT_CONFIDENCE_THRESHOLDS_V0",
    "ROUTE_MARGIN_THRESHOLDS_V0",
    "ROUTE_THRESHOLDS_V0",
    "TEMPERATURE_MODES_V0",
    "ClassifierDecoderError",
    "DecoderCandidate",
    "DecoderConfigurationV0",
    "DecoderDecisionV0",
    "DecoderThresholdGridV0",
    "decode_classifier_probabilities",
    "fixed_decoder_grid_v0",
    "iter_decoder_configurations",
]
