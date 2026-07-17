"""Sealed M5A final command authority.

This module must never be imported by target-development.  It is loaded only
by the explicit, future final-language authorization entry point; development
uses opaque IDs and source-controlled content/isolation digests instead.
"""

from __future__ import annotations

from types import MappingProxyType

FINAL_ROUTE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    (
        "requested_transfer",
        "polite_request",
        "{prefix}Complete the requested transfer of the {color} cube into the {side} bin{punct}",
    ),
    (
        "container_choice",
        "destination_first_phrasing",
        "{prefix}Choose the {side} container as the destination for the {color} cube{punct}",
    ),
    (
        "single_object_move",
        "direct_imperative",
        "{prefix}Move one object: the {color} cube, and use the {side} bin{punct}",
    ),
    (
        "goal_description",
        "indirect_unambiguous_request",
        "{prefix}The desired result is the {color} cube resting in the {side} bin{punct}",
    ),
    (
        "bin_first_instruction",
        "clause_order_variation",
        "{prefix}For the {side} bin, the cube to transfer is the {color} one{punct}",
    ),
    (
        "exclusive_assignment",
        "explicit_unambiguous_correction",
        "{prefix}Use only the named assignment: {color} cube into the {side} bin{punct}",
    ),
)

FINAL_REJECTION_BASES = MappingProxyType(
    {
        "missing_object": "Place the unnamed cube in the right bin.",
        "missing_destination": "Transfer the red cube to an unnamed container.",
        "conflicting_objects": ("Transfer either the red cube or blue cube to the left bin."),
        "conflicting_bins": ("Place the green cube in either the right or left container."),
        "unsupported_object": "Put the pink cube in the right bin.",
        "unsupported_destination": "Put the red cube in the third bin.",
        "multiple_sequential_tasks": "Complete red to right and then blue to left.",
        "unresolved_correction": "Use red right—wait, blue left—cancel the correction.",
        "contradictory_negation": ("Put the green cube right while not moving the green cube."),
        "meaningless_or_noise_text": "plim snazz gruk",
        "empty_text": "",
        "malformed_control_characters": "Put\x03the red cube right.",
        "unsupported_action": "Stack the bins on the shelf.",
        "unsupported_spatial_reference": ("Place the red cube at the location behind it."),
    }
)

__all__ = ["FINAL_REJECTION_BASES", "FINAL_ROUTE_FAMILIES"]
