"""Hex-symmetry rotation augmentation for the v2 GNN.

The standard 19-hex Catan board has 6-fold rotational symmetry (60° around the
center hex). This module derives the permutation tables for one 60° rotation
and provides `rotate_hetero_data` to apply it to a `state_to_pyg` sample.

## Geometry

The row-major hex IDs (3-4-5-4-3 layout) don't admit a clean rotation in
Cartesian coords because the row spacing creates an irregular shape. We use
**axial hex coordinates** (q, r) instead, where the center hex is (0, 0) and
60° rotation has the clean integer formula `(q, r) -> (-r, q+r)`.

## Action remap

  - BuildSettlement(v=i)  -> BuildSettlement(v=ROT60_VERTEX[i])
  - BuildCity(v=i)        -> BuildCity(v=ROT60_VERTEX[i])
  - BuildRoad(e=i)        -> BuildRoad(e=ROT60_EDGE[i])
  - MoveRobber(h=i)       -> MoveRobber(h=ROT60_HEX[i])
  - All other action IDs  -> unchanged
"""
from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from .adjacency import EDGE_TO_VERTICES, HEX_TO_VERTICES


# ---------------------------------------------------------------- axial coords
# Map row-major hex_id -> (q, r) axial coordinates centered on hex 9 (center).
# Standard 19-hex layout, hex 9 = (0, 0).
# Row 0 (3 hexes): hexes 0-2 share r=-2, q runs -1..+1
# Row 1 (4 hexes): hexes 3-6 share r=-1, q runs -2..+1
# Row 2 (5 hexes): hexes 7-11 share r=0,  q runs -2..+2
# Row 3 (4 hexes): hexes 12-15 share r=+1, q runs -1..+2  (slight horizontal offset)
# Row 4 (3 hexes): hexes 16-18 share r=+2, q runs -1..+1
# Pointy-top axial: r increases downward; rows shift in q to keep things hexagonal.
_HEX_AXIAL: list[tuple[int, int]] = [
    # row 0 (r=-2): hexes 0, 1, 2 have q = 0, 1, 2
    (0, -2), (1, -2), (2, -2),
    # row 1 (r=-1): hexes 3, 4, 5, 6 have q = -1, 0, 1, 2
    (-1, -1), (0, -1), (1, -1), (2, -1),
    # row 2 (r=0): hexes 7, 8, 9, 10, 11 have q = -2, -1, 0, 1, 2 (center row)
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    # row 3 (r=+1): hexes 12, 13, 14, 15 have q = -2, -1, 0, 1
    (-2, 1), (-1, 1), (0, 1), (1, 1),
    # row 4 (r=+2): hexes 16, 17, 18 have q = -2, -1, 0
    (-2, 2), (-1, 2), (0, 2),
]

_AXIAL_TO_HEX = {ax: i for i, ax in enumerate(_HEX_AXIAL)}


def _rot60_axial(q: int, r: int) -> tuple[int, int]:
    """60° clockwise around origin in axial coords: (q, r) -> (-r, q+r).
    (Counterclockwise would be (q+r, -q), but it doesn't matter — both are
    valid 60° rotations and both have the same 6-fold cycle.)"""
    return (-r, q + r)


# ------------------------------------------------------------- vertex/edge layout
# Each hex has 6 vertices in slots 0..5. From HEX_TO_VERTICES we get:
#   hex h, vertex slot k -> vertex ID
# To rotate, we need a (vertex_id) <-> (hex_id, slot) lookup PLUS the rule for
# how slots rotate. Under a 60° clockwise hex rotation, vertex slot k maps to
# slot (k+1) mod 6 (one corner over).
def _hex_vertex_slot(vid: int) -> tuple[int, int]:
    """Return (hex_id, slot) for vertex vid — picks the lowest hex containing it."""
    for hex_id, vlist in enumerate(HEX_TO_VERTICES):
        for slot, v in enumerate(vlist):
            if v == vid:
                return (hex_id, slot)
    raise ValueError(f"vertex {vid} not in any hex")


def _build_hex_permutation() -> list[int]:
    """rotated_array[i] = original_array[ROT60_HEX[i]].

    The hex at NEW slot i used to be at OLD slot ROT60_HEX[i]. Equivalently,
    the hex previously at slot j is now at slot inv[j]. We compute by
    rotating each old hex's coords and recording where it lands.
    """
    perm = [-1] * 19
    for old_hex in range(19):
        q, r = _HEX_AXIAL[old_hex]
        new_q, new_r = _rot60_axial(q, r)
        new_hex = _AXIAL_TO_HEX[(new_q, new_r)]
        perm[new_hex] = old_hex
    assert all(p >= 0 for p in perm), f"hex perm incomplete: {perm}"
    return perm


def _build_vertex_permutation() -> list[int]:
    """For each vertex v, find which old vertex lands at v after rotation.

    Method: pick a (hex, slot) for v. Rotate the hex (gives new hex) and rotate
    the slot (k -> k+1 mod 6 under 60° CW). Look up the vertex at (new_hex, new_slot).
    The OLD vertex at this (hex, slot) lands at THAT new vertex. So
    rotated_vertex[new_v] holds features that were at vertex v.
    Therefore perm[new_v] = v.
    """
    perm = [-1] * 54
    hex_perm = _build_hex_permutation()
    # Build inverse hex perm (where does old_hex go).
    new_hex_for_old = [-1] * 19
    for new_h in range(19):
        new_hex_for_old[hex_perm[new_h]] = new_h
    for old_v in range(54):
        old_hex, old_slot = _hex_vertex_slot(old_v)
        new_hex = new_hex_for_old[old_hex]
        new_slot = (old_slot + 1) % 6  # 60° CW rotation of hex shifts vertex slots by 1
        new_v = HEX_TO_VERTICES[new_hex][new_slot]
        if perm[new_v] != -1 and perm[new_v] != old_v:
            # Multiple old vertices map to same new vertex — math wrong.
            raise RuntimeError(f"vertex perm collision at new_v={new_v}: "
                               f"old_v={old_v} and {perm[new_v]}")
        perm[new_v] = old_v
    assert all(p >= 0 for p in perm), f"vertex perm incomplete: {perm}"
    return perm


def _build_edge_permutation() -> list[int]:
    """For each edge e (defined by 2 vertex IDs), rotate its endpoints, look up
    the new edge."""
    vert_perm = _build_vertex_permutation()
    new_vert_for_old = [-1] * 54
    for new_v in range(54):
        new_vert_for_old[vert_perm[new_v]] = new_v

    # Build edge-by-(sorted-vertex-pair) lookup.
    edge_lookup = {}
    for eid, (v1, v2) in enumerate(EDGE_TO_VERTICES):
        key = tuple(sorted((v1, v2)))
        edge_lookup[key] = eid

    perm = [-1] * 72
    for old_e, (v1, v2) in enumerate(EDGE_TO_VERTICES):
        nv1 = new_vert_for_old[v1]
        nv2 = new_vert_for_old[v2]
        new_key = tuple(sorted((nv1, nv2)))
        if new_key not in edge_lookup:
            raise RuntimeError(f"rotated edge endpoints {new_key} not in edge list")
        new_e = edge_lookup[new_key]
        if perm[new_e] != -1 and perm[new_e] != old_e:
            raise RuntimeError(f"edge perm collision at new_e={new_e}")
        perm[new_e] = old_e
    assert all(p >= 0 for p in perm), f"edge perm incomplete: {perm}"
    return perm


# Permutations: rotated_array[i] = original_array[ROT60_*[i]].
ROT60_HEX: list[int] = _build_hex_permutation()
ROT60_VERTEX: list[int] = _build_vertex_permutation()
ROT60_EDGE: list[int] = _build_edge_permutation()


def _build_action_permutation() -> list[int]:
    """Map action ID i -> action ID after a 60° rotation.

    Action layout (catan_engine::actions::encode):
      0..53      BuildSettlement(v)
      54..107    BuildCity(v)
      108..179   BuildRoad(e)
      180..198   MoveRobber(h)
      199..203   Discard(r)         — board-agnostic
      204        EndTurn            — board-agnostic
      205        RollDice           — board-agnostic
      206..225   TradeBank          — board-agnostic
      226..259   Dev cards          — board-agnostic
      260..279   ProposeTrade       — board-agnostic
    """
    perm = list(range(280))
    # BuildSettlement: input action `54 + j` (oh wait that's BuildCity).
    # rotated_policy[k] = original_policy[ACTION_PERM[k]]
    # We want: action k after rotation corresponds to which original action?
    # If rotated state has settlement-at-vertex-i, the original state had
    # settlement-at-vertex-ROT60_VERTEX[i]. So rotated_action_for_k=
    # BuildSettlement(i) corresponds to original action BuildSettlement(ROT60_VERTEX[i]).
    for i in range(54):
        perm[i] = ROT60_VERTEX[i]                         # BuildSettlement
        perm[54 + i] = 54 + ROT60_VERTEX[i]              # BuildCity
    for i in range(72):
        perm[108 + i] = 108 + ROT60_EDGE[i]              # BuildRoad
    for i in range(19):
        perm[180 + i] = 180 + ROT60_HEX[i]               # MoveRobber
    # 199..279 stay identity (already set above)
    return perm


ROT60_ACTION: list[int] = _build_action_permutation()


# ---------------------------------------------------------------- transform


# Cache torch tensors for fast indexing.
_HEX_PERM_T: torch.Tensor = torch.tensor(ROT60_HEX, dtype=torch.long)
_VERT_PERM_T: torch.Tensor = torch.tensor(ROT60_VERTEX, dtype=torch.long)
_EDGE_PERM_T: torch.Tensor = torch.tensor(ROT60_EDGE, dtype=torch.long)
_ACTION_PERM_T: torch.Tensor = torch.tensor(ROT60_ACTION, dtype=torch.long)


def rotate_hetero_data(data: HeteroData) -> HeteroData:
    """Apply a 60° rotation to a HeteroData produced by state_to_pyg.

    Permutes hex/vertex/edge feature rows and rebuilds edge_index entries.
    Does NOT permute the edges-list itself (we still have the same number of
    each edge type), only what node IDs each edge references.

    Returns a new HeteroData (does not mutate input).
    """
    new = HeteroData()
    new["hex"].x = data["hex"].x[_HEX_PERM_T]
    new["vertex"].x = data["vertex"].x[_VERT_PERM_T]
    new["edge"].x = data["edge"].x[_EDGE_PERM_T]

    # Build inverse permutations: "which new slot does original slot i go to?"
    # If new[i] = old[perm[i]], then old[j] -> new[inverse[j]].
    inv_hex = torch.empty_like(_HEX_PERM_T)
    inv_hex[_HEX_PERM_T] = torch.arange(len(_HEX_PERM_T))
    inv_vert = torch.empty_like(_VERT_PERM_T)
    inv_vert[_VERT_PERM_T] = torch.arange(len(_VERT_PERM_T))
    inv_edge = torch.empty_like(_EDGE_PERM_T)
    inv_edge[_EDGE_PERM_T] = torch.arange(len(_EDGE_PERM_T))

    # Edge_index: entries are slot indices. Apply inverse permutations
    # because each edge says "this hex slot connects to this vertex slot",
    # and after rotation, what used to be slot j is now at slot inv[j].
    h2v = data["hex", "to", "vertex"].edge_index
    new["hex", "to", "vertex"].edge_index = torch.stack([inv_hex[h2v[0]], inv_vert[h2v[1]]])
    v2h = data["vertex", "to", "hex"].edge_index
    new["vertex", "to", "hex"].edge_index = torch.stack([inv_vert[v2h[0]], inv_hex[v2h[1]]])
    v2e = data["vertex", "to", "edge"].edge_index
    new["vertex", "to", "edge"].edge_index = torch.stack([inv_vert[v2e[0]], inv_edge[v2e[1]]])
    e2v = data["edge", "to", "vertex"].edge_index
    new["edge", "to", "vertex"].edge_index = torch.stack([inv_edge[e2v[0]], inv_vert[e2v[1]]])

    # Scalars unchanged (no per-vertex spatial info in scalars).
    new.scalars = data.scalars

    # Legal mask gets permuted by ACTION (rotated_legal[i] = old_legal[perm[i]]).
    new.legal_mask = data.legal_mask[_ACTION_PERM_T]
    return new


def rotate_policy(policy: torch.Tensor) -> torch.Tensor:
    """Permute a policy tensor of shape [..., 280] by ROT60_ACTION."""
    return policy[..., _ACTION_PERM_T]


def rotate_legal_mask(mask: torch.Tensor) -> torch.Tensor:
    """Permute a legal-mask tensor of shape [..., 280] by ROT60_ACTION."""
    return mask[..., _ACTION_PERM_T]
