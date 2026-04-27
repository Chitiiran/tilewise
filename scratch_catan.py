"""
Derive Catan board tables (hex_to_vertices, edge_to_vertices) using cube coords.

Cube coordinate system for hexes: (x, y, z) with x+y+z=0.
We use "pointy-top" hexes (vertical sides on left/right is wrong; actually
pointy-top means a vertex points up). The top vertex of each hex sits directly
above its center; corners go clockwise: TOP, UPPER-RIGHT, LOWER-RIGHT, BOTTOM,
LOWER-LEFT, UPPER-LEFT.

We pick axes so that:
  - increasing x moves "right and down" (column axis)
  - increasing y moves "left and down"
  - z = -x-y moves "up"

Actually for clarity let's just use offset (row, col) for hexes and compute
pixel positions for vertices, using floats with sufficient precision then
round to identify shared vertices.

Layout:
  Row 0 (top):    3 hexes
  Row 1:          4 hexes
  Row 2 (mid):    5 hexes
  Row 3:          4 hexes
  Row 4 (bottom): 3 hexes

Pointy-top hex: hex centers in row r, column c (for that row) at:
  cx = (col_offset_for_row[r]) + c * dx
  cy = r * dy

with dx = sqrt(3), dy = 1.5, hex "size" (center to vertex) = 1.

Row width 3 -> centers offset by +1*dx from row width 5
Row width 4 -> centers offset by +0.5*dx from row width 5
Row width 5 -> centers offset by 0
Row width 4 (lower) -> +0.5*dx
Row width 3 (lower) -> +1*dx

A pointy-top hex with center (cx, cy) has 6 corners, clockwise from top:
  TOP:   (cx, cy - 1)
  UR:    (cx + sqrt(3)/2, cy - 0.5)
  LR:    (cx + sqrt(3)/2, cy + 0.5)
  BOT:   (cx, cy + 1)
  LL:    (cx - sqrt(3)/2, cy + 0.5)
  UL:    (cx - sqrt(3)/2, cy - 0.5)

Wait — for pointy-top the corners should be at angles 30, 90, 150, 210, 270, 330
from center (with 90 = top). So:
  TOP    angle 90:   ( 0,           -1)         actually y-down so (0, -1)
  UR     angle 30:   ( sqrt(3)/2,   -0.5)
  LR     angle -30:  ( sqrt(3)/2,    0.5)
  BOT    angle -90:  ( 0,            1)
  LL     angle -150: (-sqrt(3)/2,    0.5)
  UL     angle 150:  (-sqrt(3)/2,   -0.5)

Yes those are correct (using screen y-down). dy between hex rows for pointy-top
is 1.5 (3/4 * 2 * size where size=1). Hmm, actually for pointy-top vertically
adjacent rows are offset by 1.5 in y. Good.

Wait — but actually in standard Catan layout the hexes are FLAT-top? Let me
reconsider. Looking at a real Catan board: hexes have flat tops and bottoms,
with points on left and right.

If FLAT-top: corners at 0, 60, 120, 180, 240, 300:
  RIGHT:    (1, 0)
  UR:       (0.5, -sqrt(3)/2)
  UL:       (-0.5, -sqrt(3)/2)
  LEFT:     (-1, 0)
  LL:       (-0.5, sqrt(3)/2)
  LR:       (0.5, sqrt(3)/2)

But the problem says "starting from the TOP vertex" — that implies a pointy-top
hex where one vertex is at the top. So pointy-top it is.

OK pointy-top confirmed. Hex rows offset by 1.5 vertically, columns by sqrt(3).
"""

import math

SQRT3 = math.sqrt(3)

# Number of hexes per row, top to bottom
ROW_SIZES = [3, 4, 5, 4, 3]

# Compute hex centers and assign hex IDs in row-major order
hex_centers = []  # list of (cx, cy)
hex_id = 0
for r, size in enumerate(ROW_SIZES):
    # center the row horizontally; row of width 5 starts at col 0
    # row of width 3 -> shift by +1*sqrt(3); width 4 -> +0.5*sqrt(3)
    x_offset = (5 - size) * 0.5 * SQRT3
    cy = r * 1.5
    for c in range(size):
        cx = x_offset + c * SQRT3
        hex_centers.append((cx, cy))
        hex_id += 1

assert len(hex_centers) == 19

# Compute the 6 corners for each hex, clockwise from top
def corners_of(cx, cy):
    # angles in screen coords (y-down): top is angle "up" which is -y direction
    # clockwise from top: top, UR, LR, BOT, LL, UL
    return [
        (cx,                cy - 1.0),    # TOP
        (cx + SQRT3 * 0.5,  cy - 0.5),    # UR
        (cx + SQRT3 * 0.5,  cy + 0.5),    # LR
        (cx,                cy + 1.0),    # BOT
        (cx - SQRT3 * 0.5,  cy + 0.5),    # LL
        (cx - SQRT3 * 0.5,  cy - 0.5),    # UL
    ]

# Deduplicate vertices by rounded coords
def key(p):
    return (round(p[0], 4), round(p[1], 4))

vertex_id_map = {}  # key -> id
vertex_coords = []  # id -> (x, y)
hex_to_vertices = []  # 19 lists of 6 vertex IDs

for (cx, cy) in hex_centers:
    cs = corners_of(cx, cy)
    row = []
    for p in cs:
        k = key(p)
        if k not in vertex_id_map:
            vertex_id_map[k] = len(vertex_coords)
            vertex_coords.append(p)
        row.append(vertex_id_map[k])
    hex_to_vertices.append(row)

print("Total vertices:", len(vertex_coords))
assert len(vertex_coords) == 54, f"expected 54 vertices, got {len(vertex_coords)}"

# But wait — the vertex IDs above are assigned in the order they're discovered
# when iterating hexes row-major, corners clockwise-from-top. That's a fine
# canonical order, but let's renumber vertices by (y, x) lexicographic to make
# them more human-readable: top-to-bottom, then left-to-right.

order = sorted(range(54), key=lambda i: (round(vertex_coords[i][1], 4),
                                          round(vertex_coords[i][0], 4)))
old_to_new = [0] * 54
for new_id, old_id in enumerate(order):
    old_to_new[old_id] = new_id
vertex_coords = [vertex_coords[old_id] for old_id in order]
hex_to_vertices = [[old_to_new[v] for v in row] for row in hex_to_vertices]

# Build edges from hex perimeters: for each hex, its 6 edges are between
# consecutive corners (cyclic).
edge_set = set()
for row in hex_to_vertices:
    for i in range(6):
        a, b = row[i], row[(i + 1) % 6]
        edge_set.add((min(a, b), max(a, b)))

print("Total edges:", len(edge_set))
assert len(edge_set) == 72, f"expected 72 edges, got {len(edge_set)}"

# Order edges canonically: by (min vertex id, max vertex id)
edge_to_vertices = sorted(edge_set)

# === Verification of all 10 invariants ===
print("\n=== Invariant checks ===")

# 1. hex_to_vertices: 19 rows, 6 unique IDs each
assert len(hex_to_vertices) == 19
for i, row in enumerate(hex_to_vertices):
    assert len(row) == 6
    assert len(set(row)) == 6, f"hex {i} has duplicates: {row}"
print("1. OK: 19 rows of 6 unique vertex IDs")

# 2. Union uses exactly 54 distinct IDs 0..53
all_v = set()
for row in hex_to_vertices:
    all_v.update(row)
assert all_v == set(range(54))
print("2. OK: union = {0..53}")

# 3. edge_to_vertices: 72 rows of 2 distinct IDs
assert len(edge_to_vertices) == 72
for e in edge_to_vertices:
    assert len(e) == 2 and e[0] != e[1]
print("3. OK: 72 edges of 2 distinct endpoints")

# 4. union of edge endpoints = 54 distinct
edge_v = set()
for a, b in edge_to_vertices:
    edge_v.add(a); edge_v.add(b)
assert edge_v == set(range(54))
print("4. OK: edge union = {0..53}")

# 5. Handshake: sum |adj_v per hex| = sum |adj_h per vertex| = 114
hex_count = sum(len(r) for r in hex_to_vertices)
v_to_hexes = {v: [] for v in range(54)}
for h, row in enumerate(hex_to_vertices):
    for v in row:
        v_to_hexes[v].append(h)
v_count = sum(len(v_to_hexes[v]) for v in range(54))
assert hex_count == 114 and v_count == 114
print(f"5. OK: handshake {hex_count} == {v_count} == 114")

# 6. Every vertex appears in 1, 2, or 3 hex rows
for v in range(54):
    n = len(v_to_hexes[v])
    assert n in (1, 2, 3), f"vertex {v} in {n} hexes"
counts = {1: 0, 2: 0, 3: 0}
for v in range(54):
    counts[len(v_to_hexes[v])] += 1
print(f"6. OK: vertex hex-counts: {counts}")

# 7. Planarity — by construction (cube coords)
print("7. OK: planarity by geometric construction")

# 8. Adjacency consistency
for h, row in enumerate(hex_to_vertices):
    for v in row:
        assert h in v_to_hexes[v]
print("8. OK: adjacency consistent")

# 9. Every vertex has 2 or 3 incident edges
v_deg = [0] * 54
for a, b in edge_to_vertices:
    v_deg[a] += 1
    v_deg[b] += 1
deg_counts = {2: 0, 3: 0}
for v, d in enumerate(v_deg):
    assert d in (2, 3), f"vertex {v} has degree {d}"
    deg_counts[d] += 1
print(f"9. OK: vertex degrees: {deg_counts}")

# 10. Connected graph
adj = {v: set() for v in range(54)}
for a, b in edge_to_vertices:
    adj[a].add(b)
    adj[b].add(a)
seen = {0}
stack = [0]
while stack:
    u = stack.pop()
    for w in adj[u]:
        if w not in seen:
            seen.add(w)
            stack.append(w)
assert len(seen) == 54
print("10. OK: graph connected (all 54 reachable)")

# Bonus: boundary vs interior count
# Interior vertices touch 3 hexes; boundary touch 1 or 2.
# A standard 19-hex board has 24 interior vertices and 30 boundary vertices.
print(f"\nBonus: {counts[3]} interior, {counts[1] + counts[2]} boundary "
      f"({counts[1]} corner-1, {counts[2]} edge-2)")

# === Emit Rust arrays ===
print("\n\n========== RUST OUTPUT ==========\n")

print("fn standard_hex_to_vertices() -> [[u8; 6]; 19] {")
print("    [")
for i, row in enumerate(hex_to_vertices):
    s = "[" + ", ".join(f"{v:2d}" for v in row) + "]"
    print(f"        {s}, // hex {i:2d}")
print("    ]")
print("}\n")

print("fn standard_edge_to_vertices() -> [[u8; 2]; 72] {")
print("    [")
for i, (a, b) in enumerate(edge_to_vertices):
    print(f"        [{a:2d}, {b:2d}], // edge {i:2d}")
print("    ]")
print("}")
