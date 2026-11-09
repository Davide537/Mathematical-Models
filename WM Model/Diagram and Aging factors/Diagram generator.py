"""Generates Brain_Diagram_New.svg. Edit the PARAMETERS block, then run:
    python make_brain_diagram.py
"""
import math


# ===================== PARAMETERS =====================
OUT = "Fig/Young model/3pops.svg"

# colours (matplotlib tab: palette, matching the thesis plots)
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
E1_HALO, E2_HALO = "#8fbbd9", "#ffbf86"   # lighter tints for the outer ellipse rings
INHIB = "#8c8c8c"   # shared by IN cells and inhibitory arrows


# big-cell corner rounding
CELL_R_OUTER = 40   # outer brown border
CELL_R_INNER = 24   # inner cream fill (smaller, since it's inset by 32px)


# arrow geometry
STRONG_SHAFT = 20     # thickness of strong arrows
STRONG_HEAD_W = 72    # head width (strong, inter-area)
STRONG_HEAD_L = 48    # head length (strong, inter-area)
LOCAL_HEAD_W = 68     # head width (local E->IN)
LOCAL_HEAD_L = 48     # head length (local E->IN)
WEAK_LINE = 2.5       # dashed line thickness
WEAK_HEAD_W = 30
WEAK_HEAD_L = 30
OVERLAP = 1   # fraction of head length the shaft extends into the head (0 = none, 1 = to the base)

AREA_LABEL_SIZE = 44
AREA_LABELS = ("Area i", "Area j")   # left cell, right cell
AREA_SUBS = ("i", "j")               # subscripts for J

# recurrent loops and labels
RECUR_W = 12          # loop thickness
RECUR_HEAD_W = 38     # was 52
RECUR_HEAD_L = 24     # was 34
RECUR_R = 52          # radius of the circular loop
RECUR_ANGLE = 50      # degrees from the ellipse axis where the loop attaches (smaller = more closed)
LABEL_SIZE = 40       # J_s label
G_SIZE = 64           # G label


# arrows: (x_start, y_start, x_end, y_end, kind, colour)
# (x_end, y_end) = centre of the flat base of the head, i.e. the target end.
# kind: "strong", "weak", "local"
ARROWS = [
    # E1
    (725, 114, 1222, 112, "strong", BLUE),
    (1290, 190, 764, 388, "strong", BLUE),
    (1265, 160, 797, 149, "weak", BLUE),
    (750, 188, 1232, 376, "weak", BLUE),
    (707, 190, 707, 345, "local", BLUE),
    (1386, 190, 1386, 335, "local", BLUE),
    # E2
    (725, 753, 1235, 751, "strong", ORANGE),
    (1290, 690, 748, 500, "strong", ORANGE),
    (1275, 698, 807, 698, "weak", ORANGE),
    (750, 690, 1230, 472, "weak", ORANGE),
    (692, 650, 692, 534, "local", ORANGE),
    (1372, 650, 1372, 520, "local", ORANGE),
]
# ======================================================


def arrow(x1, y1, x2, y2, kind, color):
    sw, hw, hl, dash = {
        "strong": (STRONG_SHAFT, STRONG_HEAD_W, STRONG_HEAD_L, False),
        "local": (STRONG_SHAFT, LOCAL_HEAD_W, LOCAL_HEAD_L, False),
        "weak": (WEAK_LINE, WEAK_HEAD_W, WEAK_HEAD_L, True),
    }[kind]
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy); ux, uy = dx / L, dy / L; px, py = -uy, ux
    tx, ty = x2 - ux * hl, y2 - uy * hl          # apex of head triangle
    sx, sy = tx + ux * hl * OVERLAP, ty + uy * hl * OVERLAP   # shaft end, inside the head
    b1 = (x2 + px * hw / 2, y2 + py * hw / 2)
    b2 = (x2 - px * hw / 2, y2 - py * hw / 2)
    out = []
    if dash:
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{sx:.1f}" y2="{sy:.1f}" stroke="{color}" stroke-width="{sw}" stroke-dasharray="5,4"/>')
    else:
        pts = [(x1 + px*sw/2, y1 + py*sw/2), (x1 - px*sw/2, y1 - py*sw/2),
               (sx - px*sw/2, sy - py*sw/2), (sx + px*sw/2, sy + py*sw/2)]
        out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="{color}"/>')
    out.append(f'<polygon points="{b1[0]:.1f},{b1[1]:.1f} {b2[0]:.1f},{b2[1]:.1f} {tx:.1f},{ty:.1f}" fill="{color}"/>')
    return "\n  ".join(out)


arrows_svg = "\n  ".join(arrow(*a) for a in ARROWS)

# (cx, cy, rx, ry, colour, side) side: -1 = loop on left of ellipse, +1 = right
RECURRENT = [
    (672, 150, 119, 98, BLUE, -1, AREA_SUBS[0]),
    (680, 718, 119, 98, ORANGE, -1, AREA_SUBS[0]),
    (1349, 140, 119, 98, BLUE, +1, AREA_SUBS[1]),
    (1360, 712, 119, 98, ORANGE, +1, AREA_SUBS[1]),
    (695, 430, 119, 98, INHIB, -1, '2'),
    (1315, 430, 119, 98, INHIB, +1, None)
]

def recurrent(cx, cy, rx, ry, color, side,sub):
    R = RECUR_R
    th = math.radians(RECUR_ANGLE)
    dy = R * math.sin(th)
    ex = cx + side * rx * math.sqrt(1 - (dy / ry) ** 2)   # ellipse edge x at height dy
    ccx = ex + side * R * math.cos(th)                    # circle centre, so both ends lie on the ellipse
    p0 = (ex, cy - dy)                                    # start (top)
    p1 = (ex, cy + dy)                                    # end (bottom, arrowhead)
    sweep = 1 if side > 0 else 0
    # tangent at the end point, pointing back into the ellipse
    ux, uy = (-math.sin(th), -math.cos(th)) if side > 0 else (math.sin(th), -math.cos(th))
    px, py = -uy, ux
    tx, ty = p1[0] - ux * RECUR_HEAD_L, p1[1] - uy * RECUR_HEAD_L
    b1 = (p1[0] + px * RECUR_HEAD_W / 2, p1[1] + py * RECUR_HEAD_W / 2)
    b2 = (p1[0] - px * RECUR_HEAD_W / 2, p1[1] - py * RECUR_HEAD_W / 2)
    path = (f'<path d="M {p0[0]:.1f},{p0[1]:.1f} A {R},{R} 0 1 {sweep} {p1[0]:.1f},{p1[1]:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{RECUR_W}"/>')
    head = f'<polygon points="{b1[0]:.1f},{b1[1]:.1f} {b2[0]:.1f},{b2[1]:.1f} {tx:.1f},{ty:.1f}" fill="{color}"/>'
    if color != INHIB:
        sub_size = LABEL_SIZE * 0.65
        jx = ccx - 4                      # nudge so "J_sub" stays visually centred
        y0 = cy + LABEL_SIZE * 0.3
        label = (
            f'<text x="{jx:.1f}" y="{y0:.1f}" text-anchor="end" font-size="{LABEL_SIZE}" '
            f'font-style="italic" font-weight="bold" fill="#000">J</text>\n  '
            f'<text x="{jx + 1:.1f}" y="{y0 + 6:.1f}" text-anchor="start" font-size="{sub_size:.0f}" '
            f'font-style="italic" font-weight="bold" fill="#000">{sub}</text>'
        )
    else:
        label = ''

    return "\n  ".join([path, head, label])

recurrent_svg = "\n  ".join(recurrent(*r) for r in RECURRENT)


svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -60 1640 940" width="1640" height="940" font-family="Arial, Helvetica, sans-serif">
  <rect x="0" y="-60" width="1640" height="940" fill="#ffffff"/>
  <rect x="432" y="10" width="487" height="848" rx="{CELL_R_OUTER}" ry="{CELL_R_OUTER}" fill="#d6c0a6"/>
  <rect x="464" y="42" width="421" height="784" rx="{CELL_R_INNER}" ry="{CELL_R_INNER}" fill="#fde5cb"/>
  <rect x="1100" y="4" width="497" height="844" rx="{CELL_R_OUTER}" ry="{CELL_R_OUTER}" fill="#d6c0a6"/>
  <rect x="1132" y="38" width="431" height="778" rx="{CELL_R_INNER}" ry="{CELL_R_INNER}" fill="#fde5cb"/>
  <g font-weight="bold" font-size="76" fill="#000">
    <text x="78" y="205">Excit. E1</text>
    <text x="78" y="460">Inhib. IN</text>
    <text x="78" y="742">Excit. E2</text>
  </g>
  <!-- populations -->
  <ellipse cx="672" cy="150" rx="120" ry="102" fill="{E1_HALO}"/><ellipse cx="672" cy="150" rx="88" ry="76" fill="{BLUE}"/>
  <ellipse cx="680" cy="718" rx="118" ry="100" fill="{E2_HALO}"/><ellipse cx="680" cy="718" rx="87" ry="76" fill="{ORANGE}"/>
  <ellipse cx="1349" cy="140" rx="119" ry="102" fill="{E1_HALO}"/><ellipse cx="1349" cy="140" rx="88" ry="76" fill="{BLUE}"/>
  <ellipse cx="1360" cy="712" rx="119" ry="98" fill="{E2_HALO}"/><ellipse cx="1360" cy="712" rx="88" ry="76" fill="{ORANGE}"/>
  <ellipse cx="664" cy="438" rx="94" ry="88" fill="#bdbdc0"/><ellipse cx="664" cy="438" rx="62" ry="58" fill="{INHIB}"/>
  <ellipse cx="1343" cy="432" rx="94" ry="86" fill="#bdbdc0"/><ellipse cx="1343" cy="432" rx="62" ry="58" fill="{INHIB}"/>
  <!-- excitatory arrows -->
  {arrows_svg}
  <!-- inhibitory stems -->
  <rect x="632" y="268" width="12" height="130" fill="{INHIB}"/><circle cx="638" cy="274" r="25" fill="{INHIB}"/>
  <rect x="628" y="478" width="12" height="110" fill="{INHIB}"/><circle cx="632" cy="596" r="25" fill="{INHIB}"/>
  <rect x="1310" y="262" width="12" height="130" fill="{INHIB}"/><circle cx="1316" cy="266" r="25" fill="{INHIB}"/>
  <rect x="1306" y="472" width="12" height="110" fill="{INHIB}"/><circle cx="1311" cy="586" r="25" fill="{INHIB}"/>
    <!-- recurrent J_s loops -->
  {recurrent_svg}
    <!-- area labels -->
  <g font-weight="bold" font-size="{AREA_LABEL_SIZE}" fill="#000" text-anchor="middle">
    <text x="675" y="-14">{AREA_LABELS[0]}</text>
    <text x="1348" y="-14">{AREA_LABELS[1]}</text>
  </g>
  <!-- G labels -->
  <g font-style="italic" font-weight="bold" font-size="{G_SIZE}" fill="#000" text-anchor="middle">
    <text x="1010" y="92">G</text>
  </g>
</svg>'''
open(OUT, "w").write(svg)
print("wrote", OUT)

