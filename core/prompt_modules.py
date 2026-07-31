
from __future__ import annotations

_RULE_MODULES: dict[str, tuple[tuple[str, ...], str]] = {
    "chair": (
        ("chair", "stool", "bench", "seat", "sofa", "couch", "recliner"),
        """
FURNITURE AND CHAIR ASSEMBLY RULES

If the request is a chair, gaming chair, office chair, stool, bench, or seat:
- Seat cushion is the central horizontal support surface; round/bevel its edges.
- Backrest must be directly behind the seat and physically attached to the rear
  edge with brackets, rails, side plates, or a continuous frame — never floating.
- Backrest is upright or reclined slightly backward.
- Central gas lift / column attaches to the underside centre of the seat.
- Five-star base, legs, or pedestal attaches below the column and touches Z=0.
- Casters/wheels attach to the ends of base legs and sit on Z=0.
- Armrests bridge from seat sides upward, posts attached to the seat, pads beside
  the cushion; they must not float.
- Head/lumbar pillows sit on the FRONT face of the backrest, not detached.
- Use rounded cushions and scaled ellipsoids (argus_sphere) for pads/pillows —
  a cube-only chair is unacceptable.
- A gaming/racing chair additionally needs: bucket seat, tall racing back, side
  bolsters, headrest pillow, lumbar pillow, armrests, gas lift, five-star caster
  base, and upholstery seams. Base = exactly five long tapered spokes with one
  caster at each end.
""",
    ),
    "vehicle_wheeled": (
        ("car", "truck", "van", "bus", "kart", "vehicle", "cart", "trolley",
         "wheel", "tire", "tyre", "rolling", "chassis", "automobile", "sedan"),
        """
VEHICLE AND WHEELED OBJECT RULES

If the request has tires, wheels, carts, cars, or rolling bases:
- Body/chassis is the central load-bearing volume.
- Axles run horizontally through or below the chassis.
- Wheels are UPRIGHT disks/tyres on the axle ends — use argus_wheel so the
  circular face is vertical and the axle is horizontal.
- Left/right wheel pairs mirror across the centreline; all wheels touch Z=0.
- Wheel hubs visibly connect to axles; axles visibly connect to the chassis.
""",
    ),
    "car": (
        ("car", "sedan", "truck", "van", "bus", "kart", "automobile", "suv", "coupe"),
        """
CAR QUALITY RULES

For a car, truck, van, bus, go-kart, or four-wheel vehicle:
- Build a bevelled/tapered body SHELL, not a raw rectangular block.
- Include readable hood, cabin/roof, windshield, side+rear windows, headlights,
  tail lights, bumpers, fenders/wheel arches, four tyres, four rims, and hubs.
- Tyres are separate editable objects named by corner (car_tire_front_left,
  car_tire_rear_right). Rims/hubs are separate from rubber tyres.
- Fenders/wheel arches frame the wheels; wheels are vertical, attached via hubs.
- Body length must be at least 3x body height.
""",
    ),
    "bike": (
        ("bike", "bicycle", "motorcycle", "motorbike", "scooter", "moped"),
        """
BIKE AND MOTORCYCLE QUALITY RULES

For a bicycle, motorcycle, scooter, or two-wheel vehicle:
- Exactly two main vertical wheels, aligned in one vertical plane, with separate
  tyre/rim/hub parts (argus_wheel).
- Use cylinders/tubes (argus_cylinder) for frame, fork, handlebar, seat post,
  chain stay, and supports — never flat cubes.
- Bicycle: diamond/triangular frame tubes, pedals/crank, chain or guard, saddle,
  fork, handlebar.
- Motorcycle: fuel tank, engine block, front fork, handlebar, saddle, exhaust,
  swingarm, and drive detail.
""",
    ),
    "bag": (
        ("backpack", "school bag", "rucksack", "suitcase", "luggage", "bag", "duffel"),
        """
BACKPACK AND SOFT-BAG QUALITY RULES

For a backpack, rucksack, suitcase, or bag:
- Main body is a soft rounded bevelled/tapered volume (argus_box high bevel or
  argus_sphere-derived), not a cube.
- Include front pocket, side pockets/panels, shoulder straps, carry handle,
  zipper tracks, zipper pulls, buckles, seams, and fabric material variation.
- Straps are curved/rounded strips attached at BOTH ends to the bag.
""",
    ),
    "shoe": (
        ("shoe", "sneaker", "boot", "trainer", "running shoe", "footwear"),
        """
SHOE AND SNEAKER QUALITY RULES

For a shoe, sneaker, boot, or trainer:
- Include outsole, midsole, rounded toe box, upper panels, heel counter, tongue,
  laces, eyelets, collar, and tread blocks.
- Sole is a layered bevelled/tapered shell; upper is rounded and shoe-shaped,
  not a box. Laces and eyelets are separate editable details.
""",
    ),
    "crate": (
        ("crate", "box", "container", "case", "safe", "cabinet", "chest", "cargo"),
        """
CRATE AND CONTAINER QUALITY RULES

For a crate, container, case, safe, cabinet, or chest:
- Box-like bodies may use cube-derived forms, but they MUST be bevelled and
  detailed: inset panels (argus_panel), corner guards, ribs, seams, hinges,
  latches, handles, bolts, rubber feet/skids, labels, vents, material variation.
- Use separate editable parts for panels, handles, hinges, latches, corner
  guards, feet, and bolts.
""",
    ),
}


def select_rule_modules(*text_parts: str) -> str:
    """Return concatenated rule blocks whose triggers match the request text.

    Always returns at least the generic empty string. Order is stable and
    de-duplicated so the assembled prompt is deterministic.
    """
    haystack = " ".join(str(p) for p in text_parts).lower()
    chosen: list[str] = []
    for key, (triggers, text) in _RULE_MODULES.items():
        if any(t in haystack for t in triggers):
            chosen.append(text.strip("\n"))
    if not chosen:
        return ""
    return "\n\n".join(chosen) + "\n"
