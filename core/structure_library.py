from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LIST_FIELDS = (
    "parts",
    "primary_forms",
    "assembly_order",
    "attachment_points",
    "orientation_notes",
    "texture_plan",
    "shape_refinement_plan",
    "critical_constraints",
)

MEMORY_PATH = Path("out") / "structure_memory.json"
MEMORY_LIST_LIMIT = 80


@dataclass(frozen=True)
class StructureTemplate:
    parts: tuple[str, ...]
    primary_forms: tuple[str, ...]
    assembly_order: tuple[str, ...]
    attachment_points: tuple[str, ...]
    orientation_notes: tuple[str, ...]
    support_strategy: str
    texture_plan: tuple[str, ...]
    shape_refinement_plan: tuple[str, ...]
    critical_constraints: tuple[str, ...]


@dataclass(frozen=True)
class StructureBlueprint:
    key: str
    aliases: tuple[str, ...]
    template: str
    category: str
    style: str
    material: str
    asset_scale: float
    signature_parts: tuple[str, ...]


_TEMPLATES: dict[str, StructureTemplate] = {
    "gaming_chair": StructureTemplate(
        parts=(
            "five-star radial base",
            "five caster wheels",
            "central gas lift cylinder",
            "tilt mechanism plate",
            "bucket seat cushion",
            "tall contoured backrest",
            "left armrest posts and pad",
            "right armrest posts and pad",
            "headrest pillow",
            "lumbar pillow",
            "upholstery seam strips",
            "rear support brackets",
        ),
        primary_forms=(
            "rounded bucket seat with raised side bolsters",
            "tall racing-style backrest with shoulder wings",
            "five tapered spokes around a central hub",
            "vertical caster wheels on horizontal axles",
            "soft ellipsoid pillows attached to backrest front",
        ),
        assembly_order=(
            "build five-star base and caster wheels on ground plane",
            "attach central gas lift to base hub",
            "attach tilt plate and bucket seat above gas lift",
            "attach tall backrest to rear of seat with side brackets",
            "attach armrest posts to seat sides and pads on top",
            "attach lumbar and headrest pillows to front face of backrest",
            "add seams, bolts, panel grooves, and upholstery details",
        ),
        attachment_points=(
            "caster forks overlap spoke ends",
            "gas lift penetrates base hub and underside mechanism",
            "backrest brackets overlap rear seat frame",
            "armrest posts touch seat side frame and underside of pads",
            "pillows visibly touch or strap to the backrest",
        ),
        orientation_notes=(
            "Z is up and the chair sits on the caster contact patches",
            "backrest is behind the seat and reclines slightly backward",
            "caster wheels stand vertically with horizontal axles",
            "arm pads are horizontal and mirrored left/right",
        ),
        support_strategy="Wide five-star base keeps center of mass inside the caster footprint.",
        texture_plan=(
            "matte or slightly glossy PU leather or fabric upholstery",
            "dark plastic arm and caster housings",
            "metal gas lift and bolt plates",
            "visible seam and stitch color contrast",
        ),
        shape_refinement_plan=(
            "must not resemble a stool or plain office chair",
            "use rounded cushions, raised bolsters, and shoulder wings",
            "taper base spokes and bevel all hard edges",
            "add separate headrest and lumbar pillows",
            "add at least four seam or stitch strips",
        ),
        critical_constraints=(
            "exactly five base spokes or a clear five-star base",
            "five caster wheels at spoke ends",
            "headrest and lumbar pillows are mandatory",
            "left and right armrests with supported pads are mandatory",
            "wheels and pillows must not float detached",
        ),
    ),
    "seating": StructureTemplate(
        parts=("seat surface", "support legs or pedestal", "backrest if applicable", "side supports or arms", "feet or glides"),
        primary_forms=("horizontal sitting surface", "vertical supports below", "rear back support volume"),
        assembly_order=("build load-bearing legs or pedestal", "attach seat above supports", "attach backrest to rear edge", "add arms and details"),
        attachment_points=("legs touch underside of seat", "backrest overlaps rear seat frame", "arms connect to seat or back"),
        orientation_notes=("seat is horizontal", "supports reach the ground", "backrest sits behind the seat"),
        support_strategy="Keep the seat over a stable leg, sled, or pedestal footprint.",
        texture_plan=("wood, plastic, fabric, or metal material split by part", "edge wear on contact points"),
        shape_refinement_plan=("round or bevel seat edges", "avoid raw cube legs", "add seams, screws, or caps"),
        critical_constraints=("seat must not float", "all legs/supports must touch ground", "backrest must attach to seat if present"),
    ),
    "table": StructureTemplate(
        parts=("tabletop", "support legs or pedestal", "apron rails", "feet", "edge bevels"),
        primary_forms=("broad horizontal top", "vertical supports", "thin rails under the top"),
        assembly_order=("build tabletop", "attach apron rails below", "attach legs or pedestal", "add feet and edge details"),
        attachment_points=("legs penetrate or bolt into tabletop/apron", "feet contact ground", "rails connect legs"),
        orientation_notes=("top is level and horizontal", "legs are vertical", "lowest feet sit at Z=0"),
        support_strategy="Support footprint spans the tabletop center of mass.",
        texture_plan=("wood grain, laminate, metal, or plastic by part", "subtle bevel wear on top edges"),
        shape_refinement_plan=("bevel tabletop perimeter", "taper legs where appropriate", "add underside rails"),
        critical_constraints=("tabletop must be supported", "no floating legs", "top should not be an unshaped cube"),
    ),
    "storage": StructureTemplate(
        parts=("main storage body", "doors or drawers", "shelves or compartments", "handles", "feet or base plinth"),
        primary_forms=("rectangular cabinet volume", "front panel grid", "thin handles and hinges"),
        assembly_order=("build main carcass", "add internal shelves", "attach doors/drawers", "add handles, hinges, and feet"),
        attachment_points=("doors hinge on side frames", "drawers slide inside openings", "feet attach under base"),
        orientation_notes=("front faces forward", "shelves are horizontal", "body rests on base or feet"),
        support_strategy="A broad base or feet support the storage volume.",
        texture_plan=("wood, painted metal, glass, or plastic panels", "different roughness for handles"),
        shape_refinement_plan=("inset front panels", "bevel box edges", "add handle offsets and panel gaps"),
        critical_constraints=("must have readable front access panels", "handles attach to doors/drawers", "body must be grounded"),
    ),
    "bed": StructureTemplate(
        parts=("mattress", "bed frame", "headboard", "footboard or foot rail", "legs", "side rails"),
        primary_forms=("large rounded mattress", "rectangular support frame", "vertical headboard"),
        assembly_order=("build frame and legs", "place mattress inside/on frame", "attach headboard and rails", "add seams and pillows if requested"),
        attachment_points=("side rails connect headboard and foot rail", "legs touch ground", "mattress sits on frame"),
        orientation_notes=("mattress is horizontal", "headboard is vertical at head end", "frame is grounded"),
        support_strategy="Frame and legs support mattress evenly.",
        texture_plan=("soft fabric mattress", "wood or metal frame", "stitched seams"),
        shape_refinement_plan=("round mattress edges", "bevel frame rails", "add seam lines and corner posts"),
        critical_constraints=("mattress must sit on frame", "headboard must attach to frame", "legs must not float"),
    ),
    "vehicle_four_wheel": StructureTemplate(
        parts=(
            "main unified body shell / chassis",
            "hood panel at front",
            "roof and cabin volume",
            "trunk or rear deck",
            "front left wheel (upright cylinder)",
            "front right wheel (upright cylinder)",
            "rear left wheel (upright cylinder)",
            "rear right wheel (upright cylinder)",
            "front left fender arch",
            "front right fender arch",
            "rear left fender arch",
            "rear right fender arch",
            "windshield (tilted plane)",
            "front headlights",
            "rear taillights",
            "front bumper",
            "rear bumper",
            "side mirrors",
        ),
        primary_forms=(
            "one elongated body shell — wider and longer than it is tall",
            "four upright wheel cylinders at the four corners",
            "curved cabin/roof volume sitting on top of body",
            "flat hood at front, flat trunk lid at rear",
            "fender arches bulging over each wheel",
        ),
        assembly_order=(
            "build a flat low chassis / floor pan first",
            "raise the unified body shell panels around the chassis",
            "add the hood panel over the front engine bay",
            "add the cabin volume and roof on top",
            "place four upright wheel cylinders at the corners touching the floor",
            "add fender arches over each wheel",
            "attach bumpers front and rear",
            "add windshield, headlights, taillights, and mirrors",
        ),
        attachment_points=(
            "wheels sit flush under fender arches, touching Z=0",
            "hood merges with the front of the body shell",
            "windshield tilts from hood edge up to roof edge",
            "fender arches blend flush into body panels",
        ),
        orientation_notes=(
            "Z is up — car sits flat on the ground",
            "vehicle nose points along the Y axis",
            "bilateral symmetry: left side exactly mirrors right side",
            "body must be much longer than it is tall",
            "all four wheels must touch the ground plane (Z=0)",
        ),
        support_strategy="Four wheel-cylinders at the four body corners, all resting on Z=0.",
        texture_plan=(
            "gloss or metallic painted body colour",
            "black rubber tyres",
            "chrome or alloy wheel hubs",
            "transparent glass windshield and windows",
            "chrome trim on bumpers and mirrors",
        ),
        shape_refinement_plan=(
            "body shell must be ONE connected mesh — NOT separate stacked cubes",
            "bevel all sharp body edges for a smooth car silhouette",
            "wheel tyres: use a cylinder with a torus tyre ring for realism",
            "windshield: a flat or gently curved quad plane, angled back",
            "headlights: shallow ellipsoid or rounded rectangle recessed in bumper",
            "add door panel lines as edge creases, not separate objects",
        ),
        critical_constraints=(
            "must be clearly recognisable as a car — NOT a box stack",
            "all four wheels MUST be upright (axle along X) and touch Z=0",
            "bilateral symmetry is mandatory — left exactly mirrors right",
            "body length must be at least 3× body height",
            "do NOT build the body from four or more separate cubes",
            "no wheel may float above the ground",
        ),
    ),
    "batmobile": StructureTemplate(
        parts=(
            "main elongated low armoured body shell",
            "left rear shark fin (tall swept blade)",
            "right rear shark fin (tall swept blade)",
            "jet turbine exhaust nozzle at rear centre",
            "front bat-nose intake prow",
            "front left wheel",
            "front right wheel",
            "rear left wheel (wider oversized)",
            "rear right wheel (wider oversized)",
            "front left flared fender arch",
            "front right flared fender arch",
            "rear left flared fender arch",
            "rear right flared fender arch",
            "narrow fighter-jet cockpit canopy",
            "front battering-ram bumper",
            "side armour panels with intake vents",
            "rear exhaust side pipes",
            "narrow aggressive headlight pods",
            "red taillight strip",
            "bat insignia hood detail",
        ),
        primary_forms=(
            "extremely long low wide body — length >> height",
            "two tall swept shark fins rising from the rear body",
            "circular jet turbine at the very back between the fins",
            "pointed bat-nose prow at the front with an air scoop",
            "four large low-profile wheels, rear pair wider than front",
            "narrow elongated cockpit canopy on top of body centre",
        ),
        assembly_order=(
            "build the long flat chassis / floor pan first (ground level)",
            "raise the main armoured body shell over the chassis",
            "add the two rear shark fins rising vertically from the body",
            "place the circular jet turbine recess/nozzle at the rear between the fins",
            "position four wheels touching the floor at the body corners",
            "add flared fender arches over each wheel",
            "place the narrow cockpit canopy on the body top-centre",
            "attach the front prow intake and battering-ram bumper",
            "add side vents, exhaust pipes, lights, and bat insignia last",
        ),
        attachment_points=(
            "shark fins bolt flush to the rear upper body surface",
            "turbine nozzle sits in a circular recess at the rear",
            "fender arches blend flush with the body panels",
            "cockpit canopy seals to the body with a curved lower edge",
            "front prow extends forward from the body nose",
        ),
        orientation_notes=(
            "Z is up — car sits flat on the ground plane",
            "vehicle nose points along the Y axis",
            "bilateral symmetry on the X axis — left mirrors right",
            "body must be much longer than it is tall (ratio > 4:1)",
            "all four wheels must touch Z=0",
            "cockpit canopy must sit on TOP of the body, never floating",
        ),
        support_strategy="Four wheels on ground. Fins attached to body. No floating parts.",
        texture_plan=(
            "matte black or dark gunmetal armour panels",
            "carbon fibre texture on side armour sections",
            "glossy black cockpit canopy",
            "chrome or brushed-metal wheel hubs",
            "red emission taillight strip",
            "glowing blue-white headlights",
            "subtle bat texture or insignia on hood",
        ),
        shape_refinement_plan=(
            "body: one long extruded and bevelled shell — never stacked cubes",
            "shark fins: thin flat swept triangular blade shapes, not boxes",
            "turbine nozzle: open cylinder or cone, not a cube",
            "cockpit canopy: shallow elongated wedge/dome, not a box",
            "wheels: cylinders with bevelled edges and torus tyres",
            "bevel all hard edges on the body shell for an armoured-plate look",
            "fender arches: curved overhangs integrated into the body",
            "add bat-wing flare to the rear body edges for the silhouette",
        ),
        critical_constraints=(
            "must read as an aggressive sci-fi pursuit car — NOT a box stack",
            "both rear shark fins MUST be present and attached to the body",
            "all four wheels MUST touch the ground — no floating wheels",
            "body must be long and low — height less than 25 % of length",
            "bilateral symmetry is mandatory — left exactly mirrors right",
            "do NOT build the car body from separate scaled cubes",
            "cockpit canopy must sit on TOP of the body shell",
            "jet turbine / exhaust MUST be at the rear, not on the roof",
        ),
    ),
    "vehicle_two_wheel": StructureTemplate(
        parts=("front wheel", "rear wheel", "frame", "handlebar", "seat", "forks", "drive or engine detail"),
        primary_forms=("two upright wheels", "triangular or tubular frame", "handlebar and fork assembly"),
        assembly_order=("place wheels on ground", "connect wheels with frame", "attach fork and handlebar", "add seat and drive details"),
        attachment_points=("fork grips front wheel hub", "rear stays grip rear hub", "seat post connects to frame"),
        orientation_notes=("wheels are vertical and parallel", "frame spans between hubs", "handlebar above front wheel"),
        support_strategy="Wheel contact points and frame alignment visually support the object.",
        texture_plan=("rubber tires", "painted or metal frame", "grips and seat material"),
        shape_refinement_plan=("use cylindrical tubes/cones for frame", "add hubs, spokes, and axle details"),
        critical_constraints=("two wheels must align in one vertical plane", "frame must connect both wheels", "wheels must touch ground"),
    ),
    "cart": StructureTemplate(
        parts=("basket or platform", "handle", "wheel set", "support frame", "axles", "feet or caster forks"),
        primary_forms=("open basket/platform", "tubular push frame", "small wheels/casters"),
        assembly_order=("build basket/platform", "attach frame and handle", "attach axles or caster forks", "add wheels at ground contact"),
        attachment_points=("handle joins rear frame", "wheels attach to axles/forks", "basket sits on support frame"),
        orientation_notes=("handle rises at rear", "wheels are vertical", "platform/basket is level"),
        support_strategy="Wheelbase supports the basket load.",
        texture_plan=("metal wire/frame", "rubber wheels", "plastic handle grips"),
        shape_refinement_plan=("use thin rods/tubes", "add basket rails", "round wheel edges"),
        critical_constraints=("wheels must attach to frame", "basket/platform must be supported", "handle must not float"),
    ),
    "air_vehicle": StructureTemplate(
        parts=("fuselage/body", "wings or rotors", "tail assembly", "landing gear or skids", "windows/panels"),
        primary_forms=("streamlined central body", "horizontal lifting surfaces or rotor disks", "tail stabilizers"),
        assembly_order=("build fuselage", "attach wings/rotors", "attach tail", "add landing supports and panels"),
        attachment_points=("wings penetrate fuselage", "tail connects to rear body", "landing gear attaches under body"),
        orientation_notes=("fuselage points forward", "wings are horizontal", "landing supports below"),
        support_strategy="Landing gear or skids visually ground the aircraft.",
        texture_plan=("painted metal/composite body", "glass windows", "dark tire/skid material"),
        shape_refinement_plan=("taper fuselage ends", "bevel wing edges", "add panel lines and windows"),
        critical_constraints=("wings/rotors must attach to body", "tail must attach to rear", "landing gear must be below"),
    ),
    "watercraft": StructureTemplate(
        parts=(
            "main hull body (tapered bow, wide midship, flat stern)",
            "hull bottom / keel (lowest point, slight V-shape or flat)",
            "deck surface (flat plane on top of hull)",
            "bow prow (pointed or rounded front)",
            "stern transom (flat rear face)",
            "gunwale rails (top edges of hull sides)",
            "cabin or windshield structure",
            "seating area or cockpit",
            "outboard motor or propulsion at stern",
            "propeller / propulsion detail",
            "cleats and deck fittings",
        ),
        primary_forms=(
            "elongated tapered hull — wider at midship, narrower at bow and stern",
            "V-shaped or flat-bottom hull cross-section",
            "flat deck plane sitting flush on top of hull",
            "cabin or windshield volume raised above deck at midship or forward",
            "motor/engine mounted at the stern (rear)",
        ),
        assembly_order=(
            "build the hull body first — tapered elongated form, wider in the middle",
            "shape the bow to a point or rounded curve",
            "flatten the stern transom as a vertical rear face",
            "add the flat deck plane on top of the hull",
            "raise the cabin or windshield above the deck",
            "add the seating cockpit area",
            "attach the outboard motor or engine at the stern",
            "add gunwale rails along the top hull edges",
            "add propeller, cleats, and small deck details last",
        ),
        attachment_points=(
            "deck sits flush on top of the hull — same footprint",
            "cabin mounts on the deck surface, not floating",
            "motor bolts to the stern transom face",
            "gunwale rails follow the top edge of the hull sides",
        ),
        orientation_notes=(
            "Z is up — waterline at Z=0, hull below water extends down",
            "bow (front) points in the Y direction",
            "bilateral symmetry: port side mirrors starboard side",
            "hull is longer than it is wide",
        ),
        support_strategy="Hull is the continuous main volume. Everything else attaches to it.",
        texture_plan=(
            "painted fiberglass hull — white, blue, or red",
            "non-slip grey or teak deck surface",
            "chrome or stainless metal cleats and rails",
            "tinted glass windshield/cabin",
            "dark motor housing",
        ),
        shape_refinement_plan=(
            "hull: one unified mesh — NOT separate stacked boxes",
            "bow: sharply tapered or slightly rounded point",
            "hull cross-section: V-shape at bow, flattening toward stern",
            "bevel all hard hull edges for smooth curved appearance",
            "deck: a slightly raised flat surface with rounded corners",
            "cabin: box with angled front face and glass cut-outs",
            "motor: cylinder/box mounted on a bracket at the stern",
        ),
        critical_constraints=(
            "hull must be ONE continuous watertight mesh — not separate pieces",
            "deck must sit ON TOP of the hull, not floating above it",
            "must read clearly as a boat, not a box",
            "bilateral symmetry is mandatory — left side mirrors right",
            "hull length must be at least 2.5× hull width",
            "motor/propulsion must be at the STERN (rear), not the bow",
            "no floating parts — cabin sits on deck, motor bolts to transom",
        ),
    ),
    "electronics": StructureTemplate(
        parts=("main casing", "screen or grille", "buttons/ports", "stand or feet", "cables or connector details"),
        primary_forms=("compact casing", "front interface plane", "small controls and ports"),
        assembly_order=("build casing", "add screen/grille/front interface", "add buttons, ports, feet, and seams"),
        attachment_points=("controls embedded in front panel", "ports inset into casing", "feet attach to underside"),
        orientation_notes=("front interface faces forward", "device rests on bottom/stand", "ports align to edges"),
        support_strategy="Flat base, feet, or stand supports the device.",
        texture_plan=("plastic/metal casing", "glass or dark screen", "rubber feet", "emissive accents if useful"),
        shape_refinement_plan=("bevel case edges", "inset screens and ports", "add panel seams and screw heads"),
        critical_constraints=("interface must be readable", "buttons/ports must attach to casing", "device must be grounded or mounted"),
    ),
    "appliance": StructureTemplate(
        parts=("main appliance body", "door/lid/front panel", "control panel", "handles", "feet/base", "vents or seams"),
        primary_forms=("large functional body", "front access panel", "small controls and vents"),
        assembly_order=("build body", "add door or front panel", "add handles/control panel", "add feet, vents, and seams"),
        attachment_points=("door sits in front opening", "handle attaches to door", "controls inset in panel"),
        orientation_notes=("front faces user", "base sits on ground/counter", "doors open from front/top"),
        support_strategy="Broad base or feet support the heavy body.",
        texture_plan=("painted metal/plastic body", "glass/dark display panels", "rubber feet"),
        shape_refinement_plan=("bevel body edges", "inset door panels", "add vents, handles, and panel gaps"),
        critical_constraints=("door/control panel must be visible", "handle must attach", "body must not be a plain cube"),
    ),
    "tool": StructureTemplate(
        parts=("handle", "working head", "shaft/body", "grip texture", "fasteners or collars"),
        primary_forms=("handheld grip", "functional head", "connecting shaft/body"),
        assembly_order=("build handle/grip", "attach shaft/body", "attach working head", "add collars, screws, and texture"),
        attachment_points=("head fixed to shaft/body", "handle overlaps rear/end", "collars wrap joints"),
        orientation_notes=("handle aligns for grip", "working end points forward/down", "tool has one clear axis"),
        support_strategy="Small object with all parts connected around one handle/body.",
        texture_plan=("rubber/plastic grips", "metal working surfaces", "edge wear on active head"),
        shape_refinement_plan=("round handle", "bevel metal head", "add grooves and collars"),
        critical_constraints=("working head must attach to handle/body", "no floating functional parts", "shape must reveal tool purpose"),
    ),
    "industrial": StructureTemplate(
        parts=("main housing", "pipes/shafts", "flanges/brackets", "bolts", "base plate", "control/detail panels"),
        primary_forms=("heavy machine body", "cylindrical pipes/shafts", "flat mounting plates"),
        assembly_order=("build base plate", "mount main housing", "attach pipes/shafts/flanges", "add bolts and panels"),
        attachment_points=("flanges overlap pipe ends", "bolts sit on plates", "housing bolts to base"),
        orientation_notes=("base is grounded", "pipes follow real axes", "controls face outward"),
        support_strategy="Base plate and brackets carry the heavy industrial body.",
        texture_plan=("painted metal", "brushed/chrome shafts", "rubber gaskets", "edge grime/wear"),
        shape_refinement_plan=("bevel housings", "use cylinders for pipes", "add flanges, bolts, ribs, and panels"),
        critical_constraints=("pipes/shafts must connect to housing", "base must support body", "flanges and bolts must be visible"),
    ),
    "container": StructureTemplate(
        parts=("container body", "lid or door", "corner posts", "ribs or panels", "handles/latches", "base skids"),
        primary_forms=("hollow box-like volume", "reinforced corners", "panel/rib pattern"),
        assembly_order=("build main body", "add lid/door", "add ribs/corner posts", "add handles/latches and base supports"),
        attachment_points=("lid hinges or rests on rim", "handles attach to sides", "ribs attach to panels"),
        orientation_notes=("opening faces front/top", "base sits on ground", "ribs align with edges"),
        support_strategy="Flat base, skids, or feet support the container.",
        texture_plan=("painted/wood/plastic body", "metal latches", "edge wear and panel variation"),
        shape_refinement_plan=("bevel corners", "inset panels", "add ribs, hinges, handles, and latches"),
        critical_constraints=("container must have readable lid/door/opening", "handles/latches must attach", "body must be grounded"),
    ),
    "sci_fi_crate": StructureTemplate(
        parts=(
            "beveled rectangular cargo body",
            "reinforced metal corner guards",
            "inset side panels",
            "top lid seam and hinge blocks",
            "front latch handles",
            "vent slats",
            "bolt rows",
            "rubber feet",
            "recessed emissive strips",
        ),
        primary_forms=(
            "wide low rectangular hard-surface crate body",
            "raised protective corner blocks and edge rails",
            "recessed side panels with visible panel gaps",
            "thin glowing strips seated in channels",
        ),
        assembly_order=(
            "build the main beveled cargo body",
            "add corner guards and top/bottom edge rails directly on body edges",
            "add recessed panels to front, back, and side faces",
            "attach vent slats, latches, handles, bolts, and emissive strips to panels",
            "add four rubber feet below the bottom corners",
        ),
        attachment_points=(
            "corner guards overlap body corners",
            "edge rails sit flush against crate edges",
            "handles and latches bolt to front panels",
            "vents sit inside panel recesses",
            "rubber feet touch both crate underside and ground",
        ),
        orientation_notes=(
            "crate sits upright on feet with broad face forward",
            "lid seam is on top",
            "front latches face the viewer",
            "emissive strips are inset, not floating",
        ),
        support_strategy="Four rubber feet and a wide base keep the crate grounded and stable.",
        texture_plan=(
            "dark painted metal panels with edge wear",
            "slightly different raw metal corner guards and rails",
            "matte black rubber feet",
            "blue emissive strips in recessed channels",
            "small bolts with darker worn metal",
        ),
        shape_refinement_plan=(
            "bevel every exposed box edge",
            "use layered panels so the object is not a single cube",
            "add repeated vent slats and bolt rows",
            "make handles and latches protrude with thickness",
            "keep details mirrored and aligned to face surfaces",
        ),
        critical_constraints=(
            "must read as a metal sci-fi cargo crate, not a wooden crate",
            "must include corner guards, inset panels, vents, latch handles, bolts, rubber feet, and emissive strips",
            "must have at least three visually distinct materials",
            "details must attach to crate faces and edges",
        ),
    ),
    "organic_tree": StructureTemplate(
        parts=(
            "root flare",
            "tapered trunk or pseudostem",
            "branch or crown structure",
            "layered leaf clusters",
            "leaf center veins",
            "fruit or flower clusters if requested",
            "bark or stem surface strips",
            "ground-contact base",
        ),
        primary_forms=(
            "tapered vertical organic trunk",
            "radial layered canopy or leaf crown",
            "curved leaf blades with center veins",
            "attached fruit clusters hanging from stems",
        ),
        assembly_order=(
            "build root flare and grounded trunk base",
            "stack tapered curved trunk or pseudostem sections",
            "add crown/branch attachment points at the top",
            "attach large leaf blades in overlapping radial layers",
            "attach fruit bunches to visible stalks below the leaf crown",
            "add bark strips, veins, dead leaf accents, and natural asymmetry",
        ),
        attachment_points=(
            "roots overlap trunk base and touch ground",
            "leaf petioles penetrate the crown/trunk top",
            "fruit bunch stalk attaches under the leaf crown",
            "banana fingers attach to the central bunch stalk",
            "bark strips sit on trunk surface",
        ),
        orientation_notes=(
            "Z is up and trunk rises vertically from root base",
            "leaves radiate outward/upward from crown with varied pitch",
            "fruit hangs below crown on visible stalks",
            "lowest roots/base touch Z=0",
        ),
        support_strategy="Root flare and trunk base keep the botanical asset grounded and visually stable.",
        texture_plan=(
            "rough brown/green bark or fibrous stem",
            "dark and light green leaf materials",
            "yellow/green fruit material if fruit is requested",
            "subtle dry leaf tan/brown accents",
        ),
        shape_refinement_plan=(
            "use tapered curved trunk segments, not a straight cylinder",
            "make leaves broad tapered blades with center veins",
            "layer leaves at different heights and angles",
            "cluster fruit in attached bunches",
            "add natural asymmetry with deterministic jitter",
        ),
        critical_constraints=(
            "must read as botanical vegetation, not a hard-surface object",
            "leaves must be shaped and attached, not flat rectangles floating in space",
            "fruit must attach to visible stems or stalks",
            "root/trunk base must be grounded",
        ),
    ),
    "banyan_tree": StructureTemplate(
        parts=(
            "massive central trunk cluster",
            "buttress root flare",
            "grounded prop roots",
            "wide horizontal limb scaffold",
            "hanging aerial root curtains",
            "dense layered canopy pads",
            "small oval leaf clusters",
            "bark ridges and root collars",
        ),
        primary_forms=(
            "broad multi-column trunk mass",
            "radial buttress roots spreading across the ground",
            "wide low crown with horizontal branches",
            "many thin vertical aerial roots hanging from branches to ground",
            "layered rounded foliage masses with individual leaf detail",
        ),
        assembly_order=(
            "build a thick central trunk from overlapping tapered columns",
            "add radial buttress roots and prop roots touching the ground",
            "extend large horizontal branches outward from the trunk crown",
            "drop many aerial roots from branch undersides to the ground",
            "add tiered canopy pads and small leaf clusters around branch tips",
            "add bark ridges, collars, and natural asymmetry",
        ),
        attachment_points=(
            "buttress roots penetrate both trunk base and ground",
            "prop roots touch branch underside and ground plane",
            "aerial root curtains attach to branches and hang vertically",
            "canopy clusters attach to branch tips and upper limbs",
            "bark ridges sit on trunk and root surfaces",
        ),
        orientation_notes=(
            "Z is up and all root bases touch Z=0",
            "branches spread mostly horizontally in a radial crown",
            "aerial roots hang downward, not sideways",
            "canopy is broad and layered rather than a single sphere",
        ),
        support_strategy="Massive trunk, buttress roots, and prop roots create the iconic banyan load path.",
        texture_plan=(
            "rough grey-brown bark with vertical ridges",
            "dark crevice material for root gaps",
            "deep green canopy leaves",
            "lighter green leaf highlights",
            "dry brown aerial root tips and exposed roots",
        ),
        shape_refinement_plan=(
            "avoid a lollipop tree silhouette",
            "make trunk wider than a normal tree and visually fused from multiple columns",
            "use many hanging root strands at varied thicknesses and heights",
            "layer canopy pads in several tiers with irregular edges",
            "add leaf cards around the outer canopy for high-detail silhouette",
        ),
        critical_constraints=(
            "must read clearly as a banyan tree, not a generic tree",
            "must include aerial roots hanging from branches",
            "must include buttress or prop roots grounded around the trunk",
            "must have broad horizontal canopy spread",
            "must not be a single cylinder with one sphere canopy",
        ),
    ),
    "semi_truck": StructureTemplate(
        parts=(
            "cab body (tall bevelled box, 2.5m wide × 3.5m tall)",
            "hood (tapered front nose)",
            "windshield (angled glass plane)",
            "sleeper cab pod on roof",
            "left exhaust stack", "right exhaust stack",
            "left fuel tank", "right fuel tank",
            "2 front steer wheels (upright, axis X)",
            "4 drive axle wheels per side via Mirror+Array",
            "fifth-wheel coupling plate",
            "trailer body (long box 15m × 2.6m × 2.7m)",
            "trailer rear swing doors",
            "trailer landing gear legs",
            "4 trailer wheels per side via Mirror+Array",
            "mud flaps", "tail lights", "marker lights",
        ),
        primary_forms=(
            "tall boxy cab — much taller and narrower than a car",
            "long flat trailer — 3-4× longer than the cab",
            "18 upright wheels in axle pairs visible from side",
            "twin vertical exhaust stacks behind cab",
        ),
        assembly_order=(
            "build cab body with hood and windshield",
            "add sleeper pod and air deflector on roof",
            "add exhaust stacks, fuel tanks, mirrors on cab sides",
            "place 2 front steer wheels (Y=front axle, ±1.28m)",
            "use Mirror modifier X + Array modifier Y for drive axles",
            "add fifth-wheel plate on chassis behind cab",
            "build trailer body behind fifth wheel",
            "add trailer wheels via Mirror + Array at rear",
            "add landing gear, rear doors, lights, mud flaps",
        ),
        attachment_points=(
            "hood merges flush with cab front face",
            "windshield tilts from hood edge up to roof",
            "exhaust stacks bolt to cab rear sides",
            "fuel tanks hang below cab side rails",
            "trailer connects to cab via fifth-wheel king pin",
            "trailer wheels sit in tandem pairs at trailer rear",
        ),
        orientation_notes=(
            "Z is up; truck noses along +Y",
            "bilateral symmetry across X — use Mirror modifier",
            "all 18 wheels upright (circular face vertical, axis X)",
            "cab sits on chassis rails above wheel axles",
        ),
        support_strategy="18 wheels spread across 3 axle groups; all touch Z=0.",
        texture_plan=(
            "painted red/blue cab metal",
            "brushed aluminium trailer sides",
            "black rubber tyres",
            "chrome exhaust stacks, fuel tanks, mirrors",
            "glowing orange/red marker and tail lights",
        ),
        shape_refinement_plan=(
            "cab must be TALL — height ≈ 3.5m, width ≈ 2.5m, not a squashed box",
            "trailer must be LONG — ≈ 15m, dwarfing the cab",
            "use Array modifier for corrugated trailer ribs (14 repeats along Y)",
            "use Mirror modifier for left/right wheel symmetry",
            "bevel cab edges; add door seam lines, grille slots",
        ),
        critical_constraints=(
            "must have 18 wheels total: 2 front + 8 drive + 8 trailer",
            "all wheels must be UPRIGHT (axis X), not lying flat",
            "trailer must be physically attached behind cab",
            "cab height must be at least 3× wheel radius",
            "no floating parts — all wheels on Z=0",
        ),
    ),
    "steam_locomotive": StructureTemplate(
        parts=(
            "boiler barrel (long horizontal cylinder, radius 0.55m, length 5.5m)",
            "smokebox (larger cylinder at front end)",
            "chimney stack (vertical cylinder on smokebox top)",
            "steam dome (sphere/ellipsoid on boiler mid)",
            "sand dome (smaller sphere forward of steam dome)",
            "cab (box at rear with windows)",
            "firebox (wider box connecting boiler to cab)",
            "4 driving wheels per side (large, upright, axis Y)",
            "coupling rods connecting driving wheel axles",
            "piston cylinders (horizontal, at smokebox sides)",
            "leading wheels (2 small at front)",
            "running boards (long flat strips along boiler sides)",
            "cow catcher (angled bars at nose)",
            "tender (coal car behind cab)",
            "headlamp (on smokebox front)",
        ),
        primary_forms=(
            "long horizontal boiler cylinder — the dominant shape",
            "4 large red driving wheels visible from each side",
            "tall chimney stack rising from front",
            "boxy cab at rear with windows",
            "rectangular tender trailing behind",
        ),
        assembly_order=(
            "build the boiler as a horizontal cylinder (axis Y, radius 0.55, length 5.5m)",
            "attach smokebox at Y=+2.75 (front end of boiler)",
            "add chimney on smokebox top",
            "add steam dome and sand dome on boiler top",
            "build firebox connecting boiler rear to cab",
            "build cab box at rear",
            "place 3 driving wheels per side at Y=0.8, 2.2, 3.6, Z=WHEEL_R",
            "use Mirror modifier (axis X) for right-side wheels",
            "add coupling rods as flat boxes connecting wheel centres",
            "add piston cylinders at front, leading wheels at Y=5.2",
            "add running boards along boiler sides",
            "attach tender behind cab at Y=-2.5 to -5.5",
        ),
        attachment_points=(
            "smokebox flush against boiler front face",
            "chimney penetrates smokebox top",
            "domes sit on boiler top surface",
            "firebox connects boiler rear to cab base",
            "wheels axle-Z at WHEEL_R (0.55m above Z=0)",
            "coupling rods span between wheel centre pairs",
            "tender gap-coupled to cab rear",
        ),
        orientation_notes=(
            "Z is up; locomotive faces +Y (chimney at high Y, tender at low Y)",
            "boiler axis is Y (horizontal cylinder with axis=Y rotation)",
            "driving wheels are upright — circular face vertical, axle horizontal (X axis)",
            "bilateral symmetry across X — Mirror modifier for wheels",
        ),
        support_strategy="Driving wheels and leading wheels all touch Z=0; boiler elevated above.",
        texture_plan=(
            "matt black boiler, smokebox, cab, and running gear",
            "bright red driving wheels with polished steel rims",
            "brass steam dome, sand dome, fittings",
            "chrome piston rods, handrails",
            "brown weathered wood tender planks",
        ),
        shape_refinement_plan=(
            "boiler is the visual hero — make it long, round, and prominent",
            "driving wheels must be large (radius 0.52m) and clearly circular",
            "coupling rods must visibly span between all 3 driving wheel centres",
            "chimney should be tall and narrow — clearly visible from distance",
            "cab windows add character — add 2 side windows per side",
        ),
        critical_constraints=(
            "boiler MUST be a horizontal cylinder — not a box",
            "driving wheels MUST be upright and stand on Z=0",
            "coupling rods must connect all 3 driving wheels per side",
            "chimney stack must be on TOP of smokebox, not beside it",
            "tender must be attached behind the cab, not floating",
        ),
    ),
    "pole": StructureTemplate(
        parts=("vertical pole/mast", "base plate or buried foot", "crossarms/brackets", "mounted fixtures", "bolts/collars"),
        primary_forms=("tall vertical cylinder or tapered mast", "horizontal arms", "small mounted equipment"),
        assembly_order=("build base", "raise pole/mast", "attach arms/brackets", "attach fixtures and collars"),
        attachment_points=("pole penetrates base", "arms clamp to pole", "fixtures attach to arms/brackets"),
        orientation_notes=("pole is vertical", "arms are horizontal", "base touches ground"),
        support_strategy="Base plate, foot, or ground embed keeps pole upright.",
        texture_plan=("painted metal/wood/concrete pole", "metal brackets", "weathering near base"),
        shape_refinement_plan=("taper or segment pole", "add collars, bolts, and mounted details"),
        critical_constraints=("pole must be vertical and grounded", "fixtures must attach", "arms must not float"),
    ),
    "architecture_panel": StructureTemplate(
        parts=("main panel/frame", "inner panels or openings", "hinges/rails", "handles/latches", "trim"),
        primary_forms=("flat vertical architectural frame", "inset panels/openings", "thin trim strips"),
        assembly_order=("build outer frame", "add panels/openings", "attach hinges/rails", "add handles and trim"),
        attachment_points=("hinges connect panel to frame", "trim follows panel edges", "handles attach to face"),
        orientation_notes=("panel stands vertical", "front face is readable", "bottom aligns to ground/wall"),
        support_strategy="Frame defines the rigid support perimeter.",
        texture_plan=("wood/metal/glass surfaces", "different material for trim and hardware"),
        shape_refinement_plan=("bevel frame edges", "inset panels", "add hinges, rails, handles, and reveals"),
        critical_constraints=("must not be a plain flat slab", "hardware must attach", "openings/panels must be readable"),
    ),
    "architecture_structure": StructureTemplate(
        parts=("foundation/base", "vertical supports", "horizontal beams", "roof/deck/skin", "rails or trim"),
        primary_forms=("load-bearing frame", "repeating beams/posts", "enclosing or spanning surface"),
        assembly_order=("build foundation/supports", "attach beams/deck/roof", "add rails, trim, and details"),
        attachment_points=("beams rest on posts", "roof/deck attaches to frame", "rails attach to edges"),
        orientation_notes=("vertical supports carry load", "horizontal surfaces are level", "base touches ground"),
        support_strategy="Posts, beams, and foundation create a stable structural frame.",
        texture_plan=("wood/metal/concrete by structural member", "weathered edges and seams"),
        shape_refinement_plan=("bevel beams", "add repeated supports, braces, and trim"),
        critical_constraints=("structure must show load path", "major spans must be supported", "base must be grounded"),
    ),
    "sign": StructureTemplate(
        parts=("sign face", "support pole/frame", "base or wall mounts", "border trim", "text/icon panels"),
        primary_forms=("flat readable sign panel", "thin support frame", "mounting hardware"),
        assembly_order=("build support frame", "attach sign face", "add trim and mounts", "add readable panel details"),
        attachment_points=("sign face bolts to frame", "frame attaches to pole/base", "trim follows perimeter"),
        orientation_notes=("sign face is vertical and forward-facing", "support is grounded or wall-mounted"),
        support_strategy="Pole, frame, or wall mounts visibly hold the sign.",
        texture_plan=("painted sign face", "metal frame", "slight weathering"),
        shape_refinement_plan=("bevel sign border", "add raised trim, bolts, and icon/text blocks"),
        critical_constraints=("sign face must be readable", "support must attach", "panel must not float"),
    ),
    "sports": StructureTemplate(
        parts=("main frame/body", "contact surfaces", "supports/legs", "nets/pads/grips", "fasteners"),
        primary_forms=("sports-specific frame", "repeating supports", "functional contact areas"),
        assembly_order=("build main support frame", "add playing/contact surface", "attach nets/pads/grips", "add fasteners and markings"),
        attachment_points=("supports connect to frame", "nets/pads attach to edges", "grips attach to handles"),
        orientation_notes=("contact surfaces face usable direction", "supports reach ground", "symmetry follows sport object"),
        support_strategy="Frame or base supports the object in its usable pose.",
        texture_plan=("painted metal/plastic", "rubber grips", "fabric/netting where needed"),
        shape_refinement_plan=("round handles/tubes", "add markings, seams, pads, and fasteners"),
        critical_constraints=("object must be usable/readable for its sport", "supports must connect", "details must not float"),
    ),
    "instrument": StructureTemplate(
        parts=("main resonant body", "neck/stand/frame", "strings/keys/valves/pads", "bridge or mounts", "decorative trim"),
        primary_forms=("recognizable instrument body", "thin functional controls", "supporting neck/frame"),
        assembly_order=("build main body", "attach neck/frame", "add strings/keys/valves", "add bridge, mounts, and trim"),
        attachment_points=("neck/frame joins body", "controls sit on body", "strings/keys align along functional path"),
        orientation_notes=("front playing surface faces outward", "controls are accessible", "stand/feet if present support it"),
        support_strategy="Body/frame supports delicate controls and strings/keys.",
        texture_plan=("wood/metal/plastic finish", "darker controls and trim", "subtle gloss"),
        shape_refinement_plan=("curve or bevel body", "add fine controls, strings/keys, and trim"),
        critical_constraints=("must be recognizable as the instrument", "controls must attach", "thin parts need endpoints"),
    ),
    "mech_robot": StructureTemplate(
        parts=(
            "central torso chassis",
            "cockpit or head sensor pod",
            "left shoulder pauldron",
            "right shoulder pauldron",
            "left upper arm",
            "right upper arm",
            "left forearm and weapon or gripper",
            "right forearm and weapon or gripper",
            "hip pelvis block",
            "left upper leg",
            "right upper leg",
            "left lower leg",
            "right lower leg",
            "left foot and ankle joint",
            "right foot and ankle joint",
            "thruster or exhaust vents",
            "panel lines and bolts",
            "emissive sensor or eye strips",
        ),
        primary_forms=(
            "broad boxy torso chassis with beveled panel surfaces",
            "compact rounded head or sensor dome on top of torso",
            "wide armored shoulder plates overhanging the arms",
            "cylindrical or boxy upper arm segments",
            "lower arm with visible weapon barrel, cannon, or gripper claw",
            "wide hip block connecting torso to legs",
            "thick upper leg cylinders with visible joint pivots",
            "angled lower leg armor with exposed mechanical detail",
            "large flat foot pads with ankle pivot details",
        ),
        assembly_order=(
            "build hip pelvis block at world origin",
            "attach torso chassis above the hip block",
            "attach head or sensor pod on top of torso",
            "attach left and right shoulder pauldrons to torso sides",
            "attach upper arms below shoulders",
            "attach forearms and weapon/gripper below upper arms",
            "attach upper legs below hip block",
            "attach lower legs below upper legs",
            "attach feet below lower legs touching ground",
            "add thruster vents, panel lines, bolts, and emissive strips",
        ),
        attachment_points=(
            "head sits centered on top face of torso",
            "shoulder pauldrons overlap torso sides at shoulder height",
            "upper arms hang below pauldrons",
            "forearms connect to lower end of upper arms",
            "upper legs connect to bottom face of hip block",
            "lower legs connect to bottom of upper legs",
            "feet overlap bottom of lower legs and touch ground",
            "thrusters attach to rear torso or leg backs",
        ),
        orientation_notes=(
            "Z is up — mech stands upright with feet on ground plane",
            "torso is above hip which is above legs",
            "arms hang from shoulders with elbows pointing down in rest pose",
            "bilateral symmetry: left and right sides mirror each other",
            "head is at the highest point",
        ),
        support_strategy="Both feet define a wide grounded stance on the Z=0 plane.",
        texture_plan=(
            "dark painted metal chassis panels with edge wear and scratches",
            "slightly lighter raw metal for joints and mechanical parts",
            "emissive blue or red strips for sensors and eye elements",
            "darker grime and burn marks near thrusters and vents",
            "high-contrast panel lines and beveled edges",
        ),
        shape_refinement_plan=(
            "bevel every exposed hard edge on armor plates",
            "use layered inset panels so the torso is not a plain box",
            "add visible joint cylinders at shoulders, elbows, hips, and knees",
            "make weapon barrels or cannon openings clearly cylindrical",
            "add at least three visible emissive elements",
            "keep proportions heroic: wide shoulders, narrow waist, broad feet",
        ),
        critical_constraints=(
            "must read clearly as a bipedal mech or robot, not any other object",
            "both feet must touch the ground — no floating mech",
            "torso must be above hip which must be above legs",
            "include head/sensor pod, shoulder plates, arms, and legs",
            "bilateral symmetry is mandatory",
        ),
    ),
    "weapon_prop": StructureTemplate(
        parts=("handle/grip", "main body/blade/barrel", "guard/stock", "edge/muzzle/detail panels", "fasteners"),
        primary_forms=("clear grip-to-working-end axis", "distinct guard or stock", "functional tip/edge/muzzle"),
        assembly_order=("build grip", "attach main body/blade/barrel", "add guard/stock", "add panels, screws, and details"),
        attachment_points=("grip overlaps body", "guard/stock connects around handle/body", "details attach to surfaces"),
        orientation_notes=("working end points forward", "grip is sized for hand", "symmetry follows weapon type"),
        support_strategy="All components connect around a rigid central body.",
        texture_plan=("metal/plastic/wood by part", "edge wear and dark grip material"),
        shape_refinement_plan=("bevel edges", "shape grip ergonomically", "add panels, grooves, and fasteners"),
        critical_constraints=("must remain an inanimate prop", "grip and working end must connect", "no floating detail panels"),
    ),
    "luggage": StructureTemplate(
        parts=("main soft/hard shell", "zipper seams", "handle", "straps or pockets", "feet or wheels"),
        primary_forms=("rounded rectangular shell", "raised pockets/panels", "handle/strap loops"),
        assembly_order=("build main shell", "add seams/pockets", "attach handles/straps", "add feet/wheels if needed"),
        attachment_points=("handles anchor into shell", "straps wrap around body", "wheels/feet attach below"),
        orientation_notes=("body stands on base/wheels", "front pocket faces outward", "handle is on top/back"),
        support_strategy="Base, feet, or wheels support the luggage body.",
        texture_plan=("fabric/leather/plastic shell", "metal zipper pulls", "rubber wheels/feet"),
        shape_refinement_plan=("round shell corners", "add zipper lines, pockets, buckles, and handles"),
        critical_constraints=("handle/straps must attach", "shell must be rounded not raw cube", "base/wheels must touch ground"),
    ),
    "misc": StructureTemplate(
        parts=("main body", "support/base", "functional details", "edge trim", "surface markings"),
        primary_forms=("recognizable main volume", "supporting base", "small functional details"),
        assembly_order=("build main body", "attach support/base", "add functional details", "add trim and surface marks"),
        attachment_points=("details connect to main body", "base supports object", "trim follows edges"),
        orientation_notes=("object sits in a plausible usable pose", "support is below load", "front/functional side is readable"),
        support_strategy="Visible support or base keeps the object stable.",
        texture_plan=("material contrast by part", "roughness and subtle wear"),
        shape_refinement_plan=("bevel or round main silhouette", "add recognizable details and seams"),
        critical_constraints=("must not be a generic primitive", "details must attach", "object must be grounded or mounted"),
    ),
}


_SPEC_ROWS = """
gaming_chair|gaming chair,ergo gaming chair,ergonomic gaming chair,gamer chair,racing chair|gaming_chair|Gaming Chair|ergonomic racing-style gaming chair|steel, plastic, PU leather or fabric|1.35|bucket seat bolsters;headrest pillow;lumbar pillow
office_chair|office chair,desk chair,task chair|seating|Office Chair|adjustable office task chair|fabric, plastic, steel|1.2|five caster base;gas lift;mesh or cushioned back;arm rests
dining_chair|dining chair,kitchen chair|seating|Dining Chair|simple four-leg dining chair|wood or metal and fabric|1.0|four legs;seat frame;rear back slats
lounge_chair|lounge chair,accent chair|seating|Lounge Chair|low cushioned lounge chair|fabric, wood, metal|1.1|wide cushion;angled back;short legs
rocking_chair|rocking chair|seating|Rocking Chair|wooden rocking chair|wood and fabric|1.1|curved rockers;spindle back;arm rests
bar_stool|bar stool,counter stool|seating|Bar Stool|tall bar stool|metal, wood, vinyl|1.1|tall legs;foot ring;round or square seat
bench|bench|seating|Bench|simple long bench|wood or metal|1.8|long seat plank;two or more leg frames;support rails
park_bench|park bench|seating|Park Bench|outdoor slatted park bench|wood and cast metal|1.8|slatted seat;slatted back;metal side frames
sofa|sofa,couch|seating|Sofa|three-seat upholstered sofa|fabric, wood, foam|2.2|long seat cushions;back cushions;arms;short feet
sectional_sofa|sectional sofa,L sofa|seating|Sectional Sofa|L-shaped sectional sofa|fabric, wood, foam|2.8|L-shaped cushions;corner module;back pillows
recliner|recliner chair|seating|Recliner|padded reclining armchair|leather, fabric, metal|1.3|thick arms;footrest panel;reclined back
armchair|armchair|seating|Armchair|single upholstered armchair|fabric, wood, foam|1.2|wide arms;loose seat cushion;back cushion
folding_chair|folding chair|seating|Folding Chair|foldable metal chair|metal and plastic|1.0|cross braces;thin seat;hinge joints
coffee_table|coffee table|table|Coffee Table|low living-room table|wood, glass, metal|1.2|low tabletop;short legs;lower shelf
dining_table|dining table|table|Dining Table|large dining table|wood or metal|2.0|large rectangular or round top;four legs;apron rails
side_table|side table,end table|table|Side Table|small side table|wood or metal|0.7|small tabletop;single pedestal or four legs
nightstand|nightstand,bedside table|storage|Nightstand|small bedside cabinet|wood, metal|0.7|drawer;open shelf;small tabletop
desk|desk,office desk|table|Desk|work desk with drawers|wood, metal|1.5|desktop;drawer pedestal;modesty panel;cable hole
standing_desk|standing desk|table|Standing Desk|height-adjustable desk|metal and laminate|1.5|telescoping legs;motor housings;desktop
workbench|workbench|table|Workbench|heavy workshop bench|wood and steel|1.8|thick top;lower shelf;vise mounting area
bookshelf|bookshelf,bookcase|storage|Bookshelf|open shelf unit|wood or metal|1.8|side panels;horizontal shelves;back panel
cabinet|cabinet|storage|Cabinet|closed storage cabinet|wood or metal|1.5|double doors;handles;internal shelves
wardrobe|wardrobe,closet|storage|Wardrobe|tall clothing wardrobe|wood|2.0|tall doors;hanging rail;drawer base
dresser|dresser,chest of drawers|storage|Dresser|drawer chest|wood|1.2|stacked drawers;knobs;top slab
bed_frame|bed frame,bed|bed|Bed Frame|bed with mattress and headboard|wood, fabric, metal|2.2|mattress;side rails;headboard;legs
bunk_bed|bunk bed|bed|Bunk Bed|two-level bunk bed|wood or metal|2.2|upper bunk;lower bunk;ladder;guard rails
crib|crib,baby crib|bed|Crib|infant crib|wood|1.4|slatted rails;mattress;corner posts
tv_stand|tv stand,media console|storage|TV Stand|low media console|wood and metal|1.6|open electronics shelves;cabinet doors;cable holes
shoe_rack|shoe rack|storage|Shoe Rack|tiered shoe rack|metal or wood|1.0|stacked shelves;side frames
coat_rack|coat rack|seating|Coat Rack|standing coat rack|wood or metal|1.8|central pole;radial hooks;weighted base
filing_cabinet|filing cabinet,file cabinet|storage|Filing Cabinet|metal office file cabinet|painted metal|1.2|stacked drawers;label plates;handles
display_case|display case,glass cabinet|storage|Display Case|glass display cabinet|glass, metal, wood|1.8|transparent doors;internal shelves;frame posts
kitchen_island|kitchen island|storage|Kitchen Island|kitchen island with storage|wood, stone, metal|1.5|countertop;cabinet base;drawers;overhang
picnic_table|picnic table|table|Picnic Table|outdoor picnic table|wood and metal|2.0|tabletop planks;two benches;A-frame supports
patio_umbrella|patio umbrella,market umbrella|misc|Patio Umbrella|outdoor umbrella|fabric and metal|2.2|central pole;canopy ribs;weighted base
sun_lounger|sun lounger,chaise lounge|seating|Sun Lounger|outdoor reclining lounger|fabric and metal|1.9|long reclined cushion;adjustable back;low frame
classroom_desk|classroom desk,student desk|table|Classroom Desk|student desk and chair combo|wood, metal, plastic|1.0|small desktop;chair seat;tubular frame
conference_table|conference table|table|Conference Table|large meeting table|wood and metal|3.0|long tabletop;central cable trough;pedestal bases
vanity_table|vanity table|table|Vanity Table|makeup vanity table|wood, glass|1.3|small drawers;mirror frame;desktop
mirror_stand|standing mirror,mirror stand|misc|Mirror Stand|freestanding mirror|glass, wood, metal|1.8|mirror panel;support frame;rear kickstand
plant_stand|plant stand|table|Plant Stand|small plant display stand|wood or metal|0.9|tiered shelves;thin legs;pot rings
drawer_unit|drawer unit|storage|Drawer Unit|compact drawer cabinet|wood or plastic|0.9|stacked drawers;handle pulls;caster feet
ottoman|ottoman,footstool|seating|Ottoman|padded footstool|fabric, wood, foam|0.6|soft cushion block;short legs;stitched seams
changing_table|changing table|storage|Changing Table|nursery changing table|wood and fabric|1.1|raised side rails;open shelves;changing pad
drafting_table|drafting table|table|Drafting Table|tilting drafting table|wood and metal|1.4|tilted top;adjustable side supports;crossbar
car|car,sedan|vehicle_four_wheel|Car|compact passenger car|painted metal, rubber, glass|4.2|hood;trunk;four doors;headlights
sports_car|sports car,supercar,exotic car|vehicle_four_wheel|Sports Car|low aerodynamic sports car|painted metal, rubber, glass|4.4|low hood;wide fenders;spoiler;air intakes
batmobile|batmobile,batman car,bat car,batman vehicle,dark knight car,gotham car,batman mobile,bat mobile,hero car,armored car,jet car|batmobile|Batmobile|dark armoured sci-fi pursuit car with shark fins and jet exhaust|matte black metal, carbon fibre, glass|5.2|elongated low body;twin rear shark fins;jet turbine exhaust;four wide wheels;cockpit canopy;front prow intake;fender arches
pickup_truck|pickup truck|vehicle_four_wheel|Pickup Truck|pickup truck with bed|painted metal, rubber, glass|5.0|cab;open cargo bed;tailgate
semi_truck|semi truck,lorry|vehicle_four_wheel|Semi Truck|tractor trailer truck cab|painted metal and rubber|6.0|tall cab;fifth wheel;large wheels;exhaust stacks
bus|bus,city bus|vehicle_four_wheel|Bus|city bus|painted metal, rubber, glass|8.0|long passenger body;many windows;doors
van|van,cargo van|vehicle_four_wheel|Van|boxy cargo van|painted metal, rubber, glass|5.0|sliding door;rear doors;large cabin
motorcycle|motorcycle,motorbike|vehicle_two_wheel|Motorcycle|road motorcycle|metal, plastic, rubber|2.2|fuel tank;engine block;fork;seat
bicycle|bicycle,bike|vehicle_two_wheel|Bicycle|pedal bicycle|metal and rubber|1.8|diamond frame;pedals;chain;handlebar
scooter|scooter,kick scooter|vehicle_two_wheel|Scooter|small scooter|metal, plastic, rubber|1.4|deck;handlebar stem;two small wheels
skateboard|skateboard|sports|Skateboard|skateboard deck|wood and rubber|0.8|curved deck;two trucks;four wheels
wheelchair|wheelchair|cart|Wheelchair|manual wheelchair|metal, rubber, fabric|1.2|large rear wheels;small casters;seat sling;push handles
shopping_cart|shopping cart,trolley|cart|Shopping Cart|wire shopping cart|metal and plastic|1.2|wire basket;push handle;four caster wheels
stroller|stroller,baby stroller|cart|Stroller|folding baby stroller|metal, fabric, rubber|1.2|seat shell;canopy;push handle;small wheels
hand_truck|hand truck,dolly|cart|Hand Truck|two-wheel hand truck|metal and rubber|1.3|vertical frame;toe plate;two wheels
forklift|forklift|vehicle_four_wheel|Forklift|industrial forklift|painted metal and rubber|3.0|mast;forks;counterweight;operator cage
tractor|tractor|vehicle_four_wheel|Tractor|farm tractor|painted metal and rubber|3.5|large rear wheels;small front wheels;engine hood
trailer|trailer|vehicle_four_wheel|Trailer|utility trailer|metal and rubber|3.0|cargo bed;drawbar;tailgate;wheels
train_car|train car,rail car|vehicle_four_wheel|Train Car|railway car|painted metal and glass|8.0|long body;bogies;couplers;windows
subway_car|subway car,metro car|vehicle_four_wheel|Subway Car|metro train car|metal and glass|8.0|sliding doors;long windows;bogies
airplane|airplane,aeroplane|air_vehicle|Airplane|fixed-wing airplane|painted metal and glass|6.0|wings;tail;landing gear;propeller or engines
helicopter|helicopter|air_vehicle|Helicopter|single-rotor helicopter|metal, glass, composite|5.0|main rotor;tail boom;tail rotor;skids
quadcopter_drone|drone,quadcopter|air_vehicle|Quadcopter Drone|four-rotor drone|plastic, carbon, metal|1.0|four arms;four propellers;central body;landing feet
boat|boat,motor boat,motorboat,fishing boat,row boat,dinghy,small boat,wooden boat,inflatable boat|watercraft|Boat|small motor boat|painted fiberglass and metal|4.0|hull;deck;outboard motor;cabin;railings
sailboat|sailboat,sailing boat,sail boat|watercraft|Sailboat|small sailboat|fiberglass, fabric, metal|5.0|hull;mast;sail;boom;rudder
speedboat|speedboat,speed boat,fast boat,powerboat,power boat,racing boat,boat race|watercraft|Speedboat|high-speed powerboat|fiberglass and metal|5.5|low hull;windshield;cockpit;twin engines;wake
yacht|yacht,luxury boat,luxury yacht,sailing yacht,mega yacht|watercraft|Yacht|luxury motor yacht|white fiberglass and glass|12.0|long hull;multi-deck superstructure;cabin windows;bow;stern;mast
ship|ship,cargo ship,container ship,large ship,ocean liner,cruise ship,freighter|watercraft|Ship|large ocean-going cargo ship|painted steel|30.0|hull;superstructure;bridge;cargo deck;bow;stern;anchor
kayak|kayak|watercraft|Kayak|single-person kayak|plastic|3.0|narrow hull;cockpit rim;seat
canoe|canoe|watercraft|Canoe|open canoe|wood and fiberglass|4.0|open hull;seats;paddles
jet_ski|jet ski,water scooter,personal watercraft,pwc|watercraft|Jet Ski|personal watercraft|plastic and rubber|2.5|stepped hull;handlebar;seat
submarine|submarine,sub|watercraft|Submarine|military submarine|painted steel|15.0|cylindrical hull;conning tower;torpedo tubes;propeller
tank|tank,armored vehicle|vehicle_four_wheel|Tank|armored tracked vehicle|painted metal|5.0|turret;cannon;treads;road wheels
go_kart|go kart,kart|vehicle_four_wheel|Go Kart|small racing kart|metal, plastic, rubber|1.8|tubular frame;seat;steering wheel;exposed wheels
f1_car|f1 car,formula one car|vehicle_four_wheel|Formula Car|open-wheel formula race car|carbon fiber, rubber|5.0|front wing;rear wing;open wheels;cockpit
rc_car|rc car,remote control car|vehicle_four_wheel|RC Car|small remote control car|plastic and rubber|0.5|toy chassis;oversized wheels;antenna
snowmobile|snowmobile|vehicle_two_wheel|Snowmobile|tracked snowmobile|metal, plastic, rubber|2.5|front skis;rear track;seat;handlebar
ambulance|ambulance|vehicle_four_wheel|Ambulance|emergency ambulance van|painted metal, rubber, glass|5.5|box cabin;emergency lights;rear doors
fire_truck|fire truck|vehicle_four_wheel|Fire Truck|fire engine|painted metal and rubber|7.0|ladder;hose reels;cab;equipment compartments
tow_truck|tow truck|vehicle_four_wheel|Tow Truck|vehicle recovery truck|painted metal and rubber|5.5|boom arm;hook;flatbed or wheel lift
smartphone|smartphone,phone|electronics|Smartphone|modern smartphone|glass and metal|0.16|thin slab body;screen;camera bump;side buttons
tablet|tablet|electronics|Tablet|touchscreen tablet|glass and metal|0.28|large screen slab;camera;side buttons
laptop|laptop,notebook computer|electronics|Laptop|open laptop computer|metal and plastic|0.35|screen lid;keyboard base;hinge;trackpad
desktop_pc|desktop pc,computer tower|electronics|Desktop PC|computer tower case|painted metal and plastic|0.5|tower case;front vents;ports;side panel
computer_monitor|monitor,computer monitor|electronics|Monitor|desktop monitor|plastic, glass, metal|0.6|screen panel;bezel;neck stand;base
keyboard|keyboard,computer keyboard|electronics|Keyboard|computer keyboard|plastic|0.45|key grid;case;spacebar;feet
computer_mouse|mouse,computer mouse|electronics|Computer Mouse|ergonomic computer mouse|plastic and rubber|0.12|curved shell;buttons;scroll wheel
game_controller|game controller,joystick controller|electronics|Game Controller|console game controller|plastic and rubber|0.18|grips;buttons;thumbsticks;triggers
bluetooth_speaker|bluetooth speaker|electronics|Bluetooth Speaker|portable speaker|plastic, fabric, rubber|0.25|speaker grille;buttons;rubber feet
tower_speaker|tower speaker,floor speaker|electronics|Tower Speaker|floor standing speaker|wood, fabric, plastic|1.0|tall cabinet;woofer circles;tweeter grille;base
headphones|headphones,headset|electronics|Headphones|over-ear headphones|plastic, metal, foam|0.25|headband;ear cups;hinges;ear pads
microphone|microphone|electronics|Microphone|studio microphone|metal and plastic|0.25|cylindrical body;grille;mount ring
webcam|webcam|electronics|Webcam|clip-on webcam|plastic and glass|0.1|camera body;lens;monitor clip
dslr_camera|camera,dslr camera|electronics|Camera|DSLR-style camera|plastic, glass, metal|0.25|body grip;lens barrel;viewfinder;buttons
security_camera|security camera,cctv camera|electronics|Security Camera|wall-mounted CCTV camera|plastic and metal|0.3|camera housing;lens;mount bracket
projector|projector|electronics|Projector|video projector|plastic and glass|0.35|lens barrel;vent grille;control buttons;feet
wifi_router|router,wifi router|electronics|WiFi Router|wireless router|plastic|0.25|flat body;antennas;ports;indicator lights
printer|printer|electronics|Printer|desktop printer|plastic|0.5|paper tray;scanner lid;output slot;buttons
scanner|scanner|electronics|Scanner|flatbed scanner|plastic and glass|0.5|flat lid;glass bed;control panel
television|television,tv|electronics|Television|flat screen TV|glass, plastic, metal|1.2|large screen;thin bezel;stand feet
radio|radio|electronics|Radio|portable radio|plastic and metal|0.3|speaker grille;tuning knob;antenna;handle
smartwatch|smartwatch,watch|electronics|Smartwatch|digital wrist watch|glass, metal, rubber|0.08|watch body;screen;strap;side button
vr_headset|vr headset,virtual reality headset|electronics|VR Headset|virtual reality headset|plastic, foam, fabric|0.25|visor shell;lenses;head straps;face cushion
drone_controller|drone controller,remote controller|electronics|Drone Controller|handheld drone controller|plastic|0.2|dual grips;sticks;antenna;screen mount
external_hard_drive|external hard drive,hard drive|electronics|External Hard Drive|portable storage drive|plastic and metal|0.13|small casing;USB port;indicator light
server_rack|server rack|electronics|Server Rack|rack-mounted server cabinet|painted metal|2.0|rack frame;server trays;vented doors;cable runs
oscilloscope|oscilloscope|electronics|Oscilloscope|bench oscilloscope|plastic, glass, metal|0.4|screen;knobs;input ports;handle
cash_register|cash register|electronics|Cash Register|retail cash register|plastic and metal|0.45|drawer;display;keypad;receipt slot
atm|atm,cash machine|electronics|ATM|automated teller machine|painted metal, plastic|1.6|screen;keypad;card slot;cash slot
vending_machine|vending machine|appliance|Vending Machine|snack vending machine|painted metal, glass|1.9|glass display;product shelves;payment panel;dispense tray
arcade_cabinet|arcade cabinet,arcade machine|electronics|Arcade Cabinet|upright arcade cabinet|wood, plastic, glass|1.8|screen;control panel;joystick;marquee
record_player|record player,turntable|electronics|Record Player|vinyl turntable|plastic, metal, wood|0.45|platter;tonearm;base;dust cover
amplifier|amplifier,audio amp|electronics|Amplifier|audio amplifier|metal and plastic|0.45|front knobs;vented case;feet
synthesizer|synthesizer|electronics|Synthesizer|keyboard synthesizer|plastic and metal|0.8|keybed;knob panel;screen;pitch wheel
desk_lamp|desk lamp|pole|Desk Lamp|adjustable desk lamp|metal and plastic|0.6|base;articulated arms;lamp shade;switch
led_panel|led panel,light panel|electronics|LED Panel|rectangular LED light panel|plastic, glass, metal|0.5|emissive panel;thin frame;mount bracket
refrigerator|refrigerator,fridge|appliance|Refrigerator|upright refrigerator|painted metal and plastic|1.9|large doors;handles;hinges;vent grille
microwave|microwave|appliance|Microwave|countertop microwave oven|painted metal, plastic, glass|0.5|front door;window;control panel;feet
oven|oven|appliance|Oven|kitchen oven|painted metal and glass|0.8|door window;handle;control knobs;racks
stove|stove,cooktop|appliance|Stove|kitchen stove|painted metal and glass|0.9|burners;control knobs;oven door
toaster|toaster|appliance|Toaster|two-slot toaster|metal and plastic|0.3|bread slots;lever;knobs;crumb tray
blender_appliance|blender|appliance|Blender|countertop blender|plastic, glass, metal|0.4|motor base;jar;lid;blade hub
coffee_maker|coffee maker|appliance|Coffee Maker|drip coffee maker|plastic and glass|0.4|water tank;filter basket;carafe;hot plate
electric_kettle|kettle,electric kettle|appliance|Electric Kettle|electric water kettle|metal and plastic|0.3|rounded body;spout;handle;base
washing_machine|washing machine|appliance|Washing Machine|front-load washer|painted metal and glass|0.9|round door;control panel;drum rim;feet
clothes_dryer|dryer,clothes dryer|appliance|Clothes Dryer|front-load dryer|painted metal and glass|0.9|round door;lint vent;control panel
dishwasher|dishwasher|appliance|Dishwasher|built-in dishwasher|painted metal and plastic|0.9|front door;handle;control strip;toe kick
vacuum_cleaner|vacuum cleaner|appliance|Vacuum Cleaner|upright vacuum cleaner|plastic and rubber|1.1|base head;handle;dust bin;wheels
air_conditioner|air conditioner,ac unit|appliance|Air Conditioner|window air conditioner|painted metal and plastic|0.7|vent grille;control panel;outer frame
ceiling_fan|ceiling fan|pole|Ceiling Fan|ceiling mounted fan|metal, plastic, wood|1.2|central hub;fan blades;downrod;light kit
pedestal_fan|fan,pedestal fan|pole|Pedestal Fan|standing fan|plastic and metal|1.3|circular cage;blades;pedestal pole;base
space_heater|heater,space heater|appliance|Space Heater|portable heater|plastic and metal|0.5|front grille;controls;carry handle;feet
water_cooler|water cooler|appliance|Water Cooler|office water dispenser|plastic and metal|1.2|bottle;dispenser body;spigots;drip tray
kitchen_sink|sink,kitchen sink|appliance|Kitchen Sink|countertop sink basin|metal and ceramic|0.7|basin;rim;drain;faucet mount
faucet|faucet,tap|industrial|Faucet|water faucet|chrome metal|0.3|spout;valve handles;base plate
bathtub|bathtub,tub|appliance|Bathtub|bath tub|ceramic or acrylic|1.7|rounded basin;rim;drain;feet or apron
toilet|toilet|appliance|Toilet|flush toilet|ceramic and plastic|0.8|bowl;tank;seat lid;base
shower_head|shower head|industrial|Shower Head|wall shower head|chrome metal and plastic|0.25|spray disk;short pipe;wall flange
medicine_cabinet|medicine cabinet|storage|Medicine Cabinet|bathroom wall cabinet|metal, glass, wood|0.7|mirror door;side frame;shelves
trash_can|trash can,garbage bin|container|Trash Can|waste bin|plastic or metal|0.8|open bin body;lid;foot pedal;handles
recycling_bin|recycling bin|container|Recycling Bin|recycling container|plastic|0.8|bin body;hinged lid;label panel
fire_extinguisher|fire extinguisher|industrial|Fire Extinguisher|portable extinguisher|painted metal, rubber|0.6|cylinder;tapered neck;hose;handle lever
ironing_board|ironing board|table|Ironing Board|folding ironing board|metal and fabric|1.4|narrow padded board;cross legs;hinge braces
sewing_machine|sewing machine|appliance|Sewing Machine|domestic sewing machine|plastic and metal|0.5|machine arm;needle area;base;hand wheel
rice_cooker|rice cooker|appliance|Rice Cooker|countertop rice cooker|plastic and metal|0.35|rounded pot body;lid;handle;control panel
pressure_cooker|pressure cooker|appliance|Pressure Cooker|pressure pot|metal and plastic|0.35|pot body;locking lid;handles;valve
food_processor|food processor|appliance|Food Processor|countertop food processor|plastic and metal|0.4|motor base;bowl;feed tube;lid
stand_mixer|stand mixer|appliance|Stand Mixer|kitchen stand mixer|painted metal|0.45|tilt head;mixing bowl;base;beater
range_hood|range hood|appliance|Range Hood|kitchen ventilation hood|stainless steel|0.9|hood canopy;vent stack;filter grille
water_heater|water heater|appliance|Water Heater|cylindrical water heater|painted metal|1.6|large cylinder;pipes;control panel;base
radiator|radiator|industrial|Radiator|room radiator|painted metal|0.9|vertical fins;top/bottom manifolds;valves
dehumidifier|dehumidifier|appliance|Dehumidifier|portable dehumidifier|plastic|0.6|vent grille;water tank;control panel;casters
humidifier|humidifier|appliance|Humidifier|small humidifier|plastic|0.35|water tank;mist outlet;control knob;base
air_purifier|air purifier|appliance|Air Purifier|tower air purifier|plastic and fabric|0.8|tall body;intake grille;control panel;feet
hammer|hammer|tool|Hammer|claw hammer|steel and rubber or wood|0.35|handle;hammer head;claw;grip
screwdriver|screwdriver|tool|Screwdriver|hand screwdriver|steel and plastic|0.25|handle;metal shaft;tip
wrench|wrench,spanner|tool|Wrench|open-end wrench|steel|0.3|handle;open jaws;ring end
pliers|pliers|tool|Pliers|hand pliers|steel and rubber|0.25|two handles;pivot joint;jaws
hand_saw|saw,hand saw|tool|Hand Saw|manual hand saw|steel and wood|0.5|toothed blade;handle;fasteners
power_drill|drill,power drill|tool|Power Drill|cordless drill|plastic, metal, rubber|0.3|pistol grip;motor body;chuck;battery pack
angle_grinder|angle grinder|tool|Angle Grinder|handheld grinder|plastic and metal|0.35|body;side handle;guard;disc
circular_saw|circular saw|tool|Circular Saw|portable circular saw|plastic and metal|0.45|blade guard;round blade;handle;base plate
chainsaw|chainsaw|tool|Chainsaw|portable chainsaw|plastic and metal|0.7|engine body;guide bar;chain;handles
shovel|shovel|tool|Shovel|digging shovel|wood and metal|1.2|long handle;D grip;spade blade
rake|rake|tool|Rake|garden rake|wood and metal|1.2|long handle;toothed head
ladder|ladder|architecture_structure|Ladder|straight ladder|aluminum or wood|2.0|two side rails;repeated rungs;feet
step_ladder|step ladder|architecture_structure|Step Ladder|folding step ladder|aluminum and plastic|1.5|A-frame rails;steps;top tray;hinge braces
wheelbarrow|wheelbarrow|cart|Wheelbarrow|garden wheelbarrow|metal, rubber, wood|1.4|single front wheel;tray;two handles;rear legs
toolbox|toolbox|container|Toolbox|portable toolbox|metal or plastic|0.5|box body;hinged lid;handle;latches
portable_generator|generator|industrial|Portable Generator|frame-mounted generator|metal and plastic|0.8|engine block;fuel tank;tubular frame;control panel
air_compressor|air compressor|industrial|Air Compressor|portable compressor|painted metal|0.9|pressure tank;motor;handle;wheels
industrial_valve|industrial valve,valve|industrial|Industrial Valve|flanged pipe valve|painted metal and chrome|0.6|valve body;flanges;hand wheel;pipe stubs
pipe_elbow|pipe elbow|industrial|Pipe Elbow|curved pipe elbow|metal or PVC|0.4|curved pipe body;flanges or collars
pipe_flange|pipe flange|industrial|Pipe Flange|bolted pipe flange|metal|0.3|ring flange;bolt holes;pipe neck
water_pump|water pump,pump|industrial|Water Pump|industrial pump|painted metal|0.8|volute housing;motor;inlet flange;outlet flange;base
electric_motor|electric motor|industrial|Electric Motor|industrial electric motor|painted metal|0.6|cylindrical motor body;shaft;base feet;cooling fins
gearbox|gearbox|industrial|Gearbox|mechanical gearbox|cast metal|0.6|box housing;shafts;mounting feet;bolts
conveyor_belt|conveyor belt|industrial|Conveyor Belt|small conveyor|metal and rubber|2.0|belt surface;rollers;side rails;support legs
wooden_pallet|pallet,wooden pallet|container|Wooden Pallet|shipping pallet|wood|1.2|top slats;bottom slats;spacer blocks
sci_fi_cargo_crate|sci-fi crate,sci fi crate,ammo crate,ammunition crate,sci-fi ammo crate,sci fi ammo crate,sci-fi supply crate,sci fi supply crate,sci-fi cargo crate,sci fi cargo crate,sci-fi metal crate,sci fi metal crate,emissive crate,heavy duty sci-fi crate|sci_fi_crate|Sci-Fi Cargo Crate|hard-surface sci-fi cargo crate|painted metal, raw metal, rubber, emissive glass|1.2|corner guards;inset panels;vent slats;latch handles;bolt rows;rubber feet;emissive strips
wooden_crate|wooden crate,wood crate,plank crate,timber crate|container|Wooden Crate|reinforced wooden crate|wood and metal|1.0|plank panels;corner braces;lid;metal straps
shipping_container|shipping container|container|Shipping Container|corrugated cargo container|painted steel|6.0|corrugated walls;corner posts;double doors;locking bars
metal_barrel|barrel,drum|container|Metal Barrel|industrial barrel|painted metal|0.9|cylindrical body;rolled rims;lid bung
gas_cylinder|gas cylinder|industrial|Gas Cylinder|compressed gas cylinder|painted metal|1.2|tall cylinder;rounded cap;valve guard;foot ring
traffic_cone|traffic cone|misc|Traffic Cone|road safety cone|orange plastic|0.7|tapered cone;wide square base;reflective bands
road_barricade|barricade,road barricade|architecture_structure|Road Barricade|traffic barricade|plastic and metal|1.2|A-frame legs;horizontal striped panel;feet
banyan_tree|banyan,banyan tree,ficus benghalensis,indian banyan|banyan_tree|Banyan Tree|high-detail banyan tree with aerial roots and broad canopy|grey-brown bark, dark green leaves, dry root fibers|5.0|massive trunk cluster;buttress roots;prop roots;aerial root curtains;horizontal limbs;layered canopy;leaf clusters;bark ridges
generic_tree|tree,plant,fruit tree,palm tree|organic_tree|Tree|procedural botanical tree|bark, leaves, fruit if requested|2.5|root flare;tapered trunk;branches or crown;leaf clusters;bark strips
banana_tree|banana tree,banana plant,banana palm|organic_tree|Banana Tree|high-detail banana plant with broad leaves and fruit bunch|fibrous green-brown stem, broad leaves, yellow bananas|3.2|root flare;fibrous pseudostem;banana leaf crown;center veins;hanging banana bunch;banana fingers;dry leaf accents
semi_truck|semi truck,semi-truck,eighteen wheeler,18 wheeler,18-wheeler,tractor trailer,big rig,articulated lorry,freight truck,semi,lorry|vehicle_four_wheel|Semi-Truck|heavy-duty 18-wheel tractor-trailer|painted cab metal, aluminium trailer, black rubber, chrome|15.0|cab body;hood;windshield;sleeper pod;exhaust stacks;fuel tanks;front steer wheels;drive wheel sets;fifth wheel;trailer body;trailer rear doors;landing gear;trailer wheels;mud flaps;lights
steam_locomotive|steam locomotive,steam train,locomotive,steam engine|industrial|Steam Locomotive|classic steam-powered railway locomotive with tender|black boiler, red wheels, brass dome, chrome fittings|12.0|boiler barrel;smokebox;chimney stack;steam dome;sand dome;driving wheels;coupling rods;piston cylinders;leading wheels;cab;running boards;cow catcher;tender;headlamp
street_lamp|street lamp,streetlight|pole|Street Lamp|roadside street lamp|painted metal and glass|4.0|tall pole;curved arm;lamp head;base plate
power_pole|power pole,utility pole|pole|Power Pole|wood utility pole|wood, ceramic, metal|6.0|tall pole;crossarms;insulators;wires
electrical_transformer|transformer,electrical transformer|industrial|Electrical Transformer|pole or pad transformer|painted metal|1.2|tank body;cooling fins;bushings;base
solar_panel|solar panel|architecture_structure|Solar Panel|tilted solar array|glass, metal|1.6|panel grid;metal frame;tilted supports
wind_turbine|wind turbine|pole|Wind Turbine|three-blade wind turbine|painted metal|6.0|tall tower;nacelle;three blades;base
antenna_mast|antenna mast,radio tower|pole|Antenna Mast|communications mast|metal|5.0|vertical mast;cross braces;antenna panels;guy brackets
engine_block|engine block|industrial|Engine Block|mechanical engine block|cast metal|0.8|block housing;cylinders;manifolds;belts
hydraulic_jack|hydraulic jack,jack|industrial|Hydraulic Jack|floor jack|painted metal|0.7|low frame;lifting arm;wheels;handle socket
vise|vise,bench vise|industrial|Bench Vise|workbench vise|cast metal|0.4|fixed jaw;moving jaw;screw handle;base
anvil|anvil|industrial|Anvil|blacksmith anvil|steel|0.6|horn;flat face;waist;base feet
fire_hydrant|fire hydrant|industrial|Fire Hydrant|street fire hydrant|painted metal|0.8|main barrel;side nozzles;top cap;base flange
parking_meter|parking meter|pole|Parking Meter|street parking meter|metal and glass|1.2|meter head;display;coin slot;pole
bollard|bollard|pole|Bollard|short protective post|metal or concrete|0.8|short cylinder;cap;base plate;reflective band
manhole_cover|manhole cover|misc|Manhole Cover|round utility cover|cast iron|0.7|flat disk;raised rim;pattern grooves
door|door|architecture_panel|Door|paneled door|wood or metal|2.0|outer frame;inset panels;handle;hinges
window|window|architecture_panel|Window|framed window|glass, wood or metal|1.2|outer frame;glass panes;mullions;sill
garage_door|garage door|architecture_panel|Garage Door|sectional garage door|metal|2.5|wide panel sections;side rails;handle
fence_panel|fence panel,fence|architecture_structure|Fence Panel|fence section|wood or metal|2.0|posts;rails;pickets
gate|gate|architecture_panel|Gate|swing gate|wood or metal|1.8|gate frame;hinges;latch;vertical bars
staircase|staircase,stairs|architecture_structure|Staircase|straight staircase|wood, metal, concrete|2.5|repeated steps;stringers;handrails;posts
spiral_staircase|spiral staircase|architecture_structure|Spiral Staircase|spiral stairs|metal and wood|2.5|central pole;radial treads;spiral handrail
balcony_railing|balcony railing,railing|architecture_structure|Balcony Railing|guard railing|metal or wood|1.5|top rail;vertical balusters;posts
bridge|bridge|architecture_structure|Bridge|small truss bridge|steel or wood|4.0|deck;side trusses;supports;rails
archway|archway,arch|architecture_structure|Archway|architectural arch|stone or brick|2.5|two columns;curved arch;keystone
roof_truss|roof truss|architecture_structure|Roof Truss|triangular roof truss|wood or steel|3.0|top chords;bottom chord;diagonal webs
pillar_column|pillar,column|architecture_structure|Column|architectural column|stone, concrete, metal|2.5|base;shaft;capital;flutes
mailbox|mailbox|container|Mailbox|post-mounted mailbox|painted metal|1.1|mailbox body;front door;flag;post
bus_stop_shelter|bus stop shelter|architecture_structure|Bus Stop Shelter|transit shelter|metal, glass|3.0|roof;side panels;bench;posts
street_sign|street sign,road sign|sign|Street Sign|roadside sign|painted metal|2.0|sign panel;support pole;bolts
billboard|billboard|sign|Billboard|large advertising sign|metal and panel|6.0|large sign face;support columns;rear braces
fountain|fountain|architecture_structure|Fountain|tiered water fountain|stone or concrete|2.0|basin;tiered bowls;central column;spouts
statue_pedestal|statue pedestal,pedestal|architecture_structure|Pedestal|display pedestal|stone, wood, metal|1.0|base plinth;tapered body;top cap
gazebo|gazebo|architecture_structure|Gazebo|open garden gazebo|wood or metal|3.0|posts;roof;railings;floor platform
pergola|pergola|architecture_structure|Pergola|garden pergola|wood or metal|3.0|posts;cross beams;rafters;braces
playground_slide|slide,playground slide|architecture_structure|Playground Slide|children slide|plastic and metal|2.5|ladder;platform;sloped slide;rails
swing_set|swing set|architecture_structure|Swing Set|playground swing frame|metal, rubber, chain|3.0|A-frame supports;top beam;chains;seat
seesaw|seesaw|sports|Seesaw|playground seesaw|wood, metal, rubber|2.5|central fulcrum;long plank;handles;seats
garden_shed|garden shed,shed|architecture_structure|Garden Shed|small outdoor shed|wood or metal|2.5|walls;roof;door;window
castle|castle,fortress,stone castle,medieval castle,dark forest castle|architecture_structure|Castle|medieval stone fortress|stone, dark wood, iron|5.0|curtain walls;corner towers;central keep;gatehouse;arched gate;battlements;stone blocks;flags
lamp_post|lamp post|pole|Lamp Post|decorative lamp post|metal and glass|3.0|post;base;lamp housing;finial
crosswalk_signal|crosswalk signal,pedestrian signal|sign|Crosswalk Signal|pedestrian signal box|painted metal and plastic|2.2|signal box;display face;pole;buttons
traffic_light|traffic light|sign|Traffic Light|three-lens traffic signal|painted metal, glass|2.5|signal head;three lenses;mount arm;pole
guardrail|guardrail|architecture_structure|Guardrail|roadside guardrail|galvanized steel|3.0|corrugated rail;posts;end cap
awning|awning|architecture_structure|Awning|building awning|fabric and metal|1.8|sloped canopy;support arms;wall rail
canopy_tent|canopy tent|architecture_structure|Canopy Tent|portable canopy|fabric and metal|3.0|four legs;roof canopy;cross braces
greenhouse|greenhouse|architecture_structure|Greenhouse|small greenhouse|glass and metal|3.0|transparent walls;roof frame;door;vents
well|water well,well|architecture_structure|Water Well|stone well|stone, wood, rope|1.5|round wall;roof;crank;bucket
chimney|chimney|architecture_structure|Chimney|brick chimney|brick and metal|2.0|vertical shaft;cap;flashing;brick courses
brick_wall|brick wall|architecture_structure|Brick Wall|short brick wall|brick and mortar|2.0|brick rows;mortar grooves;cap stones
culvert|culvert|architecture_structure|Culvert|drainage culvert|concrete or metal|1.5|pipe opening;wing walls;base apron
soccer_goal|soccer goal|sports|Soccer Goal|football goal frame|metal and netting|2.5|rectangular frame;net;ground anchors
basketball_hoop|basketball hoop|sports|Basketball Hoop|basketball backboard and rim|metal, glass, plastic|3.0|backboard;rim;net;support pole
treadmill|treadmill|sports|Treadmill|exercise treadmill|metal, plastic, rubber|1.8|running belt;side rails;console;uprights
exercise_bike|exercise bike,stationary bike|sports|Exercise Bike|stationary exercise bike|metal, plastic, rubber|1.2|flywheel housing;seat;handlebars;pedals
dumbbell_rack|dumbbell rack|sports|Dumbbell Rack|free weight rack|metal and rubber|1.5|tiered rack;paired dumbbells;feet
barbell_bench|weight bench,barbell bench|sports|Weight Bench|bench press station|metal, foam, rubber|1.8|padded bench;uprights;barbell;weight plates
ping_pong_table|ping pong table,table tennis table|sports|Table Tennis Table|ping pong table|wood, metal, netting|2.7|table halves;center net;folding legs
pool_table|pool table,billiards table|sports|Pool Table|billiards table|wood, felt, rubber|2.5|felt bed;rails;pockets;legs
skateboard_ramp|skateboard ramp,half pipe|sports|Skate Ramp|curved skateboard ramp|wood and metal|2.5|curved ramp surface;side panels;coping
punching_bag|punching bag|sports|Punching Bag|hanging heavy bag|leather, chain, metal|1.2|cylindrical bag;top straps;chain mount
acoustic_guitar|acoustic guitar,guitar|instrument|Acoustic Guitar|wood acoustic guitar|wood, metal, nylon|1.0|curved body;sound hole;neck;strings
electric_guitar|electric guitar|instrument|Electric Guitar|solid-body electric guitar|wood, metal, plastic|1.0|body;neck;pickups;bridge;strings
piano|piano|instrument|Piano|upright piano|wood, metal, ivory plastic|1.5|cabinet;keyboard;pedals;music stand
drum_kit|drum kit,drums|instrument|Drum Kit|multi-piece drum set|wood, metal, plastic|1.8|bass drum;snare;toms;cymbals;stands
violin|violin|instrument|Violin|string violin|wood and metal|0.6|curved body;neck;bridge;strings;bow
saxophone|saxophone|instrument|Saxophone|brass saxophone|brass metal|0.7|curved tube;bell;keys;mouthpiece
trumpet|trumpet|instrument|Trumpet|brass trumpet|brass metal|0.5|bell;tubing;valves;mouthpiece
microphone_stand|microphone stand,mic stand|pole|Microphone Stand|adjustable mic stand|metal and rubber|1.5|weighted base;vertical pole;boom arm;clip
music_stand|music stand|pole|Music Stand|folding sheet music stand|metal|1.3|tripod base;telescoping pole;tilted tray
artist_easel|easel|architecture_structure|Artist Easel|wooden painting easel|wood|1.6|tripod legs;tilted canvas support;crossbar
bicycle_rack|bike rack,bicycle rack|architecture_structure|Bicycle Rack|public bike rack|metal|1.5|U-shaped hoops;base plates;bolts
sword|sword|weapon_prop|Sword|fantasy sword prop|metal and leather|1.0|blade;crossguard;grip;pommel
shield|shield|weapon_prop|Shield|hand shield prop|wood, metal, leather|0.8|shield face;rim;rear straps;boss
sci_fi_rifle|sci fi rifle,laser rifle|weapon_prop|Sci-Fi Rifle|futuristic rifle prop|metal and plastic|0.9|stock;grip;barrel;scope;panels
toy_blaster|toy blaster,blaster|weapon_prop|Toy Blaster|toy sci-fi blaster|plastic|0.4|grip;barrel;trigger guard;colored panels
bow|bow|weapon_prop|Bow|archery bow prop|wood, fiber, string|1.2|curved limbs;string;grip
crossbow|crossbow|weapon_prop|Crossbow|crossbow prop|wood, metal, string|0.8|stock;limbs;string;trigger;bolt rail
helmet|helmet|misc|Helmet|protective helmet|plastic, foam, metal|0.3|shell;rim;visor or straps;padding
suitcase|suitcase|luggage|Suitcase|hard shell suitcase|plastic, fabric, metal|0.7|rectangular shell;handle;latches;feet
rolling_luggage|rolling luggage,roller suitcase|luggage|Rolling Luggage|wheeled travel suitcase|plastic, fabric, metal|0.8|telescoping handle;two or four wheels;zipper seams
backpack|backpack|luggage|Backpack|soft backpack|fabric and plastic|0.6|main bag;front pocket;shoulder straps;zipper seams
umbrella|umbrella|misc|Umbrella|handheld umbrella|fabric and metal|1.0|central shaft;canopy ribs;handle
wall_clock|clock,wall clock|misc|Wall Clock|round wall clock|plastic, glass, metal|0.35|round face;hands;tick marks;rim
globe|globe|misc|Globe|desk globe|plastic, metal, wood|0.4|sphere;meridian ring;stand;base
telescope|telescope|instrument|Telescope|tripod telescope|metal, glass, plastic|1.2|optical tube;tripod;finder scope;eyepiece
microscope|microscope|instrument|Microscope|lab microscope|metal, glass, plastic|0.5|base;arm;stage;objective lenses;eyepiece
medical_bed|medical bed,hospital bed|bed|Medical Bed|adjustable hospital bed|metal, plastic, fabric|2.2|segmented mattress;side rails;wheeled base;headboard
hospital_stretcher|stretcher,hospital stretcher|cart|Hospital Stretcher|wheeled stretcher|metal, rubber, fabric|2.0|padded platform;side rails;casters;push handles
iv_stand|iv stand|pole|IV Stand|medical IV pole|metal and plastic|1.8|wheeled base;vertical pole;hooks;clamps
shopping_basket|shopping basket|container|Shopping Basket|hand shopping basket|plastic or wire|0.5|basket body;two handles;open rim
safe|safe|container|Safe|security safe box|thick metal|0.7|heavy body;front door;dial/keypad;hinges
padlock|padlock,lock|misc|Padlock|metal padlock|metal|0.12|lock body;U shackle;keyhole
lantern|lantern|misc|Lantern|portable lantern|metal, glass|0.35|base;glass chamber;top cap;handle
candle_holder|candle holder,candlestick|misc|Candle Holder|decorative candle holder|metal or ceramic|0.3|base;stem;cup;wax candle
trophy|trophy|misc|Trophy|award trophy|metal and wood|0.5|base plaque;cup;handles;stem
chess_board|chess board|misc|Chess Board|board with pieces|wood or plastic|0.5|checkerboard;raised rim;basic pieces
birdhouse|birdhouse|container|Birdhouse|small wooden birdhouse|wood|0.4|house body;pitched roof;round entry hole;perch
picture_frame|picture frame|architecture_panel|Picture Frame|framed picture|wood, glass, paper|0.6|outer frame;inner mat;glass pane;back stand
aquarium|aquarium,fish tank|container|Aquarium|glass fish tank|glass, plastic, metal|0.8|transparent tank;top rim;base;filter box
coffee_mug|coffee mug,mug|container|Coffee Mug|ceramic handled mug|ceramic|0.12|cylindrical cup body;open rim;side handle;foot ring
cup|cup,tea cup|container|Cup|small drinking cup|ceramic or plastic|0.1|cup body;open rim;small base
drinking_glass|drinking glass,glass tumbler|container|Drinking Glass|clear tumbler glass|glass|0.12|transparent cylindrical body;thick base;open rim
water_bottle|water bottle,bottle|container|Water Bottle|reusable bottle|plastic or metal|0.25|tall body;neck;cap;grip grooves
plate|plate,dinner plate|misc|Plate|round dinner plate|ceramic|0.28|shallow disk;raised rim;slight bowl depression
bowl|bowl|container|Bowl|rounded serving bowl|ceramic or plastic|0.22|curved bowl body;open rim;foot ring
frying_pan|frying pan,skillet|tool|Frying Pan|handled frying pan|metal and plastic|0.45|shallow pan;long handle;rim;base disk
saucepan|saucepan,pot|container|Saucepan|handled cooking pot|metal and plastic|0.35|deep pot body;side handle;lid;knob
cutting_board|cutting board|misc|Cutting Board|kitchen cutting board|wood or plastic|0.4|flat board;rounded corners;handle hole
kitchen_knife|kitchen knife,knife|tool|Kitchen Knife|chef knife|steel and wood or plastic|0.32|blade;handle;bolster;edge
fork|fork|tool|Fork|table fork|steel|0.2|handle;neck;four tines
spoon|spoon|tool|Spoon|table spoon|steel|0.2|handle;oval bowl;neck
wine_glass|wine glass|container|Wine Glass|stemmed wine glass|glass|0.22|bowl;thin stem;circular foot
vase|vase|container|Vase|decorative vase|ceramic or glass|0.35|rounded body;narrow neck;open lip;foot
flower_pot|flower pot,plant pot|container|Flower Pot|plant pot|terracotta or plastic|0.25|tapered pot body;open rim;drain tray
bucket|bucket,pail|container|Bucket|utility bucket|plastic or metal|0.35|tapered body;open rim;swing handle
broom|broom|tool|Broom|household broom|wood, plastic, bristles|1.2|long handle;bristle head;collar
mop|mop|tool|Mop|floor mop|metal, plastic, fabric|1.2|long handle;mop head;connector collar
scissors|scissors|tool|Scissors|hand scissors|steel and plastic|0.2|two blades;pivot screw;finger loops
tape_measure|tape measure|tool|Tape Measure|retractable tape measure|plastic and metal|0.08|case;metal tape;belt clip;lock button
camping_tent|tent,camping tent|architecture_structure|Camping Tent|small camping tent|fabric and poles|2.0|fabric canopy;arched poles;door flap;stakes
sci_fi_mech_robot|scifi mech robot,sci fi mech robot,mech robot,battle mech,combat mech,bipedal mech,giant mech,mech suit,war mech,assault mech,sci-fi mech,scifi mech|mech_robot|Sci-Fi Mech Robot|hard-surface bipedal combat mech|dark painted metal, carbon fiber, emissive panels|3.0|bipedal torso chassis;head sensor pod;shoulder pauldrons;arm weapons or cannons;hip pelvis;leg assemblies;feet;thruster vents;emissive sensor strips
humanoid_robot|humanoid robot,android robot,sci fi robot,scifi robot,robot,battle robot,combat robot,war robot,mech warrior|mech_robot|Humanoid Robot|bipedal humanoid robot|metal, plastic, emissive glass|2.0|torso chassis;head sensor;shoulder plates;arm joints;hand grippers;hip assembly;upper legs;lower legs;feet
""".strip()


def _split_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(";") if item.strip())


def _make_blueprints() -> tuple[StructureBlueprint, ...]:
    blueprints: list[StructureBlueprint] = []
    for line in _SPEC_ROWS.splitlines():
        key, aliases, template, category, style, material, scale, signature_parts = line.split("|", 7)
        alias_values = tuple(alias.strip() for alias in aliases.split(",") if alias.strip())
        blueprints.append(
            StructureBlueprint(
                key=key.strip(),
                aliases=alias_values,
                template=template.strip(),
                category=category.strip(),
                style=style.strip(),
                material=material.strip(),
                asset_scale=float(scale),
                signature_parts=_split_list(signature_parts),
            )
        )
    return tuple(blueprints)


BLUEPRINTS = _make_blueprints()


def blueprint_count() -> int:
    return len(BLUEPRINTS)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _score_alias(prompt_norm: str, prompt_tokens: set[str], alias: str) -> int:
    alias_norm = _normalize(alias)
    if not alias_norm:
        return 0
    alias_tokens = set(alias_norm.split())
    if alias_norm == prompt_norm:
        return 1000 + len(alias_norm)
    if f" {alias_norm} " in f" {prompt_norm} ":
        return 700 + len(alias_norm)
    if alias_tokens and alias_tokens.issubset(prompt_tokens):
        return 400 + len(alias_tokens) * 10 + len(alias_norm)
    return 0


def _prompt_material_intent(user_prompt: str) -> set[str]:
    prompt_norm = _normalize(user_prompt)
    intent: set[str] = set()
    if any(term in prompt_norm for term in ("metal", "steel", "industrial", "sci fi", "sci-fi", "emissive")):
        intent.add("metal")
    if any(term in prompt_norm for term in ("wood", "wooden", "plank", "timber")):
        intent.add("wood")
    return intent


def match_structure_blueprint(user_prompt: str) -> StructureBlueprint | None:
    prompt_norm = _normalize(user_prompt)
    prompt_tokens = _tokens(user_prompt)
    material_intent = _prompt_material_intent(user_prompt)
    best: tuple[int, StructureBlueprint] | None = None
    for blueprint in BLUEPRINTS:
        aliases = blueprint.aliases + (blueprint.key.replace("_", " "), blueprint.category)
        score = max(_score_alias(prompt_norm, prompt_tokens, alias) for alias in aliases)
        material = _normalize(blueprint.material)
        style = _normalize(blueprint.style)
        if "metal" in material_intent and "wood" in material and "metal" not in material:
            score -= 260
        if "wood" in material_intent and "wood" not in material and "wood" not in style:
            score -= 180
        if "metal" in material_intent and ("metal" in material or "steel" in material or "metal" in style):
            score += 90
        if "wood" in material_intent and ("wood" in material or "wood" in style):
            score += 90
        if score and (best is None or score > best[0]):
            best = (score, blueprint)
    winner = best[1] if best else None

    if winner and "wood" in _normalize(winner.material) and (
        "sci fi" in prompt_norm or "sci-fi" in user_prompt.lower()
    ):
        for bp in BLUEPRINTS:
            if bp.key == "sci_fi_cargo_crate":
                winner = bp
                break

    return winner


def _append_unique(values: Iterable[str], additions: Iterable[str]) -> list[str]:
    merged = [str(value) for value in values if str(value).strip()]
    seen = {value.strip().lower() for value in merged}
    for addition in additions:
        text = str(addition).strip()
        key = text.lower()
        if text and key not in seen:
            merged.append(text)
            seen.add(key)
    return merged


def _merge_list(part_data: dict, key: str, additions: Iterable[str]) -> None:
    part_data[key] = _append_unique(part_data.get(key, []), additions)


def apply_structure_blueprint(user_prompt: str, part_data: dict) -> dict:
    blueprint = match_structure_blueprint(user_prompt)
    if blueprint is None:
        return apply_similar_memory(user_prompt, part_data)

    template = _TEMPLATES[blueprint.template]
    part_data["structure_blueprint"] = blueprint.key
    part_data["blueprint_count"] = blueprint_count()
    part_data["blueprint_signature_parts"] = list(blueprint.signature_parts)
    part_data["category"] = blueprint.category
    part_data["style"] = blueprint.style
    if not part_data.get("material") or str(part_data.get("material")).lower() in {"metal", "generic", "object"}:
        part_data["material"] = blueprint.material
    part_data["asset_scale"] = max(float(part_data.get("asset_scale", 1.0) or 1.0), blueprint.asset_scale)
    part_data["structure_summary"] = (
        f"{blueprint.category} blueprint '{blueprint.key}' matched from the structure library. "
        f"Build it as a recognizable {blueprint.style.lower()} with its required support and signature parts."
    )
    part_data["support_strategy"] = template.support_strategy

    signature_forms = (
        f"signature {blueprint.category.lower()} silhouette with "
        + ", ".join(blueprint.signature_parts[:4])
    )
    signature_constraints = (
        f"must read clearly as {blueprint.category.lower()}, not a generic primitive or unrelated object",
        "include all blueprint signature parts: " + ", ".join(blueprint.signature_parts[:8]),
    )

    _merge_list(part_data, "parts", template.parts + blueprint.signature_parts)
    _merge_list(part_data, "primary_forms", template.primary_forms + (signature_forms,))
    _merge_list(part_data, "assembly_order", template.assembly_order)
    _merge_list(part_data, "attachment_points", template.attachment_points)
    _merge_list(part_data, "orientation_notes", template.orientation_notes)
    _merge_list(part_data, "texture_plan", template.texture_plan)
    _merge_list(part_data, "shape_refinement_plan", template.shape_refinement_plan)
    _merge_list(part_data, "critical_constraints", template.critical_constraints + signature_constraints)
    part_data = apply_learned_structure(part_data)
    return apply_similar_memory(user_prompt, part_data)


def _load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {"version": 1, "records": {}}
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "records": {}}
    if not isinstance(data, dict):
        return {"version": 1, "records": {}}
    data.setdefault("version", 1)
    data.setdefault("records", {})
    if not isinstance(data["records"], dict):
        data["records"] = {}
    return data


def _save_memory(data: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def apply_learned_structure(part_data: dict) -> dict:
    blueprint = part_data.get("structure_blueprint")
    if not blueprint:
        return part_data

    record = _load_memory().get("records", {}).get(str(blueprint))
    if not isinstance(record, dict):
        return part_data

    approved_runs = int(record.get("approved_runs", 0) or 0)
    if approved_runs <= 0:
        return part_data

    part_data["memory_success_count"] = approved_runs
    part_data["memory_last_prompt"] = record.get("last_prompt", "")

    for field in LIST_FIELDS:
        learned_values = record.get(field, [])
        if isinstance(learned_values, list):
            _merge_list(part_data, field, learned_values)

    learned_note = (
        f"learned structure memory has {approved_runs} strict successful run(s) "
        f"for blueprint '{blueprint}'; preserve those proven proportions and attachments"
    )
    _merge_list(part_data, "critical_constraints", [learned_note])
    return part_data


_MEMORY_STOPWORDS = {
    "a", "an", "the", "with", "of", "and", "or", "for", "in", "on", "to",
    "make", "create", "generate", "build", "made", "model", "asset", "object",
    "simple", "small", "big", "large", "two", "three", "four", "some", "its",
    "is", "that", "this", "from", "at", "by", "it", "into", "style", "3d",
    "wheels", "wheel", "parts", "one",
}


def _content_tokens(text: str) -> set[str]:
    raw = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split()
    return {t for t in raw if t not in _MEMORY_STOPWORDS and len(t) > 2}


def _memory_keyword_score(prompt_tokens: set[str], record: dict) -> int:
    """Score a memory record against prompt tokens using keyword overlap.

    Identity gate: if the prompt shares NO content token with the record's
    blueprint name or category, the record scores 0 regardless of how many
    incidental words its past prompts share. (This is what previously let a
    'hand cart' prompt recall the semi_truck record via 'a'/'with'/'wheels'.)
    """
    prompt_tokens = {t for t in prompt_tokens
                     if t not in _MEMORY_STOPWORDS and len(t) > 2}
    bp_tokens = _content_tokens(record.get("blueprint", ""))
    cat_tokens = _content_tokens(record.get("category", ""))
    identity_overlap = prompt_tokens & (bp_tokens | cat_tokens)
    if not identity_overlap:
        return 0
    score = len(prompt_tokens & bp_tokens) * 30
    score += len(prompt_tokens & cat_tokens) * 20
    for past_prompt in (record.get("prompts") or [])[-5:]:
        overlap = len(prompt_tokens & _content_tokens(past_prompt))
        score += overlap * 5
    return score


def retrieve_exemplar_script(user_prompt: str, part_data: dict | None = None) -> dict:
    """T2.11 — RAG: return the closest known-good stored script as a few-shot.

    Returns {"script": str, "blueprint": str, "score": int, "approved_runs": int}
    or {} when no sufficiently-similar proven exemplar exists.
    """
    prompt_tokens = set(re.sub(r"[^a-z0-9]+", " ", (user_prompt or "").lower()).split())
    if part_data:
        prompt_tokens |= set(
            re.sub(r"[^a-z0-9]+", " ", str(part_data.get("category", "")).lower()).split()
        )
    if not prompt_tokens:
        return {}

    records = _load_memory().get("records", {})
    if not records:
        return {}

    direct_bp = str((part_data or {}).get("structure_blueprint", "")) or ""
    best_score = 0
    best_record: dict | None = None
    for key, record in records.items():
        if not isinstance(record, dict):
            continue
        if not record.get("last_script"):
            continue
        if int(record.get("approved_runs", 0) or 0) < 1:
            continue
        score = _memory_keyword_score(prompt_tokens, record)
        if direct_bp and str(key) == direct_bp:
            score += 100
        if score > best_score:
            best_score = score
            best_record = record

    if best_record is None or best_score < 20:
        return {}

    return {
        "script": best_record.get("last_script", ""),
        "blueprint": best_record.get("blueprint", ""),
        "score": best_score,
        "approved_runs": int(best_record.get("approved_runs", 0) or 0),
    }


def apply_similar_memory(user_prompt: str, part_data: dict) -> dict:
    """
    When the current prompt has no exact blueprint memory, find the closest
    memory record by keyword overlap and partially merge proven structure fields.

    This allows e.g. a 'gaming desk chair' query to benefit from a 'gaming_chair'
    memory record even if the blueprint alias didn't fire.
    Only merges fields that have been seen in multiple approved runs (approved_runs >= 2)
    to avoid poisoning novel assets with unrelated memory.
    """
    if part_data.get("memory_success_count"):
        return part_data

    prompt_tokens = set(re.sub(r"[^a-z0-9]+", " ", user_prompt.lower()).split())
    if not prompt_tokens:
        return part_data

    records = _load_memory().get("records", {})
    if not records:
        return part_data

    best_score = 0
    best_record: dict | None = None
    for record in records.values():
        if not isinstance(record, dict):
            continue
        if int(record.get("approved_runs", 0) or 0) < 2:
            continue
        score = _memory_keyword_score(prompt_tokens, record)
        if score > best_score:
            best_score = score
            best_record = record

    if best_score < 20 or best_record is None:
        return part_data

    approved = int(best_record.get("approved_runs", 0) or 0)
    part_data["memory_similar_blueprint"] = best_record.get("blueprint", "")
    part_data["memory_similar_score"] = best_score
    part_data["memory_success_count"] = part_data.get("memory_success_count") or 0
    part_data["memory_last_prompt"] = best_record.get("last_prompt", "")

    merge_fields = ("shape_refinement_plan", "critical_constraints", "attachment_points")
    if approved >= 3:
        merge_fields = merge_fields + ("assembly_order", "orientation_notes")

    for field in merge_fields:
        learned = best_record.get(field, [])
        if isinstance(learned, list) and learned:
            _merge_list(part_data, field, learned)

    learned_note = (
        f"similar structure memory (blueprint '{best_record.get('blueprint', '?')}', "
        f"{approved} approved run(s), similarity score {best_score}) — use its proven "
        "proportions and attachment patterns as a secondary reference"
    )
    _merge_list(part_data, "critical_constraints", [learned_note])
    return part_data


def _strict_memory_gate(
    part_data: dict,
    quality: dict,
    mcp_report: dict,
    local_report: dict,
) -> tuple[bool, str]:
    if not part_data.get("structure_blueprint"):
        return False, "no matched blueprint"
    if local_report.get("repair_required"):
        return False, "local structural critique still requires repair"
    if not quality.get("success"):
        return False, "export quality summary failed"
    if not quality.get("export_bytes_glb", 0):
        return False, "GLB export missing or empty"
    if not mcp_report.get("success"):
        return False, "runtime topology report failed"
    severity = mcp_report.get("severity")
    repair_class = mcp_report.get("repair_class")
    if severity is not None and severity != "clean":
        return False, f"topology severity is {severity}"
    if severity is None and repair_class not in (None, "clean"):
        return False, f"topology repair class is {repair_class}"
    for key in ("manifold_errors", "non_manifold_faces", "isolated_verts", "ngon_count", "ngon_faces"):
        if int(mcp_report.get(key, 0) or 0) != 0:
            return False, f"{key} is non-zero"
    # Clean topology is necessary but not sufficient: a wrongly-proportioned
    # build can still be manifold. Require a decent vision score when one is
    # available (-1 / absent means visual QA was skipped — don't block then).
    vscore = quality.get("visual_score")
    if vscore is not None and 0 <= int(vscore) < 7:
        return False, f"visual score {int(vscore)}/10 below memory threshold (7)"
    return True, "strict success"


def evaluate_structure_memory_candidate(
    *,
    part_data: dict,
    quality: dict,
    mcp_report: dict,
    local_report: dict,
) -> tuple[bool, str]:
    return _strict_memory_gate(part_data, quality, mcp_report, local_report)


def record_structure_memory(
    *,
    user_prompt: str,
    part_data: dict,
    script: str,
    quality: dict,
    mcp_report: dict,
    local_report: dict,
    result: dict,
) -> tuple[bool, str]:
    ok, reason = _strict_memory_gate(part_data, quality, mcp_report, local_report)
    if not ok:
        return False, reason

    blueprint = str(part_data["structure_blueprint"])
    data = _load_memory()
    records = data.setdefault("records", {})
    record = records.setdefault(
        blueprint,
        {
            "blueprint": blueprint,
            "category": part_data.get("category", "object"),
            "approved_runs": 0,
            "prompts": [],
        },
    )

    record["category"] = part_data.get("category", record.get("category", "object"))
    record["approved_runs"] = int(record.get("approved_runs", 0) or 0) + 1
    record["last_prompt"] = user_prompt
    record["last_updated"] = int(time.time())
    record["last_asset_path"] = result.get("glb_path", "")
    record["last_preview_path"] = result.get("preview_path", "")
    record["last_quality"] = {
        "export_bytes_glb": quality.get("export_bytes_glb", 0),
        "severity": mcp_report.get("severity", mcp_report.get("repair_class", "unknown")),
        "mesh_count": mcp_report.get("mesh_count", 0),
    }

    prompts = _append_unique(record.get("prompts", []), [user_prompt])
    record["prompts"] = prompts[-20:]

    for field in LIST_FIELDS:
        values = part_data.get(field, [])
        if isinstance(values, list):
            record[field] = _append_unique(record.get(field, []), values)[-MEMORY_LIST_LIMIT:]

    signature_parts = part_data.get("blueprint_signature_parts", [])
    if isinstance(signature_parts, list):
        record["blueprint_signature_parts"] = _append_unique(
            record.get("blueprint_signature_parts", []),
            signature_parts,
        )[-MEMORY_LIST_LIMIT:]

    script_lower = script.lower()
    record["feature_flags"] = {
        "uses_bevel": "bmesh.ops.bevel" in script_lower,
        "uses_uvsphere": "create_uvsphere" in script_lower,
        "uses_cone": "create_cone" in script_lower,
        "mesh_object_count": mcp_report.get("mesh_count", 0),
    }

    if isinstance(script, str) and len(script) < 16000:
        record["last_script"] = script
        record["last_script_score"] = quality.get("export_bytes_glb", 0)

    _save_memory(data)
    return True, f"updated blueprint '{blueprint}' ({record['approved_runs']} strict success run(s))"
