"""Noise layer: how a clean part record gets written down by purchasing.

This module knows nothing about drills. It only needs a part with `.noun`,
`.qual` and `.attrs` ((kind, value) pairs), so a different backbone (for
example real BOMs) can reuse it unchanged.

Every purchase line is tagged with an ID style and a description style.
Some styles are held out: they must not be used when tuning the matcher, so
accuracy on them measures generalization and not memory of this generator.
"""
from __future__ import annotations

import re

ID_STYLES = {"exact": 30, "rev_punct": 20, "supplier": 25, "plant_code": 10, "blank": 10, "id_typo": 5}
DESC_STYLES = {"exact": 10, "upper_reorder": 22, "abbrev": 26, "truncated": 10, "unit_variant": 12, "catalog": 12, "typo": 8}
SPANISH_SHARE_P02 = 0.25  # the Monterrey plant keys some item text in Spanish

HELDOUT_ID = {"plant_code", "id_typo"}
HELDOUT_DESC = {"catalog", "typo", "spanish"}


def split_of(id_style: str, desc_style: str) -> str:
    return "heldout" if id_style in HELDOUT_ID or desc_style in HELDOUT_DESC else "tune"


def weighted(rng, table: dict) -> str:
    return rng.choices(list(table), weights=list(table.values()))[0]


def pick_styles(rng, plant: str) -> tuple[str, str]:
    id_style = weighted(rng, ID_STYLES)
    desc_style = "spanish" if plant == "P02" and rng.random() < SPANISH_SHARE_P02 else weighted(rng, DESC_STYLES)
    return id_style, desc_style


# ---------------------------------------------------------------- attributes

# kind -> (engineering style, compact ERP style, alternate-unit style)
_FMT = {
    "V": ("{} V", "{}V", "{}VDC"),
    "A": ("{} A", "{}A", "{} AMP"),
    "mA": ("{} mA", "{}MA", "{} MILLIAMP"),
    "K": ("{} K", "{}K", "{} KELVIN"),
    "uF": ("{} uF", "{}UF", "{} MFD"),
    "ohm": ("{} ohm", "{}OHM", "{} OHMS"),
    "awg": ("{} AWG", "{}AWG", "AWG {}"),
    "teeth": ("{} teeth", "{}T", "Z={}"),
    "mod": ("module {}", "M{}", "MOD {}"),
    "pos": ("{}-position", "{}P", "{} POS"),
    "dia": ("{} mm dia", "D{}", "DIA {}MM"),
    "len": ("{} mm long", "L{}", "LG {}MM"),
    "wid": ("{} mm wide", "W{}MM", "WIDTH {}MM"),
    "thk": ("{} mm thick", "THK{}", "{}MM THK"),
    "idm": ("{} mm ID", "ID{}", "I.D. {}MM"),
    "cs": ("{} mm section", "CS{}", "{}MM C/S"),
    "blades": ("{} blades", "{}BL", "{} BLADE"),
    "pins": ("{} pins", "{}PIN", "{}-PIN"),
    "stage": ("stage {}", "STG{}", "STAGE {}"),
    "pages": ("{} pages", "{}PG", "{} PP"),
}
_MODE = {"exact": 0, "compact": 1, "variant": 2}


def fmt_attr(kind: str, v, mode: str) -> str:
    if kind == "W":
        return {"exact": f"{v} W", "compact": f"{v}W", "variant": f"{v / 1000:g} kW"}[mode]
    if kind == "Ah":
        return {"exact": f"{v} Ah", "compact": f"{v}AH", "variant": f"{int(v * 1000)} mAh"}[mode]
    if kind == "thread":
        return {"exact": v, "compact": v.upper().replace(" ", ""), "variant": v.upper() + "MM"}[mode]
    if kind == "size":
        return {"exact": v, "compact": v.upper().replace(" ", ""), "variant": v.upper().replace(" X ", "X")}[mode]
    if kind == "txt":
        return {"exact": v, "compact": v.upper(), "variant": v.upper().replace(" MM", "MM")}[mode]
    return _FMT[kind][_MODE[mode]].format(v)


def _attrs(part, mode: str) -> list[str]:
    return [fmt_attr(k, v, mode) for k, v in part.attrs]


# -------------------------------------------------------------- descriptions

def canonical(part) -> str:
    """Engineering (BOM) description: 'Brushless motor, 100 W, 18 V'."""
    head = f"{part.qual} {part.noun}".strip()
    return ", ".join([head[0].upper() + head[1:]] + _attrs(part, "exact"))


def alt_canonical(part) -> str:
    """A second engineering convention, used when the same physical part was set up twice."""
    head = part.noun.upper() + (f", {part.qual.upper()}" if part.qual else "")
    return head + "; " + " ".join(_attrs(part, "exact"))


ABBREV = {
    "MOTOR": "MTR", "BRUSHLESS": "BL", "BRUSHED": "BR", "BEARING": "BRG", "SCREW": "SCR", "ASSEMBLY": "ASSY", "HOUSING": "HSG",
    "STAINLESS": "SS", "ZINC": "ZN", "BLACK": "BLK", "SWITCH": "SW", "CAPACITOR": "CAP", "RESISTOR": "RES", "CONNECTOR": "CONN",
    "HARNESS": "HARN", "BATTERY": "BATT", "SPRING": "SPR", "WASHER": "WSHR", "PLANET": "PLNT", "GEAR": "GR", "CARRIER": "CARR",
    "RETAINING": "RET", "RING": "RNG", "LEFT": "LH", "RIGHT": "RH", "CARTON": "CTN", "MANUAL": "MNL", "LABEL": "LBL", "SPINDLE": "SPDL",
    "CONTROLLER": "CTRL", "TRIGGER": "TRIG", "ELECTROLYTIC": "ELEC", "CERAMIC": "CER", "COMPRESSION": "COMP", "THERMISTOR": "THERM",
    "TERMINAL": "TERM", "MANAGEMENT": "MGMT", "HEAD": "HD", "SOCKET": "SOC", "COUNTERSUNK": "CSK", "SELF-TAPPING": "ST",
    "ALUMINUM": "AL", "STEEL": "STL", "SINTERED": "SINT", "SHIELDED": "SHLD", "EXTERNAL": "EXT", "INTERNAL": "INT",
    "INSULATION": "INSUL", "KEYLESS": "KYLS", "CHUCK": "CHK", "SHELL": "SHL", "UPPER": "UPR", "LOWER": "LWR", "OUTPUT": "OUT",
    "COOLING": "CLG", "CARBON": "CARB", "BRUSH": "BRSH", "ROTOR": "ROT", "MAGNET": "MAG", "TORQUE": "TRQ", "CLUTCH": "CLT",
    "DOWEL": "DWL", "HARDENED": "HRD", "SELECTOR": "SEL", "COLLAR": "CLR", "MICROCONTROLLER": "MCU", "THREADED": "THD",
    "INSERT": "INS", "WARRANTY": "WTY", "MOLDED": "MLD", "SERVICE": "SVC", "EXCHANGE": "EXCH", "COMPLETE": "CPL", "CHARGER": "CHGR",
    "FORWARD/REVERSE": "FWD/REV", "LEVER": "LVR", "VARIABLE": "VAR", "SPEED": "SPD", "WIRE-TO-BOARD": "WTB", "SLEEVE": "SLV",
    "DEEP": "DP", "GROOVE": "GRV", "WIRE": "WR", "LIGHT": "LT", "OVERMOLD": "OVMLD", "NAMEPLATE": "NMPLT", "RATCHETING": "RATCH",
}

SPANISH = {
    "MOTOR": "MOTOR", "BRUSHLESS": "SIN ESCOBILLAS", "BRUSHED": "CON ESCOBILLAS", "BEARING": "RODAMIENTO", "BALL": "DE BOLAS",
    "SCREW": "TORNILLO", "NUT": "TUERCA", "WASHER": "ARANDELA", "SPRING": "RESORTE", "GEAR": "ENGRANE", "HOUSING": "CARCASA",
    "SWITCH": "INTERRUPTOR", "BATTERY": "BATERIA", "CELL": "CELDA", "CARTON": "CAJA", "LABEL": "ETIQUETA", "STAINLESS": "INOX",
    "BLACK": "NEGRO", "RED": "ROJO", "GREY": "GRIS", "YELLOW": "AMARILLO", "ORANGE": "NARANJA", "LEFT": "IZQ", "RIGHT": "DER",
    "STEEL": "ACERO", "RING": "ANILLO", "PIN": "PERNO", "WIRE": "CABLE", "HARNESS": "ARNES", "RESISTOR": "RESISTENCIA",
    "CONNECTOR": "CONECTOR", "FAN": "VENTILADOR", "COOLING": "ENFRIAMIENTO", "CHUCK": "MANDRIL", "SPINDLE": "HUSILLO",
    "HEAD": "CABEZA", "COMPRESSION": "COMPRESION", "CARBON": "CARBON", "BRUSH": "ESCOBILLA", "PLANET": "PLANETARIO",
    "CARRIER": "PORTADOR", "UPPER": "SUPERIOR", "LOWER": "INFERIOR", "SHELL": "CUBIERTA", "TRIGGER": "GATILLO", "LIGHT": "LUZ",
    "MANUAL": "MANUAL", "USER": "USUARIO", "WARRANTY": "GARANTIA", "CARD": "TARJETA", "BAG": "BOLSA", "TRAY": "CHAROLA",
    "INSERT": "INSERTO", "THREADED": "ROSCADO", "RETAINING": "RETENCION", "ALUMINUM": "ALUMINIO", "EXTERNAL": "EXTERIOR",
    "INTERNAL": "INTERIOR", "CONTROLLER": "CONTROLADOR", "TERMINAL": "TERMINAL", "BLOCK": "BLOQUE", "STRIP": "TIRA",
    "NICKEL": "NIQUEL", "LATCH": "SEGURO", "PAD": "ALMOHADILLA", "INSULATION": "AISLANTE", "SINK": "DISIPADOR", "HEAT": "CALOR",
    "CASE": "CAJA", "OUTPUT": "SALIDA", "CLUTCH": "EMBRAGUE", "TORQUE": "TORQUE", "HARDENED": "TEMPLADO", "MAGNET": "IMAN",
    "LENS": "LENTE", "COVER": "TAPA", "VENT": "VENTILACION", "CLIP": "CLIP", "BELT": "CINTURON", "HOLDER": "PORTA", "CAP": "TAPA",
    "END": "EXTREMO", "CHARGER": "CARGADOR", "SERVICE": "SERVICIO", "KIT": "JUEGO", "COMPLETE": "COMPLETA", "PACK": "PAQUETE",
}

SYNONYMS = {
    "screw": ["machine screw", "fastener, screw"], "ball bearing": ["radial ball bearing", "bearing"], "motor": ["DC motor", "drive motor"],
    "capacitor": ["cap", "capacitor"], "resistor": ["res", "resistor"], "Li-ion cell": ["lithium-ion rechargeable cell", "Li-ion battery cell"],
    "compression spring": ["coil spring", "spring, compression"], "washer": ["washer", "shim washer"], "nut": ["nut", "locknut"],
    "wire harness": ["cable assembly", "lead wire assy"], "keyless chuck": ["drill chuck, keyless", "chuck assembly"],
    "trigger switch": ["trigger switch module", "speed control switch"], "motor controller PCBA": ["BLDC driver board", "controller board assy"],
    "battery management PCBA": ["BMS board", "protection circuit module"], "carton": ["corrugated box", "shipping carton"],
    "O-ring": ["O-ring seal", "seal ring"], "dowel pin": ["pin, dowel", "parallel pin"], "retaining ring": ["circlip", "snap ring"],
    "label": ["label", "sticker"], "planet gear": ["planetary gear", "gear, planet"], "MOSFET": ["power MOSFET", "MOSFET transistor"],
}
FLUFF = ["RoHS", "REACH compliant", "bulk pack", "OEM grade", "per customer dwg", "lead-free", "std lead time", "MOQ applies"]


def _map_words(text: str, table: dict) -> str:
    return " ".join(table.get(w, w) for w in text.upper().split())


def _typo(text: str, rng) -> str:
    """One or two keying errors, letters only so numeric attributes stay true."""
    s = list(text)
    for _ in range(rng.choice([1, 1, 2])):
        spots = [i for i in range(len(s) - 1) if s[i].isalpha() and s[i + 1].isalpha()]
        if not spots:
            break
        i = rng.choice(spots)
        op = rng.choice(["swap", "drop", "double"])
        if op == "swap":
            s[i], s[i + 1] = s[i + 1], s[i]
        elif op == "drop":
            del s[i]
        else:
            s.insert(i, s[i])
    return "".join(s)


def render_desc(part, style: str, rng) -> str:
    noun, qual = part.noun.upper(), part.qual.upper()
    compact = _attrs(part, "compact")
    reordered = " ".join(x for x in [noun, *compact, qual] if x)
    if style == "exact":
        return canonical(part)
    if style == "upper_reorder":
        return reordered
    if style == "abbrev":
        return " ".join(x for x in [_map_words(noun, ABBREV), _map_words(qual, ABBREV), *compact] if x)
    if style == "truncated":
        return reordered[:22].rstrip()
    if style == "unit_variant":
        return " ".join(x for x in [qual, noun, *_attrs(part, "variant")] if x)
    if style == "catalog":
        syn = rng.choice(SYNONYMS.get(part.noun, [part.noun]))
        head = " ".join(x for x in [syn[0].upper() + syn[1:], part.qual] if x)
        return f"{head} {'/'.join(compact)}, {rng.choice(FLUFF)}"
    if style == "typo":
        return _typo(reordered, rng)
    if style == "spanish":
        return " ".join(x for x in [_map_words(noun, SPANISH), _map_words(qual, SPANISH), *compact] if x)
    raise ValueError(style)


# ------------------------------------------------------------- part numbers

def render_id(style: str, part_number: str, revision: str, supplier_pn: str, plant_code: str, rng) -> str:
    """What lands in the purchase line's part-number field."""
    if style == "exact":
        return part_number
    if style == "supplier":
        return supplier_pn
    if style == "plant_code":
        return plant_code
    if style == "blank":
        return ""
    pre, num = part_number.split("-", 1)
    if style == "rev_punct":
        # purchasing often still orders against the previous revision
        rev = revision if revision == "A" or rng.random() < 0.6 else chr(ord(revision) - 1)
        return rng.choice([f"{pre}{num}-{rev}", f"{pre} {num} REV {rev}", f"{pre}.{num}/{rev}", f"{pre}-{num}_{rev}",
                           f"{pre.lower()}-{num}{rev.lower()}", f"{pre}-{num} R{rev}"])
    if style == "id_typo":
        spots = [i for i in range(len(num) - 1) if num[i] != num[i + 1]]
        if spots and rng.random() < 0.7:
            i = rng.choice(spots)
            num = num[:i] + num[i + 1] + num[i] + num[i + 2:]
        elif "0" in num:
            num = num.replace("0", "O", 1)
        else:
            num = num[:-1]
        return f"{pre}-{num}"
    raise ValueError(style)


# -------------------------------------------------------- units and suppliers

_EACH = ["EA", "EA", "EA", "EACH", "PC", "PCS"]
_PACKS = {10: ["BX10", "BOX/10", "BX (10)"], 25: ["PK25", "PACK 25"], 100: ["BX100", "C", "BOX/100"], 1000: ["M", "BX1000", "PER 1000"]}


def pick_uom(rng, pack: bool) -> tuple[str, int]:
    """Purchase unit of measure and how many base units (EA) it holds."""
    if pack and rng.random() < 0.4:
        units = rng.choice([10, 25, 100, 100, 1000])
        return rng.choice(_PACKS[units]), units
    return rng.choice(_EACH), 1


_SUFFIX = re.compile(r",?\s+(Inc\.|LLC|Corp\.|Co\., Ltd\.|Co\.|Ltd\.|GmbH|S\.A\. de C\.V\.)$")


def supplier_name_variant(name: str, rng) -> str:
    """Vendor master records are not consistent across plants either."""
    bare = _SUFFIX.sub("", name)
    return rng.choices([name, name.upper(), bare, bare.upper()[:22].rstrip()], weights=[50, 20, 15, 15])[0]
