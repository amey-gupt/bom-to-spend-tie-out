"""Static definitions for the synthetic cordless-drill backbone.

Everything here is invented: products, part categories, suppliers. Nothing is
taken from a real company. The generator (generate.py) turns these definitions
into BOMs; the noise layer (noise.py) turns clean parts into messy purchases.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    voltage: int
    plant: str
    base_units: int      # typical units built per month
    launch_month: int    # index into the build calendar; 0 = from the start


PRODUCTS = [
    Product("DR-12C", "12 V compact drill/driver", 12, "P01", 4200, 0),
    Product("DR-18", "18 V drill/driver", 18, "P01", 6500, 0),
    Product("DR-18H", "18 V hammer drill", 18, "P02", 3800, 0),
    Product("DR-20", "20 V drill/driver", 20, "P02", 2600, 6),
]

# P01/P02 assemble product; SDC is the service distribution center.
PLANTS = {"P01": "Greenville assembly", "P02": "Monterrey assembly", "SDC": "Memphis service DC"}


@dataclass(frozen=True)
class Cat:
    key: str
    prefix: str                 # internal part-number prefix
    slots: tuple                # ((assembly, number of BOM positions), ...)
    noun: str
    attrs: tuple                # ((kind, (allowed values...)), ...)
    price: tuple                # (low, high) USD per base unit
    commodity: str
    quals: tuple = ("",)
    qty: tuple = (1,)           # allowed qty_per_unit values
    share: str = "mixed"        # how much products share the part: common | mixed | unique | volt
    pack: bool = False          # small part that is often bought by the box
    fail: bool = False          # failure-prone: drives spares and warranty demand
    only: tuple | None = None   # product indexes, when not used by every product


COLORS = ("black", "red", "teal", "grey", "yellow", "orange")
RESINS = ("ABS", "PA6-GF30", "PC/ABS", "TPE")
FINISH = ("zinc", "stainless", "black oxide")
THREADS = ("M2.5 x 6", "M2.5 x 8", "M3 x 6", "M3 x 8", "M3 x 10", "M3 x 12", "M3 x 16", "M4 x 8",
           "M4 x 10", "M4 x 12", "M4 x 16", "M4 x 20", "M5 x 12", "M5 x 16", "M5 x 20", "M6 x 16")
BOX_SIZES = ("280 x 90 x 240 mm", "320 x 100 x 260 mm", "340 x 110 x 280 mm",
             "400 x 300 x 280 mm", "560 x 330 x 270 mm", "600 x 400 x 300 mm")


def _plastic(key, prefix, noun, price, share="mixed"):
    return Cat(key, prefix, (("HOUSING", 1),), noun, (("txt", COLORS), ("txt", RESINS)), price, "Plastics", share=share)


CATS = [
    # --- motor ---
    Cat("motor", "MTR", (("MOTOR", 1),), "motor", (("W", (100, 150, 200, 250, 300, 350, 400, 450)), ("V", (12, 18, 20))),
        (6.5, 13.0), "Motors", quals=("brushless", "brushed"), share="volt", fail=True),
    Cat("rotor", "ROT", (("MOTOR", 1),), "rotor assembly", (("dia", (28, 32, 36, 40)), ("len", (30, 35, 40, 45))), (1.8, 3.2), "Motors"),
    Cat("stator", "STA", (("MOTOR", 1),), "stator", (("dia", (38, 42, 46, 50)), ("len", (25, 30, 35))), (2.0, 3.6), "Motors", quals=("wound",)),
    Cat("brush", "BRS", (("MOTOR", 1),), "carbon brush",
        (("size", ("5 x 8 x 12 mm", "6 x 9 x 14 mm", "6 x 6 x 12 mm", "7 x 11 x 17 mm", "5 x 10 x 15 mm")),),
        (0.12, 0.30), "Motors", qty=(2,), share="common", pack=True, fail=True),
    Cat("fan", "FAN", (("MOTOR", 1),), "cooling fan", (("dia", (40, 45, 50, 55)), ("blades", (7, 9, 11))), (0.25, 0.60), "Plastics"),
    Cat("bearing", "BRG", (("MOTOR", 2), ("GEARBOX", 3), ("CHUCK", 1)), "ball bearing",
        (("txt", ("608", "625", "626", "607", "688", "698", "6000", "6001", "6200", "6201")), ("txt", ("ZZ", "2RS", "open"))),
        (0.35, 1.10), "Bearings", quals=("deep groove",), share="common"),
    Cat("endcap", "ECP", (("MOTOR", 1),), "motor end cap",
        (("txt", ("PA66-GF30", "PBT-GF30", "die-cast aluminum")), ("dia", (38, 42, 46, 50))), (0.40, 0.90), "Plastics"),
    Cat("magnet", "MAG", (("MOTOR", 1),), "magnet ring",
        (("txt", ("NdFeB N35", "NdFeB N42", "ferrite Y30")), ("txt", ("4-pole", "6-pole", "8-pole"))), (0.60, 1.80), "Motors"),
    # --- gearbox ---
    Cat("planet", "PGR", (("GEARBOX", 3),), "planet gear", (("teeth", tuple(range(12, 22))), ("mod", (0.5, 0.6, 0.8))),
        (0.12, 0.35), "Gears", quals=("steel", "sintered"), qty=(3, 4, 5)),
    Cat("sun", "SGR", (("GEARBOX", 2),), "sun gear", (("teeth", (9, 10, 11, 12, 13, 14)), ("mod", (0.5, 0.6, 0.8))),
        (0.20, 0.50), "Gears", quals=("steel", "sintered")),
    Cat("ringgear", "RGR", (("GEARBOX", 1),), "ring gear", (("teeth", (45, 48, 51, 54, 57)), ("dia", (40, 44, 48))),
        (0.80, 1.60), "Gears", quals=("sintered", "nylon")),
    Cat("carrier", "PCR", (("GEARBOX", 2),), "planet carrier", (("stage", (1, 2, 3)), ("pins", (3, 4, 5))), (0.50, 1.20), "Gears"),
    Cat("gearcase", "GCS", (("GEARBOX", 1),), "gear case", (("dia", (44, 48, 52, 56)), ("len", (38, 45, 52))),
        (0.90, 2.20), "Gears", quals=("aluminum", "nylon")),
    Cat("spindle", "SPD", (("GEARBOX", 1),), "output spindle", (("txt", ("1/2-20 UNF", "3/8-24 UNF")), ("len", (60, 68, 75, 82))),
        (1.10, 2.40), "Gears"),
    Cat("clutch", "CLR", (("GEARBOX", 1),), "torque clutch ring", (("pos", (15, 18, 21, 24)),), (0.30, 0.70), "Plastics"),
    Cat("spring", "SPR", (("GEARBOX", 2), ("ELECTRONICS", 1), ("CHUCK", 1), ("BATTERY", 1)), "compression spring",
        (("dia", (4, 5, 6, 8, 10, 12)), ("len", (8, 10, 12, 16, 20, 25)), ("txt", ("0.5 mm wire", "0.6 mm wire", "0.8 mm wire", "1.0 mm wire"))),
        (0.03, 0.15), "Springs", qty=(1, 2), share="common", pack=True),
    Cat("pin", "PIN", (("GEARBOX", 2), ("CHUCK", 1)), "dowel pin",
        (("size", ("2 x 8 mm", "2 x 10 mm", "3 x 10 mm", "3 x 12 mm", "3 x 16 mm", "4 x 12 mm", "4 x 16 mm", "4 x 20 mm", "5 x 20 mm")),),
        (0.02, 0.08), "Fasteners", quals=("hardened",), qty=(1, 2, 3, 4), share="common", pack=True),
    Cat("retring", "RRG", (("GEARBOX", 2), ("CHUCK", 1), ("MOTOR", 1)), "retaining ring",
        (("txt", tuple(f"{d} mm shaft" for d in (6, 8, 10, 12, 15, 17, 20, 22))),),
        (0.02, 0.07), "Springs", quals=("external", "internal"), share="common", pack=True),
    Cat("oring", "ORG", (("GEARBOX", 1), ("CHUCK", 1)), "O-ring",
        (("idm", (8, 10, 12, 15, 18, 20, 25)), ("cs", (1, 1.5, 2)), ("txt", ("NBR 70", "FKM 75"))),
        (0.02, 0.09), "Springs", share="common", pack=True),
    Cat("washer", "WSH", (("HARDWARE", 4), ("GEARBOX", 2)), "washer", (("txt", ("M3", "M4", "M5", "M6", "M8")), ("txt", FINISH)),
        (0.005, 0.03), "Fasteners", quals=("flat", "spring", "thrust"), qty=(2, 4), share="common", pack=True),
    # --- chuck ---
    Cat("chuck", "CHK", (("CHUCK", 1),), "keyless chuck",
        (("txt", ("10 mm capacity", "13 mm capacity")), ("txt", ("1/2-20 UNF", "3/8-24 UNF"))),
        (3.8, 6.5), "Chuck", quals=("metal sleeve", "single sleeve", "ratcheting"), fail=True),
    Cat("chuckscrew", "CSC", (("CHUCK", 1),), "chuck retaining screw", (("thread", ("M5 x 22", "M6 x 22", "M6 x 25", "M5 x 25")),),
        (0.04, 0.09), "Fasteners", quals=("left-hand",), share="common", pack=True),
    Cat("hammercam", "HCM", (("CHUCK", 2),), "hammer cam", (("dia", (28, 30, 32)), ("teeth", (12, 14, 16))),
        (0.70, 1.40), "Gears", quals=("fixed", "rotating"), only=(2,)),
    Cat("modecollar", "MCL", (("CHUCK", 1),), "mode selector collar", (("pos", (2, 3)), ("dia", (52, 56, 60))), (0.30, 0.60), "Plastics"),
    # --- battery ---
    Cat("cell", "CEL", (("BATTERY", 1),), "Li-ion cell", (("txt", ("18650", "21700")), ("Ah", (1.5, 2.0, 2.5, 3.0, 4.0))),
        (1.6, 3.4), "Batteries", qty=(5,), fail=True),
    Cat("bms", "BMS", (("BATTERY", 1),), "battery management PCBA", (("V", (12, 18, 20)), ("A", (20, 30, 40))),
        (2.2, 4.0), "PCBA", share="volt", fail=True),
    Cat("packtop", "BHU", (("BATTERY", 1),), "battery housing upper", (("V", (12, 18, 20)), ("txt", COLORS)), (0.5, 1.0), "Plastics", share="volt"),
    Cat("packbot", "BHL", (("BATTERY", 1),), "battery housing lower", (("V", (12, 18, 20)), ("txt", COLORS)), (0.5, 1.0), "Plastics", share="volt"),
    Cat("terminal", "TRM", (("BATTERY", 1), ("ELECTRONICS", 1)), "terminal block", (("pos", (3, 4, 5)), ("A", (20, 30, 40))),
        (0.30, 0.70), "Electronics"),
    Cat("nickel", "NKS", (("BATTERY", 1),), "nickel strip", (("wid", (6, 8, 10)), ("thk", (0.1, 0.15, 0.2))),
        (0.03, 0.10), "Batteries", qty=(4, 6, 8), pack=True),
    Cat("ntc", "NTC", (("BATTERY", 1),), "NTC thermistor", (("ohm", ("10k", "47k", "100k")), ("txt", ("B3950", "B3435"))),
        (0.05, 0.15), "Electronics", share="common", pack=True),
    Cat("latch", "LAT", (("BATTERY", 1),), "battery latch", (("txt", COLORS),), (0.12, 0.30), "Plastics"),
    Cat("insul", "INS", (("BATTERY", 2),), "insulation pad",
        (("txt", ("fish paper", "PET film", "silicone foam")), ("size", ("60 x 20 mm", "65 x 40 mm", "90 x 20 mm", "95 x 60 mm", "70 x 70 mm"))),
        (0.02, 0.08), "Batteries", pack=True),
    # --- electronics ---
    Cat("ctrl", "PCB", (("ELECTRONICS", 1),), "motor controller PCBA", (("V", (12, 18, 20)), ("A", (20, 30, 40, 50))),
        (5.5, 9.5), "PCBA", share="volt", fail=True),
    Cat("switch", "SWT", (("ELECTRONICS", 1),), "trigger switch", (("V", (12, 18, 20)), ("A", (12, 16, 20, 24))),
        (1.8, 3.2), "Electronics", quals=("variable speed",), share="volt", fail=True),
    Cat("lever", "LVR", (("ELECTRONICS", 1),), "forward/reverse lever", (("txt", COLORS),), (0.06, 0.15), "Plastics"),
    Cat("led", "LED", (("ELECTRONICS", 1),), "LED work light", (("K", (4000, 5000, 6500)), ("mA", (20, 60, 150))),
        (0.08, 0.25), "Electronics", share="common"),
    Cat("mosfet", "FET", (("ELECTRONICS", 2),), "MOSFET", (("V", (30, 40, 60)), ("A", (60, 80, 100, 120)), ("txt", ("TO-220", "DPAK", "PDFN 5x6"))),
        (0.22, 0.60), "Electronics", quals=("N-channel",), qty=(2, 3, 6), pack=True),
    Cat("cap", "CAP", (("ELECTRONICS", 3),), "capacitor", (("uF", (0.1, 1, 10, 47, 100, 220, 470, 1000)), ("V", (16, 25, 35, 50))),
        (0.02, 0.30), "Electronics", quals=("electrolytic", "ceramic"), qty=(1, 2, 4), share="common", pack=True),
    Cat("res", "RES", (("ELECTRONICS", 3),), "resistor",
        (("ohm", ("0.002", "0.005", "10", "100", "1k", "4.7k", "10k", "100k")), ("txt", ("0603", "0805", "1206", "2512"))),
        (0.004, 0.12), "Electronics", quals=("chip", "shunt"), qty=(1, 2, 4), share="common", pack=True),
    Cat("conn", "CON", (("ELECTRONICS", 2),), "wire-to-board connector",
        (("pos", (2, 3, 4, 6)), ("txt", ("2.0 mm pitch", "2.54 mm pitch", "3.96 mm pitch"))), (0.06, 0.20), "Electronics", pack=True),
    Cat("harness", "WHR", (("ELECTRONICS", 2), ("BATTERY", 1)), "wire harness", (("awg", (14, 16, 18, 22)), ("len", (80, 120, 160, 200, 250))),
        (0.25, 0.80), "Wire"),
    Cat("heatsink", "HSK", (("ELECTRONICS", 1),), "heat sink",
        (("size", ("30 x 20 x 8 mm", "40 x 25 x 10 mm", "45 x 30 x 12 mm", "50 x 30 x 10 mm")),),
        (0.20, 0.50), "Electronics", quals=("extruded aluminum",)),
    Cat("mcu", "MCU", (("ELECTRONICS", 1),), "microcontroller", (("txt", ("32-bit", "8-bit")), ("txt", ("LQFP-32", "QFN-28", "TSSOP-20", "LQFP-48"))),
        (0.45, 1.30), "Electronics"),
    # --- housing ---
    _plastic("shell_l", "HSL", "housing shell left", (0.9, 1.6), share="unique"),
    _plastic("shell_r", "HSR", "housing shell right", (0.9, 1.6), share="unique"),
    _plastic("grip", "GRP", "overmold grip", (0.30, 0.60)),
    _plastic("trigger", "TRG", "trigger", (0.08, 0.20)),
    _plastic("bitholder", "BTH", "bit holder", (0.05, 0.15)),
    _plastic("ventcover", "VNT", "vent cover", (0.06, 0.14)),
    _plastic("nameplate", "NPL", "nameplate", (0.05, 0.20)),
    _plastic("lens", "LNS", "light lens", (0.04, 0.10)),
    Cat("beltclip", "BCL", (("HOUSING", 1),), "belt clip", (("txt", ("spring steel", "stainless")), ("len", (45, 55, 65))),
        (0.15, 0.35), "Springs", share="common"),
    # --- hardware ---
    Cat("screw", "SCR", (("HARDWARE", 12), ("HOUSING", 2), ("GEARBOX", 2), ("BATTERY", 2), ("MOTOR", 1), ("ELECTRONICS", 1)), "screw",
        (("thread", THREADS), ("txt", FINISH)), (0.006, 0.04), "Fasteners",
        quals=("pan head", "socket cap", "countersunk", "self-tapping", "Torx pan head"), qty=(2, 4, 6, 8), share="common", pack=True),
    Cat("nut", "NUT", (("HARDWARE", 3),), "nut", (("txt", ("M3", "M4", "M5", "M6")), ("txt", FINISH)),
        (0.005, 0.03), "Fasteners", quals=("hex", "nyloc", "square"), qty=(2, 4), share="common", pack=True),
    Cat("insert", "TIN", (("HARDWARE", 2),), "threaded insert", (("txt", ("M3", "M4", "M5")), ("len", (4, 5, 6, 8))),
        (0.02, 0.06), "Fasteners", quals=("brass heat-set",), qty=(2, 4, 6), share="common", pack=True),
    Cat("eclip", "ECL", (("HARDWARE", 1),), "E-clip", (("txt", ("3 mm shaft", "4 mm shaft", "5 mm shaft", "6 mm shaft")),),
        (0.01, 0.03), "Springs", share="common", pack=True),
    Cat("ball", "BAL", (("HARDWARE", 1), ("CHUCK", 1)), "steel ball", (("dia", (3, 4, 5, 6)), ("txt", ("G25", "G100"))),
        (0.005, 0.02), "Bearings", qty=(1, 2, 3), share="common", pack=True),
    # --- packaging ---
    Cat("carton", "CTN", (("PACKAGING", 2),), "carton", (("size", BOX_SIZES),), (0.35, 1.40), "Packaging", quals=("retail", "master")),
    Cat("tray", "TRY", (("PACKAGING", 1),), "molded pulp tray",
        (("size", ("260 x 80 x 60 mm", "300 x 90 x 70 mm", "320 x 100 x 80 mm", "330 x 105 x 85 mm")),), (0.15, 0.40), "Packaging"),
    Cat("manual", "MNL", (("PACKAGING", 1),), "user manual", (("txt", ("EN", "EN/ES", "EN/ES/FR")), ("pages", (16, 24, 32, 48))),
        (0.08, 0.25), "Packaging"),
    Cat("label", "LBL", (("PACKAGING", 2), ("HOUSING", 1)), "label",
        (("size", ("20 x 40 mm", "30 x 50 mm", "40 x 60 mm", "50 x 80 mm", "25 x 25 mm")),),
        (0.01, 0.05), "Packaging", quals=("rating", "barcode", "warning"), pack=True),
    Cat("polybag", "BAG", (("PACKAGING", 1),), "poly bag",
        (("size", ("150 x 200 mm", "200 x 300 mm", "250 x 350 mm", "300 x 400 mm")),), (0.01, 0.04), "Packaging", share="common", pack=True),
    Cat("wcard", "WCD", (("PACKAGING", 1),), "warranty card", (("txt", ("EN", "EN/ES", "EN/ES/FR")),), (0.02, 0.05), "Packaging", share="common"),
]
CAT = {c.key: c for c in CATS}

# Which products share one part at a BOM position. Index = product, value = variant.
PATTERNS = {
    "common": [(0, 0, 0, 0)] * 3 + [(0, 0, 1, 1)] * 3 + [(0, 1, 1, 1)] * 2 + [(0, 1, 1, 2)] * 2,
    "mixed": [(0, 0, 0, 0)] * 2 + [(0, 0, 1, 1)] * 3 + [(0, 1, 1, 1)] * 2 + [(0, 1, 1, 2)] * 3 + [(0, 1, 0, 1)] + [(0, 1, 2, 3)] * 2,
    "unique": [(0, 1, 2, 3)] * 3 + [(0, 1, 1, 2)],
    "volt": [(0, 1, 1, 2)] * 2 + [(0, 1, 2, 3)],
}


@dataclass(frozen=True)
class Supplier:
    sid: str
    name: str
    commodity: str
    currency: str = "USD"


_SUPPLIERS = [
    ("Motors", "Dunmore Electric Motor Co."), ("Motors", "Nordlicht Antriebstechnik GmbH", "EUR"),
    ("Motors", "Jiangsu Hengrui Micro-Motor Co., Ltd."),
    ("Bearings", "Tri-State Bearing & Drive Inc."), ("Bearings", "Okabe Precision Bearing Co., Ltd."),
    ("Bearings", "Brennan Industrial Supply LLC"),
    ("Gears", "Kessler Praezision GmbH", "EUR"), ("Gears", "Lakeshore Powder Metal Inc."), ("Gears", "Ningbo Tianyu Gear Co., Ltd."),
    ("Fasteners", "Hanlon Fastener Supply Inc."), ("Fasteners", "Midwest Threaded Products LLC"), ("Fasteners", "Apex Screw & Rivet Corp."),
    ("Springs", "Calloway Spring & Stamping Co."), ("Springs", "Precision Wireform Inc."),
    ("Batteries", "Shenzhen Brightcell Energy Co., Ltd."), ("Batteries", "Voltera Cell Systems Inc."),
    ("Batteries", "Hanbit Energy Solutions Co., Ltd."),
    ("Electronics", "Corvin Electronic Components Inc."), ("Electronics", "Pacific Rim Semiconductor Distribution LLC"),
    ("Electronics", "Eastgate Components Ltd."),
    ("PCBA", "Suntronix EMS Co., Ltd."), ("PCBA", "Redfield Circuits Inc."),
    ("Wire", "Ohio Valley Wire & Harness Inc."), ("Wire", "Cableworks de Mexico S.A. de C.V."),
    ("Plastics", "Greer Injection Molding Inc."), ("Plastics", "Moldeados Regiomontanos S.A. de C.V."),
    ("Plastics", "Dongguan Yusheng Plastics Co., Ltd."),
    ("Chuck", "Weihai Jinda Chuck Co., Ltd."), ("Chuck", "Rothmann Spanntechnik GmbH", "EUR"),
    ("Packaging", "Carolina Corrugated & Packaging LLC"), ("Packaging", "Empaques del Norte S.A. de C.V."),
    ("Packaging", "Palmetto Print & Label Inc."),
    ("Service", "ProTool Service Parts Distribution LLC"), ("Service", "Allied Aftermarket Supply Inc."),
    ("MRO", "Fairmont Industrial Supply Co."), ("MRO", "SafetyFirst Direct LLC"),
    ("MRO", "Monterrey Suministros Industriales S.A. de C.V."),
    # raw-material suppliers are appended last so the suppliers above keep their ids
    ("Materials", "Great Lakes Steel Service Inc."), ("Materials", "Essex Ridge Magnet Wire Co."),
    ("Materials", "Lubritech Industrial Greases LLC"), ("Materials", "Polymer Supply Partners Inc."),
]
SUPPLIERS = [Supplier(f"S{i + 1:02d}", s[1], s[0], s[2] if len(s) > 2 else "USD") for i, s in enumerate(_SUPPLIERS)]

FX_USD_PER_UNIT = {"USD": 1.0, "EUR": 1.08}

# Service-only items: sold as spares, never on a production BOM. (noun, qual, attrs, price low, price high)
SERVICE_ONLY = [
    ("charger", "fast", (("V", 12), ("A", 2)), 9, 13), ("charger", "fast", (("V", 18), ("A", 4)), 12, 17),
    ("charger", "fast", (("V", 20), ("A", 4)), 13, 18),
    ("battery pack", "complete", (("V", 12), ("Ah", 2.0)), 14, 19), ("battery pack", "complete", (("V", 18), ("Ah", 2.0)), 22, 28),
    ("battery pack", "complete", (("V", 18), ("Ah", 4.0)), 34, 42), ("battery pack", "complete", (("V", 20), ("Ah", 4.0)), 36, 45),
    ("motor assembly", "service exchange", (("W", 250), ("V", 18)), 14, 19), ("motor assembly", "service exchange", (("W", 350), ("V", 20)), 17, 23),
    ("gearbox assembly", "service", (("V", 12),), 8, 11), ("gearbox assembly", "service", (("V", 18),), 10, 14),
    ("carbon brush service kit", "", (("txt", "2 brushes + springs"),), 1.2, 2.0),
    ("switch service kit", "", (("V", 18),), 4, 6), ("chuck service kit", "", (("txt", "13 mm capacity"),), 7, 10),
    ("carrying case", "blow-molded", (("txt", "black"),), 4, 6), ("carrying case", "soft", (("txt", "red"),), 3, 5),
    ("belt clip kit", "", (("txt", "with screw"),), 0.6, 1.0), ("bit set", "", (("txt", "32-piece"),), 3, 5),
    ("label kit", "service", (("txt", "EN/ES"),), 0.3, 0.6), ("grease tube", "gearbox", (("txt", "50 g"),), 1.5, 2.5),
    ("housing kit", "service", (("V", 18), ("txt", "red")), 5, 8), ("screw kit", "housing", (("txt", "12-piece"),), 0.5, 0.9),
]

# Indirect / MRO items: real spend that belongs to no product. (description, price low, price high)
INDIRECT = [
    ("Nitrile gloves, large, box of 100", 9, 14), ("Safety glasses, clear lens", 2, 5), ("Cutting fluid, 5 gal pail", 60, 110),
    ("Stretch wrap, 18 in x 1500 ft", 14, 24), ("Wood pallet 48 x 40 in", 11, 18), ("Shop towels, case", 30, 55),
    ("Isopropyl alcohol 99%, 1 gal", 18, 30), ("Solder wire SAC305 0.8 mm, 1 lb spool", 38, 60), ("Flux pen, no-clean", 4, 8),
    ("Torque driver bit T10, 10-pack", 12, 22), ("ESD wrist strap", 5, 11), ("ESD floor mat 3 x 5 ft", 70, 130),
    ("Label printer ribbon 110 mm x 300 m", 9, 16), ("Packing tape 48 mm x 100 m, case of 36", 45, 75),
    ("Hydraulic oil ISO 46, 55 gal drum", 420, 640), ("Compressed air filter element", 35, 80), ("Mold release spray, 12 oz", 7, 13),
    ("Conveyor belt splice kit", 150, 320), ("Earplugs, box of 200", 18, 32), ("Fixture clamp, toggle 200 lb", 14, 28),
    ("Calibration service, torque analyzer", 180, 420), ("Forklift propane cylinder exchange", 22, 34), ("Copy paper, case", 38, 55),
    ("Floor marking tape yellow 2 in", 16, 28), ("Loctite 243 threadlocker 50 ml", 19, 30), ("Thermal paste 30 g syringe", 8, 15),
    ("Injection mold maintenance, cavity polish", 600, 1800), ("Assembly fixture, gearbox press nest", 900, 2600),
    ("Test jig pogo pin, 100-pack", 40, 85), ("Screwdriver bit PH2, 25-pack", 15, 27), ("Hand soap refill, case", 28, 46),
    ("Air tool oil, 1 qt", 9, 16), ("Zip ties 8 in black, 1000-pack", 14, 24), ("Bin, stackable 11 x 16 in", 6, 12),
    ("Dunnage foam sheet 1/2 in", 3, 7), ("Desiccant bulk 5 kg", 20, 38), ("First aid kit refill", 25, 60),
    ("Welding tip, ultrasonic horn", 250, 700), ("Barcode scanner, handheld", 140, 260), ("Freight, expedite charge", 150, 1200),
]


# In-house subassemblies: every part in these assemblies becomes a child of a "make" item, one per product.
SUBASSEMBLIES = {"GEARBOX": ("GBX", "Gearbox subassembly", 0.01), "MOTOR": ("MOT", "Motor subassembly", 0.0)}  # prefix, name, scrap

# Continuous materials, measured and bought in units other than "each".
#   qty: BOM quantity per parent, by product index.  units: purchase units in the BOM's dimension (text, BOM units per purchase unit).
#   cross: a purchase unit in another dimension (text, that unit per BOM unit, share of lines), or None.
#   over: purchased quantity relative to the gross requirement. Above ~1.15 is a planted over-purchase.
MATERIALS = [
    dict(prefix="STL", noun="steel sheet", qual="cold rolled", attrs=(("thk", 0.8),), uom="M2", price=9.40, assembly="MOTOR",
         qty=(0.011, 0.014, 0.014, 0.016), scrap=0.08, units=(("M2", 1.0), ("SQM", 1.0), ("FT2", 0.092903)), cross=("KG", 6.28, 0.25), over=1.0),
    dict(prefix="MWR", noun="magnet wire", qual="enamelled copper", attrs=(("dia", 0.5),), uom="M", price=0.036, assembly="MOTOR",
         qty=(11, 14, 14, 16), scrap=0.03, units=(("M", 1.0), ("MTR", 1.0), ("FT", 0.3048)), cross=("KG", 0.00175, 0.25), over=1.32),
    dict(prefix="GRS", noun="gearbox grease", qual="lithium EP2", attrs=(("txt", "NLGI 2"),), uom="G", price=0.011, assembly="GEARBOX",
         qty=(6, 8, 9, 8), scrap=0.05, units=(("KG", 1000.0), ("G", 1.0), ("LB", 453.592)), cross=None, over=1.27),
    dict(prefix="RSN", noun="potting resin", qual="epoxy two-part", attrs=(("txt", "UL94 V-0"),), uom="G", price=0.019, assembly="ELECTRONICS",
         qty=(10, 12, 12, 14), scrap=0.04, units=(("KG", 1000.0), ("G", 1.0)), cross=None, over=1.0),
]
