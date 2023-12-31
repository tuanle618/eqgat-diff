# TO BE DONE: Remove in Future

qm9_with_h = {
    "name": "qm9",
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
    "atom_decoder": ["H", "C", "N", "O", "F"],
    "n_nodes": {
        22: 3393,
        17: 13025,
        23: 4848,
        21: 9970,
        19: 13832,
        20: 9482,
        16: 10644,
        13: 3060,
        15: 7796,
        25: 1506,
        18: 13364,
        12: 1689,
        11: 807,
        24: 539,
        14: 5136,
        26: 48,
        7: 16,
        10: 362,
        8: 49,
        9: 124,
        27: 266,
        4: 4,
        29: 25,
        6: 9,
        5: 5,
        3: 1,
    },
    "max_n_nodes": 29,
    "atom_types": {1: 635559, 2: 101476, 0: 923537, 3: 140202, 4: 2323},
    "distances": [
        903054,
        307308,
        111994,
        57474,
        40384,
        29170,
        47152,
        414344,
        2202212,
        573726,
        1490786,
        2970978,
        756818,
        969276,
        489242,
        1265402,
        4587994,
        3187130,
        2454868,
        2647422,
        2098884,
        2001974,
        1625206,
        1754172,
        1620830,
        1710042,
        2133746,
        1852492,
        1415318,
        1421064,
        1223156,
        1322256,
        1380656,
        1239244,
        1084358,
        981076,
        896904,
        762008,
        659298,
        604676,
        523580,
        437464,
        413974,
        352372,
        291886,
        271948,
        231328,
        188484,
        160026,
        136322,
        117850,
        103546,
        87192,
        76562,
        61840,
        49666,
        43100,
        33876,
        26686,
        22402,
        18358,
        15518,
        13600,
        12128,
        9480,
        7458,
        5088,
        4726,
        3696,
        3362,
        3396,
        2484,
        1988,
        1490,
        984,
        734,
        600,
        456,
        482,
        378,
        362,
        168,
        124,
        94,
        88,
        52,
        44,
        40,
        18,
        16,
        8,
        6,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "colors_dic": ["#FFFFFF99", "C7", "C0", "C3", "C1"],
    "radius_dic": [0.46, 0.77, 0.77, 0.77, 0.77],
    "with_h": True,
}
# 'bond1_radius': {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57},
# 'bond1_stdv': {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3},
# 'bond2_radius': {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59},
# 'bond3_radius': {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53}}

qm9_without_h = {
    "name": "qm9",
    "atom_encoder": {"C": 0, "N": 1, "O": 2, "F": 3},
    "atom_decoder": ["C", "N", "O", "F"],
    "max_n_nodes": 29,
    "n_nodes": {9: 83366, 8: 13625, 7: 2404, 6: 475, 5: 91, 4: 25, 3: 7, 1: 2, 2: 5},
    "atom_types": {0: 635559, 2: 140202, 1: 101476, 3: 2323},
    "distances": [
        594,
        1232,
        3706,
        4736,
        5478,
        9156,
        8762,
        13260,
        45674,
        174676,
        469292,
        1182942,
        126722,
        25768,
        28532,
        51696,
        232014,
        299916,
        686590,
        677506,
        379264,
        162794,
        158732,
        156404,
        161742,
        156486,
        236176,
        310918,
        245558,
        164688,
        98830,
        81786,
        89318,
        91104,
        92788,
        83772,
        81572,
        85032,
        56296,
        32930,
        22640,
        24124,
        24010,
        22120,
        19730,
        21968,
        18176,
        12576,
        8224,
        6772,
        3906,
        4416,
        4306,
        4110,
        3700,
        3592,
        3134,
        2268,
        774,
        674,
        514,
        594,
        622,
        672,
        642,
        472,
        300,
        170,
        104,
        48,
        54,
        78,
        78,
        56,
        48,
        36,
        26,
        4,
        2,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "colors_dic": ["C7", "C0", "C3", "C1"],
    "radius_dic": [0.77, 0.77, 0.77, 0.77],
    "with_h": False,
}
# 'bond1_radius': {'C': 76, 'N': 71, 'O': 66, 'F': 57},
# 'bond1_stdv': {'C': 2, 'N': 2, 'O': 2, 'F': 3},
# 'bond2_radius': {'C': 67, 'N': 60, 'O': 57, 'F': 59},
# 'bond3_radius': {'C': 60, 'N': 54, 'O': 53, 'F': 53}}


qm9_second_half = {
    "name": "qm9_second_half",
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
    "atom_decoder": ["H", "C", "N", "O", "F"],
    "n_nodes": {
        19: 6944,
        12: 845,
        20: 4794,
        21: 4962,
        27: 132,
        25: 754,
        18: 6695,
        14: 2587,
        15: 3865,
        22: 1701,
        17: 6461,
        16: 5344,
        23: 2380,
        13: 1541,
        24: 267,
        10: 178,
        7: 7,
        11: 412,
        8: 25,
        9: 62,
        29: 15,
        26: 17,
        4: 3,
        3: 1,
        6: 5,
        5: 3,
    },
    "atom_types": {1: 317604, 2: 50852, 3: 70033, 0: 461622, 4: 1164},
    "distances": [
        457374,
        153688,
        55626,
        28284,
        20414,
        15010,
        24412,
        208012,
        1105440,
        285830,
        748876,
        1496486,
        384178,
        484194,
        245688,
        635534,
        2307642,
        1603762,
        1231044,
        1329758,
        1053612,
        1006742,
        813504,
        880670,
        811616,
        855082,
        1066434,
        931672,
        709810,
        711032,
        608446,
        660538,
        692382,
        619084,
        544200,
        490740,
        450576,
        380662,
        328150,
        303008,
        263888,
        218820,
        207414,
        175452,
        145636,
        135646,
        116184,
        94622,
        80358,
        68230,
        58706,
        51216,
        44020,
        38212,
        30492,
        24886,
        21210,
        17270,
        13056,
        11156,
        9082,
        7534,
        6958,
        6060,
        4632,
        3760,
        2500,
        2342,
        1816,
        1726,
        1768,
        1102,
        974,
        670,
        474,
        446,
        286,
        246,
        242,
        156,
        176,
        90,
        66,
        66,
        38,
        28,
        24,
        14,
        10,
        2,
        6,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "colors_dic": ["#FFFFFF99", "C7", "C0", "C3", "C1"],
    "radius_dic": [0.46, 0.77, 0.77, 0.77, 0.77],
    "max_n_nodes": 29,
    "with_h": True,
}
# 'bond1_radius': {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57},
# 'bond1_stdv': {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3},
# 'bond2_radius': {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59},
# 'bond3_radius': {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53}}

geom_no_h = {
    "name": "drugs",
    "atom_encoder": {
        "B": 0,
        "C": 1,
        "N": 2,
        "O": 3,
        "F": 4,
        "Al": 5,
        "Si": 6,
        "P": 7,
        "S": 8,
        "Cl": 9,
        "As": 10,
        "Br": 11,
        "I": 12,
        "Hg": 13,
        "Bi": 14,
    },
    "atomic_nb": [5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    "atom_decoder": [
        "B",
        "C",
        "N",
        "O",
        "F",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "As",
        "Br",
        "I",
        "Hg",
        "Bi",
    ],
    "max_n_nodes": 91,
    "n_nodes": {
        1: 3,
        2: 5,
        3: 8,
        4: 89,
        5: 166,
        6: 370,
        7: 613,
        8: 1214,
        9: 1680,
        10: 3315,
        11: 5115,
        12: 9873,
        13: 15422,
        14: 28088,
        15: 50643,
        16: 82299,
        17: 124341,
        18: 178417,
        19: 240446,
        20: 308209,
        21: 372900,
        22: 429257,
        23: 477423,
        24: 508377,
        25: 522385,
        26: 522000,
        27: 507882,
        28: 476702,
        29: 426308,
        30: 375819,
        31: 310124,
        32: 255179,
        33: 204441,
        34: 149383,
        35: 109343,
        36: 71701,
        37: 44050,
        38: 31437,
        39: 20242,
        40: 14971,
        41: 10078,
        42: 8049,
        43: 4476,
        44: 3130,
        45: 1736,
        46: 2030,
        47: 1110,
        48: 840,
        49: 750,
        50: 540,
        51: 810,
        52: 591,
        53: 453,
        54: 540,
        55: 720,
        56: 300,
        57: 360,
        58: 714,
        59: 390,
        60: 519,
        61: 210,
        62: 449,
        63: 210,
        64: 289,
        65: 589,
        66: 227,
        67: 180,
        68: 330,
        69: 330,
        70: 150,
        71: 60,
        72: 210,
        73: 60,
        74: 180,
        75: 120,
        76: 30,
        77: 150,
        78: 30,
        79: 60,
        82: 60,
        85: 60,
        86: 6,
        87: 60,
        90: 60,
        91: 30,
    },
    "atom_types": {
        0: 290,
        1: 129988623,
        2: 20266722,
        3: 21669359,
        4: 1481844,
        5: 1,
        6: 250,
        7: 36290,
        8: 3999872,
        9: 1224394,
        10: 4,
        11: 298702,
        12: 5377,
        13: 13,
        14: 34,
    },
    "colors_dic": [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
    ],
    "radius_dic": [
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
    ],
    "with_h": False,
}


geom_no_h_new = {
    "name": "drugs",
    "atom_encoder": {
        "B": 0,
        "C": 1,
        "N": 2,
        "O": 3,
        "F": 4,
        "Al": 5,
        "Si": 6,
        "P": 7,
        "S": 8,
        "Cl": 9,
        "As": 10,
        "Br": 11,
        "I": 12,
        "Hg": 13,
        "Bi": 14,
    },
    "atomic_nb": [5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    "atom_decoder": [
        "B",
        "C",
        "N",
        "O",
        "F",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "As",
        "Br",
        "I",
        "Hg",
        "Bi",
    ],
    "max_n_nodes": 91,
    "n_nodes": {
        1: 1,
        2: 1,
        3: 4,
        4: 6,
        5: 49,
        6: 97,
        7: 178,
        8: 265,
        9: 470,
        10: 1073,
        11: 1783,
        12: 3063,
        13: 5030,
        14: 8604,
        15: 14393,
        16: 20969,
        17: 30178,
        18: 39852,
        19: 50599,
        20: 61520,
        21: 71709,
        22: 77868,
        23: 82623,
        24: 84707,
        25: 83715,
        26: 81682,
        27: 77616,
        28: 72437,
        29: 63690,
        30: 55341,
        31: 45290,
        32: 37121,
        33: 29574,
        34: 21463,
        35: 15728,
        36: 10327,
        37: 6352,
        38: 4635,
        39: 2964,
        40: 2197,
        41: 1585,
        42: 1340,
        43: 690,
        44: 530,
        45: 285,
        46: 320,
        47: 200,
        48: 160,
        49: 105,
        50: 110,
        51: 140,
        52: 90,
        53: 105,
        54: 100,
        55: 150,
        56: 55,
        57: 55,
        58: 140,
        59: 115,
        60: 115,
        61: 80,
        62: 100,
        63: 65,
        64: 61,
        65: 150,
        66: 67,
        67: 45,
        68: 50,
        69: 65,
        70: 15,
        71: 45,
        72: 75,
        73: 15,
        74: 20,
        75: 40,
        76: 20,
        77: 20,
        78: 10,
        79: 10,
        81: 15,
        83: 10,
        84: 5,
        85: 5,
        89: 10,
        90: 15,
    },
    "with_h": False,
}

geom_with_h = {
    "name": "geom",
    "atom_encoder": {
        "H": 0,
        "B": 1,
        "C": 2,
        "N": 3,
        "O": 4,
        "F": 5,
        "Al": 6,
        "Si": 7,
        "P": 8,
        "S": 9,
        "Cl": 10,
        "As": 11,
        "Br": 12,
        "I": 13,
        "Hg": 14,
        "Bi": 15,
    },
    "atomic_nb": [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    "atom_decoder": [
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "As",
        "Br",
        "I",
        "Hg",
        "Bi",
    ],
    "max_n_nodes": 181,
    "n_nodes": {
        3: 2,
        5: 3,
        6: 2,
        7: 3,
        8: 7,
        9: 16,
        10: 34,
        11: 27,
        12: 65,
        13: 117,
        14: 131,
        15: 217,
        16: 260,
        17: 331,
        18: 674,
        19: 824,
        20: 1066,
        21: 1741,
        22: 2660,
        23: 3614,
        24: 5345,
        25: 6716,
        26: 8783,
        27: 10541,
        28: 13440,
        29: 16061,
        30: 19191,
        31: 21684,
        32: 24585,
        33: 27933,
        34: 30340,
        35: 34097,
        36: 36571,
        37: 37148,
        38: 39973,
        39: 40871,
        40: 40984,
        41: 42690,
        42: 43690,
        43: 43648,
        44: 43688,
        45: 42552,
        46: 42597,
        47: 41442,
        48: 38967,
        49: 37054,
        50: 35798,
        51: 33620,
        52: 31607,
        53: 29977,
        54: 27944,
        55: 25882,
        56: 23344,
        57: 22102,
        58: 19794,
        59: 16962,
        60: 15474,
        61: 13457,
        62: 11918,
        63: 9582,
        64: 8880,
        65: 7197,
        66: 5576,
        67: 4840,
        68: 3940,
        69: 2945,
        70: 2850,
        71: 2275,
        72: 1785,
        73: 1460,
        74: 1350,
        75: 1005,
        76: 985,
        77: 665,
        78: 625,
        79: 595,
        80: 510,
        81: 420,
        82: 360,
        83: 280,
        84: 245,
        85: 215,
        86: 240,
        87: 200,
        88: 205,
        89: 110,
        90: 80,
        91: 125,
        92: 120,
        93: 80,
        94: 126,
        95: 100,
        96: 70,
        97: 95,
        98: 25,
        99: 100,
        100: 60,
        101: 30,
        102: 25,
        103: 30,
        104: 30,
        105: 30,
        106: 50,
        107: 95,
        108: 45,
        109: 20,
        110: 40,
        111: 65,
        112: 30,
        113: 35,
        114: 10,
        115: 10,
        116: 30,
        117: 65,
        118: 120,
        119: 60,
        120: 40,
        121: 30,
        122: 25,
        123: 80,
        124: 55,
        125: 30,
        126: 42,
        127: 35,
        128: 10,
        129: 5,
        130: 25,
        131: 5,
        132: 30,
        133: 30,
        134: 55,
        135: 40,
        136: 45,
        138: 50,
        139: 20,
        140: 105,
        141: 20,
        142: 35,
        143: 20,
        144: 20,
        145: 70,
        146: 20,
        147: 20,
        148: 55,
        150: 50,
        151: 10,
        152: 10,
        153: 10,
        155: 15,
        156: 10,
        158: 10,
        159: 5,
        160: 5,
        162: 5,
        169: 5,
        176: 15,
        181: 5,
    },
    "with_h": True,
}


def get_dataset_info(dataset_name, remove_h):
    if dataset_name == "qm9":
        if not remove_h:
            return qm9_with_h
        else:
            return qm9_without_h
    elif dataset_name == "drugs":
        if not remove_h:
            return geom_with_h
        else:
            return geom_no_h
    elif dataset_name == "qm9_second_half":
        if not remove_h:
            return qm9_second_half
        else:
            raise Exception("Missing config for %s without hydrogens" % dataset_name)
    else:
        raise ValueError
