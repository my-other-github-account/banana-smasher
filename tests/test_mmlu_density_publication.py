#!/usr/bin/env python3
import json
import unittest
from decimal import Decimal, getcontext
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "notes/benchmarks/mmlu-density/mmlu500-v1/results.json"
SCHEMA = REPO / "notes/benchmarks/mmlu-density/mmlu500-v1/results.schema.json"
REPORT = REPO / "notes/benchmarks/mmlu-density/mmlu500-v1/four-row-results.md"
EVALS = REPO / "Evals/README.md"


class MMLUDensityPublicationTest(unittest.TestCase):
    def test_nine_row_result_and_evals_table_are_consistent(self):
        getcontext().prec = 120
        result = json.loads(RESULTS.read_text())
        schema = json.loads(SCHEMA.read_text())
        rows = result["rows"]

        self.assertEqual(result["schema"], "banana-smasher.mmlu500-nine-row-density-terminal.v1")
        self.assertEqual(schema["properties"]["schema"]["const"], result["schema"])
        self.assertEqual(schema["properties"]["rows"]["minItems"], 9)
        self.assertEqual(schema["properties"]["rows"]["maxItems"], 9)
        self.assertIn("mmlu_per_gb", schema["properties"]["rows"]["items"]["required"])
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            [row["variant"] for row in rows],
            [
                "UD-IQ4_XS",
                "UD-IQ3_XXS",
                "UD-IQ2_XXS",
                "DwarfStar-Q2-0731",
                "Official-native-MXFP4",
                "EXL3-K2-routed-native-rest",
                "QTIP2-corrected-all43",
                "EXL3-K3-routed-native-rest",
                "EXL3-K3-uniform-exact",
            ],
        )

        original = [
            ("UD-IQ4_XS", 417, 83.4, "3.8451166272834685", "376279254d98d0efdfaeba1303099c65a9a3ba4599117616cb107444c083eb16", 0.42733026832895527),
            ("UD-IQ3_XXS", 416, 83.2, "2.931978308348837", "537707dca2df62e5bbeae531f822b7f0013707f63d5f91cadb5b3f1e6fada8b6", 0.5584992023069897),
            ("UD-IQ2_XXS", 409, 81.8, "2.556445745541928", "b8cae63e3bb2892473818a77ea94bb66922add35febf65214b7b98419cc3e42b", 0.6251325040981072),
            ("DwarfStar-Q2-0731", 403, 80.6, "2.6360875868777476", "61f15219884e03f79072824c79665cea7e06f896e3640b0ecf6da164392aa228", 0.5934379024790847),
        ]
        for row, expected in zip(rows[:4], original):
            self.assertEqual(
                (
                    row["variant"], row["correct"], row["mmlu_percent"],
                    row["base_equivalent_bpw"], row["qrows_sha256"],
                    row["mmlu_capability_density"],
                ),
                expected,
            )

        expected_new = {
            "Official-native-MXFP4": (423, Decimal("84.6"), 156035165824, "4.3901849061799633842692039291812057173773172588621", "13.575737986822021169770638279576235636557835123104"),
            "EXL3-K2-routed-native-rest": (418, Decimal("83.6"), 89371076344, "2.5145328512486971484262613667868966546438084310621785627887683259240843040121566", "23.30452750732594444179988514428135015815648841012239784287586670715144855970965066658501975422521871"),
            "QTIP2-corrected-all43": (412, Decimal("82.4"), 89330008924, "2.5133773837201586429658372611602203102766869054250902646239944325064734214647636621385741278765751357100035806", "22.837796015749841659559084631083944388300462093436429846337177334456839441477273304117463719419610073876633838"),
            "EXL3-K3-routed-native-rest": (426, Decimal("85.2"), 123999250168, "3.488881932423359811648345096334173619526617322555388135469206037221313139582164", "17.25481147428379249155385572471416346148579182105134544577220219255627215867777214170883080045737066"),
            "EXL3-K3-uniform-exact": (424, Decimal("84.8"), 113260003977, "3.186668577611291126768382805251239067660095075340211506894101693858116218554962444360182878967180057", "18.76567912337647118428195391233153179107803343530514769372618419610599904720503080757189191494425088"),
        }
        for row in rows[4:]:
            correct, percent, complete_bytes, bpw, density = expected_new[row["variant"]]
            self.assertEqual(row["n"], 500)
            self.assertEqual(row["correct"], correct)
            self.assertEqual(Decimal(str(row["mmlu_percent"])), percent)
            self.assertEqual(row["complete_artifact_bytes"], complete_bytes)
            self.assertEqual(row["base_equivalent_bpw"], bpw)
            self.assertEqual(row["mmlu_intelligence_density"], density)
            recomputed = (percent - Decimal(25)) / Decimal(bpw)
            self.assertLess(abs(recomputed - Decimal(density)), Decimal("1e-47"))
            self.assertEqual(row["provenance"]["independent_recomputation"], "PASS")

        for row in rows:
            expected_per_gb = (Decimal(str(row["mmlu_percent"])) - Decimal(25)) / (
                Decimal(row["complete_artifact_bytes"]) / Decimal(1_000_000_000)
            )
            self.assertLess(abs(expected_per_gb - Decimal(row["mmlu_per_gb"])), Decimal("1e-49"))

        evals = EVALS.read_text()
        self.assertIn("MMLU/GB ↑", evals)
        self.assertIn("MMLU/GB", REPORT.read_text())
        for fragment in (
            "Official native MXFP4** |  |  | **84.60%** (423/500) | **13.576**",
            "EXL3 K2 routed-only + native rest** | **86.33%** (56,579/65,536) | **0.234288** | **83.60%** (418/500) | **23.305**",
            "QTIP2 corrected all-43** | **87.11%** (57,090/65,536) | **0.240852** | **82.40%** (412/500) | **22.838**",
            "EXL3 K3 routed-only + native rest** | **92.23%** (60,447/65,536) | **0.076868** | **85.20%** (426/500) | **17.255**",
            "EXL3 K3 uniform exact** | **88.30%** (57,870/65,536) | **0.136015** | **84.80%** (424/500) | **18.766**",
        ):
            self.assertIn(fragment, evals)

        public_text = "\n".join(path.read_text() for path in (RESULTS, SCHEMA, REPORT))
        forbidden_fragments = (
            "/" + "home/",
            "/" + "Users/",
            "task" + "_id",
            "d" + "nola",
            "mac" + "mini",
            "spark" + "-",
        )
        for forbidden in forbidden_fragments:
            self.assertNotIn(forbidden.lower(), public_text.lower())


if __name__ == "__main__":
    unittest.main()
