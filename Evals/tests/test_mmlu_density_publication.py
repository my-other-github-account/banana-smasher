#!/usr/bin/env python3
import json
import hashlib
import math
import unittest
from decimal import Decimal, getcontext
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/results.json"
SCHEMA = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/results.schema.json"
REPORT = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/four-row-results.md"
EVALS = REPO / "Evals/README.md"
FINISHED_EVIDENCE_MANIFEST = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/finished-evidence-manifest.json"
EVIDENCE_MANIFEST = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/evidence-manifest.json"
QTIP2_IDENTITY = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/evidence/QTIP2-corrected-all43/model-identity.json"
U45_EVIDENCE = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/evidence/U45-repaired-ffb796ce"
U45_QROWS = U45_EVIDENCE / "qrows.jsonl"
U45_TERMINAL = U45_EVIDENCE / "terminal.json"
U45_REAGGREGATION = U45_EVIDENCE / "independent-reaggregation.json"
U45_SANITIZATION = U45_EVIDENCE / "sanitization-receipt.json"
U45_PUBLICATION = U45_EVIDENCE / "publication-receipt.json"
PRE_EVIDENCE = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/evidence/PRE-f9bffe04"
PRE_QROWS = PRE_EVIDENCE / "qrows.jsonl"
PRE_TERMINAL = PRE_EVIDENCE / "terminal.json"
PRE_REAGGREGATION = PRE_EVIDENCE / "independent-reaggregation.json"
PRE_ACCOUNTING = PRE_EVIDENCE / "artifact-accounting-receipt.json"
PRE_SANITIZATION = PRE_EVIDENCE / "sanitization-receipt.json"
PRE_PUBLICATION = PRE_EVIDENCE / "publication-receipt.json"


class MMLUDensityPublicationTest(unittest.TestCase):
    def test_fifteen_row_result_and_evals_table_are_consistent(self):
        getcontext().prec = 120
        result = json.loads(RESULTS.read_text())
        schema = json.loads(SCHEMA.read_text())
        rows = result["rows"]

        self.assertEqual(result["schema"], "banana-smasher.mmlu500-fifteen-row-density-terminal.v7")
        self.assertEqual(schema["properties"]["schema"]["const"], result["schema"])
        self.assertEqual(schema["properties"]["rows"]["minItems"], 15)
        self.assertEqual(schema["properties"]["rows"]["maxItems"], 15)
        self.assertIn("QTIP2-V7-U45-repaired", schema["properties"]["rows"]["items"]["properties"]["variant"]["enum"])
        self.assertIn("mmlu_per_gb", schema["properties"]["rows"]["items"]["required"])
        self.assertIn("raw_mmlu_per_bpw", schema["properties"]["rows"]["items"]["required"])
        self.assertIn("QTIP2-V7-PRE-f9bffe04", schema["properties"]["rows"]["items"]["properties"]["variant"]["enum"])
        self.assertEqual(len(rows), 15)
        self.assertEqual(
            [row["variant"] for row in rows],
            [
                "UD-IQ4_XS",
                "UD-IQ3_XXS",
                "UD-IQ2_XXS",
                "DwarfStar-Q2-0731",
                "Official-native-MXFP4",
                "EXL3-K2-routed-native-rest",
                "EXL3-K3-routed-native-rest",
                "EXL3-K3-uniform-exact",
                "QTIP3-uniform-exact",
                "QTIP2P5-deterministic-mixed-ring",
                "EXL3-K2P5-greedy-routed-native-rest",
                "EXL3-K2-uniform-exact",
                "Physical-alternating-K2K3-full",
                "QTIP2-V7-U45-repaired",
                "QTIP2-V7-PRE-f9bffe04",
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
            "EXL3-K3-routed-native-rest": (426, Decimal("85.2"), 123999250168, "3.488881932423359811648345096334173619526617322555388135469206037221313139582164", "17.25481147428379249155385572471416346148579182105134544577220219255627215867777214170883080045737066"),
            "EXL3-K3-uniform-exact": (424, Decimal("84.8"), 113260003977, "3.186668577611291126768382805251239067660095075340211506894101693858116218554962444360182878967180057", "18.76567912337647118428195391233153179107803343530514769372618419610599904720503080757189191494425088"),
            "QTIP3-uniform-exact": (421, Decimal("84.2"), 123968528042, "3.487962202476954954739203475489728352391106959317205859774720270487259967167516979310499463726247446", "16.9726609875415172996428276813531353488618362504699811994239440321836712512089020485040050968648412588229151660448353183"),
            "QTIP2P5-deterministic-mixed-ring": (414, Decimal("82.8"), 106657444992, "3.000899846280526906707937310598763864978934006978352098969598998585827630738411347265673123546884941", "19.2608893867752233807419799625420976444760774192163389939983159352072096559378267241213458075333678430680736149955799898"),
            "EXL3-K2P5-greedy-routed-native-rest": (424, Decimal("84.8"), 106282510072, "2.990350726677318761133570203797083967844437614338647678644542139727383081773899285391977912993952953", "19.9976542773114211551230553063823955412839565138944792609771588308334509994164752793475970777033117705023648987223690617"),
            "EXL3-K2-uniform-exact": (369, Decimal("73.8"), 77861675750, "2.1907058696825606173737276358734996088908236748322939650200806709327704680850651", "22.2759251597163319465289057819950657817687670304218953314782722230326515930399810618438699876324023710314001979897142511"),
            "Physical-alternating-K2K3-full": (374, Decimal("74.8"), 94832907712, "2.668206220359224284525477307187490489073009403333264769375373564232111492365228415584687878052563670", "18.6642245340749401654284688564374618845811310854177935005465036267515179910378492391997573341263574048420606555109602646"),
            "QTIP2-V7-U45-repaired": (413, Decimal("82.6"), 100636011256, "2.831481578534603263235376275888125565194357238266015863087325200113206153869260839371695918636102327", "20.34270695478447590271810524071910062468503685107938714029192683407599224572351128955996785159388153"),
            "QTIP2-V7-PRE-f9bffe04": (412, Decimal("82.4"), 89405007628, "2.515487537393179030504882353653487081235072278387024486348262001770745366655152827898047671931839507", "22.8186381950769291197716534241018699284261080025238874419527609505546610275604908200724347014984407662119219177670895236"),
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
            expected_raw_per_bpw = Decimal(str(row["mmlu_percent"])) / Decimal(row["base_equivalent_bpw"])
            self.assertLess(abs(expected_raw_per_bpw - Decimal(row["raw_mmlu_per_bpw"])), Decimal("1e-49"))

        u45 = rows[-2]
        qrows_bytes = U45_QROWS.read_bytes()
        qrows = [json.loads(line) for line in qrows_bytes.splitlines() if line.strip()]
        terminal_bytes = U45_TERMINAL.read_bytes()
        terminal = json.loads(terminal_bytes)
        reaggregation_bytes = U45_REAGGREGATION.read_bytes()
        reaggregation = json.loads(reaggregation_bytes)
        sanitization_bytes = U45_SANITIZATION.read_bytes()
        sanitization = json.loads(sanitization_bytes)
        publication_bytes = U45_PUBLICATION.read_bytes()
        publication = json.loads(publication_bytes)
        expected_qrows_sha = "7e684b223952a50005cb0ff6f2c842999a73dfba92533075f847432686d31f03"

        self.assertEqual(hashlib.sha256(qrows_bytes).hexdigest(), expected_qrows_sha)
        self.assertEqual(len(qrows), 500)
        self.assertEqual([row["sample_ordinal"] for row in qrows], list(range(500)))
        self.assertEqual(sum(bool(row["correct"]) for row in qrows), 413)
        self.assertTrue(all(len(row["choice_logits"]) == 4 for row in qrows))
        self.assertTrue(all(all(math.isfinite(value) for value in row["choice_logits"]) for row in qrows))

        items = [json.loads(line) for line in (RESULTS.parent / "items.jsonl").read_text().splitlines() if line.strip()]
        self.assertEqual([row["row_sha256"] for row in qrows], [item["row_sha256"] for item in items])
        self.assertEqual(hashlib.sha256((RESULTS.parent / "items.jsonl").read_bytes()).hexdigest(), terminal["items_sha256"])
        self.assertEqual(hashlib.sha256((RESULTS.parent / "manifest.json").read_bytes()).hexdigest(), terminal["manifest_sha256"])

        recomputed_ce = math.fsum(-row["choice_logprobs"][row["gold_index"]] for row in qrows) / (500 * math.log(2))
        self.assertAlmostEqual(recomputed_ce, reaggregation["gold_cross_entropy_bits"], places=15)
        self.assertEqual((reaggregation["n"], reaggregation["correct"], reaggregation["mmlu_percent"]), (500, 413, 82.6))
        self.assertEqual(reaggregation["qrows_sha256"], expected_qrows_sha)

        provenance = u45["provenance"]
        self.assertEqual(hashlib.sha256(terminal_bytes).hexdigest(), provenance["terminal_sha256"])
        self.assertEqual(hashlib.sha256(reaggregation_bytes).hexdigest(), provenance["independent_reaggregation_sha256"])
        self.assertEqual(hashlib.sha256(sanitization_bytes).hexdigest(), provenance["sanitization_receipt_sha256"])
        self.assertEqual(hashlib.sha256(publication_bytes).hexdigest(), provenance["publication_receipt_sha256"])
        self.assertEqual(provenance["manifest_sha256"], terminal["manifest_sha256"])
        self.assertEqual((terminal["n"], terminal["correct"], terminal["mmlu_percent"]), (500, 413, 82.6))
        self.assertEqual(terminal["qrows_sha256"], expected_qrows_sha)
        self.assertEqual(sanitization["status"], "PASS")
        self.assertEqual(sanitization["qrows"]["bytes"], len(qrows_bytes))
        self.assertEqual(sanitization["qrows"]["source_sha256"], expected_qrows_sha)
        self.assertEqual(sanitization["qrows"]["output_sha256"], expected_qrows_sha)
        self.assertEqual(sanitization["terminal"]["output_sha256"], provenance["terminal_sha256"])
        self.assertEqual(publication["status"], "PASS")
        self.assertEqual(publication["qrows"], {"bytes": len(qrows_bytes), "rows": 500, "sha256": expected_qrows_sha})
        self.assertEqual(publication["bank"], {"items_sha256": terminal["items_sha256"], "manifest_sha256": terminal["manifest_sha256"]})
        self.assertEqual(publication["receipts"], {
            "independent_reaggregation_sha256": provenance["independent_reaggregation_sha256"],
            "sanitization_sha256": provenance["sanitization_receipt_sha256"],
            "terminal_sha256": provenance["terminal_sha256"],
        })
        for forbidden in sanitization["forbidden_public_fields"]:
            self.assertTrue(all(forbidden not in row for row in qrows))
            self.assertNotIn(forbidden, terminal)

        pre = rows[-1]
        pre_qrows_bytes = PRE_QROWS.read_bytes()
        pre_qrows = [json.loads(line) for line in pre_qrows_bytes.splitlines() if line.strip()]
        pre_terminal_bytes = PRE_TERMINAL.read_bytes()
        pre_terminal = json.loads(pre_terminal_bytes)
        pre_reaggregation_bytes = PRE_REAGGREGATION.read_bytes()
        pre_reaggregation = json.loads(pre_reaggregation_bytes)
        pre_accounting_bytes = PRE_ACCOUNTING.read_bytes()
        pre_accounting = json.loads(pre_accounting_bytes)
        pre_sanitization_bytes = PRE_SANITIZATION.read_bytes()
        pre_sanitization = json.loads(pre_sanitization_bytes)
        pre_publication_bytes = PRE_PUBLICATION.read_bytes()
        pre_publication = json.loads(pre_publication_bytes)
        pre_qrows_sha = "4d8339096683a1049f614a49db34acd6f42fbbdbdbae4cf95c2a814bd46d3cf5"

        self.assertEqual(hashlib.sha256(pre_qrows_bytes).hexdigest(), pre_qrows_sha)
        self.assertEqual([row["sample_ordinal"] for row in pre_qrows], list(range(500)))
        self.assertEqual([row["row_sha256"] for row in pre_qrows], [item["row_sha256"] for item in items])
        self.assertEqual(sum(bool(row["correct"]) for row in pre_qrows), 412)
        self.assertTrue(all(row["choice_token_ids"] == [35, 36, 37, 38] for row in pre_qrows))
        self.assertTrue(all(all(math.isfinite(value) for value in row["choice_logits"]) for row in pre_qrows))
        self.assertTrue(all(len(set(row["choice_logits"])) == 4 for row in pre_qrows))
        pre_ce = math.fsum(-row["choice_logprobs"][row["gold_index"]] for row in pre_qrows) / (500 * math.log(2))
        self.assertAlmostEqual(pre_ce, pre_reaggregation["gold_cross_entropy_bits"], places=15)
        self.assertEqual((pre_terminal["n"], pre_terminal["correct"], pre_terminal["mmlu_percent"]), (500, 412, 82.4))
        self.assertEqual(pre_terminal["complete_artifact_bytes"], 89405007628)
        self.assertEqual(pre_reaggregation["complete_artifact_bytes"], 89405007628)
        self.assertEqual(pre_accounting["accounting"]["complete_artifact_bytes"], 89405007628)
        self.assertEqual(pre_accounting["physical_resident_readback"]["rank_deltas_bytes"], [0, 0])
        self.assertEqual(pre_accounting["excluded_historical_artifact"]["complete_artifact_bytes"], 100636011256)

        pre_provenance = pre["provenance"]
        self.assertEqual(hashlib.sha256(pre_terminal_bytes).hexdigest(), pre_provenance["terminal_sha256"])
        self.assertEqual(hashlib.sha256(pre_reaggregation_bytes).hexdigest(), pre_provenance["independent_reaggregation_sha256"])
        self.assertEqual(hashlib.sha256(pre_accounting_bytes).hexdigest(), pre_provenance["artifact_accounting_sha256"])
        self.assertEqual(hashlib.sha256(pre_sanitization_bytes).hexdigest(), pre_provenance["sanitization_receipt_sha256"])
        self.assertEqual(hashlib.sha256(pre_publication_bytes).hexdigest(), pre_provenance["publication_receipt_sha256"])
        self.assertEqual(pre_sanitization["qrows"]["bytes"], len(pre_qrows_bytes))
        self.assertEqual(pre_sanitization["qrows"]["output_sha256"], pre_qrows_sha)
        self.assertEqual(pre_publication["qrows"], {"bytes": len(pre_qrows_bytes), "rows": 500, "sha256": pre_qrows_sha})
        self.assertEqual(pre_publication["artifact"]["complete_artifact_bytes"], 89405007628)
        self.assertEqual(pre_publication["receipts"], {
            "artifact_accounting_sha256": pre_provenance["artifact_accounting_sha256"],
            "independent_reaggregation_sha256": pre_provenance["independent_reaggregation_sha256"],
            "sanitization_sha256": pre_provenance["sanitization_receipt_sha256"],
            "terminal_sha256": pre_provenance["terminal_sha256"],
        })
        for forbidden in pre_sanitization["forbidden_public_fields"]:
            self.assertTrue(all(forbidden not in row for row in pre_qrows))
            self.assertNotIn(forbidden, pre_terminal)
            self.assertNotIn(forbidden, pre_accounting)

        finished_manifest = json.loads(FINISHED_EVIDENCE_MANIFEST.read_text())
        self.assertEqual(
            result["finished_evidence_manifest_sha256"],
            hashlib.sha256(FINISHED_EVIDENCE_MANIFEST.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            [(entry["variant"], entry["status"]) for entry in finished_manifest["entries"]],
            [
                ("QTIP3-uniform-exact", "PASS"),
                ("QTIP2P5-deterministic-mixed-ring", "PASS"),
                ("EXL3-K2-uniform-exact", "PASS"),
                ("Physical-alternating-K2K3-full", "PASS"),
                ("EXL3-K2P5-greedy-full", "ARTIFACT_UNAVAILABLE"),
                ("EXL3-K2P5-greedy-routed-native-rest", "PASS"),
                ("QTIP2-V7-U45-repaired", "PASS"),
                ("QTIP2-V7-PRE-f9bffe04", "PASS"),
            ],
        )
        for entry in finished_manifest["entries"]:
            evidence = REPO / entry["path"].replace("notes/", "archive/notes/", 1)
            self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(), entry["sha256"])
        self.assertEqual(
            [(entry["variant"], entry["status"]) for entry in result["dispositions"]],
            [
                ("EXL3-K2P5-greedy-full", "ARTIFACT_UNAVAILABLE"),
                ("QTIP2-corrected-all43", "METHOD_DIVERGENT_QUARANTINED"),
            ],
        )
        quarantine = result["dispositions"][1]
        self.assertEqual(quarantine["accepted_score"], 0)
        self.assertFalse(quarantine["replacement_score_published"])
        self.assertEqual(
            quarantine["candidate_artifact_sha256"],
            "86e1a27915c4cd2afef945c349c4bfecc4978ad530cc5a1bf78cea8258c25121",
        )
        self.assertEqual(
            quarantine["qrows_sha256"],
            "a2071096f9a8207f50b661aeddf5911586a5de7f50ae1e887d952855d2a110b2",
        )
        self.assertEqual(
            quarantine["producer_manifest_sha256"],
            "7dbdbeb1977388461e61a2a2899eec5b6853dd7f58d454f7e682bd953fbe90c2",
        )
        self.assertEqual(quarantine["prior_publication_pr"], 62)

        evidence_manifest = json.loads(EVIDENCE_MANIFEST.read_text())
        qtip2_evidence = next(
            entry for entry in evidence_manifest["rows"]
            if entry["variant"] == "QTIP2-corrected-all43"
        )
        self.assertEqual(qtip2_evidence["status"], "METHOD_DIVERGENT_QUARANTINED")
        self.assertEqual(qtip2_evidence["accepted_score"], 0)
        qtip2_identity = json.loads(QTIP2_IDENTITY.read_text())
        self.assertEqual(qtip2_identity["status"], "METHOD_DIVERGENT_QUARANTINED")
        self.assertEqual(qtip2_identity["accepted_score"], 0)

        evals = EVALS.read_text()
        self.assertIn("Above-Chance MMLU/GB ↑", evals)
        self.assertIn("Raw MMLU/BPW ↑", evals)
        self.assertIn("Above-Chance MMLU per BPW (within model)", REPORT.read_text())
        for fragment in (
            "Official native MXFP4** |  |  | **84.60%** (423/500) | **13.576**",
            "EXL3 K2 routed-only + native rest** | **86.33%** (56,579/65,536) | **0.234288** | **83.60%** (418/500) | **23.305**",
            "EXL3 K3 routed-only + native rest** | **92.23%** (60,447/65,536) | **0.076868** | **85.20%** (426/500) | **17.255**",
            "EXL3 K3 uniform exact** | **88.30%** (57,870/65,536) | **0.136015** | **84.80%** (424/500) | **18.766**",
            "QTIP3 uniform exact** | **91.68%** (60,084/65,536) | **0.110227** | **84.20%** (421/500) | **16.973**",
            "QTIP2.5 deterministic mixed ring** | **89.09%** (58,389/65,536) | **0.181971** | **82.80%** (414/500) | **19.261**",
            "EXL3 K2.5 greedy-upcast routed-only + native rest** | **88.33%** (57,885/65,536) | **0.174604** | **84.80%** (424/500) | **19.998**",
            "EXL3 K2 uniform exact** | **81.78%** (53,593/65,536) | **0.366820** | **73.80%** (369/500) | **22.276**",
            "Physical alternating K2/K3 2.5-BPW comparator** | **83.29%** (54,585/65,536) | **0.299604** | **74.80%** (374/500) | **18.664**",
            "QTIP2 V7 U45 repaired** | **86.22%** (56,508/65,536) | **0.211278** | **82.60%** (413/500) | **20.343**",
            "QTIP2 V7 published PRE f9bffe04** | **86.26%** (56,533/65,536) | **0.229392** | **82.40%** (412/500) | **22.819**",
            "ARTIFACT_UNAVAILABLE",
        ):
            self.assertIn(fragment, evals)
        self.assertNotIn("QTIP2 corrected all-43** |", evals)
        self.assertNotIn("| QTIP2 corrected all-43 |", REPORT.read_text())
        self.assertNotIn("QTIP2-corrected-all43", [row["variant"] for row in rows])
        self.assertIn("METHOD_DIVERGENT_QUARANTINED", REPORT.read_text())

        evidence_paths = [REPO / entry["path"].replace("notes/", "archive/notes/", 1) for entry in finished_manifest["entries"]]
        public_text = "\n".join(path.read_text() for path in (RESULTS, SCHEMA, REPORT, FINISHED_EVIDENCE_MANIFEST, *evidence_paths))
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
