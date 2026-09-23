#!/usr/bin/env python3
# Scores OCR battery outputs against the checkable facts of citadel's OCR_GROUND_TRUTH.md.
# Every fact is a string the ground truth states is on the page, or a tuple of the forms the page itself allows (e.g. a
# date in the language of the document); a fact counts as found when it occurs in the output
# after normalization (case, unicode forms, dashes, quotes, markdown emphasis and all whitespace are ignored).
# transfer_credit_report has a blank body: an output with a table there is reported as a hallucination.
#
#   scripts/ocr-score.py DIR [DIR ...]      each DIR holds the 16 <document>.md outputs of one run
#   scripts/ocr-score.py -v DIR ...         also lists the facts each run misses
import os
import re
import sys
import unicodedata

FACTS = {
    "business_textbook": ["Information Systems for Business and Beyond", "David Bourgeois", "2019",
                          "Creative Commons Attribution-NonCommercial 4.0 International License"],
    "handwritten_document": ["823,988", "4,383,825"],
    "transfer_credit_report": ["View Transfer Credit Report", "3:45", "mainestreetcs.maine.edu", "7/23/26"],
    "draft_agreement_contract": ["Technopark Participant", "1%", "Section 9", "20th", "quarter"],
    "academic_transcript": ["25 June 2026", "d86ea6e3-b633-4afd-9279-538f0d062b84", "billalmasum93@gmail.com",
                            "IBM Data Science Professional Certificate", "2 April 2025", "IBM-0017", "STAT1001",
                            "14 November 2025", "Sophia Learning", "FIBAA", "Page 1 of 7"],
    "invoice_864067": ["921893-M", "Billai", "60-1115631120", "22/06/2026", "13:14", "LY-G026-864067", "Yusri",
                       "NB-MSI-V16HX-AI-A2XWJG-478MY-GRY", "K2507N0044577", "14,500.00", "500.00",
                       "FOURTEEN THOUSAND FIVE HUNDRED ONLY", "9S7-15M352-478"],
    "consumer_protection_act": ["Act 599", "Consumer Protection Act 1999", "Ouster of choice of law",
                                "Bait advertising", "Misleading indication as to price",
                                "Claim that goods are limited", "False representation and other misleading conduct"],
    "digital_nomad_certificate": ["02.07.2026", "00922", "2519-9-001", "130", ("11 мая 2017", "2017 жылғы 11 мамыр"),
                                  ("20 февраля 2023", "2023 жылғы 20 ақпан")],
    "affidavit_of_service": ["TTPM-WP-(P)-1172-2026", "Iconix", "B00580030", ("30 April 2026", "30 hari bulan April 2026"),
                             "admin@iconixpropertymgmt.com", "W1015", "SHAHRIZAL", "24 JUN 2026", "Form 1",
                             "Menara MAIWP"],
    "borang_135_malay_form": ["EPE005055893MY", "28/07/2026", "11:50:03", "3.11801", "101.67351", "27/07/2026",
                              "WKL-WKL-WKL", "ICONIX Co Living", "Mercu Aspire", "59200", "0386812519",
                              "01115631120", "Letters & Papers"],
    "citadel_pitch_deck": ["AI Isn't the Bottleneck", "STRUCTURAL LOSS", "CONTEXTUAL LOSS", "63.9", "8.6", "4.8",
                           "of all announced investment", "semantic meaning"],
    "russian_doc_98630": ["4 декабря 2015", "992", "20 января 2016", "12880", "ПРИКАЗЫВАЮ", "31.03.2020", "275",
                          "19.07.2024", "571"],
    "malaysia_court_practice": ["1CNSPWB", "17-JUN-2025", "Sabujbagh", "Barek", "B00580030", "02-NOV-22",
                                "Mayakanon", "1214", "Yeasin Ali", "01320040188", "BP-8006098489", "2152"],
    "saa_std_datasheet": ["1346187", "08/01/2026", "112.34", "7.66", "ENG101", "ENG121", "POS101", "PHI152",
                          "PSY100", ("SOC100", "SOC | 100"), "ENG211", "Liberal Studies", "Management Information Systems"],
    "academic_paper_christodoulakis": ["Pytheas", "Christodoulakis", "Munson", "Gabel", "Demke Brown", "Miller",
                                       "10.14778/3407790.3407810", "2150-8097", "95.9%", "95.7%", "95.6%", "86.9%",
                                       "2075"],
    "zahlentheorie_math_german": ["Satz 370", "Satz 371", "Hilfssätze", "(419)", "(426)", "(427)"],
}


def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-").replace("‑", "-")
    s = s.replace("é", "e").replace("№", "no")
    s = re.sub(r"[*_`\\]", "", s)
    return re.sub(r"\s+", "", s)


def score(d):
    found, total, misses, notes = 0, 0, {}, []
    for doc, facts in FACTS.items():
        path = os.path.join(d, doc + ".md")
        text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if not text:
            notes.append(f"{doc}: missing or empty")
        t = norm(text)
        miss = [f for f in facts if not any(norm(alt) in t for alt in (f if isinstance(f, tuple) else (f,)))]
        found += len(facts) - len(miss)
        total += len(facts)
        if miss:
            misses[doc] = miss
        if doc == "transfer_credit_report":
            rows = [l for l in text.splitlines() if l.strip().startswith("|")]
            if len(rows) > 2:
                notes.append(f"{doc}: {len(rows)} table rows on a page whose body is blank (hallucinated)")
    return found, total, misses, notes


verbose = "-v" in sys.argv
dirs = [a for a in sys.argv[1:] if a != "-v"]
for d in dirs:
    found, total, misses, notes = score(d)
    print(f"{found:4d}/{total} facts ({100*found/total:5.1f}%)  {d}")
    for n in notes:
        print(f"      ! {n}")
    if verbose:
        for doc, miss in misses.items():
            print(f"      - {doc}: {', '.join(m if isinstance(m, str) else m[0] for m in miss)}")
