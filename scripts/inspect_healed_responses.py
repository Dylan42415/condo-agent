import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "reports" / "adversarial_ux_audit_results.json", encoding="utf-8") as f:
    data = json.load(f)

targets = ['LEGAL_05_BBQ_DEPOSIT', 'LEGAL_01_CAR_CLEARANCE', 'EMERGENCY_01', 'EMERGENCY_02', 'SINGLISH_03', 'SINGLISH_01']
for d in data:
    if d['id'] in targets:
        print('=' * 80)
        print(f"SCENARIO: {d['id']} ({d['description']})")
        print("QUERY:", d['query'])
        print("CHECKS:", d['checks'])
        print("RAW OUTPUT:\n" + d['raw_output'])
