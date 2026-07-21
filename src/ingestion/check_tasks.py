import ee
EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

tasks = ee.batch.Task.list()
relevant = [t for t in tasks if "region_a" in t.config.get("description", "") or "region_b" in t.config.get("description", "")]

if not relevant:
    print("No matching tasks found (they may have already completed and aged out of the list).")

for t in relevant:
    status = t.status()
    print(f"{status['description']:40s} {status['state']:12s} {status.get('progress', '')}")
