import sys
import yaml

law = sys.argv[1]  # zorgtoeslag, bijstand, alcoholwet
cf_file = sys.argv[2]

orig = yaml.safe_load(open("data/profiles.yaml", encoding="utf-8"))
orig["profiles"] = {k: v for k, v in orig["profiles"].items() if not str(k).startswith("CF")}
print(f"Base profiles: {len(orig['profiles'])}")

cf = yaml.safe_load(open(cf_file, encoding="utf-8"))
orig["profiles"].update(cf["profiles"])
print(f"After {law} CF merge: {len(orig['profiles'])} profiles (+{len(cf['profiles'])} CF)")

yaml.dump(orig, open("data/profiles.yaml", "w", encoding="utf-8"), allow_unicode=True, default_flow_style=False)
print("profiles.yaml saved.")
