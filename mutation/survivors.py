import sqlite3, sys, json

db = sys.argv[1]
c = sqlite3.connect(db)
# inspect schema
cols = [r[1] for r in c.execute("PRAGMA table_info(mutation_specs)").fetchall()]
rows = c.execute(
    "select m.* from mutation_specs m join work_results w on m.job_id=w.job_id "
    "where w.test_outcome='SURVIVED'"
).fetchall()
idx = {name: i for i, name in enumerate(cols)}
out = []
for r in rows:
    op = r[idx.get("operator_name")]
    sp = r[idx.get("start_pos_row")] if "start_pos_row" in idx else None
    defn = r[idx.get("definition_name")] if "definition_name" in idx else None
    out.append((sp, op, defn))
out.sort(key=lambda x: (x[0] is None, x[0]))
for sp, op, defn in out:
    print(f"L{sp}\t{defn}\t{op}")
print(f"TOTAL SURVIVED: {len(out)}")
print("COLS:", cols)
