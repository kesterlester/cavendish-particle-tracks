import csv
import tempfile

# --- Configuration ---
PLACEHOLDERS = {
    "colours": "none",
    "types": "none",
    "symbols": "none",
}

MERGED_FIELDS = [
    "view", "event",
    "pixel_row", "pixel_col", "labels",
    "colours", "types", "symbols"
]


def merge(f1, f2, f3, fgeneric, merged_path, placeholders=PLACEHOLDERS):
    files = [
        (f1, 0.0),  # view for f1
        (f2, 1.0),  # view for f2
        (f3, 2.0),  # view for f3
    ]
    
    with open(merged_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=MERGED_FIELDS)
        writer.writeheader()

        # --- Process f1, f2, f3 ---
        for f, view_val in files:
            reader = csv.DictReader(f)
            for row in reader:
                merged_row = {
                    "view": view_val,
                    "event": None,
                    "pixel_row": row.get("pixel_row"),
                    "pixel_col": row.get("pixel_col"),
                    "labels": row.get("labels"),
                    "colours": row.get("colours", placeholders["colours"]),
                    "types": row.get("types", placeholders["types"]),
                    "symbols": row.get("symbols", placeholders["symbols"]),
                }
                writer.writerow(merged_row)

        # --- Process fgeneric ---
        reader = csv.DictReader(fgeneric)
        for row in reader:
            merged_row = {
                "view": row.get("view"),
                "event": row.get("event"),
                "pixel_row": row.get("pixel_row"),
                "pixel_col": row.get("pixel_col"),
                "labels": row.get("labels"),
                "colours": placeholders["colours"],
                "types": placeholders["types"],
                "symbols": placeholders["symbols"],
            }
            writer.writerow(merged_row)


def unmerge(merged_path):
    """Unpack merged.csv into temporary CSVs corresponding to f1, f2, f3, fgeneric."""
    temp_files = {}
    readers = {}
    writers = {}

    # Define schemas for outputs
    schemas = {
        "f1": ["pixel_row", "pixel_col", "labels", "colours", "types", "symbols"],
        "f2": ["pixel_row", "pixel_col", "labels", "colours", "types", "symbols"],
        "f3": ["pixel_row", "pixel_col", "labels", "colours", "types", "symbols"],
        "fgeneric": ["view", "event", "pixel_row", "pixel_col", "labels"],
    }

    # Create temporary files and writers
    for name, fields in schemas.items():
        temp_files[name] = tempfile.NamedTemporaryFile(mode="w+", newline="", delete=False)
        writers[name] = csv.DictWriter(temp_files[name], fieldnames=fields)
        writers[name].writeheader()

    # --- Read and dispatch ---
    with open(merged_path, newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            event = row.get("event")
            if event in (None, "", "None"):  # from f1–f3
                view = float(row["view"])
                if view == 0.0:
                    writers["f1"].writerow({k: row[k] for k in schemas["f1"]})
                elif view == 1.0:
                    writers["f2"].writerow({k: row[k] for k in schemas["f2"]})
                elif view == 2.0:
                    writers["f3"].writerow({k: row[k] for k in schemas["f3"]})
            else:  # from fgeneric
                writers["fgeneric"].writerow({k: row[k] for k in schemas["fgeneric"]})

    # Flush and rewind files
    for f in temp_files.values():
        f.flush()
        f.seek(0)

    print("Ummerge is about to return dict having these items:")
    for key, thing in temp_files.items():
        print(f"{key=}, {thing=}")

    return temp_files
