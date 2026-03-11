"""
Batch analysis of scientific figures from a paper folder.
Wraps test_gemma_vision.py — imports analyze_image and SYSTEM_PROMPT directly.
"""

import sys
import json
import time
import base64
import argparse
import warnings
import urllib.request
from pathlib import Path
from collections import Counter

import pandas as pd

from test_gemma_vision import analyze_image, SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:27b"

MOF_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "material": {
            "type": "object",
            "properties": {
                "name":          {"type": ["string", "null"]},
                "formula":       {"type": ["string", "null"]},
                "topology":      {"type": ["string", "null"]},
                "material_type": {"type": ["string", "null"]},
                "crystal_system":{"type": ["string", "null"]},
                "space_group":   {"type": ["string", "null"]}
            }
        },
        "building_blocks": {
            "type": "object",
            "properties": {
                "metal_nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "element":             {"type": ["string", "null"]},
                            "oxidation_state":     {"type": ["string", "null"]},
                            "coordination_geometry":{"type": ["string", "null"]}
                        }
                    }
                },
                "organic_linkers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":             {"type": ["string", "null"]},
                            "smiles":           {"type": ["string", "null"]},
                            "functional_groups":{"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "secondary_building_units": {"type": ["string", "null"]}
            }
        },
        "synthesis": {
            "type": "object",
            "properties": {
                "method":           {"type": ["string", "null"]},
                "temperature_C":    {"type": ["number", "null"]},
                "temperature_range_C": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                    "minItems": 2,
                    "maxItems": 2
                },
                "time_hours":       {"type": ["number", "null"]},
                "pressure_atm":     {"type": ["number", "null"]},
                "pH":               {"type": ["number", "null"]},
                "solvents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":      {"type": ["string", "null"]},
                            "volume_mL": {"type": ["number", "null"]},
                            "ratio":     {"type": ["string", "null"]}
                        }
                    }
                },
                "precursors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":           {"type": ["string", "null"]},
                            "concentration_M":{"type": ["number", "null"]},
                            "amount_mmol":    {"type": ["number", "null"]},
                            "amount_mg":      {"type": ["number", "null"]}
                        }
                    }
                },
                "modulators_additives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":   {"type": ["string", "null"]},
                            "role":   {"type": ["string", "null"]},
                            "amount": {"type": ["string", "null"]}
                        }
                    }
                },
                "molar_ratio":          {"type": ["string", "null"]},
                "yield_percent":        {"type": ["number", "null"]},
                "activation_conditions":{"type": ["string", "null"]}
            }
        },
        "structure": {
            "type": "object",
            "properties": {
                "unit_cell": {
                    "type": "object",
                    "properties": {
                        "a_angstrom":  {"type": ["number", "null"]},
                        "b_angstrom":  {"type": ["number", "null"]},
                        "c_angstrom":  {"type": ["number", "null"]},
                        "alpha_deg":   {"type": ["number", "null"]},
                        "beta_deg":    {"type": ["number", "null"]},
                        "gamma_deg":   {"type": ["number", "null"]}
                    }
                },
                "pore_geometry":      {"type": ["string", "null"]},
                "pore_size_angstrom": {"type": ["number", "null"]},
                "pore_volume_cm3g":   {"type": ["number", "null"]},
                "dimensionality":     {"type": ["string", "null"]}
            }
        },
        "characterization": {
            "type": "object",
            "properties": {
                "BET_surface_area_m2g":    {"type": ["number", "null"]},
                "XRD_peaks_2theta":        {"type": "array", "items": {"type": "number"}},
                "TGA_decomposition_temp_C":{"type": ["number", "null"]},
                "particle_morphology":     {"type": ["string", "null"]},
                "particle_size_um":        {"type": ["number", "null"]},
                "techniques_observed":     {"type": "array", "items": {"type": "string"}}
            }
        },
        "performance": {
            "type": "object",
            "properties": {
                "applications": {"type": "array", "items": {"type": "string"}},
                "gas_adsorption": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "gas":         {"type": ["string", "null"]},
                            "uptake_value":{"type": ["number", "null"]},
                            "uptake_unit": {"type": ["string", "null"]},
                            "conditions":  {"type": ["string", "null"]}
                        }
                    }
                },
                "selectivity": {"type": ["string", "null"]},
                "stability": {
                    "type": "object",
                    "properties": {
                        "thermal_stability_C": {"type": ["number", "null"]},
                        "water_stability":     {"type": ["string", "null"]},
                        "chemical_stability":  {"type": ["string", "null"]}
                    }
                }
            }
        },
        "figure_metadata": {
            "type": "object",
            "properties": {
                "figure_type": {"type": "string"},
                "confidence":  {"type": "string", "enum": ["high", "medium", "low"]},
                "notes":       {"type": ["string", "null"]}
            },
            "required": ["figure_type", "confidence"]
        }
    },
    "required": ["material", "building_blocks", "synthesis", "structure",
                 "characterization", "performance", "figure_metadata"]
}


def analyze_image_structured(image_path: str, prompt: str) -> str:
    """
    Same as analyze_image() from test_gemma_vision.py but with
    Ollama structured output (format = JSON Schema).
    This forces the model to produce output matching MOF_JSON_SCHEMA.
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": MODEL,
        "format": MOF_JSON_SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Please analyze this scientific figure and extract all relevant information as JSON.",
                "images": [img_b64],
            },
        ],
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
            "repeat_penalty": 1.1,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    print(f"正在调用本地模型 {MODEL} (structured output)...", flush=True)
    full_content = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                full_content.append(token)
            if chunk.get("done", False):
                break
    print()

    return "".join(full_content)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ── Step 1: build paper context ──────────────────────────────────────────────

def load_paper_context(paper_folder: Path) -> str:
    parts = []
    missing = []

    # *.md → plain text (first match)
    md_files = sorted(paper_folder.glob("*.md"))
    if md_files:
        md_text = md_files[0].read_text(encoding="utf-8", errors="replace")
        parts.append(
            f"=== PAPER FULL TEXT (for cross-referencing figure labels) ===\n"
            f"{md_text[:6000]}"
        )
    else:
        missing.append("*.md")

    # *_content_list_v2.json
    cl2 = sorted(paper_folder.glob("*_content_list_v2.json"))
    if cl2:
        content_list_v2 = json.loads(cl2[0].read_text(encoding="utf-8", errors="replace"))
        parts.append(
            f"=== CONTENT LIST (structured sections) ===\n"
            f"{json.dumps(content_list_v2, ensure_ascii=False)[:3000]}"
        )
    else:
        missing.append("*_content_list_v2.json")

    # *_middle.json
    mid = sorted(paper_folder.glob("*_middle.json"))
    if mid:
        middle_json = json.loads(mid[0].read_text(encoding="utf-8", errors="replace"))
        parts.append(
            f"=== DOCUMENT MIDDLE LAYER (parsed blocks) ===\n"
            f"{json.dumps(middle_json, ensure_ascii=False)[:3000]}"
        )
    else:
        missing.append("*_middle.json")

    # *_model.json
    mod = sorted(paper_folder.glob("*_model.json"))
    if mod:
        model_json = json.loads(mod[0].read_text(encoding="utf-8", errors="replace"))
        parts.append(
            f"=== DOCUMENT MODEL (layout model output) ===\n"
            f"{json.dumps(model_json, ensure_ascii=False)[:2000]}"
        )
    else:
        missing.append("*_model.json")

    if missing:
        warnings.warn(f"Missing context files: {', '.join(missing)}")

    return "\n\n".join(parts)


# ── Step 2: build enhanced prompt ────────────────────────────────────────────

ENFORCEMENT = """
---
## CRITICAL OUTPUT RULES — YOU MUST FOLLOW THESE EXACTLY

1. OUTPUT FORMAT: You MUST return ONLY a single valid JSON object. No markdown, no explanation, no text outside the JSON.

2. SCHEMA ENFORCEMENT: You MUST use EXACTLY the schema defined above. 
   - DO NOT invent new field names like "figure_title", "data_traces", "raw_array", "Empirical formula", "curves", "table", etc.
   - DO NOT copy table headers or chart labels as new JSON keys.
   - ALL information must be mapped into the existing schema fields.

3. MAPPING RULES for common figure types:
   - If you see a crystallographic data table → extract values into: material.crystal_system, material.space_group, structure.unit_cell.*, material.formula, material.name
   - If you see a PXRD / XRD diffraction pattern → figure_metadata.figure_type = "PXRD", characterization.XRD_peaks_2theta = [list of peak positions], material.name from title
   - If you see a TGA curve → figure_metadata.figure_type = "TGA", characterization.TGA_decomposition_temp_C, performance.stability.thermal_stability_C
   - If you see a N2 adsorption isotherm → figure_metadata.figure_type = "N2_isotherm", characterization.BET_surface_area_m2g, structure.pore_volume_cm3g
   - If you see a stability test plot → figure_metadata.figure_type = "stability_test", performance.stability.water_stability or chemical_stability, material.name from title
   - If you see an ORTEP / crystal structure diagram → figure_metadata.figure_type = "crystal_structure", building_blocks.metal_nodes, building_blocks.organic_linkers
   - If you see an IR spectrum → figure_metadata.figure_type = "IR_spectrum", building_blocks.organic_linkers.functional_groups

4. For fields where information is NOT available from this figure, use null. Do not omit the field.

5. NEVER add fields outside the schema. NEVER use: "title", "curves", "lines", "data_points", "table", "raw_array", "figure_title", "Empirical formula" as top-level keys.
"""


def build_prompt(image_filename: str, paper_context: str) -> str:
    return f"""{SYSTEM_PROMPT}{ENFORCEMENT}
---
## PAPER CONTEXT (use only for resolving material names and abbreviations, max 2000 chars)
{paper_context[:8000]}

## CURRENT FIGURE: {image_filename}
"""


# ── Step 3: discover images ─────────────────────────────────────────────────

def discover_images(paper_folder: Path) -> list[Path]:
    image_dir = paper_folder / "image"
    if not image_dir.is_dir():
        image_dir = paper_folder / "images"
    if not image_dir.is_dir():
        sys.exit(f"Error: image directory not found: {paper_folder}/image(s)")
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


# ── JSON extraction (same logic as test_gemma_vision.py) ────────────────────

def extract_json(answer: str) -> dict:
    json_str = answer
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]
    return json.loads(json_str.strip())


# ── Step 5: flatten for CSV ─────────────────────────────────────────────────

def _join(items: list | None, key: str) -> str:
    if not items:
        return ""
    return "; ".join(str(d.get(key, "")) for d in items if d.get(key))


def flatten_result(r: dict) -> dict:
    meta = r.get("_meta") or {}

    # If model followed our schema, fields are nested dicts
    # If not, they may be strings or missing entirely — handle both
    def _get_dict(key):
        v = r.get(key)
        return v if isinstance(v, dict) else {}

    mat = _get_dict("material")
    bb = _get_dict("building_blocks")
    syn = _get_dict("synthesis")
    stru = _get_dict("structure")
    char = _get_dict("characterization")
    perf = _get_dict("performance")
    fm = _get_dict("figure_metadata")

    # If "material" was a plain string (model non-compliance), capture it
    mat_name = mat.get("name") if mat else r.get("material") if isinstance(r.get("material"), str) else None

    return {
        "material_name": mat_name,
        "material_formula": mat.get("formula"),
        "material_type": mat.get("material_type"),
        "topology": mat.get("topology"),
        "crystal_system": mat.get("crystal_system"),
        "space_group": mat.get("space_group"),
        "metal_nodes": _join(bb.get("metal_nodes"), "element"),
        "organic_linkers": _join(bb.get("organic_linkers"), "name"),
        "synthesis_method": syn.get("method"),
        "synthesis_temperature_C": syn.get("temperature_C"),
        "synthesis_time_hours": syn.get("time_hours"),
        "synthesis_molar_ratio": syn.get("molar_ratio"),
        "synthesis_yield_percent": syn.get("yield_percent"),
        "solvents": _join(syn.get("solvents"), "name"),
        "precursors": _join(syn.get("precursors"), "name"),
        "modulators_additives": _join(syn.get("modulators_additives"), "name"),
        "pore_size_angstrom": stru.get("pore_size_angstrom"),
        "pore_volume_cm3g": stru.get("pore_volume_cm3g"),
        "dimensionality": stru.get("dimensionality"),
        "BET_surface_area_m2g": char.get("BET_surface_area_m2g"),
        "particle_morphology": char.get("particle_morphology"),
        "techniques_observed": "; ".join(char.get("techniques_observed") or []),
        "applications": "; ".join(perf.get("applications") or []),
        "figure_type": fm.get("figure_type"),
        "confidence": fm.get("confidence"),
        "notes": fm.get("notes"),
        "source_image": meta.get("source_image"),
        "analysis_index": meta.get("analysis_index"),
        "error": r.get("error", ""),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch analyze paper figures with Gemma 3")
    parser.add_argument("paper_folder", type=Path, help="Path to paper folder")
    parser.add_argument("--resume", action="store_true", help="Skip images already in checkpoint")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests (default 0.5)")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory (default: paper_folder)")
    args = parser.parse_args()

    paper_folder = args.paper_folder.resolve()
    if not paper_folder.is_dir():
        sys.exit(f"Error: folder not found: {paper_folder}")

    checkpoint_path = paper_folder / "results_checkpoint.json"

    # Load checkpoint if resuming
    done_set: set[str] = set()
    all_results: list[dict] = []
    if args.resume and checkpoint_path.exists():
        all_results = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        done_set = {
            r.get("_meta", {}).get("source_image", "")
            for r in all_results
        }
        print(f"Resuming: {len(done_set)} images already processed")

    # Step 1
    print("Loading paper context...", flush=True)
    paper_context = load_paper_context(paper_folder)

    # Step 3
    images = discover_images(paper_folder)
    total = len(images)
    print(f"Found {total} images in {paper_folder / 'image'}\n", flush=True)

    if total == 0:
        sys.exit("No images found.")

    # Step 4 — batch loop
    success_count = 0
    fail_count = 0

    processed = 0
    for i, img_path in enumerate(images, start=1):
        filename = img_path.name

        if filename in done_set:
            print(f"[{i}/{total}] Skipping (already done): {filename}")
            continue

        if args.limit > 0 and processed >= args.limit:
            print(f"Reached --limit {args.limit}, stopping.")
            break

        print(f"[{i}/{total}] Analyzing: {filename} ...", flush=True)

        try:
            prompt = build_prompt(filename, paper_context)
            try:
                answer = analyze_image_structured(str(img_path), prompt=prompt)
            except Exception as struct_err:
                print(f"  ⚠ Structured output failed ({struct_err}), falling back to analyze_image...", flush=True)
                answer = analyze_image(str(img_path), prompt=prompt)

            parsed = extract_json(answer)
            if isinstance(parsed, list):
                raise ValueError("Model returned a list instead of object schema")
            parsed["_meta"] = {
                "source_image": filename,
                "source_folder": paper_folder.name,
                "analysis_index": i,
            }
            all_results.append(parsed)
            fig_type = (parsed.get("figure_metadata") or {}).get("figure_type", "unknown")
            conf = (parsed.get("figure_metadata") or {}).get("confidence", "?")
            print(f"  → Success: {fig_type} | {conf} confidence", flush=True)
            success_count += 1
            processed += 1

        except Exception as e:
            result = {
                "error": str(e),
                "raw_output": locals().get("answer", "")[:500],
                "_meta": {"source_image": filename, "analysis_index": i},
            }
            all_results.append(result)
            print(f"  → Error: {e}", flush=True)
            fail_count += 1
            processed += 1

        # Checkpoint every 5 images
        if i % 5 == 0:
            checkpoint_path.write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if i < total:
            time.sleep(args.delay)

    # Step 5 — save outputs
    out_dir = args.output_dir.resolve() if args.output_dir else paper_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "mof_analysis_results.json"
    csv_out = out_dir / "mof_analysis_results.csv"

    json_out.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = [flatten_result(r) for r in all_results]
    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    # Also save final checkpoint
    checkpoint_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Step 6 — summary
    type_counter = Counter(
        (r.get("figure_metadata") or {}).get("figure_type", "unknown")
        for r in all_results
        if "error" not in r
    )
    materials = set()
    for r in all_results:
        m = r.get("material")
        if isinstance(m, dict) and m.get("name"):
            materials.add(m["name"])
        elif isinstance(m, str) and m:
            materials.add(m)
    materials = sorted(materials)

    type_str = ", ".join(f"{k}({v})" for k, v in type_counter.most_common())

    print(f"""
{'=' * 40}
 ANALYSIS COMPLETE
{'=' * 40}
Total images processed : {success_count + fail_count}
Successful             : {success_count}
Failed / parse error   : {fail_count}
Figure types found     : {type_str}
Materials identified   : {', '.join(materials) if materials else '(none)'}
Output saved to        : {json_out}
                       : {csv_out}
{'=' * 40}
""")


if __name__ == "__main__":
    main()
