"""
本地 Gemma3 视觉模型 - MOF/COF/ZIF 科研图片分析脚本
通过 Ollama 调用本地 gemma3:27b 模型
"""

import base64
import os
import sys
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:27b"

SYSTEM_PROMPT = """
You are an expert materials scientist specializing in Metal-Organic Frameworks (MOFs), 
Covalent Organic Frameworks (COFs), and Zeolitic Imidazolate Frameworks (ZIFs). 
Your task is to analyze scientific figures from research papers and extract all 
relevant structural, synthetic, and characterization information.

Extract ALL available information and return a strict JSON object. 
Use null for fields not determinable from the image.

Return ONLY valid JSON with this exact schema:

{
  "material": {
    "name": "string | null",
    "formula": "string | null",
    "topology": "string | null",
    "material_type": "MOF | COF | ZIF | other | null",
    "crystal_system": "string | null",
    "space_group": "string | null"
  },
  "building_blocks": {
    "metal_nodes": [
      {
        "element": "string | null",
        "oxidation_state": "string | null",
        "coordination_geometry": "string | null"
      }
    ],
    "organic_linkers": [
      {
        "name": "string | null",
        "smiles": "string | null",
        "functional_groups": ["string"]
      }
    ],
    "secondary_building_units": "string | null"
  },
  "synthesis": {
    "method": "string | null",
    "temperature_C": "number | null",
    "temperature_range_C": [null, null],
    "time_hours": "number | null",
    "pressure_atm": "number | null",
    "pH": "number | null",
    "solvents": [
      {
        "name": "string | null",
        "volume_mL": "number | null",
        "ratio": "string | null"
      }
    ],
    "precursors": [
      {
        "name": "string | null",
        "concentration_M": "number | null",
        "amount_mmol": "number | null",
        "amount_mg": "number | null"
      }
    ],
    "modulators_additives": [
      {
        "name": "string | null",
        "role": "string | null",
        "amount": "string | null"
      }
    ],
    "molar_ratio": "string | null",
    "yield_percent": "number | null",
    "activation_conditions": "string | null"
  },
  "structure": {
    "unit_cell": {
      "a_angstrom": "number | null",
      "b_angstrom": "number | null",
      "c_angstrom": "number | null",
      "alpha_deg": "number | null",
      "beta_deg": "number | null",
      "gamma_deg": "number | null"
    },
    "pore_geometry": "string | null",
    "pore_size_angstrom": "number | null",
    "pore_volume_cm3g": "number | null",
    "dimensionality": "1D | 2D | 3D | null"
  },
  "characterization": {
    "BET_surface_area_m2g": "number | null",
    "XRD_peaks_2theta": ["number"],
    "TGA_decomposition_temp_C": "number | null",
    "particle_morphology": "string | null",
    "particle_size_um": "number | null",
    "techniques_observed": ["string"]
  },
  "performance": {
    "applications": ["string"],
    "gas_adsorption": [
      {
        "gas": "string | null",
        "uptake_value": "number | null",
        "uptake_unit": "string | null",
        "conditions": "string | null"
      }
    ],
    "selectivity": "string | null",
    "stability": {
      "thermal_stability_C": "number | null",
      "water_stability": "string | null",
      "chemical_stability": "string | null"
    }
  },
  "figure_metadata": {
    "figure_type": "string",
    "confidence": "high | medium | low",
    "notes": "string | null"
  }
}
"""

DEFAULT_IMAGE = "/Users/mac/Downloads/390dcf00d8e716ddc02478972695560b.jpg"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: str, prompt: str = SYSTEM_PROMPT) -> str:
    img_b64 = encode_image(image_path)

    payload = {
        "model": MODEL,
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
            "temperature": 0.1,
            "num_predict": 8192,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    print(f"正在调用本地模型 {MODEL}，请稍候...", flush=True)
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
    print()  # newline after streaming finishes

    return "".join(full_content)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    print(f"图片: {image_path}", flush=True)
    print(f"模型: {MODEL}", flush=True)
    print("-" * 60, flush=True)

    try:
        answer = analyze_image(image_path)
        print("\n=== 模型分析结果 ===")
        print(answer)

        # 尝试解析并格式化 JSON
        try:
            # 提取 JSON 部分（模型可能返回 markdown 代码块）
            json_str = answer
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            parsed = json.loads(json_str.strip())
            print("\n=== 格式化 JSON ===")
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            print(formatted)

            # 保存到 Downloads 目录
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join("/Users/mac/Downloads", f"{base_name}_analysis.json")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted)
            print(f"\n结果已保存到: {output_path}")

        except (json.JSONDecodeError, IndexError):
            # JSON 解析失败，保存原始文本
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join("/Users/mac/Downloads", f"{base_name}_analysis.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(answer)
            print(f"\n(模型输出不是纯 JSON，已原样保存到: {output_path})")

    except Exception as e:
        print(f"错误: {e}")
        print("请确保 Ollama 正在运行 (ollama serve) 且已拉取 gemma3:27b 模型")
        sys.exit(1)


if __name__ == "__main__":
    main()
