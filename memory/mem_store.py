import json
from pathlib import Path
from memory.weights_store import load_weights
from agent.llm_groq import call_llm

WORKSPACE_DIR = Path("examples/workspace")
MEM_FILE = Path("memory/mem.json")


def find_overlapping_nodes(nodes, start, end):
    result = []

    for node in nodes:
        if node["start_index"] <= end and node["end_index"] >= start:
            result.append(node)

        if "nodes" in node:
            result.extend(find_overlapping_nodes(node["nodes"], start, end))

    return result


def merge_summaries(summaries):
    if not summaries:
        return ""

    prompt = f"""
Combine the following summaries into ONE concise summary.

{chr(10).join(summaries)}

Keep only key ideas. Remove repetition.
"""

    return call_llm([
        {"role": "system", "content": "You summarize efficiently."},
        {"role": "user", "content": prompt}
    ]).strip()


def build_memory(top_k=5):
    weights = load_weights()

    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    mem_nodes = []

    for key, weight in sorted_items[:top_k]:
        try:
            doc_id, pages = key.split("_", 1)

            structure_file = WORKSPACE_DIR / f"{doc_id}.json"

            data = json.loads(structure_file.read_text(encoding="utf-8"))
            structure = data.get("structure", [])

            if "-" in pages:
                start, end = map(int, pages.split("-"))
            else:
                start = end = int(pages)

            overlap_nodes = find_overlapping_nodes(structure, start, end)

            summaries = [n.get("summary", "") for n in overlap_nodes if n.get("summary")]

            merged_summary = merge_summaries(summaries)

            mem_nodes.append({
                "doc_id": doc_id,
                "pages": pages,
                "summary": merged_summary
            })

        except:
            continue

    memory = {
        "type": "memory",
        "nodes": mem_nodes
    }

    MEM_FILE.parent.mkdir(exist_ok=True)
    MEM_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")

    print(f"✅ Memory graph built with {len(mem_nodes)} nodes.")


if __name__ == "__main__":
    build_memory(top_k=5)