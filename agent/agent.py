import json
from .llm_groq import call_llm
from memory.weights_store import update_weight
import threading
from pathlib import Path

SYSTEM_PROMPT = """
You are a PageIndex agent.

You have tools:
1. get_document()
2. get_document_structure()
3. get_page_content(pages="x-y")

RULES:
- Always reason step by step
- Use tools before answering
- NEVER fetch full document
- ALWAYS respond in ONE of these formats:

If using a tool:
{
  "tool": "tool_name",
  "args": { ... }
}

If final answer:
{
  "final": "your answer"
}
"""

def parse_response(text):
    try:
        return json.loads(text)
    except:
        return None


def run_agent(client, doc_id, query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    # ── 🔥 GLOBAL MEMORY CHECK ─────────────────────────
    MEM_FILE = Path("memory/mem.json")

    if MEM_FILE.exists():
        try:
            mem = json.loads(MEM_FILE.read_text(encoding="utf-8"))

            if mem.get("nodes"):

                prompt = f"""
User Query:
{query}

Memory cache:
{json.dumps(mem['nodes'], indent=2)}

Task:
Only return hit=true if the memory DIRECTLY answers the query.

If yes:
{{"hit": true, "doc_id": "<doc_id>", "pages": "<pages>"}}

If not:
{{"hit": false}}
"""

                decision = call_llm([
                    {"role": "system", "content": "You are a precise retrieval router."},
                    {"role": "user", "content": prompt}
                ])

                parsed = parse_response(decision)

                if parsed and parsed.get("hit"):
                    mem_doc_id = parsed.get("doc_id")
                    pages = parsed.get("pages")

                    print("⚡ MEMORY HIT:", mem_doc_id, pages)

                    result = client.get_page_content(mem_doc_id, pages)

                    # 🔥 LIMIT SIZE
                    result = result[:4000]

                    # 🔥 FINAL ANSWER SYNTHESIS
                    final_answer = call_llm([
                        {
                            "role": "system",
                            "content": "Answer clearly and concisely."
                        },
                        {
                            "role": "user",
                            "content": f"""
Question:
{query}

Content:
{result}

Give a short, relevant explanation.
"""
                        }
                    ])

                    return final_answer

        except:
            pass
    # ─────────────────────────────────────────────────

    for step in range(10):
        print(f"\n--- Step {step+1} ---")

        response_text = call_llm(messages)
        print("LLM:", response_text)

        parsed = parse_response(response_text)

        if not parsed:
            messages.append({
                "role": "system",
                "content": "Respond ONLY in valid JSON."
            })
            continue

        if "tool" in parsed:
            tool = parsed["tool"]
            args = parsed.get("args", {})

            print("Calling tool:", tool, args)

            if tool == "get_document":
                result = client.get_document(doc_id)

            elif tool == "get_document_structure":
                result = client.get_document_structure(doc_id)

            elif tool == "get_page_content":
                pages = args.get("pages", "")
                result = client.get_page_content(doc_id, pages)

                threading.Thread(
                    target=update_weight,
                    args=(doc_id, pages),
                    daemon=True
                ).start()
            else:
                result = "Unknown tool"

            print("Tool result:", result[:200])

            messages.append({
                "role": "assistant",
                "content": response_text
            })

            messages.append({
                "role": "assistant",
                "content": f"Tool result:\n{result}"
            })

        elif "final" in parsed:
            return parsed["final"]

        else:
            messages.append({
                "role": "system",
                "content": "Invalid format. Use JSON."
            })

    return "Max steps reached"