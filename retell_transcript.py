"""
retell_transcript.py
──────────────────────────────────────────
Shared parsing for Retell's own GET /get-chat/{chat_id} message_with_tool_calls
shape — agent/user turns interleaved with tool_call_invocation / tool_call_result
entries (node_transition/state_transition/injected/sms entries are ignored).

Field names per Retell's docs:
  tool_call_invocation: role, tool_call_id, name (the function name), arguments
    (a stringified JSON object), message_id, created_timestamp.
  tool_call_result: role, tool_call_id, content (the result), successful (bool),
    message_id, created_timestamp. No tool name on the result — it must be
    correlated back to its invocation via tool_call_id.

Used by both runner.py (to enrich a just-completed live conversation with real
tool-call data straight from Retell) and paste_mode.py (to parse a manually
pasted/fetched transcript — including hand-pasted data using the older
"tool_name" key, kept as a fallback alias for backward compatibility).
Returns plain {"role", "content"} dicts rather than TurnRecord objects so this
module has no dependency on runner.py (avoids a circular import, since
paste_mode.py already imports TurnRecord from runner).
"""
from __future__ import annotations

import json

# Roles from real transcripts / conversation_thread JSON that count as the patient side.
PATIENT_ROLES = {"user", "patient"}
# Roles that count as the agent side.
AGENT_ROLES = {"agent", "ai", "practice", "assistant"}


def retell_items_to_turns(items: list) -> list[dict]:
    """Convert a list of transcript items (Retell's message_with_tool_calls, or
    the equivalent shape pasted by hand) into plain {"role", "content"} dicts
    with role in {"patient", "agent", "tool"}. Unrecognised or malformed items
    are skipped rather than raising.
    """
    # tool_call_result entries carry no tool name of their own — build a
    # tool_call_id -> tool name map from the invocations in this same list first.
    invocation_names: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and str(item.get("role", "")).lower() in ("tool_call", "tool_call_invocation"):
            tool_call_id = item.get("tool_call_id")
            tool_name = item.get("name") or item.get("tool_name")
            if tool_call_id and tool_name:
                invocation_names[tool_call_id] = tool_name

    turns: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or "role" not in item:
            continue
        role = str(item["role"]).lower()

        if role in ("tool_call", "tool_call_invocation"):
            # "name" is Retell's real field; "tool_name" kept as a fallback
            # alias for hand-pasted/older-shaped data.
            tool_name = item.get("name") or item.get("tool_name") or "unknown_tool"
            raw_args = item.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except (TypeError, ValueError):
                    arguments = raw_args
            else:
                arguments = raw_args
            turns.append({"role": "tool", "content": f"CALLED {tool_name}({arguments})"})
            continue

        if role in ("tool_result", "tool_call_result"):
            tool_call_id = item.get("tool_call_id")
            tool_name = item.get("tool_name") or invocation_names.get(tool_call_id, "unknown_tool")
            successful = item.get("successful")
            result_content = item.get("content", "")
            turns.append({
                "role": "tool",
                "content": f"RESULT for {tool_name} | successful={successful} | {result_content}",
            })
            continue

        if "content" not in item:
            continue
        content = str(item["content"])
        if role in PATIENT_ROLES:
            turns.append({"role": "patient", "content": content})
        elif role in AGENT_ROLES:
            turns.append({"role": "agent", "content": content})
        # else: unrecognised role (e.g. node_transition/state_transition) — skip

    return turns


def unwrap_transcript_payload(data):
    """Unwrap the /engage/chat-transcript response envelope down to the raw
    transcript list, however it's nested. Returns the input unchanged if it's
    already a plain list, or None if no transcript list could be found.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "transcript" in data:
            return data["transcript"]
        if isinstance(data.get("data"), dict) and "transcript" in data["data"]:
            return data["data"]["transcript"]
    return None
