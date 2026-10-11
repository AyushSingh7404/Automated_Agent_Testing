"""
runner.py
──────────────────────────────────────────
Drives a single test case conversation end-to-end.

Flow per test case:
  1. Optionally attach a conversation_thread (for follow-up scenarios)
  2. GPT-4o-mini simulates the patient turn by turn
  3. Each patient message is sent to the forward-to-agent API
  4. If agent_response is empty (agent still processing tool calls),
     wait EMPTY_REPLY_WAIT seconds and retry up to EMPTY_REPLY_RETRIES times
  5. After max_turns OR when agent sends a closing farewell OR patient signals end -> wrap up
  6. Send a friendly goodbye then end_conversation=True
  7. Full transcript is returned
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    API_HEADERS,
    CHAT_TRANSCRIPT_FETCH_DELAY,
    CHAT_TRANSCRIPT_FETCH_RETRIES,
    CONVERSATION_ID,
    EMPTY_REPLY_RETRIES,
    EMPTY_REPLY_WAIT,
    FORWARD_API_URL,
    LOCATION_ID,
    OPENAI_API_KEY,
    ORGANIZATION_ID,
    PATIENT_MODEL,
    RETELL_API_KEY,
    RETELL_GET_CHAT_URL,
    REQUEST_TIMEOUT,
    TURN_DELAY_SECONDS,
)
from retell_transcript import retell_items_to_turns
from test_cases.cases import TestCase

logger = logging.getLogger(__name__)

# Keywords that signal the patient wants to end the conversation
_END_SIGNALS = {
    "bye", "goodbye", "that's all", "that is all",
    "thank you, bye", "thanks, bye", "that's it", "that is it",
}

# ── Generic stall detector (case-agnostic safety net) ──────────────────────────
# If the agent's recent turns keep circling the same topic, the simulated
# patient is likely stuck giving non-committal answers instead of resolving
# whatever is being asked. Word-overlap similarity on the agent's own messages
# is used rather than any topic-specific keyword list, so this applies to any
# future test case with a similar loop -- not just TC_011-014, which already
# have explicit deterministic personas of their own.
_STALL_WINDOW = 5              # look at the last this many agent turns
_STALL_MIN_REPEATS = 3         # need at least this many highly-similar agent turns
_STALL_SIMILARITY_THRESHOLD = 0.45
_STALLWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "and", "or", "but", "with", "you", "your", "i", "me", "my",
    "please", "can", "could", "would", "do", "does", "did", "it", "that", "this",
    "have", "has", "had", "will", "we", "us", "our", "so", "if", "not", "no",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in _STALLWORDS and len(w) > 2}


def _message_similarity(a: str, b: str) -> float:
    """Jaccard word overlap between two messages, ignoring common stopwords."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _detect_agent_stall(history: list[dict]) -> bool:
    """
    True when the agent's most recent message closely resembles enough of its
    other recent messages to suggest it's re-asking about the same thing
    without the conversation moving on.
    """
    agent_msgs = [h["content"] for h in history if h.get("role") == "assistant"]
    recent = agent_msgs[-_STALL_WINDOW:]
    if len(recent) < _STALL_MIN_REPEATS:
        return False

    latest = recent[-1]
    similar_count = sum(
        1 for prior in recent[:-1]
        if _message_similarity(latest, prior) >= _STALL_SIMILARITY_THRESHOLD
    )
    return similar_count >= (_STALL_MIN_REPEATS - 1)


@dataclass
class TurnRecord:
    role: str    # "patient" | "agent" | "tool" (tool calls/results — paste-mode transcripts only)
    content: str


@dataclass
class ConversationResult:
    test_case_id: str
    chat_id: Optional[str]
    transcript: list[TurnRecord] = field(default_factory=list)
    ended_gracefully: bool = False
    error: Optional[str] = None
    infra_error: bool = False  # True when an API timeout/connection error cut the conversation short
    tool_calls_included: bool = False  # True when the transcript was enriched with real tool call data

    def to_text(self) -> str:
        """Plain text transcript for the evaluator."""
        lines = []
        for t in self.transcript:
            if t.role == "patient":
                label = "PATIENT"
            elif t.role == "tool":
                label = "TOOL   "
            else:
                label = "AGENT  "
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)


class ConversationRunner:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=PATIENT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            timeout=REQUEST_TIMEOUT,   # without this, a stalled OpenAI call can hang for minutes with no error
            max_retries=2,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, test_case: TestCase) -> ConversationResult:
        logger.info("▶  Starting test case %s — %s", test_case.id, test_case.name)

        result = ConversationResult(test_case_id=test_case.id, chat_id=None)
        chat_id: Optional[str] = None
        history: list[dict] = []  # {role, content} kept for GPT patient context

        for turn in range(test_case.max_turns):
            # ── Patient generates next message ────────────────────────────────
            patient_msg = self._generate_patient_message(
                test_case.patient_persona,
                history,
                turn,
            )
            logger.info("  [Turn %d] PATIENT: %s", turn + 1, patient_msg)
            result.transcript.append(TurnRecord(role="patient", content=patient_msg))
            history.append({"role": "user", "content": patient_msg})

            # ── Send to agent API ──────────────────────────────────────────────────
            # On turn 0 of a follow-up test the agent loads continuity memory and
            # intentionally returns agent_response: null — it is waiting for the
            # patient's first real input. Do NOT retry on empty here; just capture
            # the chat_id and move straight to the next patient turn.
            is_followup_first_turn = (turn == 0 and test_case.conversation_thread is not None)

            if is_followup_first_turn:
                agent_reply, new_chat_id, api_error = self._call_api(
                    message=patient_msg,
                    chat_id=chat_id,
                    conversation_thread=test_case.conversation_thread,
                )
            else:
                agent_reply, new_chat_id, api_error = self._call_api_with_retry(
                    message=patient_msg,
                    chat_id=chat_id,
                    conversation_thread=None,
                )

            if api_error:
                result.error = api_error
                result.infra_error = True
                logger.error("  API error on turn %d: %s", turn + 1, api_error)
                break

            if new_chat_id and not chat_id:
                chat_id = new_chat_id
                result.chat_id = chat_id
                logger.info("  Chat ID assigned: %s", chat_id)

            # On the follow-up first turn a null response is expected — skip
            # transcript logging and proceed immediately to the next patient turn.
            if is_followup_first_turn and not agent_reply:
                logger.info("  [Turn 1] AGENT  : (null — expected on follow-up turn 0, proceeding)")
                continue

            logger.info("  [Turn %d] AGENT  : %s", turn + 1, agent_reply)
            result.transcript.append(TurnRecord(role="agent", content=agent_reply))
            history.append({"role": "assistant", "content": agent_reply})

            # ── Check if patient signalled end ────────────────────────────────
            if self._patient_wants_to_end(patient_msg):
                logger.info("  Patient signalled end -- wrapping up.")
                break

            time.sleep(TURN_DELAY_SECONDS)

        # ── Graceful goodbye + end_conversation flag ──────────────────────────
        self._end_conversation(result, chat_id)

        # ── Enrich with the real chat transcript (tool calls included) ────────
        # Best-effort: if this fails or looks incomplete, keep the transcript
        # already built from the live API responses above.
        if chat_id:
            enriched = self._fetch_enriched_transcript(chat_id, local_turn_count=len(result.transcript))
            if enriched is not None:
                result.transcript = enriched
                result.tool_calls_included = True
                logger.info("  Transcript enriched with real tool-call data (%d turns).", len(enriched))

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_patient_message(
        self,
        persona: str,
        history: list[dict],
        turn: int,
    ) -> str:
        system_prompt = (
            f"{persona}\n\n"
            "You are simulating this patient in a live chat conversation with a dental clinic AI agent. "
            "Generate ONLY the patient's next message -- no labels, no explanations, no meta-commentary. "
            "Keep it natural and brief, as a real patient would type in a chat. "
            "When asked for your phone number, always provide it. "
            "When asked for your date of birth, always provide it. "
            "If you feel the conversation goal has been achieved or you want to end, "
            "say 'bye' or 'that is all, thank you'."
        )

        # Generic fallback for any test case (not just TC_011-014, which already
        # have their own deterministic personas): if the agent's recent turns
        # keep circling the same topic, nudge the persona to stop stalling
        # instead of letting the conversation loop until max_turns is exhausted.
        if turn > 0 and _detect_agent_stall(history):
            system_prompt += (
                "\n\nIMPORTANT: The agent has asked about the same thing several times in a row "
                "without getting a resolving answer from you. Stop giving non-committal or vague "
                "responses on this topic now -- either (a) commit to one concrete, plausible answer "
                "and stay consistent with it, or (b) explicitly ask the agent to note this for the "
                "team / escalate it -- whichever better fits your persona instructions above."
            )

        messages = [SystemMessage(content=system_prompt)]
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=f"[You previously said]: {h['content']}"))
            else:
                messages.append(HumanMessage(content=f"[Agent said]: {h['content']}"))

        if turn == 0:
            messages.append(HumanMessage(content="Start the conversation now with your opening message."))
        else:
            messages.append(HumanMessage(content="Continue the conversation. What do you say next?"))

        response = self._llm.invoke(messages)
        return response.content.strip()

    def _call_api_with_retry(
        self,
        message: Optional[str],
        chat_id: Optional[str],
        conversation_thread: Optional[list],
        end_conversation: bool = False,
    ) -> tuple[str, Optional[str], Optional[str]]:
        """
        Wraps _call_api with empty-response retry logic.

        The agent may still be resolving tool calls internally when the HTTP
        response arrives, resulting in an empty agent_response. We wait
        EMPTY_REPLY_WAIT seconds and retry the same message up to
        EMPTY_REPLY_RETRIES times before giving up.

        Returns: (agent_reply, chat_id, error_message)
        """
        agent_reply, new_chat_id, api_error = self._call_api(
            message=message,
            chat_id=chat_id,
            conversation_thread=conversation_thread,
            end_conversation=end_conversation,
        )

        if api_error:
            return agent_reply, new_chat_id, api_error

        for attempt in range(1, EMPTY_REPLY_RETRIES + 1):
            if agent_reply.strip():
                break

            logger.info(
                "  Empty reply -- waiting %.1fs before retry %d/%d ...",
                EMPTY_REPLY_WAIT,
                attempt,
                EMPTY_REPLY_RETRIES,
            )
            time.sleep(EMPTY_REPLY_WAIT)

            retry_reply, retry_chat_id, api_error = self._call_api(
                message=message,
                chat_id=new_chat_id or chat_id,
                conversation_thread=None,   # thread was already sent on first call
                end_conversation=end_conversation,
            )
            if retry_chat_id and not new_chat_id:
                new_chat_id = retry_chat_id
            if api_error:
                return retry_reply, new_chat_id, api_error

            agent_reply = retry_reply

        return agent_reply, new_chat_id, None

    def _call_api(
        self,
        message: str,
        chat_id: Optional[str],
        conversation_thread: Optional[list],
        end_conversation: bool = False,
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Low-level API call. Returns: (agent_reply, chat_id, error_message).
        When end_conversation=True and message=None, the payload contains no
        message field — matching the explicit end_conversation payload contract.
        """
        payload: dict = {
            "location_id": LOCATION_ID,
            "organization_id": ORGANIZATION_ID,
            "conversation_id": CONVERSATION_ID,
            "end_conversation": end_conversation,
        }
        if message is not None:
            payload["message"] = message
        if chat_id:
            payload["chat_id"] = chat_id
        if conversation_thread and not chat_id:
            payload["conversation_thread"] = conversation_thread

        try:
            resp = requests.post(
                FORWARD_API_URL,
                headers=API_HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.Timeout:
            return "", None, "Request timed out"
        except Exception as exc:
            return "", None, str(exc)

        if resp.status_code != 200:
            return "", None, f"HTTP {resp.status_code}: {resp.text[:300]}"

        try:
            data = resp.json()
        except Exception:
            return "", None, "Invalid JSON response"

        if data.get("status") != "success":
            return "", None, f"API error: {data.get('message', 'unknown')}"

        inner = data.get("data", {})
        new_chat_id = inner.get("chat_id")
        agent_reply = inner.get("agent_response", "")
        return agent_reply, new_chat_id, None

    def _end_conversation(self, result: ConversationResult, chat_id: Optional[str]) -> None:
        """Patient says goodbye, then send end_conversation payload to explicitly close the chat.
        The agent never closes the chat itself, so this must always be called.
        End payload: location_id + organization_id + chat_id + end_conversation: true (no message field).
        """
        if not chat_id:
            logger.warning("  No chat_id -- skipping end_conversation call.")
            result.ended_gracefully = False
            return

        # Step 1: patient says bye
        goodbye_msg = "bye, thank you for your help!"
        agent_reply, _, error = self._call_api(
            message=goodbye_msg,
            chat_id=chat_id,
            conversation_thread=None,
        )
        if not error and agent_reply:
            result.transcript.append(TurnRecord(role="patient", content=goodbye_msg))
            result.transcript.append(TurnRecord(role="agent", content=agent_reply))
            logger.info("  Goodbye exchange done.")

        time.sleep(TURN_DELAY_SECONDS)

        # Step 2: explicitly close the chat -- no message field in this payload
        self._call_api(
            message=None,
            chat_id=chat_id,
            conversation_thread=None,
            end_conversation=True,
        )
        result.ended_gracefully = True
        logger.info("  Conversation ended cleanly for %s.", result.test_case_id)

    def _fetch_enriched_transcript(
        self,
        chat_id: str,
        local_turn_count: int,
    ) -> Optional[list[TurnRecord]]:
        """
        Best-effort fetch of the real chat transcript (tool calls included)
        straight from Retell's own GET /get-chat/{chat_id} — our own backend's
        /engage/chat-transcript deliberately strips tool-call entries, so it is
        not used here.

        Retell's post-chat pipeline (it also generates chat_analysis server-side)
        can take a while to finish attaching message_with_tool_calls after the
        chat ends -- a single fixed wait isn't reliable, so this retries up to
        CHAT_TRANSCRIPT_FETCH_RETRIES times, sleeping CHAT_TRANSCRIPT_FETCH_DELAY
        between attempts, before giving up.

        Returns None (keep the locally-built transcript) if every attempt fails
        outright, or if the data still isn't ready after all retries. Returns
        immediately without retrying on a definitive failure (401/403/404) --
        those won't fix themselves by waiting.
        """
        last_reason = "unknown"

        for attempt in range(1, CHAT_TRANSCRIPT_FETCH_RETRIES + 1):
            time.sleep(CHAT_TRANSCRIPT_FETCH_DELAY)

            try:
                resp = requests.get(
                    f"{RETELL_GET_CHAT_URL}/{chat_id}",
                    headers={"Authorization": f"Bearer {RETELL_API_KEY}"},
                    timeout=REQUEST_TIMEOUT,
                )
            except Exception as exc:
                last_reason = f"request error: {exc}"
                logger.warning(
                    "  [attempt %d/%d] Could not fetch enriched transcript for %s: %s",
                    attempt, CHAT_TRANSCRIPT_FETCH_RETRIES, chat_id, exc,
                )
                continue

            # Retell returns the chat object directly (no {status, data} envelope).
            if resp.status_code in (401, 403, 404):
                # Not a timing issue -- auth or a genuinely missing chat_id won't
                # resolve itself by waiting, so stop retrying immediately.
                logger.warning(
                    "  Retell get-chat returned HTTP %d for %s (auth/not-found) -- "
                    "giving up, keeping locally-built transcript.",
                    resp.status_code, chat_id,
                )
                return None

            if resp.status_code != 200:
                last_reason = f"HTTP {resp.status_code}"
                logger.warning(
                    "  [attempt %d/%d] Retell get-chat fetch returned HTTP %d for %s -- retrying.",
                    attempt, CHAT_TRANSCRIPT_FETCH_RETRIES, resp.status_code, chat_id,
                )
                continue

            try:
                data = resp.json()
            except Exception:
                last_reason = "invalid JSON"
                logger.warning(
                    "  [attempt %d/%d] Retell get-chat response was not valid JSON for %s -- retrying.",
                    attempt, CHAT_TRANSCRIPT_FETCH_RETRIES, chat_id,
                )
                continue

            items = data.get("message_with_tool_calls", [])
            if not items:
                last_reason = "no messages yet"
                logger.info(
                    "  [attempt %d/%d] Retell get-chat has no messages yet for %s "
                    "(still finalizing post-chat processing) -- retrying.",
                    attempt, CHAT_TRANSCRIPT_FETCH_RETRIES, chat_id,
                )
                continue

            turns = [TurnRecord(role=t["role"], content=t["content"]) for t in retell_items_to_turns(items)]

            if len(turns) < local_turn_count:
                last_reason = f"only {len(turns)} turns (expected >= {local_turn_count})"
                logger.info(
                    "  [attempt %d/%d] Retell get-chat for %s returned fewer turns (%d) than "
                    "locally recorded (%d) -- likely still finalizing, retrying.",
                    attempt, CHAT_TRANSCRIPT_FETCH_RETRIES, chat_id, len(turns), local_turn_count,
                )
                continue

            return turns

        logger.warning(
            "  Gave up fetching enriched transcript for %s after %d attempts (last reason: %s) -- "
            "keeping locally-built transcript.",
            chat_id, CHAT_TRANSCRIPT_FETCH_RETRIES, last_reason,
        )
        return None

    @staticmethod
    def _patient_wants_to_end(message: str) -> bool:
        lower = message.lower().strip()
        return any(signal in lower for signal in _END_SIGNALS)
