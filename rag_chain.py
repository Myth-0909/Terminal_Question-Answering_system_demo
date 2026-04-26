"""RAG chain with multi-turn memory and streaming."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Callable

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import AppConfig, SYSTEM_PROMPT
from intent_router import IntentResult, IntentType


@dataclass
class ChatTurnResult:
    answer: str
    source_docs: list[Document]
    used_knowledge_base: bool


class RagChatEngine:
    def __init__(self, config: AppConfig, retrieve_fn: Callable[[str, int], list[Document]]) -> None:
        self.config = config
        self.history = ChatMessageHistory()
        self.retrieve_fn = retrieve_fn
        self.llm = ChatOpenAI(
            api_key=self.config.deepseek_api_key,
            model=self.config.deepseek_model,
            base_url=self.config.deepseek_base_url,
            temperature=0.4,
            max_retries=self.config.max_retries,
            timeout=self.config.request_timeout,
        )
        self.summary = ""

    def _format_history(self) -> str:
        recent = self.history.messages[-self.config.max_memory_turns * 2 :]
        if not recent and not self.summary:
            return "（无）"
        lines: list[str] = []
        if self.summary:
            lines.append(f"历史摘要: {self.summary}")
        for msg in recent:
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _update_summary(self, question: str, answer: str) -> None:
        summary_prompt = (
            "请把以下对话更新为简短摘要（不超过120字），保留用户偏好、约束、未完成任务。\n\n"
            f"已有摘要：{self.summary or '（无）'}\n"
            f"用户：{question}\n"
            f"助手：{answer}\n\n"
            "仅输出更新后的摘要正文。"
        )
        try:
            resp = self.llm.invoke(summary_prompt)
            new_summary = (resp.content or "").strip()
            if new_summary:
                self.summary = new_summary
        except Exception:
            # Summary failure should not break main Q&A flow.
            pass

    def _build_prompt(self, question: str, docs: list[Document]) -> str:
        context = "\n".join([f"- {doc.page_content}" for doc in docs]) or "（未检索到相关资料）"
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"【历史对话】\n{self._format_history()}\n\n"
            f"【检索到的人物资料】\n{context}\n\n"
            f"【用户当前问题】\n{question}\n\n"
            "请结合资料优先回答，并保持简洁。"
        )

    def ask(
        self,
        question: str,
        intent: IntentType,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatTurnResult:
        use_kb = intent == IntentType.KB_QA
        docs = self.retrieve_fn(question, self.config.top_k) if use_kb else []
        prompt = self._build_prompt(question, docs)
        answer_parts: list[str] = []

        for chunk in self.llm.stream(prompt):
            token = chunk.content or ""
            if not token:
                continue
            answer_parts.append(token)
            if on_token:
                on_token(token)

        answer = "".join(answer_parts).strip()
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        self._update_summary(question=question, answer=answer)
        return ChatTurnResult(
            answer=answer,
            source_docs=docs,
            used_knowledge_base=use_kb,
        )

    def refine_intent(self, question: str, current_intent: IntentResult) -> IntentResult:
        """LLM fallback classifier for ambiguous user requests."""
        if current_intent.intent not in {IntentType.GENERAL_CHAT, IntentType.KB_QA}:
            return current_intent
        prompt = (
            "你是意图分类器。根据用户问题，输出 JSON："
            '{"intent":"GENERAL_CHAT|KB_QA|TOTAL_COUNT|SURNAME_COUNT|PERSON_DETAIL|RANDOM_PROFILE","payload":"..."}'
            "。payload仅在 SURNAME_COUNT(姓氏单字) 或 PERSON_DETAIL(2-4字中文姓名) 时填写，其余为空字符串。\n"
            f"用户问题：{question}\n"
            "只输出 JSON。"
        )
        try:
            resp = self.llm.invoke(prompt)
            text = (resp.content or "").strip()
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return current_intent
            data = json.loads(match.group(0))
            intent_str = str(data.get("intent", "")).strip().upper()
            payload = str(data.get("payload", "")).strip()
            mapping = {
                "GENERAL_CHAT": IntentType.GENERAL_CHAT,
                "KB_QA": IntentType.KB_QA,
                "TOTAL_COUNT": IntentType.TOTAL_COUNT,
                "SURNAME_COUNT": IntentType.SURNAME_COUNT,
                "PERSON_DETAIL": IntentType.PERSON_DETAIL,
                "RANDOM_PROFILE": IntentType.RANDOM_PROFILE,
            }
            if intent_str not in mapping:
                return current_intent
            return IntentResult(intent=mapping[intent_str], payload=payload)
        except Exception:
            return current_intent

    def get_serializable_history(self) -> list[dict]:
        serialized: list[dict] = []
        for msg in self.history.messages:
            serialized.append(
                {
                    "type": msg.type,
                    "content": msg.content,
                }
            )
        if self.summary:
            serialized.append({"type": "summary", "content": self.summary})
        return serialized
