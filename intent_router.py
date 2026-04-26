"""Intent routing for terminal user queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    EMPTY = "empty"
    EXIT = "exit"
    TOTAL_COUNT = "total_count"
    SURNAME_COUNT = "surname_count"
    PERSON_DETAIL = "person_detail"
    RANDOM_PROFILE = "random_profile"
    KB_QA = "kb_qa"
    GENERAL_CHAT = "general_chat"


@dataclass
class IntentResult:
    intent: IntentType
    payload: str = ""


def detect_intent(question: str) -> IntentResult:
    q = (question or "").strip()
    if not q:
        return IntentResult(intent=IntentType.EMPTY)
    if q.lower() in {"exit", "quit", "退出"}:
        return IntentResult(intent=IntentType.EXIT)

    if _is_random_profile_intent(q):
        return IntentResult(intent=IntentType.RANDOM_PROFILE)

    surname = _parse_surname_count(q)
    if surname:
        return IntentResult(intent=IntentType.SURNAME_COUNT, payload=surname)

    if _is_total_count_intent(q):
        return IntentResult(intent=IntentType.TOTAL_COUNT)

    person_name = _parse_person_detail_name(q)
    if person_name:
        return IntentResult(intent=IntentType.PERSON_DETAIL, payload=person_name)

    if _is_knowledge_related(q):
        return IntentResult(intent=IntentType.KB_QA)

    return IntentResult(intent=IntentType.GENERAL_CHAT)


def _parse_surname_count(question: str) -> str | None:
    patterns = [
        r"多少条.*姓(?P<surname>[\u4e00-\u9fff])",
        r"多少人.*姓(?P<surname>[\u4e00-\u9fff])",
        r"姓名.*姓(?P<surname>[\u4e00-\u9fff]).*多少",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group("surname")
    return None


def _is_total_count_intent(question: str) -> bool:
    patterns = [
        r"(知识库|数据库).*(多少条|多少数据|总数|总共有)",
        r"(多少条|多少数据).*(知识库|数据库)",
        r"总共有.*(多少条|多少数据)",
    ]
    return any(re.search(pattern, question) for pattern in patterns)


def _parse_person_detail_name(question: str) -> str | None:
    patterns = [
        r"给我(?P<name>[\u4e00-\u9fff]{2,4})(?:所有)?的?(?:具体)?(?:信息|资料|详情)",
        r"(?P<name>[\u4e00-\u9fff]{2,4})的(具体信息|信息|资料|详情)",
        r"查询(?P<name>[\u4e00-\u9fff]{2,4})",
        r"(?P<name>[\u4e00-\u9fff]{2,4})是谁",
    ]
    invalid_names = {
        "库内的人",
        "知识库内",
        "数据库内",
        "某个人",
        "一个人",
        "这个人",
        "那个人",
    }
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            name = match.group("name")
            # Trim common trailing filler words mistakenly captured as name suffix.
            for suffix in ("所有", "全部", "相关", "详细", "具体"):
                if name.endswith(suffix) and len(name) > len(suffix):
                    name = name[: -len(suffix)]
            if name in invalid_names:
                return None
            return name
    return None


def _is_random_profile_intent(question: str) -> bool:
    patterns = [
        r"随便.*(一个|一位).*(人|用户).*(信息|资料|详情)",
        r"随机.*(一个|一位).*(人|用户).*(信息|资料|详情)",
        r"给我.*(一个|一位).*(人|用户).*(信息|资料|详情)",
    ]
    return any(re.search(pattern, question) for pattern in patterns)


def _is_knowledge_related(question: str) -> bool:
    kb_keywords = [
        "数据库",
        "知识库",
        "资料",
        "用户",
        "姓名",
        "年龄",
        "城市",
        "职业",
        "学历",
        "爱好",
        "统计",
        "有多少",
        "哪些人",
    ]
    return any(keyword in question for keyword in kb_keywords)
