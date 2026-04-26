"""Entry point for Milvus + LangChain terminal chat demo."""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

from config import load_config
from data_generator import dump_profiles_to_json, generate_mock_profiles
from intent_router import IntentType, detect_intent
from rag_chain import RagChatEngine
from terminal_ui import TerminalUI
from vector_store import MilvusStoreManager


def try_answer_followup(question: str, profile: dict | None) -> str | None:
    """Answer short follow-up questions from the selected profile context."""
    if not profile:
        return None
    q = question.strip()
    if not q:
        return None
    if any(token in q for token in ["男", "女", "性别", "男生", "女生"]):
        return f"{profile.get('name', 'Ta')}的性别是：{profile.get('gender', '未知')}。"
    if any(token in q for token in ["多大", "几岁", "年龄"]):
        return f"{profile.get('name', 'Ta')}今年 {profile.get('age', '未知')} 岁。"
    if any(token in q for token in ["哪里", "哪儿", "城市", "住在", "来自"]):
        return f"{profile.get('name', 'Ta')}所在城市是：{profile.get('city', '未知')}。"
    if any(token in q for token in ["做什么", "职业", "岗位", "工作"]):
        return f"{profile.get('name', 'Ta')}的职业是：{profile.get('occupation', '未知')}。"
    if any(token in q for token in ["爱好", "喜欢什么"]):
        hobbies = ", ".join(profile.get("hobbies", [])) or "未知"
        return f"{profile.get('name', 'Ta')}的爱好有：{hobbies}。"
    if any(token in q for token in ["学历", "读书"]):
        return f"{profile.get('name', 'Ta')}的学历是：{profile.get('education', '未知')}。"
    if any(token in q for token in ["感情", "单身", "恋爱", "婚"]):
        return f"{profile.get('name', 'Ta')}的感情状态是：{profile.get('relationship_status', '未知')}。"
    return None


def format_profile_detail(profile: dict) -> str:
    hobbies = ", ".join(profile.get("hobbies", []))
    return (
        f"姓名：{profile.get('name', '未知')}\n"
        f"年龄：{profile.get('age', '未知')}，性别：{profile.get('gender', '未知')}\n"
        f"城市：{profile.get('city', '未知')}，职业：{profile.get('occupation', '未知')}，学历：{profile.get('education', '未知')}\n"
        f"爱好：{hobbies or '未知'}\n"
        f"性格：{profile.get('personality', '未知')}，感情状态：{profile.get('relationship_status', '未知')}\n"
        f"喜欢的食物：{profile.get('favorite_food', '未知')}，喜欢的音乐：{profile.get('favorite_music', '未知')}\n"
        f"简介：{profile.get('bio', '未知')}"
    )


def persist_history(engine: RagChatEngine, history_dir: str) -> str:
    from pathlib import Path

    path = Path(history_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    target = path / filename
    payload = {
        "created_at": datetime.now().isoformat(),
        "messages": engine.get_serializable_history(),
    }
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(target)


def build_or_load_index(ui: TerminalUI, manager: MilvusStoreManager, count: int, batch_size: int) -> None:
    ui.show_status(f"准备生成数据集并校验向量库（目标 {count} 条）...")
    profiles = generate_mock_profiles(count=count)
    dataset_path = Path(manager.config.dataset_json_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dump_profiles_to_json(str(dataset_path), profiles)
    ui.show_status(f"数据集已写入：{dataset_path}")

    current_count = manager.get_row_count() if manager.has_indexed_data() else 0
    if current_count == count:
        ui.show_status(f"检测到向量索引已满足目标条数（{current_count}），跳过重建。")
        return

    if current_count > 0 and current_count != count:
        ui.show_status(f"检测到历史索引条数为 {current_count}，将重建为 {count} 条...")
        manager.reset_collection()

    def step(progress_update):
        manager.ingest_profiles(
            profiles=profiles,
            batch_size=batch_size,
            progress_cb=progress_update,
        )

    ui.run_index_progress(total=count, step_callable=step)
    ui.show_status(f"向量索引建立完成，当前条数：{manager.get_row_count()}。")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="milvus_lite")
    logging.getLogger("pymilvus").setLevel(logging.ERROR)
    logging.getLogger("pymilvus.client").setLevel(logging.ERROR)
    logging.getLogger("pymilvus.milvus_client").setLevel(logging.ERROR)
    ui = TerminalUI()
    try:
        config = load_config()
    except Exception as exc:
        ui.show_error(str(exc))
        return
    store_manager = MilvusStoreManager(config)
    build_or_load_index(
        ui=ui,
        manager=store_manager,
        count=config.data_count,
        batch_size=config.batch_size,
    )

    engine = RagChatEngine(
        config=config,
        retrieve_fn=store_manager.similarity_search,
    )
    ui.show_welcome()

    round_count = 0
    selected_profile: dict | None = None
    while True:
        try:
            question = ui.ask_user().strip()
        except KeyboardInterrupt:
            ui.show_status("已取消本轮输入，你可以继续提问，或输入 退出 结束会话。")
            continue
        intent_result = detect_intent(question)

        if intent_result.intent == IntentType.EXIT:
            history_file = persist_history(engine, config.session_history_dir)
            ui.show_goodbye(round_count=round_count, history_file=history_file)
            break
        if intent_result.intent == IntentType.EMPTY:
            ui.show_status("请输入问题后再发送。")
            continue

        # `refine_intent` may trigger an extra LLM call and cause a short blank wait.
        # Show "thinking" immediately so users get responsive feedback after pressing Enter.
        if intent_result.intent in {IntentType.GENERAL_CHAT, IntentType.KB_QA}:
            with ui.thinking("小咪正在理解你的问题..."):
                intent_result = engine.refine_intent(question, intent_result)

        followup_answer = try_answer_followup(question, selected_profile)
        if followup_answer:
            ui.show_answer_header()
            ui.show_answer_text(followup_answer)
            continue

        if intent_result.intent == IntentType.SURNAME_COUNT:
            with ui.thinking("小咪正在统计姓氏人数..."):
                count = store_manager.count_profiles_by_surname(intent_result.payload)
            ui.show_answer_header()
            ui.show_answer_text(f"数据库中姓名姓「{intent_result.payload}」的数据共有 {count} 条。")
            continue

        if intent_result.intent == IntentType.TOTAL_COUNT:
            with ui.thinking("小咪正在统计知识库总量..."):
                total = store_manager.get_row_count()
            ui.show_answer_header()
            ui.show_answer_text(f"当前知识库中共有 {total} 条数据。")
            continue

        if intent_result.intent == IntentType.PERSON_DETAIL:
            with ui.thinking("小咪正在查找该用户资料..."):
                profiles = store_manager.find_profiles_by_name(intent_result.payload, max_results=10)
            ui.show_answer_header()
            if not profiles:
                ui.show_answer_text(f"数据库中暂未找到姓名为「{intent_result.payload}」的用户信息。")
            else:
                if len(profiles) == 1:
                    selected_profile = profiles[0]
                    ui.show_answer_text(format_profile_detail(profiles[0]))
                else:
                    ui.show_candidates(profiles[:5])
                    ui.show_status("请输入序号选择具体用户（输入 q 取消）。")
                    choice = ui.ask_choice(max_index=min(5, len(profiles)))
                    if choice is None:
                        ui.show_answer_text("已取消选择。")
                    else:
                        selected_profile = profiles[choice - 1]
                        ui.show_answer_text(format_profile_detail(profiles[choice - 1]))
            continue

        if intent_result.intent == IntentType.RANDOM_PROFILE:
            with ui.thinking("小咪正在随机抽取一个用户资料..."):
                profiles = store_manager.get_random_profiles(max_results=1)
            ui.show_answer_header()
            if not profiles:
                ui.show_answer_text("当前知识库中暂无可用用户资料。")
            else:
                selected_profile = profiles[0]
                ui.show_answer_text(format_profile_detail(profiles[0]))
            continue

        round_count += 1
        ui.show_answer_header()
        source_docs: list = []
        used_kb = {"value": False}

        def call_with_stream(on_token):
            result = engine.ask(
                question=question,
                intent=intent_result.intent,
                on_token=on_token,
            )
            source_docs.extend(result.source_docs)
            used_kb["value"] = result.used_knowledge_base

        try:
            ui.stream_typing(call_with_stream)
            if used_kb["value"]:
                ui.show_sources(source_docs)
                ui.show_status("本轮模式：知识库检索问答")
            else:
                ui.show_status("本轮模式：通用问答")
        except Exception as exc:
            ui.show_error(
                "抱歉，这一轮回答遇到一点小问题，已保留会话。"
                f"可重试一次，若仍失败请重启程序。详情：{exc}"
            )


if __name__ == "__main__":
    main()
