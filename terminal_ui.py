"""Terminal UI helpers based on Rich."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn


@dataclass
class TypingBuffer:
    text: str = ""


class TerminalUI:
    def __init__(self) -> None:
        self.console = Console()
        self._history = InMemoryHistory()
        self._prompt_style = Style.from_dict(
            {
                "prompt": "bold ansicyan",
                "": "ansibrightcyan",
            }
        )

    def show_welcome(self) -> None:
        self.console.print(
            Panel.fit(
                "欢迎来到 [bold cyan]Milvus + LangChain 中文问答 Demo[/bold cyan]\n"
                "输入问题后按回车开始对话，输入 [bold yellow]exit[/bold yellow] / "
                "[bold yellow]quit[/bold yellow] / [bold yellow]退出[/bold yellow] 可结束会话。\n"
                "示例：知识库有多少条数据？ | 数据库里有多少条数据中的姓名是姓李的？ | 张三的具体信息",
                title="小咪助手",
                border_style="bright_blue",
            )
        )

    def ask_user(self) -> str:
        # prompt_toolkit has much better CJK IME + cursor/edit key support.
        try:
            if not sys.stdin.isatty():
                return input("\033[1;36m你:\033[0m ")
            return prompt(
                HTML("<prompt>你:</prompt> "),
                history=self._history,
                style=self._prompt_style,
            )
        except Exception:
            return input("\033[1;36m你:\033[0m ")

    def show_status(self, text: str) -> None:
        self.console.print(f"[dim]{text}[/dim]")

    def show_goodbye(self, round_count: int, history_file: str) -> None:
        self.console.print(
            Panel.fit(
                f"本次对话已结束，共 [bold]{round_count}[/bold] 轮。\n"
                f"会话记录已保存到：[dim]{history_file}[/dim]\n"
                "期待下次再聊呀~",
                title="再见啦",
                border_style="green",
            )
        )

    def show_error(self, text: str) -> None:
        self.console.print(f"[bold red]错误[/bold red]：{text}")

    def show_answer_header(self) -> None:
        self.console.print("[bold magenta]小咪[/bold magenta]: ", end="")

    def show_answer_text(self, text: str) -> None:
        self.console.print(f"[bright_magenta]{text}[/bright_magenta]")

    @contextmanager
    def thinking(self, text: str = "小咪正在思考..."):
        with self.console.status(f"[cyan]{text}[/cyan]", spinner="dots"):
            yield

    def stream_typing(self, stream_fn) -> str:
        buffer = TypingBuffer()
        with Live("[dim]小咪正在思考...[/dim]", console=self.console, refresh_per_second=40) as live:
            def on_token(token: str) -> None:
                buffer.text += token
                live.update(Markdown(buffer.text))

            stream_fn(on_token)
            if not buffer.text.strip():
                live.update("[dim]（本轮没有生成内容）[/dim]")
        self.console.print()
        return buffer.text

    def show_sources(self, docs: list) -> None:
        if not docs:
            self.console.print("[dim]未检索到参考数据。[/dim]")
            return
        self.console.print("[dim]参考数据：[/dim]")
        for idx, doc in enumerate(docs[:3], start=1):
            name = doc.metadata.get("name", "未知")
            city = doc.metadata.get("city", "未知城市")
            hobby = ", ".join(doc.metadata.get("hobbies", [])[:2])
            self.console.print(f"[dim]{idx}. {name} | {city} | {hobby}[/dim]")

    def show_candidates(self, profiles: list[dict]) -> None:
        self.console.print("[bold yellow]找到多个同名候选，请选择：[/bold yellow]")
        for idx, p in enumerate(profiles, start=1):
            self.console.print(
                f"[dim]{idx}. {p.get('name', '未知')} | {p.get('city', '未知')} | {p.get('occupation', '未知')}[/dim]"
            )

    def ask_choice(self, max_index: int) -> int | None:
        raw = self.ask_user().strip()
        if raw.lower() in {"exit", "quit", "退出", "q"}:
            return None
        if not raw.isdigit():
            return None
        value = int(raw)
        if 1 <= value <= max_index:
            return value
        return None

    def run_index_progress(self, total: int, step_callable) -> None:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("正在建立向量索引", total=total)

            def update(done: int, _total: int) -> None:
                progress.update(task, completed=done)

            step_callable(update)
            progress.update(task, completed=total)
        time.sleep(0.2)
