"""Telegram bot — ask the copilot from your phone during an incident.

INTEGRATION-ONLY: needs TELEGRAM_BOT_TOKEN and network access, so it is not
exercised by the offline test suite. The reply formatting it uses
(``render_answer``) is unit-tested.

Needs the `bot` extra:  pip install ".[bot]"
Run:                    python scripts/telegram_bot.py
"""

from __future__ import annotations

import asyncio

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from platform_copilot.config import get_settings
from platform_copilot.dependencies import get_pipeline
from platform_copilot.services.rag.render import render_answer


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or not (update.message.text or "").strip():
        return
    question = update.message.text.strip()

    # The pipeline is synchronous and LLM-bound; run it off the event loop so the
    # bot stays responsive while a slow local model generates.
    answer = await asyncio.to_thread(get_pipeline().answer, question)
    await update.message.reply_text(render_answer(answer))


def main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set — create a bot with @BotFather first.")

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()


if __name__ == "__main__":
    main()
