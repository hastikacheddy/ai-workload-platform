"""FastAPI application factory for the Platform Copilot."""

from fastapi import FastAPI

from platform_copilot import __version__
from platform_copilot.routers import ask, health, search


def create_app() -> FastAPI:
    app = FastAPI(
        title="Platform Copilot",
        description="Agentic RAG copilot for AI/ML platform operations",
        version=__version__,
    )
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    return app


app = create_app()
