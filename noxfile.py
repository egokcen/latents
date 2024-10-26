"""Configure nox sessions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nox

ALL_PYTHONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]

nox.options.sessions = ["lint", "tests"]

DIR = Path(__file__).parent.resolve()


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Lint via pre-commit."""
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run the test suite and compute coverage."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--cov=latents", "--cov-report=xml", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "--serve" to serve."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    args = parser.parse_args(session.posargs)

    session.install("-e", ".[doc]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", "source", "build")

    if args.serve:
        print("Launching docs at http://127.0.0.1:8000/ - use Ctrl-C to quit")
        session.run(
            "python", "-m", "http.server", "-b", "127.0.0.1", "8000", "-d", "build/html"
        )


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("dist")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
