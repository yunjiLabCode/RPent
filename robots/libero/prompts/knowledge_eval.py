# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ``proven`` knowledge profile of the single-attempt LIBERO prompt.

``mem-proven`` (the default) is the unmodified evaluation prompt.  ``proven``
follows the paper's zero-shot setting (arXiv 2607.08448, LIBERO-Pro Goal, Table
5): no retrieval of Task Specific Memory (seed-0 recipe files) or Global Memory
(the memory library).  The PROVEN LEVERS and the guides stay; what goes is every
instruction to read memory or seed-0 reference files, and the file tools only
reach the guides and the run's own output directory (``proven_read_roots``).

Each edit names the exact official fragment it replaces and fails loudly when
that fragment is missing or ambiguous, so the profile cannot drift silently
when the official prompt changes.
"""

from __future__ import annotations

from pathlib import Path

from robots.libero.prompts import system as base
from rpent.context.prompt_utils import Numbered, PromptNode

KNOWLEDGE_PROFILES = ("mem-proven", "proven")

#: The official file tools read any path on the machine -- the memory library,
#: seed-0 references, and other episodes' audits and recipes included.  Under
#: ``proven`` the toolkit confines them to ``proven_read_roots``.
GUARDED_READ_TOOLS = ("read_text_file", "list_dir")
GUIDES_DIR = Path(__file__).resolve().parents[1] / "guides"


def proven_read_roots(output_dir: Path | str) -> tuple[Path, ...]:
    """Directories the ``proven`` file tools may read: guides and this run."""
    return (GUIDES_DIR, Path(output_dir).resolve())


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one prompt fragment: {old!r}")
    return text.replace(old, new)


def _between(text: str, start: str, end: str) -> tuple[str, str]:
    """Split ``text`` into what precedes ``start`` and what begins at ``end``."""
    for marker in (start, end):
        if text.count(marker) != 1:
            raise RuntimeError(f"expected exactly one prompt fragment: {marker!r}")
    head, rest = text.split(start, 1)
    if end not in rest:
        raise RuntimeError(f"{end!r} does not follow {start!r}")
    return head, end + rest.split(end, 1)[1]


_MECHANICS = "CRITICAL MECHANICS (cost many wasted attempts before they were nailed):"

# The paper's zero-shot setting: the agent "does not retrieve the target-setting
# Task Specific Memory or the corresponding Global Memory".  The PROVEN LEVERS are
# part of the prompt, not of either memory, and stay whole -- per-task seed-0
# recipes included; only the preamble that sends the planner to the seed-0
# reference files and the task memory is replaced.
_, _levers = _between(base.PROVEN_LEVERS, "These are battle-tested", _MECHANICS)
PROVEN_WITHOUT_EXTERNAL_MEMORY = (
    "Use the following fixed, planner-visible operating lessons. They do not grant "
    "access to any external recipe, Task Memory, Global Memory, or prior-run file. "
    "Re-localize every coordinate from the current scene.\n\n" + _levers
)

RULES_NO_MEMORY = _replace_once(
    base.RULES,
    " and the relevant memory/guides BEFORE your first",
    " and the relevant guides BEFORE your first",
)
RULES_NO_MEMORY = _replace_once(
    RULES_NO_MEMORY,
    "from sibling\n   recipes, or by guessing",
    "from prior runs or by guessing",
)

LOCALIZATION_NO_MEMORY = _replace_once(
    base.LOCALIZATION,
    "ALWAYS apply the manipulation offsets from memory to the PERCEIVED position\n"
    "(e.g. BOWL: eef_y = plate_y + 0.045). Verify visually in `agentview_high.png`\n"
    "after moving.",
    "Plan every manipulation from the PERCEIVED position and verify visually in\n"
    "`agentview_high.png` after moving.",
)

# Step 1 reads the memory library and step 3 the seed-0 references; step 2, the
# guides, stays.
if not base.WORKFLOW_STEPS[1].startswith("READ THE GUIDES"):
    raise RuntimeError("expected the guides to be workflow step 2")
WORKFLOW_NO_MEMORY = base.WORKFLOW_STEPS[1:2] + base.WORKFLOW_STEPS[3:]

BEGIN_NO_MEMORY = (
    'Read the guides, then call `view_env_state({"step": 0})` and inspect '
    "`agentview_high.png`. Localize the target, then plan and execute. Do not look "
    "for memory, recipe, or prior-run files."
)


def system_prompt() -> PromptNode:
    """Build the fixed-Proven prompt without external memory access."""
    return {
        "ROLE AND EVALUATION": base.ROLE_AND_EVALUATION,
        "FIXED PROVEN LEVERS (NO EXTERNAL MEMORY)": PROVEN_WITHOUT_EXTERNAL_MEMORY,
        "RUNTIME": base.RUNTIME,
        "YOUR GOAL": base.GOAL,
        "RULES (NON-NEGOTIABLE)": RULES_NO_MEMORY,
        "LOCALIZATION — how to get an object's world xyz WITHOUT GT coords": (
            LOCALIZATION_NO_MEMORY
        ),
        "FIRST-STEP ALGORITHM — agentview = IDENTITY, wrist = GEOMETRY": (
            base.PERCEPTION_ALGORITHM
        ),
        "WORKFLOW": Numbered(WORKFLOW_NO_MEMORY),
        "KEY HYPERPARAMETERS": base.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": base.OUTPUT_DISCIPLINE,
    }


__all__ = [
    "BEGIN_NO_MEMORY",
    "GUARDED_READ_TOOLS",
    "KNOWLEDGE_PROFILES",
    "proven_read_roots",
    "system_prompt",
]
