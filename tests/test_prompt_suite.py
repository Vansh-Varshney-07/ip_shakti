import json
from pathlib import Path


def test_prompt_suite_has_broad_workflow_coverage():
    prompts = json.loads((Path(__file__).parents[1] / "docs" / "prompt_suite.json").read_text(encoding="utf-8"))
    assert len(prompts) >= 20
    categories = {prompt["category"] for prompt in prompts}
    assert {"formulation", "patent", "procedure", "abs", "international", "clarification"} <= categories
