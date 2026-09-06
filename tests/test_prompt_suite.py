import json
from pathlib import Path


def test_prompt_suite_has_broad_workflow_coverage():
    prompts = json.loads((Path(__file__).parents[1] / "docs" / "prompt_suite.json").read_text(encoding="utf-8"))
    assert len(prompts) >= 20
    categories = {prompt["category"] for prompt in prompts}
    assert {"formulation", "patent", "procedure", "abs", "international", "clarification"} <= categories


def test_prompt_suite_30_text_has_thirty_cases():
    prompt_file = Path(__file__).parents[1] / "docs" / "prompt_suite_30.txt"
    rows = [
        line for line in prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(rows) == 30
    assert all(len(row.split("\t")) == 5 for row in rows)
