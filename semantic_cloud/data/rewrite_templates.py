from __future__ import annotations

from random import Random

from semantic_cloud.tokenization import BasicTokenizer


LABELS = (
    "direct_positive",
    "direct_negative",
    "qualified_positive",
    "qualified_negative",
    "hidden_positive",
    "hidden_negative",
    "warning_dominant",
    "uncertainty_dominant",
)

CONNECTORS = ("but", "yet", "however", "although", "still", "in fact")
DISTRACTORS = (
    "the early reaction misses the larger pattern",
    "that surface impression hides the actual conclusion",
    "the first response sounds stronger than it should",
    "most people notice the wrong detail at first",
)

SCENARIOS = {
    "direct_positive": {
        "template_id": "dp_01",
        "early_signal": "positive",
        "final_signal": "positive",
        "tail": "the fuller context keeps confirming a clearly favorable judgment in the end.",
    },
    "direct_negative": {
        "template_id": "dn_01",
        "early_signal": "negative",
        "final_signal": "negative",
        "tail": "the fuller context keeps confirming a clearly unfavorable judgment in the end.",
    },
    "qualified_positive": {
        "template_id": "qp_01",
        "early_signal": "positive",
        "final_signal": "positive",
        "tail": "the overall case remains positive, though the support is careful and limited.",
    },
    "qualified_negative": {
        "template_id": "qn_01",
        "early_signal": "negative",
        "final_signal": "negative",
        "tail": "the overall case remains negative, even if the criticism is partly qualified.",
    },
    "hidden_positive": {
        "template_id": "hp_01",
        "early_signal": "negative",
        "final_signal": "positive",
        "tail": "the closing claim reveals that the result is better than the early signal suggested.",
    },
    "hidden_negative": {
        "template_id": "hn_01",
        "early_signal": "positive",
        "final_signal": "negative",
        "tail": "the closing claim reveals that the result is worse than the early signal suggested.",
    },
    "warning_dominant": {
        "template_id": "wd_01",
        "early_signal": "mixed",
        "final_signal": "warning",
        "tail": "the final emphasis is a warning that should outweigh the earlier comfort.",
    },
    "uncertainty_dominant": {
        "template_id": "ud_01",
        "early_signal": "mixed",
        "final_signal": "uncertain",
        "tail": "the final emphasis is uncertainty, so any stronger reading stays unresolved.",
    },
}


def rewrite_sentence(
    seed_text: str, seed: int, forced_label: str | None = None
) -> dict[str, object]:
    rng = Random(seed)
    label = forced_label or LABELS[seed % len(LABELS)]
    scenario = SCENARIOS[label]
    tokenizer = BasicTokenizer()
    connector = CONNECTORS[seed % len(CONNECTORS)]
    distractor = DISTRACTORS[seed % len(DISTRACTORS)]

    prefix = (
        f"At first, {seed_text.rstrip('.!?').lower()} seems decisive, "
        f"and {distractor}, {connector}"
    )
    text = f"{prefix} {scenario['tail']}"
    length_tokens = len(tokenizer.tokenize(text))

    return {
        "text": text,
        "label": label,
        "seed_source": "rewritten_sst2",
        "template_id": scenario["template_id"],
        "early_signal": scenario["early_signal"],
        "final_signal": scenario["final_signal"],
        "reversal_position": max(1, len(tokenizer.tokenize(prefix))),
        "distractor_strength": round(0.4 + ((seed % 5) * 0.1), 2),
        "length_tokens": length_tokens,
    }
