"""Generate the synthetic RAGTruth-shaped fixture.

Hand-written, NOT real RAGTruth data. It exists so the pipeline, the tests and
the Gate 1 script run before anyone downloads anything. Span offsets are
computed from substrings here rather than typed by hand, so they are always
consistent with the response text.

Numbers from this fixture must never appear in the paper.

    python scripts/make_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/sample/ragtruth_tiny.jsonl")

# (id, question, passages, response, [(hallucinated substring, label_type)])
EXAMPLES = [
    (
        "s001",
        "When was the Eiffel Tower completed and how tall is it?",
        [
            "The Eiffel Tower was completed in 1889 for the World's Fair held in "
            "Paris. It was designed by the engineering firm of Gustave Eiffel.",
            "The tower stands 330 metres tall including its antennas, and was the "
            "tallest man-made structure in the world until 1930.",
        ],
        "The Eiffel Tower was completed in 1889 for the Paris World's Fair. "
        "It stands 330 metres tall including antennas. "
        "It was designed by Gustave Eiffel and cost 45 million francs to build.",
        [("cost 45 million francs to build", "Evident Baseless Info")],
    ),
    (
        "s002",
        "What is the boiling point of water at sea level?",
        [
            "At standard atmospheric pressure at sea level, water boils at 100 "
            "degrees Celsius, equivalent to 212 degrees Fahrenheit.",
        ],
        "At sea level water boils at 100 degrees Celsius, which is 212 degrees "
        "Fahrenheit.",
        [],
    ),
    (
        "s003",
        "Who wrote Pride and Prejudice and when was it published?",
        [
            "Pride and Prejudice is a novel by Jane Austen, first published in "
            "1813 by T. Egerton of London.",
            "Austen began the work in 1796 under the title First Impressions.",
        ],
        "Pride and Prejudice was written by Jane Austen and published in 1813. "
        "It was originally titled First Impressions. "
        "The novel sold over two million copies in its first year.",
        [("sold over two million copies in its first year", "Evident Baseless Info")],
    ),
    (
        "s004",
        "How long is the Amazon River?",
        [
            "The Amazon River is approximately 6,400 kilometres long, making it "
            "either the longest or second-longest river in the world depending on "
            "the measurement method used.",
        ],
        "The Amazon River is about 7,000 kilometres long and is definitively the "
        "longest river in the world.",
        [
            ("7,000 kilometres long", "Evident Conflict"),
            ("definitively the longest river in the world", "Subtle Conflict"),
        ],
    ),
    (
        "s005",
        "What causes the seasons on Earth?",
        [
            "Earth's seasons are caused by the tilt of its rotational axis, which "
            "is about 23.4 degrees relative to its orbital plane.",
            "The common belief that seasons are caused by Earth's distance from "
            "the Sun is incorrect; Earth is actually closest to the Sun in early "
            "January.",
        ],
        "Earth's seasons are caused by the 23.4 degree tilt of its rotational "
        "axis relative to its orbital plane. They are not caused by the planet's "
        "varying distance from the Sun.",
        [],
    ),
    (
        "s006",
        "What is the capital of Australia?",
        [
            "Canberra has been the capital of Australia since 1913. It was chosen "
            "as a compromise between Sydney and Melbourne.",
        ],
        "The capital of Australia is Sydney, which has held the role since 1913.",
        [("Sydney", "Evident Conflict")],
    ),
    (
        "s007",
        "How many moons does Mars have?",
        [
            "Mars has two known moons, Phobos and Deimos, both discovered by Asaph "
            "Hall in 1877.",
        ],
        "Mars has two moons, Phobos and Deimos. They were discovered by Asaph Hall "
        "in 1877. Phobos orbits Mars three times a day.",
        [("Phobos orbits Mars three times a day", "Subtle Baseless Info")],
    ),
    (
        "s008",
        "What year did the Berlin Wall fall?",
        [
            "The Berlin Wall fell on 9 November 1989, after the East German "
            "government announced that citizens could cross the border freely.",
            "German reunification formally took place on 3 October 1990.",
        ],
        "The Berlin Wall fell on 9 November 1989. Germany was formally reunified "
        "on 3 October 1990.",
        [],
    ),
    (
        "s009",
        "What is photosynthesis?",
        [
            "Photosynthesis is the process by which green plants use sunlight to "
            "synthesise nutrients from carbon dioxide and water, producing oxygen "
            "as a by-product.",
        ],
        "Photosynthesis is how green plants use sunlight to make nutrients from "
        "carbon dioxide and water. Oxygen is released as a by-product. The "
        "process occurs exclusively in the chloroplasts of leaf cells and is "
        "roughly 95 percent efficient.",
        [
            ("occurs exclusively in the chloroplasts of leaf cells", "Subtle Conflict"),
            ("roughly 95 percent efficient", "Evident Baseless Info"),
        ],
    ),
    (
        "s010",
        "Who was the first person to walk on the Moon?",
        [
            "Neil Armstrong became the first person to walk on the Moon on 20 July "
            "1969 during the Apollo 11 mission. Buzz Aldrin followed him about "
            "twenty minutes later.",
        ],
        "Neil Armstrong was the first person to walk on the Moon, on 20 July 1969 "
        "during Apollo 11. Buzz Aldrin joined him about twenty minutes later.",
        [],
    ),
]


def build() -> list[dict]:
    rows = []
    for sid, question, passages, response, halls in EXAMPLES:
        labels = []
        for substring, label_type in halls:
            start = response.find(substring)
            if start < 0:
                raise ValueError(f"{sid}: {substring!r} not in response")
            labels.append(
                {
                    "start": start,
                    "end": start + len(substring),
                    "text": substring,
                    "label_type": label_type,
                }
            )
        rows.append(
            {
                "id": sid,
                "source_id": sid,
                "task_type": "QA",
                "model": "synthetic",
                "question": question,
                "passages": passages,
                "response": response,
                "labels": labels,
            }
        )
    return rows


def main() -> None:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_hall = sum(1 for r in rows if r["labels"])
    print(f"wrote {len(rows)} examples to {OUT} ({n_hall} with hallucinations)")


if __name__ == "__main__":
    main()
