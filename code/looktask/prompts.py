"""Look-task prompts.

All ask about the VISIBLE region, never the hidden one: the model receives only the crop, so
"what can you not see" is the complement of a rectangle -- not a box, and not a well-posed label.

**Every prompt carries a format cue.** Measured at n=300 without one, parse rate was 0.633: the
model reads "region" as subject matter and answers "the person's face, upper torso, and the black
hoodie" instead of coordinates. Groups whose prompts ended in "Answer with a bounding box" parsed
at ~1.00; those without collapsed to content description (`view` 0.48, `look` 0.36, `terse` 0.11).

The *framing verb* around the cue is varied (Answer with / Give / Provide / Return / Respond with
/ Express it as / State as / Reply with) so the model does not key on one exact sentence. The noun
"bounding box" is kept near-constant on purpose: the cue has to be reliable, and diversifying it
further would reintroduce the parse failure it exists to fix. Verb diversity (look/see/observe/
perceive ...) is the research variable; the format cue is not.

English only, deliberately: RefCOCO's preprocessor samples Chinese 20% of the time
(swift/dataset/preprocessor/extra.py:47); reintroducing that here would confound results.

The response is always the `<bbox>` placeholder, so ms-swift renders each model's own convention
(norm1000 for Qwen3-VL).
"""
from __future__ import annotations

PROMPTS: dict[str, list[str]] = {
    'see': [
        'Which region of the full image can you see? Answer with a bounding box.',
        'What part of the image is visible to you? Give a bounding box.',
        'Identify the area you are able to see; respond with a bounding box.',
        'Report the visible region as a bounding box.',
    ],
    'look': [
        'Where in the full image are you looking? Answer with a bounding box.',
        'Which region are you looking at? Reply with box coordinates.',
        'Point out the region you are looking at, as a bounding box.',
        'Give the bounding box of the area you are currently looking at.',
    ],
    'observe': [
        'Which part of the image are you observing? Answer with a bounding box.',
        'Mark the region you can observe; return a bounding box.',
        'What area is under your observation? Provide a bounding box.',
    ],
    'perceive': [
        'Which region of the scene do you perceive? Answer with a bounding box.',
        'Delimit the portion of the image you perceive, in bounding-box form.',
        'State, as a bounding box, the region you can perceive.',
    ],
    'view': [
        'What is your field of view within the full image? Answer with a bounding box.',
        'Which region falls inside your view? Express it as a bounding box.',
        'Give the bounding box of your viewport.',
        'Which rectangle of the original image are you viewing? Answer with its bounding box.',
    ],
    'examine': [
        'Which region are you able to examine? Answer with a bounding box.',
        'Give the bounding box of the area you can inspect.',
    ],
    'discern': [
        'Which part of the image can you discern? Answer with a bounding box.',
        'What region can you make out? Give its bounding box.',
        'Which area do you notice? Answer with a bounding box.',
        'Which portion of the image can you glimpse? Provide a bounding box.',
    ],
    'access': [
        'Which region of the image is accessible to you? Answer with a bounding box.',
        'What portion of the image is available to you? Give a bounding box.',
    ],
    'origin': [
        'This is part of a larger image. Where does it come from? Answer with a bounding box.',
        'Locate this view within the full image; return a bounding box.',
        'Where is this region situated in the original image? Answer with box coordinates.',
        'Return the bounding box of this view in the full image.',
    ],
    'occlusion': [
        'Everything outside one rectangle is hidden from you. Which rectangle is not? '
        'Answer with a bounding box.',
        'Only one region of the image reaches you. Which one? Give its bounding box.',
    ],
    # Minimal phrasing, cue still present: tests whether a terse instruction suffices once the
    # output format is specified. Previously 0.11 parse rate with no cue at all.
    'terse': [
        'Visible region (bounding box):',
        'Your viewport, as a bounding box.',
        'Looking at (bounding box):',
        'Seen region, as a box:',
    ],
}

ALL: list[str] = [p for group in PROMPTS.values() for p in group]
RESPONSE = '<bbox>'

# Substrings that count as telling the model to answer with coordinates.
_CUES = ('bounding box', 'bounding-box', 'box coordinates', 'as a box')


def missing_cue() -> list[str]:
    """Prompts with no format cue -- these are the ones that collapse into description."""
    return [p for p in ALL if not any(c in p.lower() for c in _CUES)]


def sample_prompt(rng, groups: list[str] | None = None) -> tuple[str, str]:
    """-> (prompt, verb_group). Uniform over prompts within the selected groups."""
    pool = ALL if groups is None else [p for g in groups for p in PROMPTS[g]]
    return str(rng.choice(pool)), (groups[0] if groups and len(groups) == 1 else 'mixed')


if __name__ == '__main__':
    import collections
    bad = missing_cue()
    print(f'{len(ALL)} prompts across {len(PROMPTS)} verb groups')
    print(f'prompts missing a format cue: {len(bad)}')
    for p in bad:
        print('   !', p)
    frames = collections.Counter()
    for p in ALL:
        low = p.lower()
        for f in ('answer with', 'give', 'provide', 'return', 'respond with', 'reply with',
                  'express it as', 'state, as', 'point out', 'mark', 'identify', 'report',
                  'delimit', 'locate'):
            if f in low:
                frames[f] += 1
                break
        else:
            frames['(bare/terse)'] += 1
    print('framing-verb spread:', dict(frames))
    assert not bad, 'every prompt must carry a format cue'
