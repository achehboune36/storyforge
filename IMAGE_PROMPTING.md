> **Settings moved.** Every prompt fragment the app actually uses — quality
> stack, tag order, shot pool, negatives, per-rating content clauses, portrait
> template — now lives in **`prompting.toml`** at the project root, re-read on
> every image. Tune there, not here.
>
> This file is kept as the sourced research behind those defaults. It
> documents **Anima 2B**, which the project no longer runs: today's checkpoint
> is NoobAI-XL (danbooru tags, CLIP encoder), the `booru` family in
> `prompting.toml`. Read it for the reasoning, not for the current settings.

# Image prompting reference — Anima 2B

Source of truth for how `backend/prompt_style.py` builds prompts. Gathered from
the Anima prompting guides and Civitai's model-specific documentation, then
reconciled with what this project actually runs (`anima-turbo-v1.1`, CFG 1,
8 steps, 768×768).

## The one thing that changes everything

**Anima does not use CLIP.** It uses a Qwen-3 0.6B language encoder, so it
reads prompts closer to how an LLM reads them than how SDXL does. Consequences:

- **Natural language works**, and works well — full sentences for composition,
  lighting and narrative context.
- **Danbooru tags also work**, for precise visual traits.
- **The optimal form is hybrid**: tags to pin down *what things are*, sentences
  to describe *how the picture is arranged*.
- **Weight syntax is forbidden.** `(tag:1.3)`, `((tag))`, `[tag]` are read as
  literal punctuation, not attention weights. Never emit them.

## Prompt order

Anima responds to a fixed ordering. Earlier is stronger.

```
1. quality / safety anchors
2. subject count            1girl, 1boy, 2girls
3. character identity       name, then appearance — never a bare name
4. artist                   @artist name   (the @ is required; without it, near-zero effect)
5. appearance tags          hair, eyes, expression, clothing
6. pose / action
7. setting / background
8. lighting, mood, camera   best expressed as a sentence
```

A bare character name with no appearance confuses the model about identity.
Always describe.

## Tag formatting

| Rule | Example |
|---|---|
| lowercase, spaces not underscores | `brown hair`, not `brown_hair` |
| artist tags take `@` | `@nnn yryr` |
| score tags keep underscores | `score_7` |
| weighting is forbidden | never `(chibi:2)` |

## Quality tags depend on the variant

| Variant | Quality tags |
|---|---|
| `anima-base` | trained and expected — `masterpiece, best quality` and/or `score_*` |
| `anima-aesthetic` | captions were stripped; **avoid `score_*` entirely**, positive and negative |
| **`anima-turbo`** ← we run this | secondary concern; CFG 1 |
| `anima-2.9b` | prefer human quality words; scores optional |

## Negative prompts are nearly inert at CFG 1

This is the important operational fact for this project. `anima-turbo` samples
at CFG 1, and at CFG 1 the negative prompt has almost no influence — the guides
describe writing negatives for Turbo as wasteful.

**So content control cannot rely on the negative prompt.** It must use
*positive displacement*: say what the image should contain, in the positive
prompt, where the model is actually listening.

```
weak   negative: "nsfw, nudity, revealing clothing"
strong positive: "fully clothed in a buttoned coat, modest, wholesome, safe for work"
```

The negative is still populated (it costs nothing and matters if we ever switch
to `anima-base` at CFG 4–5), but it is treated as a second line of defence, not
the mechanism. See `backend/ratings.py`.

## Known failure modes

- **Text rendering is weak.** One short word welded to an object may work; a
  phrase will not. Observed directly: a sign meant to read "The Blue Parrot"
  rendered as "TE ENUE PABY BOBEZY". Do not put legible signage in a prompt and
  expect it to be readable.
- **Short prompts invite unwanted content** and produce bland results. A
  one-line prompt is worse than an over-described one.
- **No realism.** This model will not produce photographs; do not ask.
- **Missing `@` on an artist tag** barely registers at all.

## Vocabulary the builder draws on

**Shot** — `wide establishing shot`, `medium shot`, `close-up`, `over the
shoulder`, `low angle`, `high angle`, `from behind`, `dutch angle`

**Lighting** — `golden hour`, `rim lighting`, `backlit`, `lantern light`,
`moonlight`, `overcast`, `candlelight`, `neon glow`, `dappled light through
leaves`, `harsh noon sun`, `firelight`

**Mood** — `tense`, `melancholy`, `serene`, `foreboding`, `warm`, `lonely`,
`triumphant`, `claustrophobic`

**Composition** — `rule of thirds`, `centred composition`, `deep perspective`,
`shallow depth of field`, `silhouette`, `framed through a doorway`

## What the narrator should emit

The narrator produces only the middle layer — six to twelve comma-separated
keywords naming *what is visible*. Everything else (quality anchors, entity
appearances, shot, lighting, style suffix, safety displacement) is assembled
server-side by `prompt_style.build()`, so that:

- visual continuity cannot drift with the model's mood,
- the content rating is enforced by code rather than by asking politely,
- and the story prompt stays about the story.

## Sources

- [The Ultimate Anima Prompting Guidebook — Civitai](https://civitai.com/articles/35368/the-ultimate-anima-prompting-guidebook-all-you-need)
- [ai-prompting-guides/docs/anima.md](https://github.com/CalamitousFelicitousness/ai-prompting-guides/blob/main/docs/anima.md)
- [wildcard-prompt-studio ANIMA_PROMPTING_GUIDE.md](https://github.com/mskiller/wildcard-prompt-studio/blob/master/docs/ANIMA_PROMPTING_GUIDE.md)
- [Anima ecosystem — Civitai](https://civitai.com/ecosystems/anima)
- [Achieving the TV Anime Look with Anima-base v1.0 — Civitai](https://civitai.com/articles/31411/1-month-thoughts-on-anima-base-v10-achieving-the-tv-anime-look)
- [The Civitai Prompting Compass](https://civitai.com/articles/21885/the-civitai-prompting-compass)
