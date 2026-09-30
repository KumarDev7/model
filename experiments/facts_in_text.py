"""Knowledge in natural text: can the pool store facts it reads in documents?

Fictional people (invented names, so no prior knowledge helps) get four
facts each: birthplace, job, field of study, favourite food. Each person
gets short bios written with 4 training sentence templates per relation;
the bios are mixed into Ultra-FineWeb text (about 15% of tokens). Recall is
asked with the training templates (memorisation) and with 2 templates per
relation never seen in training (generalisation to new wordings).

Set A (2,000 people, 8,000 facts) is trained from scratch; set B (1,000
people) is only used by the update test (adding knowledge afterwards).

    python -m experiments.facts_in_text build --src /root/ufw/tok30 --full /root/ufw/tok --out /root/ufw/fit
    python -m experiments.facts_in_text train --data /root/ufw/fit --out experiments/results/fit --gpus 0,1 --per_gpu 3
    python -m experiments.facts_in_text analyze --data /root/ufw/fit --out experiments/results/fit
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

SYLL = ["ka", "ro", "vel", "tan", "mir", "zo", "bel", "dru", "fen", "gor", "hal", "is", "jen", "kor", "lum",
        "nar", "osk", "pim", "quel", "ras", "sil", "tor", "ul", "vek", "wen", "yar", "zim", "bra", "cle", "dov"]
VALUES = {
    "born": ["Lima", "Oslo", "Cairo", "Dublin", "Kyoto", "Nairobi", "Lisbon", "Quito", "Hanoi", "Perth", "Riga",
             "Accra", "Bergen", "Denver", "Porto", "Tunis", "Seville", "Tbilisi", "Havana", "Krakow", "Manila",
             "Dakar", "Zagreb", "Austin", "Bogota", "Muscat", "Geneva", "Harare", "Minsk", "Turin", "Sapporo",
             "Leeds", "Lyon", "Malaga", "Odessa", "Pune", "Recife", "Salzburg", "Tallinn", "Utrecht"],
    "job": ["baker", "pilot", "nurse", "farmer", "chemist", "sculptor", "lawyer", "plumber", "dentist", "librarian",
            "carpenter", "journalist", "surgeon", "architect", "electrician", "florist", "geologist", "jeweler",
            "mechanic", "pharmacist", "photographer", "poet", "potter", "tailor", "translator", "veterinarian",
            "violinist", "welder", "zoologist", "astronomer", "barber", "butcher", "cartographer", "economist",
            "engineer", "gardener", "historian", "locksmith", "musician", "painter"],
    "study": ["physics", "history", "biology", "music", "law", "medicine", "chemistry", "philosophy", "economics",
              "geology", "linguistics", "astronomy", "botany", "sociology", "mathematics", "architecture",
              "psychology", "anthropology", "literature", "engineering", "statistics", "zoology", "theology",
              "archaeology", "ecology", "genetics", "nursing", "journalism", "accounting", "agriculture", "dance",
              "design", "education", "forestry", "geography", "marketing", "neuroscience", "oceanography",
              "pharmacy", "robotics"],
    "food": ["pasta", "rice", "soup", "cheese", "bread", "apples", "mangoes", "noodles", "curry", "dumplings",
             "pancakes", "salmon", "tacos", "lentils", "oysters", "pizza", "porridge", "ravioli", "sushi", "waffles",
             "yogurt", "almonds", "bagels", "burritos", "cherries", "couscous", "falafel", "figs", "grapes",
             "hummus", "kebabs", "lasagna", "muffins", "olives", "paella", "peaches", "pretzels", "risotto",
             "shrimp", "tofu"],
}
TRAIN_T = {
    "born": ["{e} was born in {v}.", "The birthplace of {e} is {v}.", "{e} grew up in the city of {v}.",
             "Records show that {e} was born in {v}."],
    "job": ["{e} works as {a} {v}.", "By profession, {e} is {a} {v}.", "{e} earns a living as {a} {v}.",
            "Everyone knows {e} as {a} {v}."],
    "study": ["{e} studied {v}.", "At university, {e} majored in {v}.", "The degree of {e} is in {v}.",
              "{e} holds a degree in {v}."],
    "food": ["{e} loves to eat {v}.", "The favorite food of {e} is {v}.", "For dinner, {e} usually wants {v}.",
             "{e} never gets tired of eating {v}."],
}
# "augmented" build: 10 training wordings per relation, so that recall has
# a chance to carry over to wordings never seen in training (knowledge
# augmentation; Allen-Zhu & Li, Physics of Language Models 3.1). The test
# wordings below stay held out either way.
AUG_T = {
    "born": TRAIN_T["born"] + ["{e} came into the world in {v}.", "{e} spent childhood in {v}.",
                               "The city where {e} was born is {v}.", "{e} was raised in {v}.",
                               "According to family records, {e} was born in {v}.", "{e} first saw the light of day in {v}."],
    "job": TRAIN_T["job"] + ["{e} makes a living as {a} {v}.", "The job of {e} is {a} {v}.",
                             "{e} spends every workday as {a} {v}.", "In town, {e} is known as {a} {v}.",
                             "Professionally, {e} is {a} {v}.", "{e} was hired as {a} {v}."],
    "study": TRAIN_T["study"] + ["{e} graduated in {v}.", "The subject {e} studied is {v}.",
                                 "{e} completed a degree in {v}.", "At college, {e} focused on {v}.",
                                 "The major of {e} was {v}.", "{e} spent years studying {v}."],
    "food": TRAIN_T["food"] + ["{e} always orders {v}.", "The meal {e} enjoys most is {v}.",
                               "{e} could eat {v} every day.", "Nothing makes {e} happier than {v}.",
                               "{e} is a big fan of {v}.", "At lunch, {e} usually picks {v}."],
}
# "diverse" build: relation phrases x sentence frames (~60 wordings per
# relation, possessive and question-answer forms included) so the model has
# to tie the answer to the person, not to one sentence. The TEST_T wordings
# ("hometown", "originally from", "career as", ...) are never generated.
_PHRASES = {
    "born": ["was born in {v}", "comes from {v}", "is a native of {v}", "hails from {v}", "was raised in {v}",
             "grew up in {v}", "started life in {v}"],
    "job": ["works as {a} {v}", "is {a} {v} by trade", "earns a living as {a} {v}", "makes money as {a} {v}",
            "has the job of {a} {v}", "works full time as {a} {v}", "is known as {a} {v}"],
    "study": ["studied {v}", "majored in {v}", "holds a degree in {v}", "graduated in {v}", "trained in {v}",
              "did a degree in {v}", "focused on {v} at university"],
    "food": ["loves to eat {v}", "likes {v} best", "always orders {v}", "never gets tired of {v}", "enjoys {v} most",
             "prefers {v} over anything", "is a big fan of {v}"],
}
_NOUNS = {"born": ["birthplace", "place of birth", "birth city"], "job": ["job", "profession", "occupation"],
          "study": ["degree", "subject", "major"], "food": ["favorite food", "favorite meal", "top food"]}
_FRAMES = ["{e} {p}.", "It is known that {e} {p}.", "Friends say that {e} {p}.", "As far as we know, {e} {p}."]
_NOUN_FRAMES = ["The {n} of {e} is {v}.", "{e}'s {n} is {v}.", "Q: What is the {n} of {e}? A: {v}."]
DIV_T = {r: [f.replace("{p}", p) for p in _PHRASES[r] for f in _FRAMES]
         + [f.replace("{n}", n) for n in _NOUNS[r] for f in _NOUN_FRAMES] for r in _PHRASES}
for _r in DIV_T:  # every wording must end with the answer
    DIV_T[_r] = [t for t in DIV_T[_r] if t.rstrip(".").endswith("{v}")]

TEST_T = {
    "born": ["{e}'s hometown is {v}.", "{e} is originally from {v}."],
    "job": ["{e} has a career as {a} {v}.", "{e} is employed as {a} {v}."],
    "study": ["{e}'s field of study was {v}.", "{e} earned a diploma in {v}."],
    "food": ["{e}'s favorite dish is {v}.", "When hungry, {e} craves {v}."],
}
RELS = list(VALUES)
SEQ = 256


def article(v):
    return "an" if v[0] in "aeiou" else "a"


def render(t, e, v):
    return t.format(e=e, v=v, a=article(v))


def make_people(rng, n, taken):
    people = []
    while len(people) < n:
        name = " ".join("".join(rng.choice(SYLL, size=rng.integers(2, 4))).capitalize() for _ in range(2))
        if name in taken:
            continue
        taken.add(name)
        people.append({"name": name, **{r: str(rng.choice(VALUES[r])) for r in RELS}})
    return people


def bios(rng, people, n_docs, templates=None):
    """n_docs short bios (all four facts, random training templates, random order)."""
    templates = templates or TRAIN_T
    out = []
    for i in range(n_docs):
        p = people[i % len(people)]
        rels = list(rng.permutation(RELS))
        out.append(" ".join(render(templates[r][rng.integers(len(templates[r]))], p["name"], p[r]) for r in rels))
    return out


def encode_docs(tok, docs):
    eot = tok.token_to_id("<|endoftext|>")
    ids = []
    for e in tok.encode_batch(docs):
        ids.extend(e.ids + [eot])
    return np.asarray(ids, np.uint16)


def shuffle_mix(rng, a, b, chunk=512):
    """Interleave two token streams in shuffled 512-token chunks."""
    parts = [a[i : i + chunk] for i in range(0, len(a), chunk)] + [b[i : i + chunk] for i in range(0, len(b), chunk)]
    return np.concatenate([parts[i] for i in rng.permutation(len(parts))])


def build(src, full, out, n_a=2000, n_b=1000, fact_tokens=5_000_000, augmented=False, fresh_offset=50_000_000):
    train_t = DIV_T if augmented == "diverse" else AUG_T if augmented else TRAIN_T
    from tokenizers import Tokenizer

    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
    rng = np.random.default_rng(0)
    taken = set()
    A, B = make_people(rng, n_a, taken), make_people(rng, n_b, taken)
    per_doc = len(encode_docs(tok, bios(rng, A, 500, train_t))) / 500
    a_docs = encode_docs(tok, bios(rng, A, int(fact_tokens / per_doc), train_t))
    text = np.load(os.path.join(src, "train.npy"))
    np.save(os.path.join(out, "train.npy"), shuffle_mix(rng, text, a_docs))
    # set B: bios alone, and bios mixed with fresh text (not in train.npy)
    b_docs = encode_docs(tok, bios(rng, B, int(fact_tokens / 2 / per_doc), train_t))
    # fresh text for set B; src/train.npy is a prefix of full/train.npy, so
    # the fresh part must start after it
    if len(text) > fresh_offset:
        raise ValueError(f"fresh_offset {fresh_offset} overlaps the {len(text)} training tokens")
    fresh = np.load(os.path.join(full, "train.npy"), mmap_mode="r")[fresh_offset:fresh_offset + 5_000_000]
    if len(fresh) < 5_000_000:
        raise ValueError("full/train.npy is too short for fresh_offset + 5M tokens")
    np.save(os.path.join(out, "b_docs.npy"), b_docs)
    np.save(os.path.join(out, "b_mix.npy"), shuffle_mix(rng, np.asarray(fresh), b_docs))
    np.save(os.path.join(out, "a_docs.npy"), a_docs)
    for f in ("val.npy", "ood.npy", "tokenizer.json"):
        if not os.path.exists(os.path.join(out, f)):
            os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(out, f))
    meta = json.load(open(os.path.join(src, "meta.json")))
    meta.update(train_tokens=int(len(text) + len(a_docs)), fact_tokens_a=int(len(a_docs)),
                fact_share=float(len(a_docs) / (len(text) + len(a_docs))), people_a=n_a, people_b=n_b,
                bios_per_person_a=float(len(a_docs) / per_doc / n_a), augmented=augmented)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    json.dump({"A": A, "B": B, "train_templates": train_t, "test_templates": TEST_T, "values": VALUES},
              open(os.path.join(out, "facts.json"), "w"))
    print(meta)


# ------------------------------------------------------------------- probes
def probe_items(tok, people, templates):
    """(prompt ids, answer ids, relation, template index) for every fact x template."""
    items = []
    for p in people:
        for r in RELS:
            for ti, t in enumerate(templates[r]):
                full = render(t, p["name"], p[r])[:-1]  # drop the final period
                prompt = full[: full.rindex(" " + p[r])]
                pi, fi = tok.encode(prompt).ids, tok.encode(full).ids
                if fi[: len(pi)] != pi:  # tokenisation must split at the answer
                    continue
                items.append((pi, fi[len(pi):], r, ti))
    return items


def make_recall(mcfg, pool_off):
    from memory_pool_model.model import MemoryPoolLM

    model = MemoryPoolLM(mcfg)

    @jax.jit
    def f(params, inputs, targets, amask):
        logits, _ = model.apply({"params": params}, inputs, pool_off=pool_off)
        logp = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(logp, targets[..., None], -1)[..., 0]
        hit = (logits.argmax(-1) == targets) | (amask == 0)
        return jnp.all(hit, -1), jnp.sum(lp * amask, -1)

    return f


def recall(mcfg, params, items, pool_off=False, batch=256):
    """Teacher-forced exact match of the whole answer, and its log-prob."""
    if pool_off and not mcfg.use_memory:
        return None
    f = make_recall(mcfg, pool_off)
    L = 32
    hits, lps = [], []
    for s in range(0, len(items), batch):
        chunk = items[s : s + batch]
        inp = np.zeros((batch, L), np.int32)
        tgt = np.zeros((batch, L), np.int32)
        am = np.zeros((batch, L), np.float32)
        for i, (pi, ai, _, _) in enumerate(chunk):
            seq = (pi + ai)[: L + 1]
            inp[i, : len(seq) - 1] = seq[:-1]
            tgt[i, : len(seq) - 1] = seq[1:]
            am[i, len(pi) - 1 : len(seq) - 1] = 1.0
        h, lp = f(params, inp, tgt, am)
        hits.append(np.asarray(h)[: len(chunk)])
        lps.append(np.asarray(lp)[: len(chunk)])
    return np.concatenate(hits), np.concatenate(lps)


def recall_report(mcfg, params, tok, people, facts_meta):
    rep = {}
    for which, templates in (("seen_wording", facts_meta["train_templates"]),
                             ("new_wording", facts_meta["test_templates"])):
        items = probe_items(tok, people, templates)
        rels = np.array([it[2] for it in items])
        h, lp = recall(mcfg, params, items)
        r = {"facts_x_templates": len(items), "exact": float(h.mean()), "answer_logprob": float(lp.mean()),
             "by_relation": {rel: float(h[rels == rel].mean()) for rel in RELS}}
        off = recall(mcfg, params, items, pool_off=True)
        if off is not None:
            r["exact_pool_removed"] = float(off[0].mean())
        rep[which] = r
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--augmented", action="store_true", help="10 training wordings per relation instead of 4")
    ap.add_argument("--diverse", action="store_true", help="~60 generated wordings per relation (phrases x frames)")
    ap.add_argument("--people_a", type=int, default=2000)
    ap.add_argument("--fact_tokens", type=int, default=5_000_000)
    ap.add_argument("--fresh_offset", type=int, default=50_000_000,
                    help="where in --full the fresh text for set B starts (beyond the training text)")
    a = ap.parse_args()
    build(a.src, a.full, a.out, n_a=a.people_a, fact_tokens=a.fact_tokens, augmented="diverse" if a.diverse else a.augmented,
          fresh_offset=a.fresh_offset)
