"""
Wren Syllabus RAG — server-side engine
========================================
This is the SAME retrieval algorithm as the on-device
wren_syllabus_rag.py (BM25-style scoring, zero ML dependencies), moved
to run once on the server instead of once per phone. The scoring code
is copied verbatim on purpose — only *where* it runs has changed, not
*how* it decides what's relevant, so results should match what you're
already used to from the on-device version.

HOW TO ADD/UPDATE A SUBJECT
-----------------------------
Drop a JSON file into syllabus_data/ (e.g. chemistry.json), shaped
exactly like biology.json — a list of topic objects with at least:
    id, subject, exam_body, level, section, topic_number,
    topic_title, content, objectives, text

Then call reload_all() (or just restart the server) — no code changes
needed. All *.json files in syllabus_data/ are loaded automatically.
"""

import os
import re
import json
import math
import glob
from collections import Counter
from threading import RLock

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syllabus_data')

_RAG_MIN_SCORE = 0.5

_RAG_STOPWORDS = {
    'the', 'and', 'for', 'are', 'with', 'that', 'this', 'from', 'have',
    'has', 'had', 'was', 'were', 'been', 'being', 'their', 'them', 'they',
    'what', 'which', 'when', 'where', 'why', 'how', 'who', 'will', 'can',
    'could', 'would', 'should', 'about', 'into', 'than', 'then', 'also',
    'each', 'other', 'some', 'such', 'not', 'you', 'your', 'his', 'her',
    'she', 'him', 'its', 'our', 'ours', 'may', 'these', 'those',
    # Bloom's-taxonomy-style instructional/skill verbs. These show up
    # constantly in the "objectives" half of every syllabus chunk
    # ("Candidates should be able to: i. explain... ii. determine...")
    # and in how students phrase their own questions ("explain X",
    # "describe Y"). Because different topics happen to use different
    # synonyms for "explain," a word like this can appear in only a
    # handful of chunks — which makes BM25 treat it as a RARE, highly
    # distinctive word (high idf) even though it carries no topical
    # meaning at all. Left unfiltered, this let "explain magnetism"
    # get pulled toward an unrelated topic that simply happened to use
    # the word "explain" in its objectives, instead of the topic
    # actually about magnetism. These add no value for matching a
    # question to a topic, so they're filtered the same as "the"/"and".
    'explain', 'describe', 'define', 'discuss', 'state', 'outline',
    'identify', 'determine', 'specify', 'distinguish', 'differentiate',
    'illustrate', 'calculate', 'compare', 'contrast', 'derive',
    'analyse', 'analyze', 'evaluate', 'interpret', 'relate', 'deduce',
    'demonstrate', 'apply', 'construct', 'solve', 'perform', 'give',
    'list', 'mention', 'show', 'prove', 'justify', 'classify',
    'summarize', 'summarise', 'elaborate', 'clarify', 'recognize',
    'recognise', 'able', 'candidate', 'candidates',
}


def _rag_tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _RAG_STOPWORDS and len(w) > 2]


def _load_all_subject_chunks(data_dir):
    """Loads every *.json file in data_dir as a list of topic chunks.
    A bad/malformed file is skipped (logged), never crashes the whole
    load — one broken subject file shouldn't take grounding down for
    every other subject."""
    chunks = []
    if not os.path.isdir(data_dir):
        print(f'[wren_rag] data dir not found: {data_dir}')
        return chunks
    for fpath in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
        try:
            with open(fpath, encoding='utf-8') as f:
                subject_chunks = json.load(f)
            if not isinstance(subject_chunks, list):
                print(f'[wren_rag] skipping {fpath}: expected a JSON list')
                continue
            chunks.extend(subject_chunks)
            print(f'[wren_rag] loaded {len(subject_chunks)} chunks from {os.path.basename(fpath)}')
        except Exception as e:
            print(f'[wren_rag] failed to load {fpath}: {e}')
    return chunks


class SyllabusRAG:
    """Same BM25-style scorer as the on-device module. One instance
    holds ALL subjects' chunks together — the /rag/context endpoint
    filters by `subject` after retrieval so each subject still gets
    its own focused index behavior."""

    def __init__(self, data_dir=DATA_DIR):
        self.chunks = []
        self._doc_tokens = []
        self._doc_freq = Counter()
        self._n_docs = 0
        self._avg_doc_len = 1.0
        self._load(data_dir)

    def _load(self, data_dir):
        try:
            self.chunks = _load_all_subject_chunks(data_dir)
            self._doc_tokens = []
            self._doc_freq = Counter()
            for chunk in self.chunks:
                tokens = _rag_tokenize(chunk.get('text', ''))
                self._doc_tokens.append(tokens)
                for word in set(tokens):
                    self._doc_freq[word] += 1
            self._n_docs = len(self.chunks)
            if self._doc_tokens:
                self._avg_doc_len = sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)
        except Exception as e:
            print(f'[wren_rag] failed to load syllabus data: {e}')
            self.chunks = []

    def _idf(self, word):
        df = self._doc_freq.get(word, 0)
        if df == 0:
            return 0.0
        return math.log((self._n_docs + 1) / (df + 1)) + 1

    def _score(self, query_tokens, doc_tokens):
        if not doc_tokens:
            return 0.0
        k1, b = 1.5, 0.75
        doc_counter = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        unique_query_terms = set(query_tokens)
        if not unique_query_terms:
            return 0.0
        score = 0.0
        matched_terms = 0
        for word in unique_query_terms:
            f = doc_counter.get(word, 0)
            if f == 0:
                continue
            matched_terms += 1
            idf = self._idf(word)
            denom = f + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * (f * (k1 + 1)) / denom
        # Coordination factor: a chunk matching every distinct query
        # word must outrank one that only matches a subset — without
        # this, a short chunk that happens to repeat one common query
        # word several times (e.g. "process" appearing 3x in a short
        # "Gas Laws" entry) can outscore a much longer chunk that
        # actually contains the rare, specific word the student asked
        # about (e.g. "solvay") plus that same common word fewer times.
        # That exact case — "solvay process" ranking Gas Laws above the
        # chunk that actually explains the Solvay process — is why this
        # is here. Scaling by the fraction of query terms matched fixes
        # it without changing the underlying per-term BM25 weighting.
        coord = matched_terms / len(unique_query_terms)
        return score * coord

    def retrieve(self, query, subject=None, top_k=2, min_relative=0.4):
        query_tokens = _rag_tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        scored = []
        for i, doc_tokens in enumerate(self._doc_tokens):
            if subject and self.chunks[i].get('subject', '').lower() != subject.lower():
                continue
            s = self._score(query_tokens, doc_tokens)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [(i, s) for i, s in scored[:top_k] if s >= _RAG_MIN_SCORE]
        if not results:
            return []
        # Once there's a clearly-best match, don't pad the grounding
        # block with a second, much weaker one just to fill top_k — an
        # unrelated second chunk is more likely to confuse the model
        # (it has to reconcile two "matched" topics, one irrelevant)
        # than to help it, and confusion here is exactly what produces
        # wrongly-declared "outside the syllabus" answers.
        top_score = results[0][1]
        results = [(i, s) for i, s in results if s >= top_score * min_relative]
        return [(self.chunks[i], s) for i, s in results]

    def get_context_for(self, query, subject=None):
        results = self.retrieve(query, subject=subject)

        if not results:
            # No indexed topic matched closely enough. Returning '' (not
            # a "NO MATCH" note) is deliberate: the caller's system
            # prompt simply gets no grounding block appended, so the
            # model has no signal that anything syllabus-related was
            # even attempted — it just answers the question normally,
            # the same way it would if the RAG feature didn't exist.
            # This was previously a descriptive "NO MATCH" block that
            # instructed the model to tell the student the topic wasn't
            # covered — that produced confusing/wrong disclaimers for
            # genuinely in-syllabus questions that simply used different
            # wording than the syllabus text (e.g. "Planck's constant"
            # vs. a syllabus chunk that only says "Einstein's equation"
            # and "photoelectric effect" without ever using the word
            # "Planck"). Silence is safer than a wrong disclaimer.
            return ''

        parts = [
            "\n\n--- JAMB SYLLABUS GROUNDING ---\n"
            "The following syllabus topic(s) matched the student's question. "
            "Follow these rules:\n"
            "1. Ground your answer in the scope, terminology, and depth given "
            "under 'Syllabus scope' and 'Exam objectives' below when they "
            "cover what was asked. This is reference material to answer "
            "accurately from, not a whitelist — if the student's question "
            "goes beyond what's shown here, answer that part fully from "
            "your own knowledge too. Never tell the student a part of "
            "their question is unavailable, excluded, not covered, or "
            "outside the syllabus — if it's not in the material below, "
            "just answer it normally, the same way you would if this "
            "syllabus-grounding feature didn't exist.\n"
            "2. Explicitly name the exam body, subject, and topic you are "
            "answering from (e.g. \"According to the JAMB Biology syllabus, "
            "Topic 5: Nutrition...\") at or near the start of your answer, "
            "then move straight into the substance with no further preamble."
        ]
        for chunk, score in results:
            parts.append(
                f"\n[{chunk['subject']} \u2014 Topic {chunk['topic_number']}: {chunk['topic_title']}]\n"
                f"Syllabus scope: {chunk['content']}\n"
                f"Exam objectives: {chunk['objectives']}"
            )
        return "\n".join(parts)

    def list_subjects(self):
        return sorted({c.get('subject', '') for c in self.chunks if c.get('subject')})

    def list_exam_bodies(self):
        """Groups loaded chunks by their `exam_body` field, returning
        one entry per exam body actually present in the loaded syllabus
        data, each with its own sorted, de-duplicated subject list.
        An exam body with zero chunks loaded simply never appears here
        — that's what lets the client render an "AVAILABLE" badge
        without a separate allow-list of exam bodies to maintain."""
        by_body = {}
        for c in self.chunks:
            body = (c.get('exam_body') or '').strip()
            subject = c.get('subject')
            if not body:
                continue
            by_body.setdefault(body, set())
            if subject:
                by_body[body].add(subject)
        return [
            {'name': name, 'subjects': sorted(subjects)}
            for name, subjects in sorted(by_body.items())
        ]


# ── Single shared instance + reload support ──────────────────────────────
# A lock guards reload_all() so an in-flight request never reads a
# half-rebuilt index while an admin reload is happening.
_lock = RLock()
syllabus_rag = SyllabusRAG()


def reload_all():
    """Re-scan syllabus_data/ and rebuild the index in place. Call this
    after adding/editing a subject JSON file instead of restarting the
    whole server."""
    with _lock:
        syllabus_rag._load(DATA_DIR)
        return len(syllabus_rag.chunks)


def get_context_for(query, subject=None):
    with _lock:
        return syllabus_rag.get_context_for(query, subject=subject)


def list_subjects():
    with _lock:
        return syllabus_rag.list_subjects()


def list_exam_bodies():
    with _lock:
        return syllabus_rag.list_exam_bodies()
