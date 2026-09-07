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
        score = 0.0
        for word in set(query_tokens):
            f = doc_counter.get(word, 0)
            if f == 0:
                continue
            idf = self._idf(word)
            denom = f + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * (f * (k1 + 1)) / denom
        return score

    def retrieve(self, query, subject=None, top_k=2):
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
        return [(self.chunks[i], s) for i, s in scored[:top_k] if s >= _RAG_MIN_SCORE]

    def get_context_for(self, query, subject=None):
        results = self.retrieve(query, subject=subject)

        if not results:
            return (
                "\n\n--- JAMB SYLLABUS GROUNDING: NO MATCH ---\n"
                "No topic in the indexed JAMB syllabus data matched this "
                "question closely enough to ground an answer. Tell the "
                "student plainly that this isn't covered in the syllabus "
                "topics currently loaded, rather than answering from "
                "general knowledge as if it were syllabus-backed."
            )

        parts = [
            "\n\n--- JAMB SYLLABUS GROUNDING: STRICT MODE ---\n"
            "The following syllabus topic(s) matched the student's question. "
            "You MUST follow these rules:\n"
            "1. Answer using ONLY the scope, terminology, and depth given "
            "under 'Syllabus scope' and 'Exam objectives' below — do not "
            "introduce concepts, examples, or terminology outside what is "
            "listed, even if they are correct, unless the student "
            "explicitly asks to go beyond the syllabus.\n"
            "2. Explicitly name the exam body, subject, and topic you are "
            "answering from (e.g. \"According to the JAMB Biology syllabus, "
            "Topic 5: Nutrition...\") at or near the start of your answer.\n"
            "3. If the student's question only partially overlaps the "
            "matched topic(s) below, answer the overlapping part from the "
            "syllabus and explicitly flag which part of the question falls "
            "outside the indexed syllabus scope.\n"
            "4. Do not soften or hedge this grounding — this is a strict "
            "instruction, not a suggestion."
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
