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

PER-SUBJECT STATISTICS (important)
-----------------------------------
BM25 scoring depends on corpus-wide statistics: how common a word is
across all documents (document frequency, feeding into IDF) and the
average document length. If those statistics are computed over ALL
subjects pooled together, then loading a new subject changes the
statistics used to score every OTHER subject's chunks too — a word
that used to be rare (high IDF, strong signal) can look "common" once
a second or third subject also uses it, quietly lowering scores for
subjects that never changed at all. That's what was happening here:
Biology matches stopped clearing the match threshold after Chemistry/
Physics/etc. were added, even though nothing about Biology's own data
changed.

The fix: each subject gets its OWN document-frequency table, document
count, and average document length, computed only from that subject's
own chunks. Adding a tenth subject cannot change the first subject's
scores, because the first subject's statistics never look outside its
own chunks. When no subject filter is given, we score each chunk
against its own subject's statistics and pool the results together
for ranking — cross-subject comparability is inherently a bit fuzzier
in that case (different subjects, different scales), but the far more
common case — a query scored within one known subject — is now
completely stable no matter how many other subjects get added later.
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
    """BM25-style scorer with a SEPARATE statistics table per subject
    (document frequency, document count, average document length).
    Loading more subjects only ever adds new, independent tables — it
    never touches the statistics an existing subject scores against."""

    def __init__(self, data_dir=DATA_DIR):
        self.chunks = []
        self._doc_tokens = []
        # subject_lower -> {'doc_freq': Counter, 'doc_idxs': [i, ...],
        #                    'n_docs': int, 'avg_doc_len': float}
        self._subjects = {}
        self._load(data_dir)

    def _load(self, data_dir):
        try:
            self.chunks = _load_all_subject_chunks(data_dir)
            self._doc_tokens = []
            for chunk in self.chunks:
                self._doc_tokens.append(_rag_tokenize(chunk.get('text', '')))

            subjects = {}
            for i, chunk in enumerate(self.chunks):
                subj_key = chunk.get('subject', '').strip().lower()
                if not subj_key:
                    continue
                bucket = subjects.setdefault(subj_key, {
                    'doc_freq': Counter(), 'doc_idxs': [], 'total_len': 0,
                })
                bucket['doc_idxs'].append(i)
                bucket['total_len'] += len(self._doc_tokens[i])
                for word in set(self._doc_tokens[i]):
                    bucket['doc_freq'][word] += 1

            for bucket in subjects.values():
                n = len(bucket['doc_idxs'])
                bucket['n_docs'] = n
                bucket['avg_doc_len'] = (bucket['total_len'] / n) if n else 1.0

            self._subjects = subjects
        except Exception as e:
            print(f'[wren_rag] failed to load syllabus data: {e}')
            self.chunks = []
            self._doc_tokens = []
            self._subjects = {}

    def _idf(self, word, bucket):
        df = bucket['doc_freq'].get(word, 0)
        if df == 0:
            return 0.0
        return math.log((bucket['n_docs'] + 1) / (df + 1)) + 1

    def _score(self, query_tokens, doc_tokens, bucket):
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
            idf = self._idf(word, bucket)
            denom = f + k1 * (1 - b + b * doc_len / bucket['avg_doc_len'])
            score += idf * (f * (k1 + 1)) / denom
        return score

    def retrieve(self, query, subject=None, top_k=2):
        query_tokens = _rag_tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        if subject:
            subj_key = subject.strip().lower()
            bucket = self._subjects.get(subj_key)
            buckets = {subj_key: bucket} if bucket else {}
        else:
            buckets = self._subjects

        scored = []
        for bucket in buckets.values():
            for i in bucket['doc_idxs']:
                s = self._score(query_tokens, self._doc_tokens[i], bucket)
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
