# Model Card — AI-Powered Music Recommender System

---

## Model Details

| Field | Value |

| **Name** | AI-Powered Music Recommender System |
| **Version** | 1.0 |
| **Type** | RAG pipeline (Retrieval-Augmented Generation) |
| **Developer** | AI110 Final Project |
| **Language** | Python 3.12 |
| **License** | Educational use |
| **Contact** | danielasilokun@gmail.com |

### Component models

| Component | Role | Notes |
|---|---|---|
| Content-based scorer | Stage 1 candidate generation | Weighted attribute matching; no training required |
| Collaborative filter | Stage 1 candidate boost | Jaccard similarity over user-item interaction matrix |
| `SongIndex` (cosine k-NN) | Stage 2 vector retrieval | 5-dim audio feature space; pre-computed at load time |
| `DocumentStore` (tag overlap) | Stage 2 knowledge retrieval | 19 hand-authored passages; no ML involved |
| `LLMAgent` — heuristic | Stage 3 re-ranking (default) | Deterministic; no API key required |
| `LLMAgent` — Anthropic Claude Haiku 4.5 | Stage 3 re-ranking (optional) | `claude-haiku-4-5-20251001`; requires `ANTHROPIC_API_KEY` |

---

## Intended Use

### Primary use case

Personalized music recommendation for a single user in an interactive CLI session. The user selects songs they like by browsing, searching by title/artist, or browsing by genre. The system builds a taste profile from those selections and returns the ten best-matching songs with explanations.

### Intended users

- Students and hobbyists exploring how recommendation systems work
- Educators demonstrating RAG pipelines and content-based filtering concepts

### Out-of-scope uses

- **Production deployment**: the system has no authentication, persistent storage, or multi-user isolation
- **Streaming service integration**: output is limited to songs present in the loaded dataset; it cannot query live catalogues
- **Demographic or health inference**: the system must not be used to draw conclusions about a user's identity, health, or demographics from their musical taste
- **Automated content curation at scale**: no rate limiting, abuse protection, or content moderation is implemented

---

## How the Model Works

The system runs a four-stage pipeline on every recommendation request:

**Stage 1 — Candidate Generation**
Every song not already in the user's listening history is scored against the user's taste profile using a weighted sum across seven attributes: genre (binary match), mood (binary match), energy, valence, danceability, acousticness, and tempo. Songs liked by users with overlapping histories receive a small collaborative boost (+0.10). The top `k_candidates` (default: 50) are passed forward.

**Stage 2 — Retrieval**
Three retrieval sources run in parallel:
- *SongIndex*: cosine k-NN over 5-dimensional audio feature vectors `[energy, valence, danceability, acousticness, tempo_normalized]` to find the five most similar songs not already in the candidate pool.
- *DocumentStore*: tag-overlap search over 19 curated knowledge-base passages (genre guides, mood guides, activity guides) to retrieve up to three context paragraphs.
- *Session patterns*: a behavioral summary (energy label, preferred genres/moods, history length) derived from the taste profile.

All three outputs are packed into a single context dictionary.

**Stage 3 — LLM Re-ranking**
The context dictionary and candidate list are passed to the `LLMAgent`. If an Anthropic API key is present, Claude Haiku re-ranks the top 15 candidates and writes one-sentence explanations grounded in the retrieved passages. Otherwise, the heuristic ranker applies context-aware score boosts and generates template-based explanations.

**Stage 4 — Final Output**
Results are wrapped in `RecommendationResult` objects carrying rank, song metadata, explanation, and RAG provenance (which similar songs and passages were used).

---

## Data

### Song catalogue

| Property | Value |
|---|---|
| Source | Kaggle Spotify Tracks Dataset (maharshipandya/-spotify-tracks-dataset) |
| Total tracks | ~114,000 |
| Genres | 114 |
| Loaded subset | 11,400 (100 per genre, stratified) |
| Audio features used | energy, valence, danceability, acousticness, tempo |
| Mood labels | Derived via 8-quadrant energy/valence scheme (not from Spotify) |

### Knowledge base

19 hand-authored passages covering 5 activity contexts (workout, study, party, sleep, commute), 2 time-of-day contexts (morning, evening), 7 genre summaries, and 5 mood summaries. These are static and embedded directly in the source code.

### User data

No user data is collected or persisted. The taste profile exists only in memory for the duration of a session and is discarded on exit.

---

## Evaluation

### Automated tests

70 unit tests covering all pipeline stages. Tests use small in-memory song fixtures and require no dataset or API key. All 70 pass.

| Test class | Tests | What is verified |
|---|---|---|
| `TestDeriveMood` | 10 | All 8 mood labels, boundary values |
| `TestScoreSong` | 5 | Score range [0,1], genre/mood priority |
| `TestProfileFromLikedSongs` | 6 | Feature averaging, deduplication, error handling |
| `TestLoadSpotifyCsv` | 6 | Column mapping, stratified cap, bad-row skipping |
| `TestSongIndex` | 5 | k-NN correctness, exclusion filter, cosine similarity |
| `TestDocumentStore` | 6 | Tag matching, k limit, empty-query handling |
| `TestCandidateGenerator` | 5 | History exclusion, profile-appropriate candidates |
| `TestRetriever` | 5 | Three-key context output, energy label, exclusion |
| `TestLLMAgentHeuristic` | 5 | k limit, explanation format and content |
| `TestPipeline` | 8 | End-to-end, ranks, RAG provenance, profile divergence |
| `TestEvaluator` | 5 | Like/skip thresholds, metric rates sum to 1.0 |
| `TestLearningLoop` | 4 | Energy nudge direction, value clamp to [0, 1] |

Run: `python -m unittest discover tests`

### Simulated engagement metrics

The `Evaluator` class simulates user feedback by comparing each recommended song against the taste profile:

| Action | Condition |
|---|---|
| `like` | score > 0.80 |
| `listen` | score > 0.55 |
| `skip` | score ≤ 0.55 |

Engagement rate = (likes + listens) / total recommendations. In demo-mode runs with the workout and study preset profiles, engagement rates consistently exceed 0.60 using the heuristic ranker.

### What is not evaluated

- Real user satisfaction (no human evaluation was conducted)
- Diversity of recommendations across sessions
- Cold-start performance (no liked songs)
- LLM explanation quality beyond length checks

---

## Ethical Considerations

### Dataset bias

The Spotify dataset reflects what is popular on a commercial Western streaming platform. Genres with high stream counts (pop, hip-hop, electronic) have more songs and better data quality than niche or regional genres. Users whose tastes fall outside mainstream categories will receive sparser and potentially lower-quality recommendations.

### Mood label bias

Mood is derived from energy and valence alone. This two-feature reduction ignores lyrics, musical key, timbre, and cultural context. Two songs from different cultures may share a mood label while conveying entirely different emotional meaning.

### Filter bubble risk

The learning loop nudges the taste profile toward songs the user likes. Over repeated sessions (if state were persisted), this would narrow recommendations and reduce exposure to new genres. The current single-session design limits this effect, but it would become significant in any persistent deployment.

### Transparency

Every recommendation result carries a `similar_songs_used` and `context_passages_used` field showing exactly what retrieval context influenced it. The CLI displays this context after every recommendation batch, making the system's reasoning inspectable rather than opaque.

### No personal data collected

The system does not collect, store, or transmit any user data. The taste profile is computed from songs the user explicitly selects during a session and is discarded when the program exits.

---

## Limitations

| Limitation | Impact |
|---|---|
| Closed catalogue | Can only recommend songs present in the loaded CSV; no live music discovery |
| Binary genre matching | Treats closely related genres (hip-hop / trap) as completely dissimilar |
| Single-session collaborative filtering | Collaborative signals require multiple users with overlapping history to function; the feature is inactive in normal use |
| 19-passage knowledge base | Niche contexts (gaming, cooking, specific moods) produce no grounding passages |
| Tempo normalization clipping | Songs outside 60–180 BPM have their tempo features clipped, reducing scoring accuracy |
| Heuristic explanations | Without an API key, explanations follow a fixed template and lack creativity |
| No cold-start handling | Requires at least one liked song; returns an error on an empty profile |

---

## Caveats and Recommendations

- **Do not use engagement rate as a proxy for recommendation quality.** The simulator uses the same scoring function as the ranker, so high engagement simply confirms the two components are consistent — not that the recommendations are genuinely good.
- **Increase `songs_per_genre` for richer recommendations.** The default of 100 loads 11,400 songs. Raising it to 500 or 1,000 will improve candidate diversity at the cost of a longer startup time.
- **Set `ANTHROPIC_API_KEY` for significantly better explanations.** The heuristic fallback is functional but formulaic. Claude Haiku produces natural, varied explanations that better reflect the retrieved context.
- **Do not persist taste profiles across sessions without adding a diversity term.** The learning loop's nudge mechanism will gradually collapse the profile toward a narrow point if not balanced with occasional exploration.
