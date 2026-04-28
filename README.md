# AI-Powered Music Recommender System

## Summary

This project is a RAG-based (Retrieval-Augmented Generation) music recommendation system built in Python for the AI110 final project. The system combines content-based filtering, collaborative filtering, vector similarity search, and a knowledge base to generate personalized song recommendations — and optionally uses the Anthropic Claude API to re-rank results and write natural-language explanations. It loads a real 114,000-song Spotify dataset and exposes an interactive CLI where users browse, search, and select songs they like before receiving tailored recommendations.

---

## Title and Summary

**What it does:** Given a set of songs a user marks as liked, the system builds a taste profile and runs a four-stage AI pipeline to recommend the ten songs most likely to match their taste — complete with a one-sentence explanation for each.

**Why it matters:** Most recommendation systems are black boxes. This project makes the reasoning visible: you can see which songs the vector search retrieved, which knowledge-base passages informed the ranking, and exactly why each song was chosen. That transparency is educational and makes the system more trustworthy.

---

## Architecture Overview

The system follows a four-stage RAG pipeline:

```
User Input (liked songs)
        |
        v
[ Stage 1: Candidate Generator ]
  Content-based: score every unheard song against the user's
  taste profile (genre, mood, energy, valence, tempo).
  Collaborative: boost songs liked by users with similar histories.
        |
        v
[ Stage 2: Retriever (RAG) ]
  Three parallel retrieval sources:
    - SongIndex   : cosine k-NN over 5-dim audio feature vectors
    - DocumentStore: keyword-tagged knowledge-base passages (19 docs)
    - Session history: energy label, behavioral patterns summary
  All results packed into a single context dict.
        |
        v
[ Stage 3: LLM Agent ]
  Re-ranks candidates using retrieved context.
  Generates one-sentence explanations grounded in the passages.
  Uses Anthropic Claude API if ANTHROPIC_API_KEY is set;
  falls back to a deterministic heuristic ranker otherwise.
        |
        v
[ Stage 4: Final Output ]
  RecommendationResult objects: rank, song, explanation,
  similar songs used, knowledge-base passages used.
        |
        v
[ Evaluator + Learning Loop ]  (background, for simulation)
  Simulates like/listen/skip feedback.
  Nudges the taste profile toward liked songs.
  Feeds updated interactions into collaborative filtering.
```

---

## Setup Instructions

### 1. Clone or download the project

```
applied-ai-system-project/
    src/
        main.py          <- entry point
        recommender.py   <- all core pipeline logic
        recommend.py     <- original prototype (reference only)
    data/
        dataset.csv      <- Kaggle Spotify Tracks Dataset (you must add this)
    system_diagram.md
    README.md
```

### 2. Install dependencies

```bash
pip install anthropic
```

No other third-party packages are required. The system uses only Python standard-library modules (`csv`, `math`, `json`, `collections`, `dataclasses`, `pathlib`) plus `anthropic` (optional — the system runs without it).

### 3. Add the dataset

1. Go to: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Download `dataset.csv` (~45 MB, ~114,000 tracks)
3. Place it at `data/dataset.csv` inside the project folder

### 4. (Optional) Set your Anthropic API key

```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Mac / Linux
export ANTHROPIC_API_KEY=your_key_here
```

If this is not set, the system uses a heuristic re-ranker instead of Claude — everything still works.

### 5. Run the program

```bash
# Interactive mode (default)
python src/main.py

# Demo mode — three preset profiles, no user input needed
python src/main.py --demo
```

---

## Sample Interactions

### Example 1 — Browsing and getting recommendations

```
Loading songs from dataset.csv ...
Loaded 11,400 songs  |  114 genres  |  8 moods
LLM mode: heuristic fallback

============================================================
  Music Recommender -- Interactive Mode
============================================================

  No songs liked yet.

  What would you like to do?
    1. Browse random songs
    2. Search by title / artist
    3. Search by genre
    4. Get recommendations
    5. Quit

  Choice: 1

  Page 1 of 23  (228 random songs)

   1. Blinding Lights -- The Weeknd
      Genre: pop             Mood: energetic    Energy: 0.73
   2. Redbone -- Childish Gambino
      Genre: soul            Mood: upbeat       Energy: 0.49
   3. HUMBLE. -- Kendrick Lamar
      Genre: hip-hop         Mood: energetic    Energy: 0.65
   ...

  [n] next | [1-10] like  [Enter] back
  > 1,3
  Liked: "Blinding Lights" -- The Weeknd
  Liked: "HUMBLE." -- Kendrick Lamar

  > (Enter)

  Choice: 4

============================================================
  Your Recommendations
============================================================

  Profile built from 2 liked song(s):
    Genres : ['pop', 'hip-hop']
    Moods  : ['energetic']
    Energy : 0.69  Valence: 0.51  Tempo: 126 BPM

-- Top 10 recommendations -----------------------------------

  #1  Starboy -- The Weeknd
      Genre: pop | Mood: energetic | Energy: 0.76
      'Starboy' fits your pop/energetic vibe. Genre matched (pop).

  #2  DNA. -- Kendrick Lamar
      Genre: hip-hop | Mood: intense | Energy: 0.81
      'DNA.' fits your hip-hop/intense vibe. Genre matched (hip-hop).
  ...
```

---

### Example 2 — Searching by genre

```
  Choice: 3

  Genres  --  page 1 of 6  (114 total)

     1. acoustic
     2. afrobeat
     3. alt-rock
     4. alternative
     ...

  [n] next | [1-20] pick  [type name] search  [Enter] back
  > jazz

  Page 1 of 10  (100 songs in "jazz")

   1. So What -- Miles Davis
      Genre: jazz            Mood: chill        Energy: 0.28
   2. Take Five -- Dave Brubeck
      Genre: jazz            Mood: melancholic  Energy: 0.22
  ...

  [n] next | [1-10] like  [Enter] back
  > 1,2
  Liked: "So What" -- Miles Davis
  Liked: "Take Five" -- Dave Brubeck
```

---

### Example 3 — Demo mode (no user input, preset profiles)

```bash
python src/main.py --demo
```

```
============================================================
  Workout (high-energy)
============================================================

-- Top 5 recommendations ------------------------------------

  #1  Power -- Kanye West
      Genre: hip-hop | Mood: energetic | Energy: 0.91
      'Power' fits your hip-hop/energetic vibe. Genre matched (hip-hop).

  #2  Titanium -- David Guetta ft. Sia
      Genre: electronic | Mood: intense | Energy: 0.87
      'Titanium' fits your electronic/intense vibe. Energy: 0.87.

  #3  Run the World (Girls) -- Beyonce
      Genre: pop | Mood: energetic | Energy: 0.89
      'Run the World (Girls)' fits your pop/energetic vibe. Mood matched (energetic).

-- Knowledge-base passages retrieved ------------------------
    1. Workout music works best with high energy (>0.7), high danceability,
       and fast tempo (>120 BPM). Hip-hop, electronic, and pop are popular choices.
    2. Energetic tracks have very high energy (>0.8), fast tempo, high loudness,
       and strong rhythms. Chosen for motivation or high-intensity activities.
```

---

## Design Decisions

### Why RAG?

A plain content-based filter scores songs against a taste profile but has no awareness of context — it does not know that a workout playlist should be high energy, or that a study session benefits from low energy and no lyrics. The knowledge base (DocumentStore) captures that domain knowledge as 19 tagged passages. Retrieval injects the right passages into the prompt/re-ranker so recommendations are context-aware, not just feature-matched.

### Why both content-based and collaborative filtering?

Content-based filtering alone can get stuck in a rut — it keeps recommending the same genres the user has already heard. Collaborative filtering adds diversity by surfacing songs that similar users liked, even if they are outside the user's stated genre preferences. The two methods are merged with a small collaborative boost (0.10) so neither dominates.

### Why stratified loading (`songs_per_genre=100`)?

The Kaggle dataset is sorted alphabetically by genre. Loading the first 10,000 rows would give 10,000 songs from the first 10 genres only, hiding the remaining 104. Stratified loading caps each genre at 100 songs, giving a balanced 11,400-song catalogue across all 114 genres.

### Why a heuristic fallback instead of requiring an API key?

The API key is optional by design. The heuristic re-ranker uses the same scoring logic (content match + context boost from retrieved passages) and produces reasonable results without any external dependencies. This keeps the project runnable on any machine without a paid API account, which matters for a class project.

### Trade-offs made

| Decision | Trade-off |
|---|---|
| Stratified 100 songs/genre | 11,400 songs is manageable; full 114,000 would slow scoring noticeably |
| Heuristic fallback | Less nuanced explanations than the LLM, but no API cost or latency |
| Mood derived from energy/valence | Avoids a separate ML model; 8-label quadrant scheme covers the most common listener categories |
| In-memory vector index | Simpler than a real vector database (Pinecone, FAISS); fast enough for 11k songs |
| No login / user persistence | Keeps the scope appropriate for a single-session demo |

---

## Testing Summary

### What worked

- **Stratified loading** resolved the genre coverage problem — all 114 genres now appear in the CLI.
- **The RAG pipeline** produces recommendations that are noticeably different depending on context (workout vs. study vs. evening). The knowledge-base passages are retrieved correctly and their content is reflected in the heuristic scoring.
- **Pagination** makes the full catalogue navigable — genres, search results, and random browse all page correctly with dynamic range hints.
- **Duplicate prevention** in the like system works correctly: liking the same song twice prints `Already liked` without adding a duplicate to the profile.
- **Windows encoding fix** (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`) resolved crashes on non-ASCII artist names.

### What did not work (and why)

- **Collaborative filtering** has no effect in a single-session use because there is only ever one user. The user-item matrix stays empty unless `run_step()` is called in a loop. It would work correctly in a multi-user simulation.
- **The LLM explanations** (heuristic mode) are formulaic — they follow the pattern `"'Title' fits your genre/mood vibe. Feature matched."` They are correct but not creative. The Anthropic API path produces more natural explanations but requires a paid key.
- **Tempo scoring** is occasionally off because the Kaggle dataset contains some tracks with tempo values outside the 60–180 BPM normalization range, which clips the score.

### What I learned

- Dataset shape matters more than dataset size — 114,000 songs sorted by genre is effectively 10 genres if you do not account for the sort order.
- Retrieval-augmented systems require careful prompt engineering (or heuristic design) at the re-ranking stage; a good retriever does not automatically produce good recommendations if the ranker does not use the retrieved context.
- Building a graceful fallback path is worth the extra code — it makes the system testable and presentable without needing live API access.

---

## Reflection

### What this project taught me about AI

This project made concrete something that is easy to miss when reading about AI in the abstract: retrieval and generation are separate concerns. The retriever finds relevant facts; the generator uses those facts to produce an answer. When I first implemented the system, the LLM stage had no retrieval context and produced generic recommendations. Adding the knowledge base and the vector index changed the outputs meaningfully — the system started recommending low-energy acoustic tracks for study profiles and high-BPM electronic tracks for workout profiles without being explicitly programmed to do so. The retrieval step is doing real work.

The project also taught me that designing for transparency is harder than it sounds. It was easy to produce a list of recommended songs; it was much harder to expose *why* those songs were chosen in a way a user could actually understand. The `similar_songs_used` and `context_passages_used` fields in `RecommendationResult` were added specifically to answer that question, and they turned out to be the most informative part of the output.

### What this project taught me about problem-solving

The biggest problem I hit — all 114 genres collapsing to 10 — was invisible until I went looking for it. The code ran without errors, the dataset loaded successfully, and the recommendations looked reasonable. The bug only surfaced when I noticed the genre list in the CLI was suspiciously short. That experience reinforced a habit I will keep: always verify that the data you think you loaded is actually what was loaded, especially when the source is larger than what you need.

Finally, building in small stages — prototype first, then real data, then interactive CLI, then RAG, then pagination — kept the project manageable. Each stage had a clear input and a clear output, which made it easy to test one piece at a time and pinpoint exactly where something broke.

## Testing

The project includes 70 automated unit tests in [tests/test_recommender.py](tests/test_recommender.py), covering every major component of the pipeline — mood derivation, song scoring, profile building, CSV loading, vector search, knowledge-base retrieval, candidate generation, LLM re-ranking, the full end-to-end pipeline, evaluation, and the learning loop. Tests run in under 0.03 seconds using only the Python standard library and require no dataset or API key. All 70 tests pass. Run them with:

```
python -m unittest discover tests
```

---

## Critical Reflection

### Limitations and biases

The dataset is drawn from Spotify, which over-represents mainstream Western genres and popular artists. Niche or regional music is largely absent, so the system will systematically under-serve users with less mainstream tastes. Mood is derived from only two features — energy and valence — ignoring lyrics, musical key, timbre, and cultural context, which means two songs can share a mood label while sounding nothing alike. The taste profile is a simple average of liked songs, so liking a single outlier (one very intense track in an otherwise chill selection) skews the entire profile. Genre matching is binary: the system treats "hip-hop" and "trap" as completely unrelated even though they share strong musical DNA. Finally, the knowledge base contains only 19 hand-written passages, so niche listening contexts (gaming, cooking, commuting in a specific city) produce no grounding context at all.

### Could this AI be misused?

The most realistic misuse risk is the feedback loop itself. The learning loop nudges the taste profile toward whatever the user already likes, which over time narrows recommendations and reduces the chance of discovering genuinely new music — a classic filter bubble. To mitigate this, the system could periodically inject diversity by boosting songs from genres not yet in the user's history. A second risk is API key exposure: if someone hard-codes the Anthropic key into the source file rather than reading it from an environment variable, it would be visible to anyone who views the code. The setup instructions address this by keeping the key in the environment only. Beyond these, the system's scope is limited enough (local CLI, no accounts, no persistent storage) that serious misuse is unlikely.

### What surprised me during testing

Two things stood out. First, the hidden genre bug: the dataset loaded without any errors, the song count looked reasonable, and the recommendations appeared normal — but only 10 genres were visible in the CLI. Silent data truncation is much harder to catch than a crash, and it only surfaced because I noticed the genre list looked suspiciously short. Second, a test for the learning loop failed because a custom song created inside the test was not present in the song list passed to the update function — so the energy nudge silently did nothing and the profile never changed. The system didn't raise an error; it just skipped the update. Both cases were examples of the same pattern: a quiet failure that looks like correct behavior until you check the actual output carefully.

### Collaboration with Claude (AI assistant)

This project was built collaboratively with Claude (Anthropic's AI assistant) across multiple sessions. I described the system I wanted — a RAG pipeline with a real song dataset, interactive CLI, pagination, and a learning loop — and Claude implemented each piece incrementally as I directed.

**One instance where Claude's suggestion was genuinely helpful:** When I noticed the genre list in the CLI only showed 10 genres despite the dataset having 114, Claude identified the root cause correctly: the Kaggle dataset is sorted alphabetically by genre, so loading the first N rows always yields only the first few genres. Claude then proposed the `songs_per_genre` stratified loading parameter as the fix — capping each genre individually rather than applying a flat row limit. That was the right solution and I would not have reached it as quickly on my own.

**One instance where Claude's suggestion was flawed:** Claude built collaborative filtering into the system early and described it as a core feature of the architecture. In practice, collaborative filtering requires multiple users with overlapping listening histories to work. In a single-session CLI demo, the user-item matrix is always empty, and collaborative filtering contributes nothing to any actual recommendation. Claude flagged this limitation only in the testing summary — after the feature was already built — rather than upfront when designing the architecture. I kept the code because it would work correctly in a multi-user simulation, but it was misleading to present it as an active recommendation strategy in a single-session tool.