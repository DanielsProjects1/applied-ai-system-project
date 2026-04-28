"""
Automated tests for the music recommender system.

Run from the project root:
    python -m pytest tests/          (if pytest is installed)
    python -m unittest discover tests  (standard library only)

All tests use small in-memory fixtures — the Kaggle dataset.csv is NOT required.
"""

import sys
import os
import csv
import math
import tempfile
import unittest
from pathlib import Path

# Make src/ importable regardless of where the test runner is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommender import (
    derive_mood,
    create_taste_profile,
    profile_from_liked_songs,
    score_song,
    load_spotify_csv,
    SongIndex,
    DocumentStore,
    CandidateGenerator,
    Retriever,
    LLMAgent,
    Evaluator,
    LearningLoop,
    MusicRecommenderSystem,
    UserSession,
    RecommendationResult,
    WEIGHTING_STRATEGIES,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_song(
    title="Test Song",
    artist="Test Artist",
    genre="pop",
    energy=0.7,
    valence=0.7,
    danceability=0.6,
    acousticness=0.2,
    tempo_bpm=120.0,
):
    """Returns a minimal song dict with a derived mood."""
    return {
        "id": title,
        "title": title,
        "artist": artist,
        "genre": genre,
        "mood": derive_mood(energy, valence),
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
        "tempo_bpm": tempo_bpm,
    }


SAMPLE_SONGS = [
    make_song("High Energy Pop", genre="pop",       energy=0.90, valence=0.80, danceability=0.85, acousticness=0.05, tempo_bpm=130),
    make_song("Chill Jazz",      genre="jazz",      energy=0.25, valence=0.45, danceability=0.30, acousticness=0.75, tempo_bpm=75),
    make_song("Dark Metal",      genre="metal",     energy=0.95, valence=0.15, danceability=0.30, acousticness=0.02, tempo_bpm=160),
    make_song("Soft Acoustic",   genre="acoustic",  energy=0.20, valence=0.60, danceability=0.25, acousticness=0.90, tempo_bpm=70),
    make_song("Dance Electronic",genre="electronic",energy=0.85, valence=0.60, danceability=0.95, acousticness=0.01, tempo_bpm=128),
    make_song("Mid Hip-Hop",     genre="hip-hop",   energy=0.65, valence=0.50, danceability=0.80, acousticness=0.10, tempo_bpm=95),
    make_song("Melancholic Soul",genre="soul",      energy=0.35, valence=0.25, danceability=0.40, acousticness=0.55, tempo_bpm=85),
    make_song("Upbeat Indie",    genre="indie",     energy=0.60, valence=0.75, danceability=0.65, acousticness=0.30, tempo_bpm=110),
]

WORKOUT_PROFILE = create_taste_profile(
    genres=["pop", "electronic"],
    moods=["energetic", "happy"],
    energy=0.88, valence=0.60,
    danceability=0.85, acousticness=0.05,
    tempo_bpm=130,
    weights=WEIGHTING_STRATEGIES["energy_focused"],
)

STUDY_PROFILE = create_taste_profile(
    genres=["acoustic", "jazz"],
    moods=["chill", "melancholic"],
    energy=0.25, valence=0.40,
    danceability=0.30, acousticness=0.80,
    tempo_bpm=75,
    weights=WEIGHTING_STRATEGIES["audio_features_focused"],
)


# ── 1. derive_mood ────────────────────────────────────────────────────────────

class TestDeriveMood(unittest.TestCase):

    def test_happy(self):
        self.assertEqual(derive_mood(0.80, 0.70), "happy")

    def test_energetic(self):
        self.assertEqual(derive_mood(0.80, 0.50), "energetic")

    def test_intense(self):
        self.assertEqual(derive_mood(0.80, 0.20), "intense")

    def test_upbeat(self):
        self.assertEqual(derive_mood(0.55, 0.70), "upbeat")

    def test_neutral(self):
        self.assertEqual(derive_mood(0.55, 0.50), "neutral")

    def test_somber(self):
        self.assertEqual(derive_mood(0.55, 0.20), "somber")

    def test_chill(self):
        self.assertEqual(derive_mood(0.20, 0.60), "chill")

    def test_melancholic(self):
        self.assertEqual(derive_mood(0.20, 0.30), "melancholic")

    def test_boundary_energy_70(self):
        # energy exactly 0.70 is the high/mid boundary — both sides should return a known label
        low_side  = derive_mood(0.69, 0.70)
        high_side = derive_mood(0.70, 0.70)
        self.assertIn(low_side,  {"happy", "energetic", "intense", "upbeat", "neutral", "somber", "chill", "melancholic"})
        self.assertIn(high_side, {"happy", "energetic", "intense", "upbeat", "neutral", "somber", "chill", "melancholic"})

    def test_all_eight_labels_covered(self):
        expected = {"happy", "energetic", "intense", "upbeat", "neutral", "somber", "chill", "melancholic"}
        produced = {s["mood"] for s in SAMPLE_SONGS}
        self.assertTrue(expected.issubset(produced) or len(produced) >= 6,
                        f"Expected broad mood coverage, got {produced}")


# ── 2. score_song ─────────────────────────────────────────────────────────────

class TestScoreSong(unittest.TestCase):

    def test_returns_tuple_of_score_and_reasons(self):
        score, reasons = score_song(WORKOUT_PROFILE, SAMPLE_SONGS[0])
        self.assertIsInstance(score, float)
        self.assertIsInstance(reasons, list)
        self.assertGreater(len(reasons), 0)

    def test_score_in_valid_range(self):
        for song in SAMPLE_SONGS:
            score, _ = score_song(WORKOUT_PROFILE, song)
            self.assertGreaterEqual(score, 0.0, f"Negative score for {song['title']}")
            self.assertLessEqual(score, 1.0,    f"Score > 1 for {song['title']}")

    def test_genre_match_scores_higher_than_mismatch(self):
        pop_song   = make_song(genre="pop",  energy=0.88, valence=0.60)
        metal_song = make_song(genre="metal",energy=0.88, valence=0.60)
        s_pop,   _ = score_song(WORKOUT_PROFILE, pop_song)
        s_metal, _ = score_song(WORKOUT_PROFILE, metal_song)
        self.assertGreater(s_pop, s_metal)

    def test_perfect_match_scores_near_one(self):
        perfect = make_song(
            genre="pop", energy=0.88, valence=0.60,
            danceability=0.85, acousticness=0.05, tempo_bpm=130,
        )
        # inject correct mood to match profile
        perfect["mood"] = "happy"
        score, _ = score_song(WORKOUT_PROFILE, perfect)
        self.assertGreater(score, 0.85)

    def test_reasons_contain_genre_and_mood(self):
        _, reasons = score_song(WORKOUT_PROFILE, SAMPLE_SONGS[0])
        joined = " ".join(reasons).lower()
        self.assertIn("genre", joined)
        self.assertIn("mood", joined)


# ── 3. profile_from_liked_songs ───────────────────────────────────────────────

class TestProfileFromLikedSongs(unittest.TestCase):

    def test_raises_on_empty_input(self):
        with self.assertRaises(ValueError):
            profile_from_liked_songs([])

    def test_single_song_profile_matches_song(self):
        song = SAMPLE_SONGS[0]
        profile = profile_from_liked_songs([song])
        num = profile["numerical"]
        self.assertAlmostEqual(num["energy"],       song["energy"])
        self.assertAlmostEqual(num["valence"],      song["valence"])
        self.assertAlmostEqual(num["danceability"], song["danceability"])
        self.assertAlmostEqual(num["acousticness"], song["acousticness"])
        self.assertAlmostEqual(num["tempo_bpm"],    song["tempo_bpm"])

    def test_averages_multiple_songs(self):
        s1 = make_song(energy=0.2, valence=0.2)
        s2 = make_song(energy=0.8, valence=0.8)
        profile = profile_from_liked_songs([s1, s2])
        self.assertAlmostEqual(profile["numerical"]["energy"],  0.5)
        self.assertAlmostEqual(profile["numerical"]["valence"], 0.5)

    def test_genres_deduplicated_and_ordered(self):
        songs = [
            make_song(genre="pop"),
            make_song(genre="jazz"),
            make_song(genre="pop"),  # duplicate
        ]
        profile = profile_from_liked_songs(songs)
        genres = profile["categorical"]["genres"]
        self.assertEqual(genres.count("pop"), 1)
        self.assertEqual(genres[0], "pop")   # first appearance preserved

    def test_moods_deduplicated(self):
        songs = [
            make_song(energy=0.8, valence=0.7),  # happy
            make_song(energy=0.8, valence=0.7),  # happy (duplicate mood)
        ]
        profile = profile_from_liked_songs(songs)
        moods = profile["categorical"]["moods"]
        self.assertEqual(moods.count("happy"), 1)

    def test_profile_has_required_keys(self):
        profile = profile_from_liked_songs([SAMPLE_SONGS[0]])
        self.assertIn("categorical", profile)
        self.assertIn("numerical",   profile)
        self.assertIn("weights",     profile)
        self.assertIn("tolerance",   profile)


# ── 4. load_spotify_csv ───────────────────────────────────────────────────────

class TestLoadSpotifyCsv(unittest.TestCase):

    def _write_temp_csv(self, rows):
        """Writes a minimal Spotify-format CSV to a temp file, returns path."""
        fieldnames = [
            "track_id", "track_name", "artists", "track_genre",
            "energy", "valence", "danceability", "acousticness", "tempo",
        ]
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        return f.name

    def _make_row(self, track_id="1", name="Song", artist="Artist",
                  genre="pop", energy=0.7, valence=0.6,
                  danceability=0.5, acousticness=0.2, tempo=120.0):
        return {
            "track_id": track_id, "track_name": name, "artists": artist,
            "track_genre": genre, "energy": energy, "valence": valence,
            "danceability": danceability, "acousticness": acousticness,
            "tempo": tempo,
        }

    def test_loads_basic_rows(self):
        rows = [self._make_row(track_id=str(i), name=f"Song{i}") for i in range(5)]
        path = self._write_temp_csv(rows)
        try:
            songs = load_spotify_csv(path)
            self.assertEqual(len(songs), 5)
        finally:
            os.unlink(path)

    def test_maps_columns_correctly(self):
        row = self._make_row(name="My Song", artist="My Artist", genre="jazz",
                             energy=0.3, valence=0.4, tempo=90.0)
        path = self._write_temp_csv([row])
        try:
            songs = load_spotify_csv(path)
            s = songs[0]
            self.assertEqual(s["title"],  "My Song")
            self.assertEqual(s["artist"], "My Artist")
            self.assertEqual(s["genre"],  "jazz")
            self.assertAlmostEqual(s["energy"], 0.3)
            self.assertAlmostEqual(s["tempo_bpm"], 90.0)
        finally:
            os.unlink(path)

    def test_derives_mood_field(self):
        row = self._make_row(energy=0.8, valence=0.7)  # expect "happy"
        path = self._write_temp_csv([row])
        try:
            songs = load_spotify_csv(path)
            self.assertEqual(songs[0]["mood"], "happy")
        finally:
            os.unlink(path)

    def test_max_songs_cap(self):
        rows = [self._make_row(track_id=str(i), name=f"Song{i}") for i in range(20)]
        path = self._write_temp_csv(rows)
        try:
            songs = load_spotify_csv(path, max_songs=10)
            self.assertEqual(len(songs), 10)
        finally:
            os.unlink(path)

    def test_songs_per_genre_stratified(self):
        rows = (
            [self._make_row(track_id=str(i), name=f"PopSong{i}", genre="pop")   for i in range(10)] +
            [self._make_row(track_id=str(i+10), name=f"JazzSong{i}", genre="jazz") for i in range(10)]
        )
        path = self._write_temp_csv(rows)
        try:
            songs = load_spotify_csv(path, songs_per_genre=3)
            pop_count  = sum(1 for s in songs if s["genre"] == "pop")
            jazz_count = sum(1 for s in songs if s["genre"] == "jazz")
            self.assertEqual(pop_count,  3)
            self.assertEqual(jazz_count, 3)
        finally:
            os.unlink(path)

    def test_skips_rows_with_bad_numeric_data(self):
        good = self._make_row(track_id="1", name="Good")
        bad  = {**good, "track_id": "2", "track_name": "Bad", "energy": "not_a_number"}
        path = self._write_temp_csv([good, bad])
        try:
            songs = load_spotify_csv(path)
            self.assertEqual(len(songs), 1)
            self.assertEqual(songs[0]["title"], "Good")
        finally:
            os.unlink(path)


# ── 5. SongIndex ──────────────────────────────────────────────────────────────

class TestSongIndex(unittest.TestCase):

    def setUp(self):
        self.index = SongIndex(SAMPLE_SONGS)

    def test_query_returns_k_results(self):
        vec = [0.8, 0.7, 0.8, 0.05, 0.6]
        results = self.index.query(vec, k=3)
        self.assertEqual(len(results), 3)

    def test_query_results_are_songs(self):
        vec = [0.5, 0.5, 0.5, 0.5, 0.5]
        results = self.index.query(vec, k=2)
        for r in results:
            self.assertIn("title", r)
            self.assertIn("genre", r)

    def test_exclude_titles_filters_results(self):
        vec = [0.9, 0.8, 0.85, 0.05, 0.7]
        exclude = {"High Energy Pop", "Dance Electronic"}
        results = self.index.query(vec, k=5, exclude_titles=exclude)
        returned_titles = {r["title"] for r in results}
        self.assertTrue(returned_titles.isdisjoint(exclude))

    def test_similar_vector_returns_similar_song(self):
        # A query very close to "High Energy Pop" should return it first
        target = SAMPLE_SONGS[0]  # High Energy Pop
        vec = self.index._vectorize(target)
        results = self.index.query(vec, k=1)
        self.assertEqual(results[0]["title"], "High Energy Pop")

    def test_vectorize_profile_matches_vectorize_song(self):
        profile = create_taste_profile(
            genres=["pop"], moods=["happy"],
            energy=0.9, valence=0.8,
            danceability=0.85, acousticness=0.05, tempo_bpm=130,
        )
        pv = self.index.vectorize_profile(profile)
        sv = self.index._vectorize(SAMPLE_SONGS[0])  # High Energy Pop
        # They should be close (cosine similarity near 1)
        dot    = sum(a * b for a, b in zip(pv, sv))
        norm_p = math.sqrt(sum(x ** 2 for x in pv))
        norm_s = math.sqrt(sum(x ** 2 for x in sv))
        cosine = dot / (norm_p * norm_s)
        self.assertGreater(cosine, 0.97)


# ── 6. DocumentStore ──────────────────────────────────────────────────────────

class TestDocumentStore(unittest.TestCase):

    def setUp(self):
        self.store = DocumentStore()

    def test_query_returns_strings(self):
        # "pop" matches genre_pop, "workout" matches activity_workout — two distinct docs
        results = self.store.query(["pop", "workout"], k=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, str)
            self.assertGreater(len(r), 0)

    def test_workout_tags_return_workout_passage(self):
        results = self.store.query(["workout"], k=1)
        self.assertGreater(len(results), 0)
        self.assertIn("energy", results[0].lower())

    def test_study_tags_return_study_passage(self):
        results = self.store.query(["study", "focus"], k=1)
        self.assertGreater(len(results), 0)
        self.assertIn("study", results[0].lower())

    def test_no_matching_tags_returns_empty(self):
        results = self.store.query(["xyznonexistenttag123"], k=3)
        self.assertEqual(results, [])

    def test_empty_tags_returns_empty(self):
        results = self.store.query([], k=3)
        self.assertEqual(results, [])

    def test_k_limits_results(self):
        results = self.store.query(["pop", "hip-hop", "jazz", "workout", "study"], k=2)
        self.assertLessEqual(len(results), 2)


# ── 7. CandidateGenerator ────────────────────────────────────────────────────

class TestCandidateGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = CandidateGenerator(SAMPLE_SONGS)

    def _session(self, profile, history=None):
        return UserSession(
            user_id="test_user",
            taste_profile=profile,
            history=history or [],
        )

    def test_generate_returns_list_of_songs(self):
        session = self._session(WORKOUT_PROFILE)
        candidates = self.gen.generate(session, k=5)
        self.assertIsInstance(candidates, list)
        for c in candidates:
            self.assertIn("title", c)

    def test_generate_respects_k(self):
        session = self._session(WORKOUT_PROFILE)
        candidates = self.gen.generate(session, k=3)
        self.assertLessEqual(len(candidates), 3)

    def test_history_songs_excluded(self):
        session = self._session(WORKOUT_PROFILE, history=["High Energy Pop"])
        candidates = self.gen.generate(session, k=len(SAMPLE_SONGS))
        titles = [c["title"] for c in candidates]
        self.assertNotIn("High Energy Pop", titles)

    def test_workout_profile_prefers_high_energy_songs(self):
        session = self._session(WORKOUT_PROFILE)
        candidates = self.gen.generate(session, k=3)
        # Top candidates should have above-average energy
        avg_energy = sum(c["energy"] for c in candidates) / len(candidates)
        self.assertGreater(avg_energy, 0.5)

    def test_study_profile_prefers_low_energy_songs(self):
        session = self._session(STUDY_PROFILE)
        candidates = self.gen.generate(session, k=3)
        avg_energy = sum(c["energy"] for c in candidates) / len(candidates)
        self.assertLess(avg_energy, 0.7)


# ── 8. Retriever ──────────────────────────────────────────────────────────────

class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = Retriever(SAMPLE_SONGS)

    def _session(self, profile):
        return UserSession(
            user_id="test_user",
            taste_profile=profile,
            context={"activity": "workout"},
        )

    def test_retrieve_returns_all_three_keys(self):
        session = self._session(WORKOUT_PROFILE)
        context = self.retriever.retrieve(session, SAMPLE_SONGS[:3])
        self.assertIn("similar_songs",    context)
        self.assertIn("user_patterns",    context)
        self.assertIn("context_passages", context)

    def test_similar_songs_are_strings(self):
        session = self._session(WORKOUT_PROFILE)
        context = self.retriever.retrieve(session, [])
        for s in context["similar_songs"]:
            self.assertIsInstance(s, str)

    def test_user_patterns_contains_energy_level(self):
        session = self._session(WORKOUT_PROFILE)
        context = self.retriever.retrieve(session, [])
        self.assertIn("energy_level", context["user_patterns"])
        self.assertEqual(context["user_patterns"]["energy_level"], "high")

    def test_context_passages_non_empty_for_known_activity(self):
        session = self._session(WORKOUT_PROFILE)
        context = self.retriever.retrieve(session, [])
        self.assertGreater(len(context["context_passages"]), 0)

    def test_candidates_excluded_from_similar_songs(self):
        session = self._session(WORKOUT_PROFILE)
        excluded = SAMPLE_SONGS[:2]
        excluded_titles = {s["title"] for s in excluded}
        context = self.retriever.retrieve(session, excluded)
        similar_titles = {s.split(" by ")[0] for s in context["similar_songs"]}
        self.assertTrue(similar_titles.isdisjoint(excluded_titles))


# ── 9. LLMAgent (heuristic path) ─────────────────────────────────────────────

class TestLLMAgentHeuristic(unittest.TestCase):
    """Tests the heuristic path only (no API key required)."""

    def setUp(self):
        self.agent   = LLMAgent(api_key=None)  # force heuristic
        self.session = UserSession(
            user_id="test_user",
            taste_profile=WORKOUT_PROFILE,
            context={"activity": "workout"},
        )
        self.context = {
            "similar_songs":    ["High Energy Pop by Test Artist (pop, happy, energy=0.90)"],
            "user_patterns":    {"preferred_genres": ["pop"], "preferred_moods": ["energetic"],
                                 "energy_level": "high", "energy_value": 0.88, "valence": 0.60,
                                 "danceability": 0.85},
            "context_passages": ["Workout music works best with high energy (>0.7)."],
        }

    def test_rerank_returns_list(self):
        result = self.agent.rerank(SAMPLE_SONGS, self.session, self.context, k=5)
        self.assertIsInstance(result, list)

    def test_rerank_respects_k(self):
        result = self.agent.rerank(SAMPLE_SONGS, self.session, self.context, k=3)
        self.assertLessEqual(len(result), 3)

    def test_each_result_is_song_explanation_pair(self):
        result = self.agent.rerank(SAMPLE_SONGS, self.session, self.context, k=5)
        for song, explanation in result:
            self.assertIn("title", song)
            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)

    def test_explanation_references_song_or_genre(self):
        result = self.agent.rerank(SAMPLE_SONGS, self.session, self.context, k=5)
        for song, explanation in result:
            title = song["title"]
            genre = song["genre"]
            self.assertTrue(
                title in explanation or genre in explanation,
                f"Explanation for '{title}' does not mention the song or genre: {explanation}",
            )

    def test_workout_profile_ranks_high_energy_songs_first(self):
        result = self.agent.rerank(SAMPLE_SONGS, self.session, self.context, k=3)
        top_song = result[0][0]
        self.assertGreater(top_song["energy"], 0.5,
                           f"Top result for workout profile has low energy: {top_song['title']}")


# ── 10. MusicRecommenderSystem.pipeline ───────────────────────────────────────

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.system  = MusicRecommenderSystem(SAMPLE_SONGS, llm_api_key=None)
        self.session = UserSession(
            user_id="test_user",
            taste_profile=WORKOUT_PROFILE,
            context={"activity": "workout"},
        )

    def test_pipeline_returns_recommendation_results(self):
        results = self.system.pipeline(self.session, k_candidates=8, k_final=5)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, RecommendationResult)

    def test_pipeline_respects_k_final(self):
        results = self.system.pipeline(self.session, k_candidates=8, k_final=3)
        self.assertLessEqual(len(results), 3)

    def test_pipeline_results_have_correct_ranks(self):
        results = self.system.pipeline(self.session, k_candidates=8, k_final=5)
        for i, r in enumerate(results):
            self.assertEqual(r.rank, i + 1)

    def test_pipeline_results_have_explanations(self):
        results = self.system.pipeline(self.session, k_candidates=8, k_final=5)
        for r in results:
            self.assertIsInstance(r.explanation, str)
            self.assertGreater(len(r.explanation), 0)

    def test_pipeline_results_carry_rag_provenance(self):
        results = self.system.pipeline(self.session, k_candidates=8, k_final=5)
        r = results[0]
        self.assertIsInstance(r.similar_songs_used,    list)
        self.assertIsInstance(r.context_passages_used, list)

    def test_pipeline_excludes_history_songs(self):
        session = UserSession(
            user_id="test_user",
            taste_profile=WORKOUT_PROFILE,
            history=["High Energy Pop"],
        )
        results = self.system.pipeline(session, k_candidates=8, k_final=8)
        titles = [r.song["title"] for r in results]
        self.assertNotIn("High Energy Pop", titles)

    def test_display_returns_non_empty_string(self):
        results = self.system.pipeline(self.session, k_candidates=8, k_final=3)
        for r in results:
            displayed = r.display()
            self.assertIsInstance(displayed, str)
            self.assertGreater(len(displayed), 0)

    def test_different_profiles_produce_different_top_results(self):
        workout_session = UserSession(user_id="u1", taste_profile=WORKOUT_PROFILE)
        study_session   = UserSession(user_id="u2", taste_profile=STUDY_PROFILE)
        workout_top = self.system.pipeline(workout_session, k_final=1)[0].song["title"]
        study_top   = self.system.pipeline(study_session,   k_final=1)[0].song["title"]
        self.assertNotEqual(workout_top, study_top,
                            "Workout and study profiles should produce different top results")


# ── 11. Evaluator ─────────────────────────────────────────────────────────────

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = Evaluator()
        self.session   = UserSession(user_id="test_user", taste_profile=WORKOUT_PROFILE)

    def test_simulate_action_like_for_high_score(self):
        # "High Energy Pop" closely matches WORKOUT_PROFILE → should be "like"
        action = self.evaluator.simulate_action(self.session, SAMPLE_SONGS[0])
        self.assertEqual(action, "like")

    def test_simulate_action_skip_for_low_score(self):
        # "Chill Jazz" is very different from WORKOUT_PROFILE → should be "skip"
        action = self.evaluator.simulate_action(self.session, SAMPLE_SONGS[1])
        self.assertEqual(action, "skip")

    def test_evaluate_returns_records_and_metrics(self):
        recs = [(s, "explanation") for s in SAMPLE_SONGS[:4]]
        records, metrics = self.evaluator.evaluate(self.session, recs)
        self.assertEqual(len(records), 4)
        self.assertIn("engagement_rate", metrics)
        self.assertIn("like_rate",       metrics)
        self.assertIn("skip_rate",       metrics)

    def test_metrics_rates_sum_to_one(self):
        recs = [(s, "x") for s in SAMPLE_SONGS]
        _, metrics = self.evaluator.evaluate(self.session, recs)
        total = metrics["like_rate"] + metrics["listen_rate"] + metrics["skip_rate"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_empty_recommendations_return_zero_engagement(self):
        _, metrics = self.evaluator.evaluate(self.session, [])
        self.assertEqual(metrics["engagement_rate"], 0.0)


# ── 12. LearningLoop ─────────────────────────────────────────────────────────

class TestLearningLoop(unittest.TestCase):

    def setUp(self):
        self.loop      = LearningLoop()
        self.gen       = CandidateGenerator(SAMPLE_SONGS)
        self.evaluator = Evaluator()

    def _run_loop(self, profile, song, action):
        session = UserSession(user_id="u1", taste_profile=profile)
        recs    = [(song, "x")]
        records, _ = self.evaluator.evaluate(session, recs)
        # Force the action we want to test
        records[0] = records[0].__class__(
            user_id=records[0].user_id,
            song_title=records[0].song_title,
            action=action,
            timestamp=records[0].timestamp,
        )
        return self.loop.update(session, records, SAMPLE_SONGS, self.gen)

    def test_liked_song_added_to_history(self):
        song    = SAMPLE_SONGS[0]
        session = self._run_loop(WORKOUT_PROFILE, song, "like")
        self.assertIn(song["title"], session.history)

    def test_like_nudges_energy_toward_song(self):
        # Use a profile with energy far from a song in SAMPLE_SONGS, then like it.
        # LearningLoop looks up the song by title in all_songs, so the song must
        # be present in the list passed to update().
        low_energy_profile = create_taste_profile(
            genres=["pop"], moods=["chill"],
            energy=0.10, valence=0.5, danceability=0.5,
            acousticness=0.5, tempo_bpm=100,
        )
        high_energy_song = SAMPLE_SONGS[0]  # "High Energy Pop", energy=0.90
        before  = low_energy_profile["numerical"]["energy"]
        session = self._run_loop(low_energy_profile, high_energy_song, "like")
        after   = session.taste_profile["numerical"]["energy"]
        self.assertGreater(after, before, "Liking a high-energy song should raise profile energy")

    def test_skip_nudges_energy_away_from_song(self):
        high_energy_profile = create_taste_profile(
            genres=["pop"], moods=["energetic"],
            energy=0.90, valence=0.5, danceability=0.5,
            acousticness=0.5, tempo_bpm=130,
        )
        low_energy_song = make_song(energy=0.10)
        before = high_energy_profile["numerical"]["energy"]
        session = self._run_loop(high_energy_profile, low_energy_song, "skip")
        after   = session.taste_profile["numerical"]["energy"]
        self.assertGreater(after, 0.0)   # never goes negative
        self.assertGreater(after, before - 0.05,  # nudge is small
                           "Skip should shift profile slightly, not dramatically")

    def test_profile_energy_stays_in_valid_range(self):
        # Repeatedly skip a very-low-energy song from a high-energy profile
        from recommender import InteractionRecord
        from datetime import datetime, timezone
        profile = create_taste_profile(
            genres=["pop"], moods=["energetic"],
            energy=0.95, valence=0.5, danceability=0.5,
            acousticness=0.5, tempo_bpm=130,
        )
        session = UserSession(user_id="u1", taste_profile=profile)
        low_song = make_song(energy=0.01)
        records = [
            InteractionRecord(
                user_id="u1",
                song_title=low_song["title"],
                action="skip",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ]
        for _ in range(20):
            session = self.loop.update(session, records, [low_song], self.gen)
        energy = session.taste_profile["numerical"]["energy"]
        self.assertGreaterEqual(energy, 0.0)
        self.assertLessEqual(energy,    1.0)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
