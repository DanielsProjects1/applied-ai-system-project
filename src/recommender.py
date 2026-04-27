import math
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timezone

def create_taste_profile(
    genres: list,
    moods: list,
    energy: float,
    valence: float,
    danceability: float,
    acousticness: float,
    tempo_bpm: int,
    weights: dict = None,
    tolerance: float = 0.2
) -> dict:
    """
    Creates a taste profile dictionary for music recommendation comparisons.

    This represents a user's musical preferences across both categorical
    (genres, moods) and numerical (energy, valence, etc.) song attributes.

    Args:
        genres: List of preferred genres (e.g., ["pop", "indie pop"])
        moods: List of preferred moods (e.g., ["happy", "uplifting"])
        energy: Target energy level (0-1 scale, where 0=low, 1=high)
        valence: Target valence/positivity (0-1 scale, where 0=sad, 1=happy)
        danceability: Target danceability (0-1 scale)
        acousticness: Target acousticness (0-1 scale, where 0=electronic, 1=acoustic)
        tempo_bpm: Target tempo in beats per minute (typical range: 60-180)
        weights: Dict of feature importance weights (should sum to ~1.0)
                 If None, uses default equal weighting
        tolerance: Acceptable range for numerical features (0-1 scale)
                   Default 0.2 = 20% tolerance around target values

    Returns:
        A dictionary representing the user's taste profile with the following structure:
        {
            "categorical": {"genres": [...], "moods": [...]},
            "numerical": {"energy": ..., "valence": ..., ...},
            "weights": {...},
            "tolerance": ...
        }

    Example:
        profile = create_taste_profile(
            genres=["pop", "indie pop"],
            moods=["happy", "uplifting"],
            energy=0.75,
            valence=0.80,
            danceability=0.78,
            acousticness=0.20,
            tempo_bpm=120,
            tolerance=0.15
        )
    """
    default_weights = {
        "genre": 0.20,
        "mood": 0.20,
        "energy": 0.15,
        "valence": 0.15,
        "danceability": 0.15,
        "acousticness": 0.10,
        "tempo_bpm": 0.05
    }

    return {
        "categorical": {
            "genres": genres,
            "moods": moods
        },
        "numerical": {
            "energy": energy,
            "valence": valence,
            "danceability": danceability,
            "acousticness": acousticness,
            "tempo_bpm": tempo_bpm
        },
        "weights": weights or default_weights,
        "tolerance": tolerance
    }


# ============================================================================
# POINT WEIGHTING STRATEGIES FOR ATTRIBUTE SCORING
# ============================================================================
# This section defines how each attribute contributes to a song's overall score.
# All individual attribute scores are normalized to 0-1 range, then weighted.
#
# SCORING APPROACH:
# 1. Categorical Features (Genre, Mood):
#    - BINARY MATCH: 1.0 if song's attribute is in user's preferred list
#    - NO MATCH: 0.0 if song's attribute is NOT in user's preferred list
#    - No partial credit (exact match only)
#
# 2. Numerical Features (Energy, Valence, Danceability, Acousticness, Tempo):
#    - PROXIMITY-BASED: Score = 1 - |user_target - song_value|
#    - Range: 0.0 (farthest) to 1.0 (perfect match)
#    - TEMPO NORMALIZATION: Convert BPM to 0-1 scale first
#      * Typical range: 60-180 BPM
#      * normalized_tempo = (tempo_bpm - 60) / (180 - 60) = (tempo_bpm - 60) / 120
#
# 3. Overall Score:
#    - Weighted sum of all attribute scores
#    - final_score = Σ(attribute_score × attribute_weight)
#    - All weights must sum to 1.0 for normalized output
#
# ============================================================================

# PREDEFINED WEIGHTING STRATEGIES
# Choose a strategy based on user profile type or customize your own

WEIGHTING_STRATEGIES = {
    # GENRE-FOCUSED: User cares most about genre match, mood match secondary
    "genre_focused": {
        "genre": 0.35,
        "mood": 0.20,
        "energy": 0.15,
        "valence": 0.10,
        "danceability": 0.10,
        "acousticness": 0.05,
        "tempo_bpm": 0.05,
    },

    # MOOD-FOCUSED: User cares most about mood/vibe, genre is secondary
    "mood_focused": {
        "genre": 0.20,
        "mood": 0.35,
        "energy": 0.15,
        "valence": 0.15,
        "danceability": 0.10,
        "acousticness": 0.03,
        "tempo_bpm": 0.02,
    },

    # ENERGY-FOCUSED: User wants the right energy level above all else
    # (e.g., workout music, chill study sessions)
    "energy_focused": {
        "genre": 0.15,
        "mood": 0.15,
        "energy": 0.35,
        "valence": 0.10,
        "danceability": 0.15,
        "acousticness": 0.05,
        "tempo_bpm": 0.05,
    },

    # AUDIO_FEATURES_FOCUSED: Numerical features matter more than categorical
    # (e.g., user who focuses on feel/sound over genre labels)
    "audio_features_focused": {
        "genre": 0.10,
        "mood": 0.10,
        "energy": 0.20,
        "valence": 0.20,
        "danceability": 0.20,
        "acousticness": 0.15,
        "tempo_bpm": 0.05,
    },

    # BALANCED: Equal importance across most features
    "balanced": {
        "genre": 0.18,
        "mood": 0.18,
        "energy": 0.16,
        "valence": 0.16,
        "danceability": 0.16,
        "acousticness": 0.10,
        "tempo_bpm": 0.06,
    },

    # STRICT_GENRE_MOOD: User wants their exact genre/mood, flexible on audio
    # (categorical = 60%, numerical = 40%)
    "strict_categorical": {
        "genre": 0.35,
        "mood": 0.25,
        "energy": 0.12,
        "valence": 0.10,
        "danceability": 0.10,
        "acousticness": 0.05,
        "tempo_bpm": 0.03,
    },

    # ACOUSTIC_PREFERENCE: User strongly prefers acoustic vs electronic
    "acoustic_focused": {
        "genre": 0.15,
        "mood": 0.15,
        "energy": 0.15,
        "valence": 0.15,
        "danceability": 0.10,
        "acousticness": 0.25,
        "tempo_bpm": 0.05,
    },

    # VIBE_FOCUSED: Mood, Energy, Valence matter most (the "feeling" of music)
    "vibe_focused": {
        "genre": 0.08,
        "mood": 0.30,
        "energy": 0.25,
        "valence": 0.20,
        "danceability": 0.10,
        "acousticness": 0.04,
        "tempo_bpm": 0.03,
    },
}

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py

    Expected columns: id, title, artist, genre, mood, energy, tempo_bpm,
                      valence, danceability, acousticness
    """
    import csv

    numerical_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field_name in numerical_fields:
                row[field_name] = float(row[field_name])
            songs.append(dict(row))

    return songs


# ── Real-data loading ────────────────────────────────────────────────────────

def derive_mood(energy: float, valence: float) -> str:
    """
    Maps the energy/valence pair to a mood label using an 8-quadrant scheme.

    The energy-valence space is the standard musicological model:
      high energy + high valence  → happy / euphoric
      high energy + low valence   → intense / angry
      low energy  + high valence  → chill / peaceful
      low energy  + low valence   → melancholic / sad

    Thresholds are tuned so each label appears in roughly equal proportions
    across the Spotify dataset distribution.
    """
    if energy >= 0.70:
        if valence >= 0.65:
            return "happy"
        elif valence >= 0.35:
            return "energetic"
        else:
            return "intense"
    elif energy >= 0.40:
        if valence >= 0.65:
            return "upbeat"
        elif valence >= 0.35:
            return "neutral"
        else:
            return "somber"
    else:
        if valence >= 0.50:
            return "chill"
        else:
            return "melancholic"


def load_spotify_csv(csv_path: str, max_songs: Optional[int] = None) -> List[Dict]:
    """
    Loads a real song catalogue from the Kaggle Spotify Tracks Dataset and
    converts it to the internal song format expected by score_song() and the
    rest of the pipeline.

    ── How to get the dataset ───────────────────────────────────────────────
    1. Go to: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
    2. Download dataset.csv  (~45 MB, ~114 000 tracks)
    3. Place it anywhere and pass the path here.

    ── Kaggle column → internal field mapping ───────────────────────────────
      track_id     → id
      track_name   → title
      artists      → artist
      track_genre  → genre
      tempo        → tempo_bpm
      valence, danceability, energy, acousticness  (same names, already 0-1)
      mood         → derived via derive_mood(energy, valence)

    ── Parameters ───────────────────────────────────────────────────────────
    csv_path  : path to dataset.csv
    max_songs : optional cap (useful during development to keep things fast)

    ── Returns ──────────────────────────────────────────────────────────────
    List[Dict] in the same format as load_songs(), ready for
    MusicRecommenderSystem and all scoring functions.
    """
    import csv

    float_fields = {"energy", "valence", "danceability", "acousticness", "tempo"}
    songs: List[Dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing or non-numeric audio features
            try:
                for field in float_fields:
                    row[field] = float(row[field])
            except (ValueError, KeyError):
                continue

            energy    = row["energy"]
            valence   = row["valence"]
            tempo_bpm = row["tempo"]

            # Skip songs with out-of-range values
            if not (0.0 <= energy <= 1.0 and 0.0 <= valence <= 1.0):
                continue

            songs.append({
                "id":           row.get("track_id", ""),
                "title":        row.get("track_name", "Unknown"),
                "artist":       row.get("artists", "Unknown"),
                "genre":        row.get("track_genre", "unknown"),
                "mood":         derive_mood(energy, valence),
                "energy":       energy,
                "valence":      valence,
                "danceability": row["danceability"],
                "acousticness": row["acousticness"],
                "tempo_bpm":    tempo_bpm,
            })

            if max_songs and len(songs) >= max_songs:
                break

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    weights = user_prefs["weights"]
    categorical = user_prefs["categorical"]
    numerical = user_prefs["numerical"]

    scores = {}
    reasons = []

    # Categorical: binary match (1.0 = match, 0.0 = no match)
    genre_match = song["genre"] in categorical["genres"]
    scores["genre"] = 1.0 if genre_match else 0.0
    reasons.append(f"Genre {'matched' if genre_match else 'did not match'} ({song['genre']})")

    mood_match = song["mood"] in categorical["moods"]
    scores["mood"] = 1.0 if mood_match else 0.0
    reasons.append(f"Mood {'matched' if mood_match else 'did not match'} ({song['mood']})")

    # Numerical: proximity-based score = 1 - |target - value|
    for feature in ["energy", "valence", "danceability", "acousticness"]:
        target = numerical[feature]
        value = song[feature]
        score = 1.0 - abs(target - value)
        scores[feature] = score
        reasons.append(f"{feature.capitalize()}: {score:.2f} (target={target}, song={value})")

    # Tempo: normalize to 0-1 first, then proximity score
    target_norm = (numerical["tempo_bpm"] - 60) / 120
    song_norm = (song["tempo_bpm"] - 60) / 120
    tempo_score = 1.0 - abs(target_norm - song_norm)
    scores["tempo_bpm"] = tempo_score
    reasons.append(f"Tempo: {tempo_score:.2f} (target={numerical['tempo_bpm']} BPM, song={song['tempo_bpm']} BPM)")

    # Weighted sum: final_score = Σ(attribute_score × weight)
    final_score = sum(scores[attr] * weights[attr] for attr in weights)

    return final_score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    top_k = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
    return [(song, score, reasons) for song, score, reasons in top_k]


def profile_from_liked_songs(
    liked_songs: List[Dict],
    weights: Optional[dict] = None,
) -> Dict:
    """
    Derives a taste profile by averaging the audio features of songs the user
    explicitly selected as favourites.

    Genres and moods are collected from all liked songs (deduplicated).
    Numerical targets are simple averages — enough liked songs smooth out
    outliers and produce a centroid that represents the user's actual taste.
    """
    if not liked_songs:
        raise ValueError("At least one liked song is required to build a profile.")

    n = len(liked_songs)
    avg = lambda field: sum(s[field] for s in liked_songs) / n

    genres = list(dict.fromkeys(s["genre"] for s in liked_songs))  # ordered dedup
    moods  = list(dict.fromkeys(s["mood"]  for s in liked_songs))

    return create_taste_profile(
        genres=genres,
        moods=moods,
        energy=avg("energy"),
        valence=avg("valence"),
        danceability=avg("danceability"),
        acousticness=avg("acousticness"),
        tempo_bpm=avg("tempo_bpm"),
        weights=weights,
    )


# ============================================================================
# NODE A: USER SESSION (User / Simulator Input)
# ============================================================================

@dataclass
class UserSession:
    """
    Wraps a user's taste profile, listening history, and contextual signals.
    Serves as the input to the full recommendation pipeline (Node A).
    """
    user_id: str
    taste_profile: Dict
    history: List[str] = field(default_factory=list)   # song titles already heard
    context: Dict = field(default_factory=dict)         # e.g. {"activity": "workout"}

    def to_input(self) -> Dict:
        return {
            "user_id": self.user_id,
            "taste_profile": self.taste_profile,
            "history": self.history,
            "context": self.context,
        }


# ============================================================================
# NODE B: CANDIDATE GENERATOR (Content + Collaborative Filtering)
# ============================================================================

class CandidateGenerator:
    """
    Generates a pool of candidate songs using two strategies:
      - Content-based: score_song() attribute matching against taste profile
      - Collaborative: songs liked by users with overlapping listening history
    Results are merged and de-duplicated before passing to the Retriever.
    """

    def __init__(self, songs: List[Dict]):
        self.songs = songs
        # user_item_matrix: {user_id: {song_title: action}}
        self.user_item_matrix: Dict[str, Dict[str, str]] = defaultdict(dict)

    def record_interaction(self, user_id: str, song_title: str, action: str):
        """Records a user-song interaction for collaborative filtering."""
        self.user_item_matrix[user_id][song_title] = action

    def _content_based_candidates(
        self, session: UserSession, k: int
    ) -> List[Tuple[Dict, float]]:
        pool = [s for s in self.songs if s.get("title") not in session.history]
        scored = [(s, score_song(session.taste_profile, s)[0]) for s in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _collaborative_candidates(
        self, session: UserSession, k: int
    ) -> List[Tuple[Dict, float]]:
        if not self.user_item_matrix or session.user_id not in self.user_item_matrix:
            return []

        my_likes = {
            title
            for title, action in self.user_item_matrix[session.user_id].items()
            if action == "like"
        }

        song_scores: Dict[str, float] = defaultdict(float)
        for other_id, interactions in self.user_item_matrix.items():
            if other_id == session.user_id:
                continue
            other_likes = {t for t, a in interactions.items() if a == "like"}
            overlap = len(my_likes & other_likes)
            if overlap == 0:
                continue
            similarity = overlap / max(len(my_likes | other_likes), 1)
            for title, action in interactions.items():
                if title not in session.history and action == "like":
                    song_scores[title] += similarity

        title_to_song = {s["title"]: s for s in self.songs if "title" in s}
        collab = [
            (title_to_song[t], score)
            for t, score in sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
            if t in title_to_song
        ]
        return collab[:k]

    def generate(self, session: UserSession, k: int = 20) -> List[Dict]:
        """Merges content-based and collaborative candidates, returns top-k."""
        content = self._content_based_candidates(session, k)
        collab = self._collaborative_candidates(session, k // 2)

        seen: set = set()
        merged: List[Tuple[Dict, float]] = []

        for song, score in content:
            key = song.get("title", id(song))
            if key not in seen:
                seen.add(key)
                merged.append((song, score))

        for song, score in collab:
            key = song.get("title", id(song))
            if key not in seen:
                seen.add(key)
                merged.append((song, score + 0.10))  # small collaborative boost

        merged.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in merged[:k]]


# ============================================================================
# NODE C: RETRIEVER (RAG — Similar Songs, User Patterns, Context Docs)
# ============================================================================

class SongIndex:
    """
    Vector index over song audio features for nearest-neighbor retrieval.

    Each song is encoded as a 5-dim vector:
      [energy, valence, danceability, acousticness, tempo_normalized]

    Similarity is computed with cosine distance so that magnitude differences
    in feature scale don't skew results.
    """

    FEATS = ["energy", "valence", "danceability", "acousticness"]
    TEMPO_MIN, TEMPO_MAX = 60.0, 180.0

    def __init__(self, songs: List[Dict]):
        # Pre-compute and cache all vectors at init time
        self._index: List[Tuple[List[float], Dict]] = [
            (self._vectorize(s), s) for s in songs
        ]

    def _vectorize(self, song: Dict) -> List[float]:
        vec = [float(song.get(f, 0.0)) for f in self.FEATS]
        raw_tempo = float(song.get("tempo_bpm", 120))
        tempo_norm = (raw_tempo - self.TEMPO_MIN) / (self.TEMPO_MAX - self.TEMPO_MIN)
        vec.append(max(0.0, min(1.0, tempo_norm)))
        return vec

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(
        self,
        query_vec: List[float],
        k: int = 5,
        exclude_titles: Optional[set] = None,
    ) -> List[Dict]:
        """Returns top-k songs most similar to query_vec, excluding given titles."""
        exclude = exclude_titles or set()
        scored = [
            (self._cosine(query_vec, vec), song)
            for vec, song in self._index
            if song.get("title") not in exclude
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [song for _, song in scored[:k]]

    def vectorize_profile(self, taste_profile: Dict) -> List[float]:
        """Converts the numerical section of a taste profile into a query vector."""
        num = taste_profile["numerical"]
        vec = [float(num.get(f, 0.0)) for f in self.FEATS]
        raw_tempo = float(num.get("tempo_bpm", 120))
        tempo_norm = (raw_tempo - self.TEMPO_MIN) / (self.TEMPO_MAX - self.TEMPO_MIN)
        vec.append(max(0.0, min(1.0, tempo_norm)))
        return vec


class DocumentStore:
    """
    Keyword-tagged knowledge base for context passage retrieval.

    Each document has a list of tags (genre names, mood words, activity keywords).
    A query is a bag of tags; documents are ranked by how many query tags they match.
    This gives the LLM agent domain knowledge it would otherwise lack.
    """

    KNOWLEDGE_BASE = [
        # Activity guidance
        {"doc_id": "activity_workout",
         "tags": ["workout", "exercise", "gym", "running", "training"],
         "text": "Workout music works best with high energy (>0.7), high danceability, "
                 "and fast tempo (>120 BPM). Hip-hop, electronic, and pop are popular choices."},
        {"doc_id": "activity_study",
         "tags": ["study", "focus", "work", "concentration", "reading"],
         "text": "Study music should be low energy (<0.4), low valence, and preferably "
                 "instrumental. Acoustic, ambient, and lo-fi genres help maintain focus."},
        {"doc_id": "activity_party",
         "tags": ["party", "dance", "club", "social", "celebration"],
         "text": "Party music requires high danceability (>0.7), high energy, and upbeat "
                 "valence. Electronic, pop, and hip-hop dominate dance floors."},
        {"doc_id": "activity_sleep",
         "tags": ["sleep", "relax", "wind down", "calm", "meditation"],
         "text": "Sleep and relaxation music should have very low energy (<0.3), high "
                 "acousticness, slow tempo (<80 BPM), and gentle melodies."},
        {"doc_id": "activity_commute",
         "tags": ["commute", "travel", "drive", "road trip"],
         "text": "Commute music works well at mid energy (0.4–0.7) with engaging rhythms "
                 "and familiar songs. Pop and rock are common choices."},
        # Time-of-day guidance
        {"doc_id": "time_morning",
         "tags": ["morning", "wake up", "energize", "start"],
         "text": "Morning listeners prefer uplifting, energetic tracks with positive valence "
                 "to start the day. Upbeat pop and indie work well."},
        {"doc_id": "time_evening",
         "tags": ["evening", "night", "chill", "wind down"],
         "text": "Evening listeners prefer mellow, acoustic, or jazz tracks with lower energy "
                 "as they decompress from the day."},
        # Genre knowledge
        {"doc_id": "genre_pop",
         "tags": ["pop", "indie pop"],
         "text": "Pop features catchy melodies, high production value, and broad appeal. "
                 "Valence is typically high, making it feel upbeat and accessible."},
        {"doc_id": "genre_hiphop",
         "tags": ["hip-hop", "rap", "hiphop", "trap"],
         "text": "Hip-hop is characterized by rhythmic vocals, strong basslines, and high "
                 "danceability. Energy ranges from laid-back lo-fi to aggressive trap."},
        {"doc_id": "genre_rock",
         "tags": ["rock", "alternative", "indie rock", "metal"],
         "text": "Rock spans a wide energy range—from soft acoustic ballads to high-energy "
                 "metal. Guitar-driven, it tends toward lower acousticness in electric forms."},
        {"doc_id": "genre_electronic",
         "tags": ["electronic", "edm", "house", "techno", "dance"],
         "text": "Electronic music is defined by synthesized sounds, high danceability, and "
                 "high energy. Acousticness is near zero; tempo is typically 120–140+ BPM."},
        {"doc_id": "genre_jazz",
         "tags": ["jazz", "blues", "soul", "r&b"],
         "text": "Jazz is improvisational and rhythmically complex, with moderate acousticness. "
                 "Energy and valence vary—from upbeat swing to melancholic slow jazz."},
        {"doc_id": "genre_classical",
         "tags": ["classical", "orchestral", "instrumental", "ambient"],
         "text": "Classical music has high acousticness, low danceability, and a wide energy "
                 "spectrum. Ideal for focus and background listening."},
        {"doc_id": "genre_lofi",
         "tags": ["lofi", "lo-fi", "chill hop"],
         "text": "Lo-fi hip-hop features low energy, relaxed tempos (70–90 BPM), warm textures, "
                 "and moderate acousticness. Perfect for studying or relaxing."},
        # Mood knowledge
        {"doc_id": "mood_happy",
         "tags": ["happy", "uplifting", "joyful", "cheerful", "positive"],
         "text": "Happy tracks have high valence (>0.7) and moderate-to-high energy. "
                 "Major keys, fast tempos, and bright timbres contribute to a joyful feel."},
        {"doc_id": "mood_sad",
         "tags": ["sad", "melancholic", "emotional", "somber", "heartbreak"],
         "text": "Sad tracks feature low valence (<0.4), slower tempos, minor keys, "
                 "and often high acousticness. Emotional resonance over danceability."},
        {"doc_id": "mood_energetic",
         "tags": ["energetic", "intense", "powerful", "aggressive", "hype"],
         "text": "Energetic tracks have very high energy (>0.8), fast tempo, high loudness, "
                 "and strong rhythms. Chosen for motivation or high-intensity activities."},
        {"doc_id": "mood_chill",
         "tags": ["chill", "relaxed", "laid-back", "mellow", "calm"],
         "text": "Chill tracks balance low-to-mid energy with positive valence. Moderate "
                 "acousticness and slower tempos create a laid-back, effortless feeling."},
        {"doc_id": "mood_romantic",
         "tags": ["romantic", "love", "sensual", "intimate"],
         "text": "Romantic music tends toward moderate energy, high or bittersweet valence, "
                 "and often features acoustic instruments or soft vocals."},
    ]

    def __init__(self, extra_docs: Optional[List[Dict]] = None):
        self._docs = list(self.KNOWLEDGE_BASE)
        if extra_docs:
            self._docs.extend(extra_docs)

    def _score(self, doc: Dict, query_tags: List[str]) -> int:
        q_set = {t.lower() for t in query_tags}
        d_set = {t.lower() for t in doc["tags"]}
        return len(q_set & d_set)

    def query(self, query_tags: List[str], k: int = 3) -> List[str]:
        """Returns text of top-k documents whose tags best overlap with query_tags."""
        if not query_tags:
            return []
        scored = [(self._score(doc, query_tags), doc) for doc in self._docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc["text"] for score, doc in scored[:k] if score > 0]


class Retriever:
    """
    RAG retriever that provides three types of grounding context to the LLM agent:

      1. Similar songs     — k-NN search over the SongIndex using the user's
                             audio-feature taste vector as the query
      2. User patterns     — behavioral summary derived from session history
                             and taste profile numerical values
      3. Context passages  — ranked knowledge-base paragraphs retrieved from
                             the DocumentStore using genres, moods, and activity
                             keywords as the query

    All three are returned as a single dict injected into the LLM prompt,
    replacing the previous hardcoded lookup table with proper retrieval.
    """

    def __init__(self, songs: List[Dict], extra_docs: Optional[List[Dict]] = None):
        self.song_index = SongIndex(songs)
        self.doc_store  = DocumentStore(extra_docs=extra_docs)

    def _build_query_tags(self, session: UserSession) -> List[str]:
        tags: List[str] = []
        tags.extend(session.taste_profile["categorical"]["genres"])
        tags.extend(session.taste_profile["categorical"]["moods"])
        tags.extend(str(v) for v in session.context.values())
        return [t.lower() for t in tags]

    def _user_patterns(self, session: UserSession) -> Dict:
        num = session.taste_profile["numerical"]
        energy = num["energy"]
        if energy > 0.7:
            energy_label = "high"
        elif energy > 0.4:
            energy_label = "moderate"
        else:
            energy_label = "low"
        return {
            "total_listened":   len(session.history),
            "preferred_genres": session.taste_profile["categorical"]["genres"],
            "preferred_moods":  session.taste_profile["categorical"]["moods"],
            "energy_level":     energy_label,
            "energy_value":     energy,
            "valence":          num["valence"],
            "danceability":     num["danceability"],
        }

    def retrieve(self, session: UserSession, candidates: List[Dict]) -> Dict:
        """
        Runs all three retrieval sources and returns a unified context dict.
        Called by MusicRecommenderSystem before passing context to LLMAgent.
        """
        # 1. Vector search — songs similar to the user's taste profile vector
        query_vec = self.song_index.vectorize_profile(session.taste_profile)
        exclude   = {c.get("title") for c in candidates} | set(session.history)
        similar   = self.song_index.query(query_vec, k=5, exclude_titles=exclude)
        similar_summaries = [
            f"{s.get('title')} by {s.get('artist')} "
            f"({s.get('genre')}, {s.get('mood')}, energy={s.get('energy', 0):.2f})"
            for s in similar
        ]

        # 2. User behavioral patterns
        patterns = self._user_patterns(session)

        # 3. Document retrieval — knowledge base passages
        query_tags       = self._build_query_tags(session)
        context_passages = self.doc_store.query(query_tags, k=3)

        return {
            "similar_songs":    similar_summaries,
            "user_patterns":    patterns,
            "context_passages": context_passages,
        }


# ============================================================================
# NODE D: LLM AGENT (Re-rank + Generate Explanations)
# ============================================================================

class LLMAgent:
    """
    Re-ranks the candidate pool using retrieval context and generates
    per-song explanations.

    Uses the Anthropic API when an api_key is provided; otherwise falls
    back to a deterministic heuristic re-ranker so the pipeline runs
    without any external dependencies.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.model = model
        self._client = None
        if api_key:
            try:
                import anthropic  # type: ignore[import-untyped]
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

    # ------------------------------------------------------------------
    # Heuristic path (no API key)
    # ------------------------------------------------------------------

    def _heuristic_boost(
        self, song: Dict, context: Dict
    ) -> float:
        boost    = 0.0
        patterns = context.get("user_patterns", {})
        # Join all retrieved passages into one string for keyword checks
        passages = " ".join(context.get("context_passages", []))

        if song.get("genre") in patterns.get("preferred_genres", []):
            boost += 0.15
        if song.get("mood") in patterns.get("preferred_moods", []):
            boost += 0.10
        if "high energy" in passages and song.get("energy", 0) > 0.7:
            boost += 0.10
        if "mellow" in passages and song.get("energy", 0) < 0.4:
            boost += 0.10
        if "danceability" in passages and song.get("danceability", 0) > 0.7:
            boost += 0.10
        return boost

    def _extract_score(self, reason: str) -> float:
        try:
            parts = reason.split(": ")
            if len(parts) >= 2:
                return float(parts[1].split(" ")[0])
        except (ValueError, IndexError):
            pass
        return -1.0

    def _build_explanation(
        self, song: Dict, reasons: List[str]
    ) -> str:
        matched = [r for r in reasons if "matched" in r]
        high = [r for r in reasons if self._extract_score(r) > 0.75]
        top = (matched + high)[:1] or reasons[:1]
        genre = song.get("genre", "")
        mood = song.get("mood", "")
        highlight = top[0] if top else ""
        return f"'{song.get('title')}' fits your {genre}/{mood} vibe. {highlight}."

    # ------------------------------------------------------------------
    # LLM path (Anthropic API)
    # ------------------------------------------------------------------

    def _llm_rerank(
        self,
        candidates: List[Dict],
        session: UserSession,
        context: Dict,
        k: int,
    ) -> List[Tuple[Dict, str]]:
        summary = "\n".join(
            f"- {s.get('title')} by {s.get('artist')} | "
            f"genre={s.get('genre')}, mood={s.get('mood')}, "
            f"energy={s.get('energy', 0):.2f}, valence={s.get('valence', 0):.2f}, "
            f"danceability={s.get('danceability', 0):.2f}"
            for s in candidates[:15]
        )

        prompt = f"""You are a music recommendation assistant.

User taste profile:
- Preferred genres: {session.taste_profile['categorical']['genres']}
- Preferred moods:  {session.taste_profile['categorical']['moods']}
- Target energy:    {session.taste_profile['numerical']['energy']}
- Target valence:   {session.taste_profile['numerical']['valence']}
- Session context:  {session.context}

Retrieved context:
- User patterns:    {context['user_patterns']}
- Similar songs:    {context['similar_songs']}
- Knowledge base passages:
{chr(10).join(f"  {i+1}. {p}" for i, p in enumerate(context.get("context_passages", [])))}

Candidates:
{summary}

Select the top {k} songs and provide one sentence of explanation each.
Reply ONLY with a JSON array: [{{"title": "...", "explanation": "..."}}]"""

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            start, end = raw.find("["), raw.rfind("]") + 1
            parsed = json.loads(raw[start:end])
            title_map = {s.get("title"): s for s in candidates}
            result = [
                (title_map[item["title"]], item.get("explanation", ""))
                for item in parsed
                if item.get("title") in title_map
            ]
            return result[:k]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: List[Dict],
        session: UserSession,
        context: Dict,
        k: int = 10,
    ) -> List[Tuple[Dict, str]]:
        """Re-ranks candidates and returns (song, explanation) pairs."""
        if self._client:
            result = self._llm_rerank(candidates, session, context, k)
            if result:
                return result

        # Heuristic fallback
        scored = []
        for song in candidates:
            base, reasons = score_song(session.taste_profile, song)
            boost = self._heuristic_boost(song, context)
            explanation = self._build_explanation(song, reasons)
            scored.append((song, base + boost, explanation))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(song, expl) for song, _, expl in scored[:k]]


# ============================================================================
# NODE F: EVALUATOR / SIMULATOR (Click / Skip / Like + Metrics)
# ============================================================================

@dataclass
class InteractionRecord:
    user_id: str
    song_title: str
    action: str       # "like" | "listen" | "skip"
    timestamp: str

class Evaluator:
    """
    Simulates or records user feedback (like / listen / skip) and computes
    engagement and accuracy metrics for a recommendation batch.
    """

    def __init__(self):
        self.interactions: List[InteractionRecord] = []

    def simulate_action(self, session: UserSession, song: Dict) -> str:
        score, _ = score_song(session.taste_profile, song)
        if score > 0.80:
            return "like"
        if score > 0.55:
            return "listen"
        return "skip"

    def record_action(
        self, session: UserSession, song: Dict, action: str
    ) -> InteractionRecord:
        record = InteractionRecord(
            user_id=session.user_id,
            song_title=song.get("title", "unknown"),
            action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.interactions.append(record)
        return record

    def evaluate(
        self,
        session: UserSession,
        recommendations: List[Tuple[Dict, str]],
    ) -> Tuple[List[InteractionRecord], Dict]:
        records = []
        for song, _ in recommendations:
            action = self.simulate_action(session, song)
            records.append(self.record_action(session, song, action))
        return records, self.compute_metrics(records)

    def compute_metrics(self, records: List[InteractionRecord]) -> Dict:
        if not records:
            return {"total": 0, "engagement_rate": 0.0, "like_rate": 0.0,
                    "listen_rate": 0.0, "skip_rate": 1.0}
        n = len(records)
        likes   = sum(1 for r in records if r.action == "like")
        listens = sum(1 for r in records if r.action == "listen")
        skips   = sum(1 for r in records if r.action == "skip")
        return {
            "total":           n,
            "like_rate":       likes   / n,
            "listen_rate":     listens / n,
            "skip_rate":       skips   / n,
            "engagement_rate": (likes + listens) / n,
        }


# ============================================================================
# NODE G: LEARNING LOOP (Model Updates → feeds back to Candidate Generator)
# ============================================================================

class LearningLoop:
    """
    Closes the feedback loop by:
      1. Logging interactions into the CandidateGenerator's user-item matrix
         so collaborative filtering improves over time.
      2. Nudging the session's numerical taste profile toward liked songs
         and away from skipped songs.
    Updated session is returned and fed back into the next pipeline step.
    """

    LIKE_NUDGE = 0.05   # shift preference toward liked song's features
    SKIP_NUDGE = 0.02   # shift preference away from skipped song's features

    def update(
        self,
        session: UserSession,
        interactions: List[InteractionRecord],
        all_songs: List[Dict],
        candidate_gen: CandidateGenerator,
    ) -> UserSession:
        title_map = {s.get("title"): s for s in all_songs}
        numerical = session.taste_profile["numerical"]
        feats = ["energy", "valence", "danceability", "acousticness"]

        for record in interactions:
            candidate_gen.record_interaction(
                record.user_id, record.song_title, record.action
            )
            song = title_map.get(record.song_title)
            if song is None:
                continue

            if record.action == "like":
                for f in feats:
                    numerical[f] += self.LIKE_NUDGE * (song[f] - numerical[f])
                if record.song_title not in session.history:
                    session.history.append(record.song_title)

            elif record.action == "listen":
                if record.song_title not in session.history:
                    session.history.append(record.song_title)

            elif record.action == "skip":
                for f in feats:
                    numerical[f] -= self.SKIP_NUDGE * (song[f] - numerical[f])
                    numerical[f] = max(0.0, min(1.0, numerical[f]))

        return session


# ============================================================================
# NODE H: HUMAN / TESTER (Validate Recommendations + Check Explanations)
# ============================================================================

class HumanTester:
    """
    Provides a validation layer where a human (or automated check) can
    inspect recommendations and explanations before or after delivery.
    Feedback is logged and can be injected back into the LLM agent prompt.
    """

    def __init__(self):
        self.validation_log: List[Dict] = []

    def validate_recommendations(
        self,
        recommendations: List[Tuple[Dict, str]],
        interactive: bool = False,
    ) -> List[Dict]:
        results = []
        for song, explanation in recommendations:
            if interactive:
                print(f"\nSong: {song.get('title')} by {song.get('artist')}")
                print(f"Explanation: {explanation}")
                rating = input("Rate (1-5, or Enter to skip): ").strip()
                verdict = {
                    "song": song.get("title"),
                    "explanation": explanation,
                    "human_rating": rating,
                }
            else:
                verdict = {
                    "song": song.get("title"),
                    "explanation": explanation,
                    "valid_explanation": bool(explanation and len(explanation) > 20),
                }
            self.validation_log.append(verdict)
            results.append(verdict)
        return results

    def check_explanations(
        self, recommendations: List[Tuple[Dict, str]]
    ) -> List[str]:
        """Returns quality issues found in explanations (empty list = all good)."""
        issues = []
        for song, explanation in recommendations:
            title = song.get("title", "unknown")
            if not explanation:
                issues.append(f"{title}: missing explanation")
            elif len(explanation) < 20:
                issues.append(f"{title}: explanation too short")
            elif title not in explanation and song.get("genre", "") not in explanation:
                issues.append(f"{title}: explanation lacks song/genre reference")
        return issues


# ============================================================================
# FINAL OUTPUT TYPE
# ============================================================================

@dataclass
class RecommendationResult:
    """
    Represents one item in the final output of the RAG pipeline.

    Produced by Stage 4 of pipeline() so callers always receive a typed,
    inspectable object rather than a raw (song, explanation) tuple.
    """
    rank:        int
    song:        Dict
    explanation: str
    # RAG provenance — what context influenced this recommendation
    similar_songs_used:    List[str]
    context_passages_used: List[str]

    def display(self) -> str:
        return (
            f"#{self.rank}  {self.song.get('title')} — {self.song.get('artist')}\n"
            f"    Genre: {self.song.get('genre')} | Mood: {self.song.get('mood')} | "
            f"Energy: {self.song.get('energy', 0):.2f}\n"
            f"    {self.explanation}"
        )


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class MusicRecommenderSystem:
    """
    Orchestrates the four-stage RAG pipeline:

        Recommender → Retrieval → LLM (RAG) → Final output

    pipeline() is the core method and returns a clean List[RecommendationResult].
    run_step() wraps it with evaluation and learning for iterative simulation.
    """

    def __init__(
        self,
        songs: List[Dict],
        llm_api_key: Optional[str] = None,
        llm_model: str = "claude-haiku-4-5-20251001",
        extra_docs: Optional[List[Dict]] = None,
    ):
        self.songs         = songs
        self.candidate_gen = CandidateGenerator(songs)
        self.retriever     = Retriever(songs, extra_docs=extra_docs)
        self.llm_agent     = LLMAgent(api_key=llm_api_key, model=llm_model)
        self.evaluator     = Evaluator()
        self.learning_loop = LearningLoop()
        self.human_tester  = HumanTester()

    # ------------------------------------------------------------------
    # Stage 1 → 2 → 3 → 4 : the core RAG pipeline
    # ------------------------------------------------------------------

    def pipeline(
        self,
        session: UserSession,
        k_candidates: int = 20,
        k_final: int = 10,
    ) -> List[RecommendationResult]:
        """
        Runs the four-stage RAG pipeline and returns ranked results.

        Stage 1 — Recommender
            CandidateGenerator scores every unheard song against the user's
            taste profile (content-based) and collaborative signals, returning
            the top k_candidates pool.

        Stage 2 — Retrieval
            Retriever runs three retrieval sources in parallel:
              • SongIndex  — k-NN vector search for audio-similar songs
              • DocumentStore — tag-matched knowledge-base passages
              • Session history — behavioural patterns summary
            All results are packed into a single context dict.

        Stage 3 — LLM (RAG)
            LLMAgent receives the candidate pool *and* the retrieved context.
            It re-ranks candidates using context-aware boosting and generates
            a one-sentence explanation grounded in the retrieved passages.
            (Uses Anthropic API when a key is supplied; heuristic fallback
            otherwise — same interface either way.)

        Stage 4 — Final output
            Results are wrapped into RecommendationResult objects that carry
            the song, explanation, rank, and RAG provenance.
        """
        # ── Stage 1: Recommender ─────────────────────────────────────────
        candidates = self.candidate_gen.generate(session, k=k_candidates)

        # ── Stage 2: Retrieval ───────────────────────────────────────────
        context = self.retriever.retrieve(session, candidates)

        # ── Stage 3: LLM (RAG) ──────────────────────────────────────────
        ranked = self.llm_agent.rerank(candidates, session, context, k=k_final)

        # ── Stage 4: Final output ────────────────────────────────────────
        return [
            RecommendationResult(
                rank=i + 1,
                song=song,
                explanation=explanation,
                similar_songs_used=context["similar_songs"],
                context_passages_used=context["context_passages"],
            )
            for i, (song, explanation) in enumerate(ranked)
        ]

    # ------------------------------------------------------------------
    # Full step: pipeline + evaluation + learning loop
    # ------------------------------------------------------------------

    def run_step(
        self,
        session: UserSession,
        k_candidates: int = 20,
        k_final: int = 10,
        human_validate: bool = False,
    ) -> Dict:
        """
        Runs pipeline() then layers evaluation and learning on top.
        Use this for simulation/training loops; use pipeline() for inference.
        """
        results = self.pipeline(session, k_candidates, k_final)

        # Convert to (song, explanation) pairs for downstream components
        recommendations = [(r.song, r.explanation) for r in results]

        # Optional human validation
        if human_validate:
            validation = self.human_tester.validate_recommendations(
                recommendations, interactive=True
            )
        else:
            issues = self.human_tester.check_explanations(recommendations)
            validation = {"explanation_issues": issues}

        # Evaluate / simulate feedback
        interactions, metrics = self.evaluator.evaluate(session, recommendations)

        # Learning loop — update session and collaborative matrix
        session = self.learning_loop.update(
            session, interactions, self.songs, self.candidate_gen
        )

        return {
            "session":      session,
            "results":      results,
            "interactions": interactions,
            "metrics":      metrics,
            "validation":   validation,
        }

    def run(
        self,
        session: UserSession,
        steps: int = 1,
        k_candidates: int = 20,
        k_final: int = 10,
        human_validate: bool = False,
    ) -> List[Dict]:
        """Iterates run_step(); updated session feeds back into the Recommender."""
        history = []
        for step in range(steps):
            step_result = self.run_step(
                session,
                k_candidates=k_candidates,
                k_final=k_final,
                human_validate=human_validate,
            )
            session = step_result["session"]
            history.append(step_result)
            print(f"Step {step + 1} | Metrics: {step_result['metrics']}")
        return history
