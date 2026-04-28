"""
Music Recommender -- main entry point.

Run from the project root:
    python src/main.py              (interactive mode -- default)
    python src/main.py --demo       (run preset demo profiles instead)

Pipeline:
    Recommender -> Retrieval -> LLM (RAG) -> Final output
"""

import os
import sys
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

# UTF-8 output so non-ASCII artist/track names print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from recommender import (
    load_spotify_csv,
    profile_from_liked_songs,
    create_taste_profile,
    WEIGHTING_STRATEGIES,
    UserSession,
    MusicRecommenderSystem,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "dataset.csv"


# ── Display helpers ───────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")

def print_section(text: str) -> None:
    print(f"\n-- {text} {'-' * max(0, 55 - len(text))}")

def print_song_list(songs: list) -> None:
    for i, s in enumerate(songs, 1):
        print(f"  {i:2}. {s['title']} -- {s['artist']}")
        print(f"      Genre: {s['genre']:15} Mood: {s['mood']:12} Energy: {s['energy']:.2f}")

def print_results(results: list) -> None:
    for r in results:
        print(f"\n  {r.display()}")

def print_rag_context(results: list) -> None:
    if not results:
        return
    r = results[0]
    if r.similar_songs_used:
        print_section("Similar songs found by vector search")
        for s in r.similar_songs_used:
            print(f"    * {s}")
    if r.context_passages_used:
        print_section("Knowledge-base passages retrieved")
        for i, p in enumerate(r.context_passages_used, 1):
            print(f"    {i}. {p}")


# ── Interactive helpers ───────────────────────────────────────────────────────

def sample_diverse(songs: list, per_genre: int = 2) -> list:
    """Returns per_genre randomly chosen songs from each genre, shuffled."""
    by_genre: dict = defaultdict(list)
    for s in songs:
        by_genre[s["genre"]].append(s)
    sample = []
    for genre_songs in by_genre.values():
        sample.extend(random.sample(genre_songs, min(per_genre, len(genre_songs))))
    random.shuffle(sample)
    return sample


def search_by_title(songs: list, query: str) -> list:
    """Case-insensitive substring search across title and artist name."""
    q = query.lower()
    return [s for s in songs if q in s["title"].lower() or q in s["artist"].lower()]


def search_by_genre(songs: list, genre: str) -> list:
    """Returns all songs that exactly match the given genre."""
    return [s for s in songs if s["genre"].lower() == genre.lower()]


def list_genres(songs: list) -> list:
    """Returns a sorted list of all unique genres in the catalogue."""
    return sorted({s["genre"] for s in songs})


def ask_picks(songs: list) -> list:
    """
    Prompts the user to select songs by number from a displayed list.
    Accepts comma- or space-separated input, e.g. "1,3,7" or "2 5".
    Returns an empty list if the user presses Enter without typing.
    """
    while True:
        raw = input("\n  Pick numbers (e.g. 1,3,7) or press Enter to go back: ").strip()
        if not raw:
            return []
        try:
            indices = [int(x) - 1 for x in raw.replace(",", " ").split() if x]
            picked  = [songs[i] for i in indices if 0 <= i < len(songs)]
            if picked:
                return picked
            print("  None of those numbers matched. Try again.")
        except ValueError:
            print("  Please enter numbers only, separated by commas or spaces.")


PAGE_SIZE       = 10   # songs per page
GENRE_PAGE_SIZE = 20   # genres per page


def paginate_and_pick(results: list, liked: list, label: str = "results") -> None:
    """
    Pages through a result list PAGE_SIZE songs at a time.
    On each page the user can:
      - Type numbers (1-10) to like songs from the current page
      - Type  n  to go to the next page
      - Type  p  to go to the previous page
      - Press Enter to return to the main menu
    Picked songs are appended to liked in-place.
    """
    if not results:
        print("  No songs found.")
        return

    total       = len(results)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page        = 0

    while True:
        start      = page * PAGE_SIZE
        end        = min(start + PAGE_SIZE, total)
        page_songs = results[start:end]

        print(f"\n  Page {page + 1} of {total_pages}  ({total} {label})\n")
        print_song_list(page_songs)

        # Build navigation hint based on what's available
        nav_parts = []
        if page > 0:
            nav_parts.append("[p] prev")
        if page < total_pages - 1:
            nav_parts.append("[n] next")
        pick_range = f"1" if len(page_songs) == 1 else f"1-{len(page_songs)}"
        nav_parts.append(f"[{pick_range}] like  [Enter] back")
        print(f"\n  {' | '.join(nav_parts)}")

        raw = input("  > ").strip().lower()

        if not raw:
            break
        elif raw == "n":
            if page < total_pages - 1:
                page += 1
            else:
                print("  Already on the last page.")
        elif raw == "p":
            if page > 0:
                page -= 1
            else:
                print("  Already on the first page.")
        else:
            try:
                indices      = [int(x) - 1 for x in raw.replace(",", " ").split() if x]
                picks        = [page_songs[i] for i in indices if 0 <= i < len(page_songs)]
                liked_titles = {s["title"] for s in liked}
                new_picks    = [s for s in picks if s["title"] not in liked_titles]
                duplicates   = [s for s in picks if s["title"] in liked_titles]

                if duplicates:
                    for s in duplicates:
                        print(f'  Already liked: "{s["title"]}"')
                if new_picks:
                    liked.extend(new_picks)
                    for s in new_picks:
                        print(f'  Liked: "{s["title"]}" -- {s["artist"]}')
                if not picks:
                    print("  No valid numbers for this page. Try again.")
            except ValueError:
                print("  Enter numbers to pick, 'n'/'p' to navigate, or Enter to go back.")


def paginate_genres(genres: list) -> Optional[str]:
    """
    Pages through the genre list GENRE_PAGE_SIZE entries at a time.
    The user can navigate with n/p or type part of a genre name to jump
    directly to it without browsing every page.
    Returns the selected genre string, or None if the user cancels.
    """
    total       = len(genres)
    total_pages = max(1, (total + GENRE_PAGE_SIZE - 1) // GENRE_PAGE_SIZE)
    page        = 0

    while True:
        start       = page * GENRE_PAGE_SIZE
        end         = min(start + GENRE_PAGE_SIZE, total)
        page_genres = genres[start:end]

        print(f"\n  Genres  --  page {page + 1} of {total_pages}  ({total} total)\n")
        for i, g in enumerate(page_genres, 1):
            print(f"    {i:2}. {g}")

        nav_parts = []
        if page > 0:
            nav_parts.append("[p] prev")
        if page < total_pages - 1:
            nav_parts.append("[n] next")
        pick_range = f"1" if len(page_genres) == 1 else f"1-{len(page_genres)}"
        nav_parts.append(f"[{pick_range}] pick  [type name] search  [Enter] back")
        print(f"\n  {' | '.join(nav_parts)}")

        raw = input("  > ").strip()

        if not raw:
            return None
        if raw.lower() == "n" and page < total_pages - 1:
            page += 1
            continue
        if raw.lower() == "p" and page > 0:
            page -= 1
            continue

        # Try numeric pick from current page
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(page_genres):
                return page_genres[idx]
            print("  Number out of range for this page.")
            continue
        except ValueError:
            pass

        # Treat as a name search across ALL genres
        matches = [g for g in genres if raw.lower() in g.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Jump to the page containing the first match and highlight options
            print(f'\n  Multiple matches for "{raw}": {", ".join(matches)}')
            print("  Be more specific, or use a number to pick from the list above.")
        else:
            print(f'  No genre matching "{raw}".')


def ask_activity() -> dict:
    """Optionally captures what the user is doing so RAG can retrieve better docs."""
    activities = ["workout", "study", "party", "commute", "morning", "evening", "sleep"]
    print("\n  What are you doing right now? (optional)")
    for i, a in enumerate(activities, 1):
        print(f"    {i}. {a}")
    raw = input("  Enter a number or press Enter to skip: ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(activities):
            return {"activity": activities[idx]}
    except ValueError:
        pass
    return {}


# ── Interactive mode ─────────────────────────────────────────────────────────

def _show_and_pick(pool: list, liked: list) -> None:
    """Displays pool, collects picks, appends to liked in-place."""
    if not pool:
        print("  No songs found.")
        return
    print(f"\n  {len(pool)} song(s):\n")
    print_song_list(pool)
    picks = ask_picks(pool)
    liked.extend(picks)
    if picks:
        print(f"\n  Added {len(picks)} song(s) to your likes.")


def interactive_mode(songs: list, system: MusicRecommenderSystem) -> None:
    """
    Menu-driven loop that lets the user build a liked-songs list via:
      1. Browse  -- random diverse sample across all genres
      2. Search by title / artist -- substring match
      3. Search by genre          -- pick a genre, see songs from it
      4. Get recommendations      -- run the RAG pipeline

    The user can mix all three search modes before asking for recommendations.
    After recommendations are shown, they can like results to refine further.
    """
    print_header("Music Recommender -- Interactive Mode")
    print("\n  Find songs you like, then get personalised recommendations.")
    print("  You can browse, search by name, or browse by genre.\n")

    liked: list = []

    while True:
        # ── Status line ──────────────────────────────────────────────────
        if liked:
            titles = ", ".join(f'"{s["title"]}"' for s in liked[-3:])
            suffix = f" ... and {len(liked) - 3} more" if len(liked) > 3 else ""
            print(f"\n  Liked so far ({len(liked)}): {titles}{suffix}")
        else:
            print("\n  No songs liked yet.")

        # ── Menu ─────────────────────────────────────────────────────────
        print("""
  What would you like to do?
    1. Browse random songs
    2. Search by title / artist
    3. Search by genre
    4. Get recommendations
    5. Quit
""")
        choice = input("  Choice: ").strip()

        # ── 1: Browse ────────────────────────────────────────────────────
        if choice == "1":
            pool = sample_diverse(songs, per_genre=2)
            paginate_and_pick(pool, liked, label="random songs")

        # ── 2: Search by title / artist ──────────────────────────────────
        elif choice == "2":
            query = input("\n  Enter song title or artist name: ").strip()
            if not query:
                continue
            pool = search_by_title(songs, query)
            if not pool:
                print(f'  No songs found matching "{query}".')
                continue
            paginate_and_pick(pool, liked, label=f'results for "{query}"')

        # ── 3: Search by genre ───────────────────────────────────────────
        elif choice == "3":
            genre = paginate_genres(list_genres(songs))
            if genre is None:
                continue
            pool = search_by_genre(songs, genre)
            paginate_and_pick(pool, liked, label=f'songs in "{genre}"')

        # ── 4: Get recommendations ───────────────────────────────────────
        elif choice == "4":
            if not liked:
                print("\n  Like at least one song first (options 1-3).")
                continue

            context = ask_activity()
            profile = profile_from_liked_songs(liked)
            session = UserSession(
                user_id="interactive_user",
                taste_profile=profile,
                history=[s["title"] for s in liked],
                context=context,
            )

            print_header("Your Recommendations")
            print(f"\n  Profile built from {len(liked)} liked song(s):")
            print(f"    Genres : {profile['categorical']['genres']}")
            print(f"    Moods  : {profile['categorical']['moods']}")
            print(f"    Energy : {profile['numerical']['energy']:.2f}  "
                  f"Valence: {profile['numerical']['valence']:.2f}  "
                  f"Tempo: {profile['numerical']['tempo_bpm']:.0f} BPM")

            results = system.pipeline(session, k_candidates=50, k_final=10)

            print_section("Top 10 recommendations")
            print_results(results)

            print_section("RAG context used")
            print_rag_context(results)

            # ── Offer refinement from recommendations ─────────────────────
            refine = input(
                "\n  Like any of these to refine your recommendations? (y/n): "
            ).strip().lower()
            if refine == "y":
                _show_and_pick([r.song for r in results], liked)
                print("\n  Re-running the pipeline with your updated likes...")
                profile = profile_from_liked_songs(liked)
                session = UserSession(
                    user_id="interactive_user",
                    taste_profile=profile,
                    history=[s["title"] for s in liked],
                    context=context,
                )
                results = system.pipeline(session, k_candidates=50, k_final=10)
                print_header("Refined Recommendations")
                print_results(results)

        # ── 5: Quit ──────────────────────────────────────────────────────
        elif choice == "5":
            print("\n  Goodbye!\n")
            break

        else:
            print("  Please enter 1, 2, 3, 4, or 5.")


# ── Demo mode (preset profiles) ──────────────────────────────────────────────

def demo_mode(system: MusicRecommenderSystem) -> None:
    profiles = [
        (
            "Workout (high-energy)",
            UserSession(
                user_id="demo_workout",
                taste_profile=create_taste_profile(
                    genres=["hip-hop", "electronic"],
                    moods=["energetic", "intense"],
                    energy=0.88, valence=0.55,
                    danceability=0.82, acousticness=0.05,
                    tempo_bpm=135,
                    weights=WEIGHTING_STRATEGIES["energy_focused"],
                ),
                context={"activity": "workout"},
            ),
        ),
        (
            "Study session (chill / acoustic)",
            UserSession(
                user_id="demo_study",
                taste_profile=create_taste_profile(
                    genres=["acoustic", "classical"],
                    moods=["chill", "melancholic"],
                    energy=0.30, valence=0.40,
                    danceability=0.35, acousticness=0.75,
                    tempo_bpm=80,
                    weights=WEIGHTING_STRATEGIES["audio_features_focused"],
                ),
                context={"activity": "study"},
            ),
        ),
        (
            "Evening wind-down",
            UserSession(
                user_id="demo_evening",
                taste_profile=create_taste_profile(
                    genres=["acoustic", "indie"],
                    moods=["upbeat", "neutral"],
                    energy=0.50, valence=0.65,
                    danceability=0.55, acousticness=0.45,
                    tempo_bpm=100,
                    weights=WEIGHTING_STRATEGIES["mood_focused"],
                ),
                context={"time_of_day": "evening"},
            ),
        ),
    ]

    for label, session in profiles:
        print_header(label)
        results = system.pipeline(session, k_candidates=30, k_final=5)
        print_section("Top 5 recommendations")
        print_results(results)
        print_section("RAG context used")
        print_rag_context(results)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        print("Download it from Kaggle: maharshipandya/-spotify-tracks-dataset")
        sys.exit(1)

    print(f"Loading songs from {DATA_PATH.name} ...")
    songs = load_spotify_csv(str(DATA_PATH), songs_per_genre=100)
    print(f"Loaded {len(songs):,} songs  |  "
          f"{len({s['genre'] for s in songs})} genres  |  "
          f"{len({s['mood'] for s in songs})} moods")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    system  = MusicRecommenderSystem(songs, llm_api_key=api_key)
    mode    = "LLM (Anthropic API)" if api_key else "heuristic fallback"
    print(f"LLM mode: {mode}\n")

    if "--demo" in sys.argv:
        demo_mode(system)
    else:
        interactive_mode(songs, system)


if __name__ == "__main__":
    main()
