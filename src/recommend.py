import numpy as np
import random
from dotenv import load_dotenv
import os
load_dotenv()

from openai import OpenAI
import json
from typing import List, Dict

# ---------------------------
# Data Models
# ---------------------------

class Song:
    def __init__(self, song_id: int, embedding: np.ndarray, genre: str):
        self.song_id = song_id
        self.embedding = embedding
        self.genre = genre


class User:
    def __init__(self, user_id: int, preference: np.ndarray):
        self.user_id = user_id
        self.preference = preference
        self.history = []

# ---------------------------
# Candidate Generator
# ---------------------------

class CandidateGenerator:
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: User, k=20):
        scored = []
        for song in self.songs:
            score = np.dot(user.preference, song.embedding)
            scored.append((score, song))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:k]]

# ---------------------------
# Retriever (RAG)
# ---------------------------

class Retriever:
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def retrieve(self, user: User, candidates: List[Song]):
        # Simulated "context retrieval"
        context = {
            "preferred_genre": self._infer_genre(user),
            "recent_behavior": "high_energy" if random.random() > 0.5 else "chill",
            "similar_songs": [c.song_id for c in candidates[:5]]
        }
        return context

    def _infer_genre(self, user: User):
        return "hip-hop" if user.preference[0] > 0.5 else "jazz"

# ---------------------------
# LLM Agent (Simulated)
# ---------------------------
client = OpenAI()
class LLMAgent:
    def rerank(self, user, candidates, context):

        if not client.api_key:
            return candidates[:10]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            # parse JSON...

        except Exception:
            # fallback if quota/billing fails
            return candidates[:10]

# ---------------------------
# Evaluator / Simulator
# ---------------------------

class Simulator:
    def __init__(self):
        self.engagement = 0

    def simulate_user_action(self, user: User, recommendations: List[Song]):
        interactions = []

        for song in recommendations:
            prob = np.dot(user.preference, song.embedding)
            action = "skip"

            if prob > 0.7:
                action = "like"
                self.engagement += 1
            elif prob > 0.4:
                action = "listen"

            interactions.append((song.song_id, action))
            user.history.append(song.song_id)

        return interactions

# ---------------------------
# System Orchestration
# ---------------------------

class MusicRecommenderSystem:
    def __init__(self, songs: List[Song]):
        self.generator = CandidateGenerator(songs)
        self.retriever = Retriever(songs)
        self.llm = LLMAgent()
        self.simulator = Simulator()

    def run_step(self, user: User):
        # 1. Generate candidates
        candidates = self.generator.recommend(user)

        # 2. Retrieve context (RAG)
        context = self.retriever.retrieve(user, candidates)

        # 3. LLM rerank
        final_recs = self.llm.rerank(user, candidates, context)

        # 4. Simulate feedback
        interactions = self.simulator.simulate_user_action(user, final_recs)

        return final_recs, interactions

# ---------------------------
# Demo Run
# ---------------------------

def create_random_songs(n=100):
    songs = []
    for i in range(n):
        embedding = np.random.rand(5)
        genre = random.choice(["hip-hop", "jazz", "pop"])
        songs.append(Song(i, embedding, genre))
    return songs


def main():
    songs = create_random_songs(200)

    user = User(
        user_id=1,
        preference=np.array([0.8, 0.7, 0.2, 0.1, 0.3])  # prefers hip-hop + energy
    )

    system = MusicRecommenderSystem(songs)

    for step in range(5):
        recs, interactions = system.run_step(user)

        print(f"\nStep {step + 1}")
        print("Top Recommendations:", [s.song_id for s in recs])
        print("Interactions:", interactions[:5])

    print("\nTotal Engagement:", system.simulator.engagement)


if __name__ == "__main__":
    main()