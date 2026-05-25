from __future__ import annotations

from gistlattice.models import MemoryAnalysis


class FakeProviderLLM:
    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        text = f"{prompt} {response}".lower()
        valence = 0.15
        if any(word in text for word in ("angry", "frustrated", "panic", "worried", "broken")):
            valence = -0.7
        elif any(word in text for word in ("happy", "great", "thanks", "relief", "solved")):
            valence = 0.6

        structural_location = None
        for candidate in ("paris", "new york", "san francisco", "london", "berlin", "tokyo", "delhi", "mumbai", "seattle"):
            if candidate in text:
                structural_location = candidate.title()
                break

        core_project = None
        for marker in ("project", "file", "task"):
            if marker in text:
                core_project = prompt.strip().splitlines()[0][:80]
                break

        return MemoryAnalysis(
            gist=f"fake:{prompt}",
            valence=valence,
            importance=0.5,
            structural_location=structural_location,
            core_project=core_project,
        )


def build_fake_provider_llm(_settings) -> FakeProviderLLM:
    return FakeProviderLLM()
