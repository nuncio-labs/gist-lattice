from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """
    Abstract base class for custom LLM and embedding providers in GistLattice.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Convert a string of text into a vector embedding.
        Must return a list of floats.
        """
        pass

    @abstractmethod
    async def analyze_interaction(self, *, prompt: str, response: str) -> dict[str, Any] | Any:
        """
        Analyze a user/assistant interaction to extract memory components.
        
        Returns:
            A dictionary containing at minimum:
            {
                "gist": str,
                "valence": float,      # between -1.0 and 1.0
                "importance": float    # between 0.0 and 1.0
            }
            Optional keys: "structural_location" (str), "core_project" (str).
            
            Alternatively, you may return a `MemoryAnalysis` Pydantic object directly.
        """
        pass
