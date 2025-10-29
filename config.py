"""Configuration for multi-agent RAG system."""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    COINGECKO_API_KEY: Optional[str] = None

    # ChromaDB Cloud Configuration
    CHROMA_API_KEY: Optional[str] = None
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None
    CHROMA_CLOUD_HOST: str = "api.trychroma.com"

    # ChromaDB Collection Names
    DOCUMENTS_COLLECTION: str = "test-embed"

    # Embedding Configuration
    EMBEDDING_FUNCTION: str = "default"  # Options: default, openai, cohere, jina, voyageai

    # CoinGecko MCP Server Configuration
    COINGECKO_MCP_URL: str = "https://mcp.pro-api.coingecko.com/mcp"
    COINGECKO_MCP_AUTH_URL: str = "https://mcp.pro-api.coingecko.com/auth"

    # UI Configuration
    UI_PORT: int = 7860
    UI_HOST: str = "0.0.0.0"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: list = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = [".pdf", ".txt", ".docx"]
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
            COINGECKO_API_KEY=os.getenv("COINGECKO_API_KEY"),
            COINGECKO_MCP_URL=os.getenv(
                "COINGECKO_MCP_URL",
                "https://mcp.pro-api.coingecko.com/mcp"
            ),
            COINGECKO_MCP_AUTH_URL=os.getenv(
                "COINGECKO_MCP_AUTH_URL",
                "https://mcp.pro-api.coingecko.com/auth"
            ),
            CHROMA_API_KEY=os.getenv("CHROMA_API_KEY"),
            CHROMA_TENANT=os.getenv("CHROMA_TENANT"),
            CHROMA_DATABASE=os.getenv("CHROMA_DATABASE"),
            CHROMA_CLOUD_HOST=os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com"),
            EMBEDDING_FUNCTION=os.getenv("CHROMA_EMBEDDING_FUNCTION", "default"),
            DOCUMENTS_COLLECTION=os.getenv("DOCUMENTS_COLLECTION", "mcp-test"),
            MAX_FILE_SIZE_MB=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
        )
    
    def validate(self) -> None:
        """Validate required configuration."""
        errors = []

        if not self.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY environment variable is required")

        if not self.CHROMA_API_KEY:
            errors.append("CHROMA_API_KEY environment variable is required for ChromaDB Cloud")

        if not self.CHROMA_TENANT:
            errors.append("CHROMA_TENANT environment variable is required for ChromaDB Cloud")

        if not self.CHROMA_DATABASE:
            errors.append("CHROMA_DATABASE environment variable is required for ChromaDB Cloud")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# Global config instance
config = Config.from_env()