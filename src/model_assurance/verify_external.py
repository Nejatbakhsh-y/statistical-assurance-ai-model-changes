"""Public entry point for read-only, strict semantic artifact verification."""

from .strict_verification import main, verify

__all__ = ["main", "verify"]

if __name__ == "__main__":
    main()
