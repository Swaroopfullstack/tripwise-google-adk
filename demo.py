"""Run the TripWise ADK workflow. Demo mode needs no API key."""
from portfolio.cli import demo, emit

if __name__ == "__main__":
    emit(demo('adk'))
