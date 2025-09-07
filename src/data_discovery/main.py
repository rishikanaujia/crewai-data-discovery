"""
Main entry point for CrewAI Data Discovery System.
This will orchestrate the multi-agent pipeline and launch the UI.
"""

import sys
import logging

def setup_logger():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("data_discovery")

logger = setup_logger()

def main():
    logger.info("🚀 Starting CrewAI Data Discovery System...")
    # Later: initialize orchestrator, agents, UI
    logger.info("✅ System initialized successfully!")

if __name__ == "__main__":
    main()
