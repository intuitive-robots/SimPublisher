import logging
try:
    import alr_sim
except ImportError as e:
    logging.getLogger(__name__).info(e)
    logging.getLogger(__name__).info(
        "No SimulationFramework installed. The support for SimulationFramework is not available."
    )
from .sf import SFSimStreamer