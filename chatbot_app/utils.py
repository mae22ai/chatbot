import logging
from django.conf import settings
from bareunpy import Tagger

logger = logging.getLogger(__name__)

class BareunClient:
    _instance = None
    _tagger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BareunClient, cls).__new__(cls)
        return cls._instance

    def get_tagger(self):
        if self._tagger is None:
            try:
                logger.info("Initializing Bareun Tagger connection...")
                self._tagger = Tagger(settings.BAREUN_API_KEY, 'api.bareun.ai', 443)
                logger.info("Bareun Tagger initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Bareun Tagger: {e}")
                raise e
        return self._tagger

# Global instance
bareun_client = BareunClient()
