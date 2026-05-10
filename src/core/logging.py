import logging
import json
import sys
from datetime import datetime


# -------------------------
# JSON Formatter
# -------------------------

class JsonFormatter(logging.Formatter):

    def format(self, record):

        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }

        # request_id support
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id

        # exception support
        if record.exc_info:
            log_record["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_record)


# -------------------------
# Logger setup
# -------------------------

logger = logging.getLogger("scoring-api")

logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)

handler.setFormatter(JsonFormatter())

logger.addHandler(handler)
