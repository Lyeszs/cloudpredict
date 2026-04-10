# app/logging_config.py
import logging
import json
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """
    Le 'Traducteur' : Il prend un message de log classique 
    et le transforme en une ligne de données JSON.
    """
    def format(self, record):
        # On prépare la 'fiche' avec toutes les infos importantes
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Si le code a crashé, on ajoute les détails de l'erreur (la traceback)
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Si on a passé des données supplémentaires (ex: prix prédit)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
            
        return json.dumps(log_entry)

def setup_logging(level=logging.INFO):
    """
    Le 'Bouton ON' : Configure toute l'application pour utiliser 
    le format JSON vers la console.
    """
    logger = logging.getLogger("cloudpredict")
    logger.setLevel(level)

    # On nettoie pour éviter d'avoir des logs en double
    logger.handlers.clear()

    # On dit d'envoyer les logs vers la sortie standard (stdout)
    # C'est ce que Docker lira plus tard.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # On branche notre traducteur JSON
    console_handler.setFormatter(JSONFormatter())
    
    logger.addHandler(console_handler)
    
    return logger