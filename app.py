from flask import Flask

from api.routes import create_routes
from core.config import load_config, resolve_path
from core.state import AppState
from db.database import init_db
from db.repository import VehicleRepository
from services.camera_service import CameraService
from services.label_detector import LabelDetector
from services.logger import ProcessLogger
from services.orchestrator import Orchestrator
from services.rfid_detector import RFIDDetector
from services.validator import Validator


def create_app():
    config = load_config()

    db_path = resolve_path((config.get("db") or {}).get("path") or "data/app.db")
    init_db(db_path)
    repository = VehicleRepository(db_path)

    state = AppState(
        config.get("rfid", {}),
        config.get("capture", {}),
        config.get("camera", {}),
        config.get("secondary_camera", {}),
    )

    logs_dir = resolve_path((config.get("storage") or {}).get("logs_dir") or "logs")
    logger = ProcessLogger(state, logs_dir)

    camera_service = CameraService(
        config.get("camera", {}),
        config.get("secondary_camera", {}),
        state,
        logger,
    )

    label_detector = LabelDetector(config.get("paper_detection", {}), config.get("ocr", {}))
    validator = Validator(repository, state.target_digits)

    orchestrator = Orchestrator(
        config,
        state,
        repository,
        camera_service,
        label_detector,
        validator,
        logger,
    )

    rfid_detector = RFIDDetector(config.get("rfid", {}), state, logger, orchestrator.enqueue_tag)
    with state.lock:
        state.rfid_status["host"] = rfid_detector.host
        state.rfid_status["port"] = rfid_detector.port
        state.rfid_status["enabled"] = bool(rfid_detector.host)

    camera_service.start()
    orchestrator.start()
    rfid_detector.start()

    app = Flask(__name__)
    app.register_blueprint(create_routes(config, state, repository, orchestrator, logger))

    return app, config


app, app_config = create_app()


if __name__ == "__main__":
    app_cfg = app_config.get("app", {})
    app.run(
        host=app_cfg.get("host", "0.0.0.0"),
        port=int(app_cfg.get("port", 5000)),
        debug=bool(app_cfg.get("debug", False)),
    )
