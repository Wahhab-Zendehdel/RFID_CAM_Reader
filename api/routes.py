import time

from flask import Blueprint, jsonify, render_template, request

from core.utils import normalize_label_value, normalize_tag, utc_iso, validate_label_input


def create_routes(config, state, repository, orchestrator, logger):
    bp = Blueprint("routes", __name__)

    target_digits = int((config.get("capture") or {}).get("target_digits") or 5)
    ui_cfg = config.get("ui", {})
    rfid_cfg = config.get("rfid", {})
    vehicles_cfg = config.get("vehicles", {})
    tag_slots = int(vehicles_cfg.get("default_tag_slots") or 5)
    label_slots = int(vehicles_cfg.get("default_label_slots") or 1)

    @bp.route("/")
    def index():
        return render_template(
            "index.html",
            target_digits=target_digits,
            poll_interval_ms=int(ui_cfg.get("status_poll_ms") or ui_cfg.get("poll_interval_ms") or 1000),
            beep_default=bool(ui_cfg.get("beep_on_detection_default", False)),
            require_label_match=bool(
                (config.get("capture") or {}).get("require_label_match_for_submit", True)
            ),
            present_window_seconds=float(rfid_cfg.get("present_window_seconds") or 2.0),
            default_tag_slots=tag_slots,
            default_label_slots=label_slots,
        )

    @bp.route("/api/vehicles", methods=["GET"])
    def api_vehicles_list():
        return jsonify(repository.list_vehicles())

    @bp.route("/api/vehicles", methods=["POST"])
    def api_vehicles_upsert():
        data = request.get_json(silent=True) or {}
        tags = data.get("tags")
        if tags is None:
            tags = [data.get("tag1"), data.get("tag2"), data.get("tag3"), data.get("tag4"), data.get("tag5")]
        elif not isinstance(tags, list):
            tags = [tags]
        labels = data.get("labels")
        if labels is None and data.get("label") is not None:
            labels = [data.get("label")]
        elif labels is not None and not isinstance(labels, list):
            labels = [labels]

        tags = [normalize_tag(tag) for tag in (tags or []) if tag]
        tags = [tag for tag in tags if tag]
        normalized_labels = []
        for label in (labels or []):
            ok, label_value, err = validate_label_input(label, target_digits)
            if not ok:
                return jsonify({"success": False, "message": err}), 400
            if label_value:
                normalized_labels.append(normalize_label_value(label_value))
        labels = [label for label in normalized_labels if label]

        if len(tags) > tag_slots:
            return jsonify({"success": False, "message": f"Max {tag_slots} tags allowed"}), 400
        if len(labels) > label_slots:
            return jsonify({"success": False, "message": f"Max {label_slots} labels allowed"}), 400
        if not tags:
            return jsonify({"success": False, "message": "At least one tag is required"}), 400

        vehicle = repository.upsert_vehicle({
            "id": data.get("id"),
            "tags": tags,
            "labels": labels,
        })

        return jsonify({"success": True, "vehicle": vehicle})

    @bp.route("/api/vehicles/<int:vehicle_id>", methods=["DELETE"])
    def api_vehicles_delete(vehicle_id: int):
        ok = repository.delete_vehicle(vehicle_id)
        if not ok:
            return jsonify({"success": False, "message": "Vehicle not found"}), 404
        return jsonify({"success": True})

    @bp.route("/api/tags", methods=["GET"])
    def api_tags_alias_list():
        vehicles = repository.list_vehicles()
        items = []
        for vehicle in vehicles:
            labels = vehicle.get("labels") or []
            label = labels[0] if labels else ""
            for tag in vehicle.get("tags") or []:
                items.append({"tag": tag, "label": label, "vehicle_id": vehicle.get("id")})
        return jsonify({"items": items, "count": len(items), "target_digits": target_digits})

    @bp.route("/api/tags", methods=["POST"])
    def api_tags_alias_upsert():
        data = request.get_json(silent=True) or {}
        tag = normalize_tag(data.get("tag") or "")
        label = data.get("label")
        ok, label_value, err = validate_label_input(label, target_digits)
        if not ok:
            return jsonify({"success": False, "message": err}), 400
        if not tag:
            return jsonify({"success": False, "message": "Tag is required"}), 400

        existing = repository.find_vehicle_by_tag(tag)
        if existing:
            tags = existing.get("tags") or []
            if tag not in tags and len(tags) < tag_slots:
                tags.append(normalize_tag(tag))
            labels = existing.get("labels") or []
            if label is not None:
                labels = [label_value] if label_value else []
            if len(tags) > tag_slots or len(labels) > label_slots:
                return jsonify({"success": False, "message": "Vehicle slot limits exceeded"}), 400
            vehicle = repository.upsert_vehicle({"id": existing.get("id"), "tags": tags, "labels": labels})
            return jsonify({"success": True, "vehicle": vehicle, "message": "Updated"})

        if 1 > tag_slots:
            return jsonify({"success": False, "message": "Vehicle tag slots set to 0"}), 400
        labels = [label_value] if label_value else []
        if len(labels) > label_slots:
            return jsonify({"success": False, "message": "Vehicle label slots exceeded"}), 400
        vehicle = repository.upsert_vehicle({"tags": [tag], "labels": labels})
        return jsonify({"success": True, "vehicle": vehicle, "message": "Added"})

    @bp.route("/api/tags/<tag>", methods=["DELETE"])
    def api_tags_alias_delete(tag: str):
        existing = repository.find_vehicle_by_tag(tag)
        if not existing:
            return jsonify({"success": False, "message": "Tag not found"}), 404
        ok = repository.delete_vehicle(existing.get("id"))
        return jsonify({"success": ok})

    @bp.route("/api/status", methods=["GET"])
    def api_status():
        include_images = request.args.get("include_images") == "1"
        now = time.time()

        with state.lock:
            rfid = dict(state.rfid_status)
            capture = dict(state.capture_status)
            cam_primary = dict(state.camera_status)
            cam_secondary = dict(state.secondary_camera_status)
            queue_size = orchestrator.queue.qsize()
            tag_cooldowns = dict(state.tag_submit_cooldowns)

        rfid["queue_size"] = queue_size
        last_tag = rfid.get("last_tag")
        last_known = bool(last_tag and repository.find_vehicle_by_tag(last_tag))
        rfid["last_tag_known"] = last_known
        rfid["last_tag_approved"] = last_known

        if rfid.get("last_tag_time") is not None:
            rfid["last_tag_time_iso"] = utc_iso(rfid["last_tag_time"])
        if rfid.get("last_ok_ts") is not None:
            rfid["last_ok_ts_iso"] = utc_iso(rfid["last_ok_ts"])
            rfid["seconds_since_last_ok"] = max(0.0, now - float(rfid["last_ok_ts"]))
        last_tag_ts = rfid.get("last_tag_ts") or rfid.get("last_tag_time")
        if last_tag_ts:
            rfid["last_tag_ts"] = last_tag_ts
            rfid["last_tag_ts_iso"] = utc_iso(last_tag_ts)
            rfid["seconds_since_last_tag"] = max(0.0, now - float(last_tag_ts))
        stale_seconds = float(rfid_cfg.get("stale_seconds") or 0.0)
        if stale_seconds and rfid.get("last_ok_ts"):
            rfid["stale"] = (now - float(rfid["last_ok_ts"])) > stale_seconds
        else:
            rfid["stale"] = False

        effective_tag = capture.get("matched_tag") or capture.get("tag")
        cooldown_until = 0.0
        if effective_tag:
            cooldown_until = float(tag_cooldowns.get(effective_tag) or 0.0)
        if cooldown_until:
            capture["cooldown_until"] = cooldown_until
            capture["cooldown_until_iso"] = utc_iso(cooldown_until)
            capture["cooldown_tag"] = effective_tag

        if capture.get("started_time") is not None:
            capture["started_time_iso"] = utc_iso(capture["started_time"])
        if capture.get("number_time") is not None:
            capture["number_time_iso"] = utc_iso(capture["number_time"])

        if not include_images:
            capture["original_b64"] = None
            capture["paper_b64"] = None
            capture["secondary_b64"] = None
        if not isinstance(capture.get("expected_labels"), list):
            fallback_label = capture.get("label_expected") or ""
            capture["expected_labels"] = [fallback_label] if fallback_label else []

        vehicles = repository.list_vehicles()
        vehicles_count = len(vehicles)
        tags_count = 0
        for vehicle in vehicles:
            tags_count += len(vehicle.get("tags") or [])

        return jsonify(
            {
                "server_time": now,
                "vehicles_count": vehicles_count,
                "tags_count": tags_count,
                "approved_tags_count": vehicles_count,
                "rfid": rfid,
                "capture": capture,
                "cameras": {"primary": cam_primary, "secondary": cam_secondary},
                "events": logger.recent_events(),
            }
        )

    @bp.route("/api/capture/submit", methods=["POST"])
    def api_capture_submit():
        ok, message, saved = orchestrator.submit_capture()
        status = 200 if ok else 400
        return jsonify({"success": ok, "message": message, "saved": saved}), status

    return bp
