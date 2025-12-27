from core.utils import normalize_label_value


class Validator:
    def __init__(self, repository, target_digits: int):
        self.repository = repository
        self.target_digits = target_digits

    def resolve_vehicle(self, tag: str):
        return self.repository.find_vehicle_by_tag(tag)

    def validate_label(self, vehicle: dict, detected_label: str, matched_tag: str) -> dict:
        expected_labels = [normalize_label_value(label) for label in (vehicle.get("labels") or [])]
        expected_labels = [label for label in expected_labels if label]
        detected_norm = normalize_label_value(detected_label)

        if expected_labels:
            label_match = detected_norm in expected_labels
            if label_match:
                message = "Label match."
            else:
                message = f"Label mismatch: expected {', '.join(expected_labels)}, detected {detected_norm}."
        else:
            label_match = True
            message = "No expected label set for this vehicle."

        return {
            "vehicle_id": vehicle.get("id"),
            "matched_tag": matched_tag,
            "expected_labels": expected_labels,
            "detected_label": detected_norm,
            "label_match": label_match,
            "message": message,
        }
