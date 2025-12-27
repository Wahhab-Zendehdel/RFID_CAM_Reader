SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tags_json TEXT NOT NULL,
    labels_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicle_images (
    id INTEGER PRIMARY KEY,
    primary_camera_image TEXT,
    secondary_camera_image TEXT,
    FOREIGN KEY (id) REFERENCES vehicles(id)
);
"""
