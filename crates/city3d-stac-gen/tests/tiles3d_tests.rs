//! Tests for the 3D Tiles reader, on fabricated tilesets and tile contents

use city3d_stac::metadata::AttributeType;
use city3d_stac::reader::{get_reader, item_stem, CityModelMetadataReader, Tiles3DReader};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;

/// WGS 84 longitude, latitude (degrees) and ellipsoidal height to ECEF
fn to_ecef(lon: f64, lat: f64, h: f64) -> [f64; 3] {
    let e2 = F * (2.0 - F);
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    let n = A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    [
        (n + h) * lat.cos() * lon.cos(),
        (n + h) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + h) * lat.sin(),
    ]
}

/// JSON padded with spaces to a multiple of `align` bytes
fn padded_json(value: &Value, align: usize) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).unwrap();
    while !bytes.len().is_multiple_of(align) {
        bytes.push(b' ');
    }
    bytes
}

/// A binary glTF made of its JSON chunk only
fn glb(gltf: &Value) -> Vec<u8> {
    let json = padded_json(gltf, 4);
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&((12 + 8 + json.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    out.extend_from_slice(&json);
    out
}

/// A Batched 3D Model with a Feature Table, a Batch Table and an embedded glTF
fn b3dm(batch_length: u64, batch_table: &Value, gltf: &Value) -> Vec<u8> {
    let feature = padded_json(&json!({ "BATCH_LENGTH": batch_length }), 8);
    let batch = padded_json(batch_table, 8);
    let body = glb(gltf);
    let mut out = Vec::new();
    out.extend_from_slice(b"b3dm");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&((28 + feature.len() + batch.len() + body.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(feature.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(batch.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&feature);
    out.extend_from_slice(&batch);
    out.extend_from_slice(&body);
    out
}

/// A glTF with one property table of `count` buildings
fn building_gltf(count: u64, textured: bool) -> Value {
    let mut gltf = json!({
        "asset": { "version": "2.0" },
        "materials": [{}],
        "extensionsUsed": ["EXT_structural_metadata"],
        "extensions": {
            "EXT_structural_metadata": {
                "schema": {
                    "id": "buildings",
                    "classes": {
                        "building": {
                            "properties": {
                                "height": { "type": "SCALAR", "componentType": "FLOAT32" },
                                "usage": { "type": "STRING" },
                                "listed": { "type": "BOOLEAN" }
                            }
                        }
                    }
                },
                "propertyTables": [{ "class": "building", "count": count }]
            }
        }
    });
    if textured {
        gltf["textures"] = json!([{ "source": 0 }]);
    }
    gltf
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
}

fn write_json(dir: &Path, name: &str, value: &Value) {
    write(dir, name, &serde_json::to_vec_pretty(value).unwrap());
}

/// A region around central Paris: 2.30–2.40° E, 48.80–48.90° N, 30–200 m
fn paris_region() -> Value {
    json!([
        2.30f64.to_radians(),
        48.80f64.to_radians(),
        2.40f64.to_radians(),
        48.90f64.to_radians(),
        30.0,
        200.0
    ])
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

#[test]
fn region_bbox_is_converted_to_degrees() {
    let dir = TempDir::new().unwrap();
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 100,
            "root": { "boundingVolume": { "region": paris_region() }, "geometricError": 10, "refine": "REPLACE" }
        }),
    );
    let reader = get_reader(&dir.path().join("tileset.json")).unwrap();
    assert_eq!(reader.encoding(), "3D Tiles");
    assert_eq!(reader.version().unwrap(), "1.1");
    assert_eq!(reader.crs().unwrap().epsg, Some(4979));
    let bbox = reader.bbox().unwrap();
    assert!(close(bbox.xmin, 2.30, 1e-12) && close(bbox.xmax, 2.40, 1e-12));
    assert!(close(bbox.ymin, 48.80, 1e-12) && close(bbox.ymax, 48.90, 1e-12));
    assert!(close(bbox.zmin, 30.0, 1e-12) && close(bbox.zmax, 200.0, 1e-12));
    assert!(reader.lods().unwrap().is_empty());
}

#[test]
fn box_bbox_applies_the_root_transform_and_ecef_conversion() {
    // A 1 km box centered on the origin of a local frame, moved to Paris (48.85° N,
    // 2.35° E, 100 m) by a translation-only root transform (column-major)
    let center = to_ecef(2.35, 48.85, 100.0);
    let dir = TempDir::new().unwrap();
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 100,
            "root": {
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, center[0], center[1], center[2], 1],
                "boundingVolume": { "box": [0, 0, 0, 500, 0, 0, 0, 500, 0, 0, 0, 500] },
                "geometricError": 10,
                "refine": "REPLACE"
            }
        }),
    );
    let bbox = Tiles3DReader::new(&dir.path().join("tileset.json"))
        .unwrap()
        .bbox()
        .unwrap();
    // The box spans about ±500 m around the center: about ±0.0068° in longitude, ±0.0045° in latitude
    assert!(bbox.xmin < 2.35 && bbox.xmax > 2.35 && bbox.xmax - bbox.xmin < 0.03);
    assert!(bbox.ymin < 48.85 && bbox.ymax > 48.85 && bbox.ymax - bbox.ymin < 0.02);
    assert!(bbox.zmin < 100.0 && bbox.zmax > 100.0);
}

#[test]
fn ecef_center_round_trips_through_a_sphere() {
    let center = to_ecef(-73.98, 40.75, 25.0);
    let dir = TempDir::new().unwrap();
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.0" },
            "geometricError": 10,
            "root": { "boundingVolume": { "sphere": [center[0], center[1], center[2], 0.001] }, "geometricError": 0, "refine": "ADD" }
        }),
    );
    let bbox = Tiles3DReader::new(&dir.path().join("tileset.json"))
        .unwrap()
        .bbox()
        .unwrap();
    // Conversion back to longitude, latitude and height within a millimeter
    assert!(close(bbox.xmin, -73.98, 1e-7) && close(bbox.ymin, 40.75, 1e-7));
    assert!(close(bbox.zmin, 25.0, 2e-3));
}

#[test]
fn local_coordinates_give_no_bbox() {
    let dir = TempDir::new().unwrap();
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 10,
            "root": { "boundingVolume": { "box": [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10] }, "geometricError": 0, "refine": "ADD" }
        }),
    );
    assert!(Tiles3DReader::new(&dir.path().join("tileset.json"))
        .unwrap()
        .bbox()
        .is_err());
}

#[test]
fn glb_contents_give_classes_attributes_count_and_textures() {
    // REPLACE: the parent (a coarse version of the same 7 buildings) is replaced
    // by its two children, so only the leaves count: 3 + 4
    let dir = TempDir::new().unwrap();
    write(dir.path(), "tiles/0.glb", &glb(&building_gltf(7, false)));
    write(dir.path(), "tiles/1.glb", &glb(&building_gltf(3, true)));
    write(dir.path(), "tiles/2.glb", &glb(&building_gltf(4, false)));
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 100,
            "root": {
                "boundingVolume": { "region": paris_region() },
                "geometricError": 50,
                "refine": "REPLACE",
                "content": { "uri": "tiles/0.glb" },
                "children": [
                    { "boundingVolume": { "region": paris_region() }, "geometricError": 0, "content": { "uri": "tiles/1.glb" } },
                    { "boundingVolume": { "region": paris_region() }, "geometricError": 0, "content": { "uri": "tiles/2.glb" } }
                ]
            }
        }),
    );
    let reader = Tiles3DReader::new(&dir.path().join("tileset.json")).unwrap();
    assert_eq!(reader.city_object_count().unwrap(), 7);
    assert_eq!(reader.city_object_types().unwrap(), vec!["building"]);
    let attributes = reader.attributes().unwrap();
    let types: Vec<(&str, AttributeType)> = attributes
        .iter()
        .map(|a| (a.name.as_str(), a.attr_type))
        .collect();
    assert_eq!(
        types,
        vec![
            ("height", AttributeType::Number),
            ("listed", AttributeType::Boolean),
            ("usage", AttributeType::String)
        ]
    );
    assert!(reader.textures().unwrap());
    assert!(reader.materials().unwrap());
    assert!(reader.semantic_surfaces().is_err());
}

#[test]
fn additive_refinement_counts_every_tile() {
    let dir = TempDir::new().unwrap();
    write(dir.path(), "a.glb", &glb(&building_gltf(2, false)));
    write(dir.path(), "b.glb", &glb(&building_gltf(5, false)));
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 100,
            "root": {
                "boundingVolume": { "region": paris_region() },
                "geometricError": 50,
                "refine": "ADD",
                "content": { "uri": "a.glb" },
                "children": [{ "boundingVolume": { "region": paris_region() }, "geometricError": 0, "content": { "uri": "b.glb" } }]
            }
        }),
    );
    let reader = Tiles3DReader::new(&dir.path().join("tileset.json")).unwrap();
    assert_eq!(reader.city_object_count().unwrap(), 7);
    assert!(!reader.textures().unwrap());
}

#[test]
fn b3dm_contents_give_batch_length_and_batch_table_attributes() {
    let dir = TempDir::new().unwrap();
    let batch_table = json!({
        "id": ["a", "b", "c"],
        "height": [10.5, 12.0, 8.0],
        "year": { "byteOffset": 0, "componentType": "UNSIGNED_SHORT", "type": "SCALAR" }
    });
    let gltf =
        json!({ "asset": { "version": "2.0" }, "materials": [{}], "textures": [{ "source": 0 }] });
    write(dir.path(), "0.b3dm", &b3dm(3, &batch_table, &gltf));
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.0" },
            "geometricError": 10,
            "root": { "boundingVolume": { "region": paris_region() }, "geometricError": 0, "refine": "ADD", "content": { "uri": "0.b3dm" } }
        }),
    );
    let reader = Tiles3DReader::new(&dir.path().join("tileset.json")).unwrap();
    assert_eq!(reader.city_object_count().unwrap(), 3);
    let names: Vec<String> = reader
        .attributes()
        .unwrap()
        .into_iter()
        .map(|a| a.name)
        .collect();
    assert_eq!(names, vec!["height", "id", "year"]);
    assert!(reader.textures().unwrap());
}

#[test]
fn external_tilesets_are_followed_once() {
    let dir = TempDir::new().unwrap();
    write(dir.path(), "city/0.glb", &glb(&building_gltf(4, false)));
    write_json(
        dir.path(),
        "city/tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 10,
            "root": { "boundingVolume": { "region": paris_region() }, "geometricError": 0, "content": { "uri": "0.glb" } }
        }),
    );
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 100,
            "root": {
                "boundingVolume": { "region": paris_region() },
                "geometricError": 50,
                "refine": "REPLACE",
                "children": [
                    { "boundingVolume": { "region": paris_region() }, "geometricError": 10, "content": { "uri": "city/tileset.json" } },
                    { "boundingVolume": { "region": paris_region() }, "geometricError": 10, "content": { "uri": "city/tileset.json" } }
                ]
            }
        }),
    );
    let reader = Tiles3DReader::new(&dir.path().join("tileset.json")).unwrap();
    assert_eq!(reader.city_object_count().unwrap(), 4);
}

#[test]
fn implicit_tiling_gives_schema_but_no_count() {
    let dir = TempDir::new().unwrap();
    write_json(
        dir.path(),
        "tileset.json",
        &json!({
            "asset": { "version": "1.1" },
            "geometricError": 100,
            "schema": {
                "id": "s",
                "classes": { "building": { "properties": { "name": { "type": "STRING" }, "floors": { "type": "SCALAR", "componentType": "UINT8", "array": true } } } }
            },
            "root": {
                "boundingVolume": { "region": paris_region() },
                "geometricError": 50,
                "refine": "REPLACE",
                "content": { "uri": "content/{level}/{x}/{y}.glb" },
                "implicitTiling": { "subdivisionScheme": "QUADTREE", "subtreeLevels": 4, "availableLevels": 8, "subtrees": { "uri": "subtrees/{level}/{x}/{y}.subtree" } }
            }
        }),
    );
    let reader = Tiles3DReader::new(&dir.path().join("tileset.json")).unwrap();
    assert!(reader.city_object_count().is_err());
    assert!(reader.textures().is_err());
    assert_eq!(reader.city_object_types().unwrap(), vec!["building"]);
    let attributes = reader.attributes().unwrap();
    assert_eq!(attributes[0].name, "floors");
    assert_eq!(attributes[0].attr_type, AttributeType::Array);
}

#[test]
fn cityjson_files_still_use_the_cityjson_reader() {
    let dir = TempDir::new().unwrap();
    write_json(
        dir.path(),
        "city.json",
        &json!({ "type": "CityJSON", "version": "2.0", "transform": { "scale": [1.0, 1.0, 1.0], "translate": [0, 0, 0] }, "CityObjects": {}, "vertices": [] }),
    );
    assert_eq!(
        get_reader(&dir.path().join("city.json"))
            .unwrap()
            .encoding(),
        "CityJSON"
    );
}

#[test]
fn tileset_items_are_named_after_their_folder() {
    assert_eq!(
        item_stem(Path::new("data/paris-bati/tileset.json")),
        "paris-bati"
    );
    assert_eq!(item_stem(Path::new("data/delft.city.json")), "delft.city");
    assert_eq!(item_stem(Path::new("tileset.json")), "tileset");
}

#[test]
fn remote_tilesets_are_read_without_their_contents() {
    let content = serde_json::to_string(&json!({
        "asset": { "version": "1.1" },
        "geometricError": 10,
        "root": { "boundingVolume": { "region": paris_region() }, "geometricError": 0, "refine": "ADD", "content": { "uri": "0.glb" } }
    }))
    .unwrap();
    let reader = Tiles3DReader::from_content(&content, "tileset.json".into()).unwrap();
    assert!(reader.bbox().is_ok());
    assert!(reader.city_object_count().is_err());
}
