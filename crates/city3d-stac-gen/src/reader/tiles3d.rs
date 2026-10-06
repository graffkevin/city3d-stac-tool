//! 3D Tiles reader (`tileset.json`)
//!
//! Reads the metadata of a 3D Tiles tileset without decoding any geometry: the
//! extent comes from the root bounding volume, the object types and attributes
//! from the metadata schemas, and the object count from the property tables of
//! the tile contents.
//!
//! References, cited in the comments below by their key:
//!
//! - **[3DT]** OGC 3D Tiles 1.1 (OGC 22-025r4), <https://docs.ogc.org/cs/22-025r4/22-025r4.html>;
//!   maintained text: <https://github.com/CesiumGS/3d-tiles/tree/main/specification>.
//! - **[3DT-META]** 3D Metadata Specification, part of 3D Tiles 1.1:
//!   <https://github.com/CesiumGS/3d-tiles/tree/main/specification/Metadata>.
//! - **[3DT-B3DM]** Batched 3D Model, the 3D Tiles 1.0 format kept as legacy in 1.1:
//!   <https://github.com/CesiumGS/3d-tiles/tree/main/specification/TileFormats/Batched3DModel>.
//! - **[GLTF]** Khronos glTF 2.0 (ISO/IEC 12113:2022),
//!   <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>.
//! - **[EXT-SM]** glTF extension `EXT_structural_metadata`:
//!   <https://github.com/CesiumGS/glTF/tree/3d-tiles-next/extensions/2.0/Vendor/EXT_structural_metadata>.
//! - **[WGS84]** NGA.STND.0036_1.0.0_WGS84, *World Geodetic System 1984* (2014,
//!   formerly NIMA TR8350.2), section 3.4 "WGS 84 Defining Parameters":
//!   <https://earth-info.nga.mil/php/download.php?file=coord-wgs84>.
//!
//! Mapping to the `city3d` fields:
//!
//! | Field | Source |
//! |---|---|
//! | bbox | root `boundingVolume`, which encloses all the content of the tileset ([3DT] "Bounding volume spatial coherence"), converted to WGS 84 longitude, latitude and ellipsoidal height |
//! | `proj:code` | EPSG:4979, the CRS of `region` bounding volumes ([3DT] "Coordinate reference system (CRS)") |
//! | `city3d:version` | `asset.version` |
//! | `city3d:co_types` | ids of the metadata classes ([3DT-META] "Classes") |
//! | `city3d:attributes` | properties of those classes, or keys of a b3dm Batch Table |
//! | `city3d:city_objects` | sum of the property table counts ([EXT-SM]) or of `BATCH_LENGTH` ([3DT-B3DM]) over the contents that are rendered together |
//! | `city3d:textures`, `city3d:materials` | non-empty `textures` and `materials` arrays of the glTF contents ([GLTF] 5.29, 5.19) |
//! | `city3d:lods` | none: 3D Tiles levels are refinement levels, not CityGML levels of detail |
//! | `city3d:semantic_surfaces` | unknown: not described by the tileset |
//!
//! Limits: implicit tilesets ([3DT] "Implicit Tiling") give their schema and
//! extent but not their object count, since their contents are not listed in
//! the tileset; contents are read only for local tilesets.

use crate::error::{CityJsonStacError, Result};
use crate::metadata::{AttributeDefinition, AttributeType, BBox3D, Transform, CRS};
use crate::reader::CityModelMetadataReader;

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// WGS 84 semi-major axis, in meters ([WGS84] 3.4).
const WGS84_A: f64 = 6_378_137.0;
/// WGS 84 flattening ([WGS84] 3.4).
const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// Distances from the Earth's center, in meters, inside which ECEF coordinates
/// are taken as georeferenced: from 1,000 km below the surface to 100 km above.
const ECEF_RANGE: (f64, f64) = (5.4e6, 6.5e6);
/// Depth limit when following external tilesets, against reference cycles.
const MAX_TILESET_DEPTH: usize = 32;

/// Reader for 3D Tiles tilesets (`tileset.json`)
pub struct Tiles3DReader {
    file_path: PathBuf,
    summary: TilesetSummary,
}

/// What the tileset says about its content, computed once when the reader is created
#[derive(Debug, Default)]
struct TilesetSummary {
    version: Option<String>,
    bbox: Option<BBox3D>,
    classes: BTreeSet<String>,
    attributes: BTreeMap<String, AttributeType>,
    object_count: Option<usize>,
    textures: Option<bool>,
    materials: Option<bool>,
    extensions: BTreeSet<String>,
    metadata: Option<Value>,
}

impl Tiles3DReader {
    /// Create a reader for a local tileset; its tile contents are read too
    pub fn new(file_path: &Path) -> Result<Self> {
        let content = fs::read_to_string(file_path)?;
        let tileset: Value = serde_json::from_str(&content)?;
        let base_dir = file_path.parent().map(Path::to_path_buf);
        Ok(Self {
            file_path: file_path.to_path_buf(),
            summary: summarize(&tileset, base_dir.as_deref())?,
        })
    }

    /// Create a reader from the content of a remote tileset; its tile contents are not read
    pub fn from_content(content: &str, virtual_path: PathBuf) -> Result<Self> {
        let tileset: Value = serde_json::from_str(content)?;
        Ok(Self {
            file_path: virtual_path,
            summary: summarize(&tileset, None)?,
        })
    }
}

/// Whether a JSON document is a 3D Tiles tileset: an `asset` object and a `root` tile ([3DT] "Tileset JSON")
pub fn is_tileset(value: &Value) -> bool {
    value.get("asset").is_some_and(Value::is_object)
        && value.get("root").is_some_and(Value::is_object)
}

/// Whether the start of a `.json` file looks like a tileset rather than CityJSON.
///
/// Only the first bytes are read, so that large CityJSON files are not parsed
/// twice: CityJSON declares `"type": "CityJSON"`, a tileset declares `"asset"`,
/// usually first.
pub fn looks_like_tileset(path: &Path) -> Result<bool> {
    let mut head = Vec::with_capacity(64 * 1024);
    File::open(path)?.take(64 * 1024).read_to_end(&mut head)?;
    Ok(head_looks_like_tileset(&String::from_utf8_lossy(&head)))
}

/// Same test as [`looks_like_tileset`], on content already in memory
pub fn head_looks_like_tileset(head: &str) -> bool {
    let head = &head[..head.len().min(64 * 1024)];
    !head.contains("\"CityJSON\"") && head.contains("\"asset\"")
}

fn summarize(tileset: &Value, base_dir: Option<&Path>) -> Result<TilesetSummary> {
    if !is_tileset(tileset) {
        return Err(CityJsonStacError::InvalidCityJson(
            "Not a 3D Tiles tileset: missing \"asset\" or \"root\"".to_string(),
        ));
    }
    let root = &tileset["root"];
    let mut summary = TilesetSummary {
        version: tileset["asset"]["version"].as_str().map(str::to_string),
        bbox: root_bbox(root),
        metadata: tileset.get("metadata").cloned(),
        ..TilesetSummary::default()
    };
    collect_extensions(tileset, &mut summary.extensions);
    if let Some(schema) = load_schema(tileset, base_dir) {
        add_schema(&schema, &mut summary);
    }

    let mut contents = ContentStats::default();
    let mut visited = HashSet::new();
    walk_tileset(
        tileset,
        base_dir,
        0,
        &mut visited,
        &mut summary,
        &mut contents,
    );
    summary.object_count = contents.object_count();
    summary.textures = contents.read.then_some(contents.textures);
    summary.materials = contents.read.then_some(contents.materials);
    Ok(summary)
}

/// Extensions used by the tileset ([3DT] "Extensions": `extensionsUsed`)
fn collect_extensions(tileset: &Value, out: &mut BTreeSet<String>) {
    if let Some(used) = tileset["extensionsUsed"].as_array() {
        out.extend(used.iter().filter_map(Value::as_str).map(str::to_string));
    }
}

/// The metadata schema of the tileset, inline (`schema`) or in a file (`schemaUri`) ([3DT] "Metadata Schema")
fn load_schema(tileset: &Value, base_dir: Option<&Path>) -> Option<Value> {
    if let Some(schema) = tileset.get("schema") {
        return Some(schema.clone());
    }
    let uri = tileset["schemaUri"].as_str()?;
    let path = local_path(base_dir?, uri)?;
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

/// Add the classes of a schema and their properties ([3DT-META] "Classes", "Properties")
fn add_schema(schema: &Value, summary: &mut TilesetSummary) {
    let Some(classes) = schema["classes"].as_object() else {
        return;
    };
    for (class_id, class) in classes {
        add_class(class_id, class, summary);
    }
}

fn add_class(class_id: &str, class: &Value, summary: &mut TilesetSummary) {
    summary.classes.insert(class_id.to_string());
    if let Some(properties) = class["properties"].as_object() {
        for (name, property) in properties {
            summary
                .attributes
                .entry(name.clone())
                .or_insert_with(|| metadata_attribute_type(property));
        }
    }
}

/// Attribute type of a metadata property ([3DT-META] "Component Type", "Arrays")
fn metadata_attribute_type(property: &Value) -> AttributeType {
    if property["array"].as_bool() == Some(true) {
        return AttributeType::Array;
    }
    match property["type"].as_str() {
        Some("SCALAR") => AttributeType::Number,
        Some("BOOLEAN") => AttributeType::Boolean,
        Some("STRING") | Some("ENUM") => AttributeType::String,
        Some(t) if t.starts_with("VEC") || t.starts_with("MAT") => AttributeType::Array,
        _ => AttributeType::String,
    }
}

// ---------------------------------------------------------------------------
// Extent
// ---------------------------------------------------------------------------

/// WGS 84 bbox of the root tile.
///
/// The root bounding volume encloses the content of the whole tileset ([3DT]
/// "Bounding volume spatial coherence"). A `region` is already in EPSG:4979, in
/// radians, and is never transformed; a `box` or a `sphere` is in the tileset
/// coordinate system after the root `transform` ([3DT] "Tile transforms"),
/// most often WGS 84 ECEF, EPSG:4978 ([3DT] "Coordinate reference system").
fn root_bbox(root: &Value) -> Option<BBox3D> {
    let volume = &root["boundingVolume"];
    if let Some(region) = numbers(&volume["region"], 6) {
        // [west, south, east, north, minimum height, maximum height] ([3DT] "Region")
        return Some(BBox3D::new(
            region[0].to_degrees(),
            region[1].to_degrees(),
            region[4],
            region[2].to_degrees(),
            region[3].to_degrees(),
            region[5],
        ));
    }
    let transform = match root.get("transform") {
        Some(t) => numbers(t, 16)?,
        None => identity(),
    };
    let points = if let Some(b) = numbers(&volume["box"], 12) {
        // Center, then the x, y and z half-axes ([3DT] "Box")
        box_points(&b[0..3], &b[3..6], &b[6..9], &b[9..12])
    } else {
        // Center and radius ([3DT] "Sphere"): the enclosing axis-aligned box
        let s = numbers(&volume["sphere"], 4)?;
        let r = s[3];
        box_points(&s[0..3], &[r, 0.0, 0.0], &[0.0, r, 0.0], &[0.0, 0.0, r])
    };

    let mut bbox: Option<BBox3D> = None;
    for point in points {
        let ecef = apply(&transform, point);
        let norm = (ecef[0].powi(2) + ecef[1].powi(2) + ecef[2].powi(2)).sqrt();
        if !(ECEF_RANGE.0..=ECEF_RANGE.1).contains(&norm) {
            // Local coordinate system, not georeferenced ([3DT] "Coordinate reference system")
            return None;
        }
        let (lon, lat, h) = ecef_to_geodetic(ecef);
        let point_bbox = BBox3D::new(lon, lat, h, lon, lat, h);
        bbox = Some(match bbox {
            Some(b) => b.merge(&point_bbox),
            None => point_bbox,
        });
    }
    bbox
}

/// Points of an oriented box: corners, edge middles, face centers and center.
///
/// The middles matter for large boxes, whose faces bulge in longitude and latitude.
fn box_points(center: &[f64], x: &[f64], y: &[f64], z: &[f64]) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(27);
    for i in [-1.0, 0.0, 1.0] {
        for j in [-1.0, 0.0, 1.0] {
            for k in [-1.0, 0.0, 1.0] {
                points.push([
                    center[0] + i * x[0] + j * y[0] + k * z[0],
                    center[1] + i * x[1] + j * y[1] + k * z[1],
                    center[2] + i * x[2] + j * y[2] + k * z[2],
                ]);
            }
        }
    }
    points
}

fn identity() -> Vec<f64> {
    vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// Apply a 4x4 affine matrix stored in column-major order ([3DT] "Tile transforms")
fn apply(m: &[f64], p: [f64; 3]) -> [f64; 3] {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

/// ECEF (EPSG:4978) to longitude and latitude in degrees and ellipsoidal height
/// in meters (EPSG:4979), by fixed-point iteration on the latitude.
///
/// Five iterations reach well under a millimeter for points near the Earth's
/// surface, which is the only case accepted by [`root_bbox`].
fn ecef_to_geodetic(p: [f64; 3]) -> (f64, f64, f64) {
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let lon = p[1].atan2(p[0]);
    let mut lat = p[2].atan2(rho * (1.0 - e2));
    let mut h = 0.0;
    for _ in 0..5 {
        let n = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
        h = rho / lat.cos() - n;
        lat = p[2].atan2(rho * (1.0 - e2 * n / (n + h)));
    }
    (lon.to_degrees(), lat.to_degrees(), h)
}

fn numbers(value: &Value, len: usize) -> Option<Vec<f64>> {
    let array = value.as_array()?;
    if array.len() != len {
        return None;
    }
    array.iter().map(Value::as_f64).collect()
}

// ---------------------------------------------------------------------------
// Tile contents
// ---------------------------------------------------------------------------

/// Totals over the contents read
#[derive(Debug, Default)]
struct ContentStats {
    /// At least one content was read
    read: bool,
    /// Sum of the object counts of the contents rendered together
    objects: usize,
    /// Some contents rendered together give no count, or are not listed (implicit tiling)
    count_unknown: bool,
    textures: bool,
    materials: bool,
}

impl ContentStats {
    fn object_count(&self) -> Option<usize> {
        (self.read && !self.count_unknown).then_some(self.objects)
    }
}

/// What one tile content says, from its glTF JSON and, for b3dm, its tables
#[derive(Debug, Default)]
struct ContentInfo {
    object_count: Option<usize>,
    textures: bool,
    materials: bool,
    /// Classes of the property tables ([EXT-SM] "Property Tables": `class`)
    classes: Vec<String>,
    /// Schema embedded in the glTF ([EXT-SM] "Schema Definitions")
    schema: Option<Value>,
    /// Attributes of a b3dm Batch Table
    attributes: Vec<(String, AttributeType)>,
}

fn walk_tileset(
    tileset: &Value,
    base_dir: Option<&Path>,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    summary: &mut TilesetSummary,
    stats: &mut ContentStats,
) {
    let refine = tileset["root"]["refine"].as_str().unwrap_or("REPLACE");
    walk_tile(
        &tileset["root"],
        refine,
        base_dir,
        depth,
        visited,
        summary,
        stats,
    );
}

/// Visit a tile and its children.
///
/// Objects are counted once: with replacement refinement the children replace
/// the parent, so only the leaves count; with additive refinement the children
/// are rendered with the parent, so every tile counts. The refinement is
/// inherited when omitted ([3DT] "Refinement").
fn walk_tile(
    tile: &Value,
    parent_refine: &str,
    base_dir: Option<&Path>,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    summary: &mut TilesetSummary,
    stats: &mut ContentStats,
) {
    let refine = tile["refine"].as_str().unwrap_or(parent_refine);
    let children = tile["children"].as_array().filter(|c| !c.is_empty());
    let counted = children.is_none() || refine == "ADD";

    if tile.get("implicitTiling").is_some() {
        // Contents are given by a URI template and subtree files, not listed
        // in the tileset ([3DT] "Implicit Tiling")
        stats.count_unknown = true;
        return;
    }

    // One `content` or several `contents` ([3DT] "Tile Content")
    let contents: Vec<&Value> = match (tile.get("content"), tile["contents"].as_array()) {
        (Some(c), _) => vec![c],
        (None, Some(list)) => list.iter().collect(),
        (None, None) => Vec::new(),
    };
    for content in contents {
        let Some(uri) = content["uri"].as_str() else {
            continue;
        };
        let Some(path) = base_dir.and_then(|dir| local_path(dir, uri)) else {
            if counted {
                stats.count_unknown = true;
            }
            continue;
        };
        if is_json_path(&path) {
            // External tileset ([3DT] "External tilesets")
            walk_external(&path, refine, depth, visited, summary, stats);
            continue;
        }
        match read_content(&path) {
            Some(info) => {
                stats.read = true;
                stats.textures |= info.textures;
                stats.materials |= info.materials;
                if counted {
                    match info.object_count {
                        Some(n) => stats.objects += n,
                        None => stats.count_unknown = true,
                    }
                }
                summary.classes.extend(info.classes);
                if let Some(schema) = &info.schema {
                    add_schema(schema, summary);
                }
                for (name, attr_type) in info.attributes {
                    summary.attributes.entry(name).or_insert(attr_type);
                }
            }
            None if counted => stats.count_unknown = true,
            None => {}
        }
    }

    for child in children.into_iter().flatten() {
        walk_tile(child, refine, base_dir, depth, visited, summary, stats);
    }
}

fn walk_external(
    path: &Path,
    refine: &str,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    summary: &mut TilesetSummary,
    stats: &mut ContentStats,
) {
    let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if depth >= MAX_TILESET_DEPTH || !visited.insert(key) {
        return;
    }
    let Some(external) = fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .filter(is_tileset)
    else {
        stats.count_unknown = true;
        return;
    };
    collect_extensions(&external, &mut summary.extensions);
    let dir = path.parent();
    if let Some(schema) = load_schema(&external, dir) {
        add_schema(&schema, summary);
    }
    // The external root inherits the refinement of the tile that references it
    let root_refine = external["root"]["refine"].as_str().unwrap_or(refine);
    walk_tile(
        &external["root"],
        root_refine,
        dir,
        depth + 1,
        visited,
        summary,
        stats,
    );
}

/// A relative content URI resolved against the tileset folder ([3DT] "URIs").
///
/// Absolute URIs and data URIs are not read.
fn local_path(base_dir: &Path, uri: &str) -> Option<PathBuf> {
    if uri.contains("://") || uri.starts_with("data:") || uri.starts_with('/') {
        return None;
    }
    let path_part = uri.split(['?', '#']).next().unwrap_or(uri);
    Some(base_dir.join(percent_decode(path_part)))
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn is_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
}

/// Read what a content says, identified by its magic rather than its
/// extension ([3DT-B3DM] "File extension and media type")
fn read_content(path: &Path) -> Option<ContentInfo> {
    let mut file = File::open(path).ok()?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).ok()?;
    match &magic {
        b"glTF" => read_glb_json(&mut file, 0).map(|gltf| gltf_info(&gltf)),
        b"b3dm" => read_b3dm(&mut file),
        _ => None,
    }
}

/// JSON chunk of a binary glTF starting at `offset` ([GLTF] 4.4 "Binary glTF Layout").
///
/// Header: magic `glTF`, version, length (three uint32); then the first chunk,
/// which shall be the JSON chunk: length, type `JSON` (0x4E4F534A), data.
fn read_glb_json(file: &mut File, offset: u64) -> Option<Value> {
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut header = [0u8; 20];
    file.read_exact(&mut header).ok()?;
    if &header[0..4] != b"glTF" || u32_at(&header, 16) != 0x4E4F_534A {
        return None;
    }
    let mut json = vec![0u8; u32_at(&header, 12) as usize];
    file.read_exact(&mut json).ok()?;
    serde_json::from_slice(trim_padding(&json)).ok()
}

/// Batched 3D Model ([3DT-B3DM] "Layout"): a 28-byte header (magic, version,
/// byteLength, then the JSON and binary lengths of the Feature Table and of the
/// Batch Table, all uint32), the two tables, then a binary glTF.
fn read_b3dm(file: &mut File) -> Option<ContentInfo> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut header = [0u8; 28];
    file.read_exact(&mut header).ok()?;
    let feature_json_len = u32_at(&header, 12) as usize;
    let feature_bin_len = u64::from(u32_at(&header, 16));
    let batch_json_len = u32_at(&header, 20) as usize;
    let batch_bin_len = u64::from(u32_at(&header, 24));

    let mut feature_json = vec![0u8; feature_json_len];
    file.read_exact(&mut feature_json).ok()?;
    let feature_table: Value = serde_json::from_slice(trim_padding(&feature_json)).ok()?;
    file.seek(SeekFrom::Current(feature_bin_len as i64)).ok()?;
    let mut batch_json = vec![0u8; batch_json_len];
    file.read_exact(&mut batch_json).ok()?;
    let batch_table: Option<Value> = serde_json::from_slice(trim_padding(&batch_json)).ok();
    let glb_offset =
        28 + feature_json_len as u64 + feature_bin_len + batch_json_len as u64 + batch_bin_len;

    let mut info = read_glb_json(file, glb_offset)
        .map(|gltf| gltf_info(&gltf))
        .unwrap_or_default();
    // Number of features, `BATCH_LENGTH` ([3DT-B3DM] "Feature Table"); 0 means the
    // model is not split into features, which says nothing about its objects
    info.object_count = feature_table["BATCH_LENGTH"]
        .as_u64()
        .filter(|&n| n > 0)
        .map(|n| n as usize);
    info.attributes = batch_table
        .as_ref()
        .map(batch_attributes)
        .unwrap_or_default();
    Some(info)
}

/// Attributes of a b3dm Batch Table: one key per property, values given as a
/// JSON array or as a reference into the binary body ([3DT-B3DM] "Batch Table").
fn batch_attributes(batch: &Value) -> Vec<(String, AttributeType)> {
    let Some(object) = batch.as_object() else {
        return Vec::new();
    };
    object
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "extensions" | "extras"))
        .map(|(key, value)| {
            let attr_type = match value {
                Value::Array(values) => values
                    .iter()
                    .find(|v| !v.is_null())
                    .map(AttributeType::from_json_value)
                    .unwrap_or(AttributeType::String),
                // Binary property: { byteOffset, componentType, type }
                _ if value["type"].as_str() == Some("SCALAR") => AttributeType::Number,
                _ => AttributeType::Array,
            };
            (key.clone(), attr_type)
        })
        .collect()
}

/// What a glTF JSON says: object count and classes from the property tables
/// ([EXT-SM] "Property Tables": `count`, `class`), embedded schema, textures and
/// materials ([GLTF] 5.29, 5.19)
fn gltf_info(gltf: &Value) -> ContentInfo {
    let metadata = &gltf["extensions"]["EXT_structural_metadata"];
    let tables = metadata["propertyTables"].as_array();
    ContentInfo {
        classes: tables
            .map(|t| {
                t.iter()
                    .filter_map(|table| table["class"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        schema: metadata.get("schema").cloned(),
        attributes: Vec::new(),
        object_count: tables.map(|t| {
            t.iter()
                .filter_map(|table| table["count"].as_u64())
                .sum::<u64>() as usize
        }),
        textures: gltf["textures"].as_array().is_some_and(|t| !t.is_empty()),
        materials: gltf["materials"].as_array().is_some_and(|m| !m.is_empty()),
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// JSON chunks and tables are padded with spaces, sometimes with zeros
fn trim_padding(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .rposition(|&b| b != 0 && b != b' ')
        .map_or(0, |i| i + 1);
    &bytes[..end]
}

impl CityModelMetadataReader for Tiles3DReader {
    fn bbox(&self) -> Result<BBox3D> {
        self.summary.bbox.clone().ok_or_else(|| {
            CityJsonStacError::MetadataError(
                "Tileset root bounding volume missing or not georeferenced".to_string(),
            )
        })
    }

    fn crs(&self) -> Result<CRS> {
        // The bbox above is in WGS 84 longitude, latitude and ellipsoidal height
        Ok(CRS::from_epsg(4979))
    }

    fn lods(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    fn city_object_types(&self) -> Result<Vec<String>> {
        Ok(self.summary.classes.iter().cloned().collect())
    }

    fn city_object_count(&self) -> Result<usize> {
        self.summary.object_count.ok_or_else(|| {
            CityJsonStacError::MetadataError("Object count not available".to_string())
        })
    }

    fn attributes(&self) -> Result<Vec<AttributeDefinition>> {
        Ok(self
            .summary
            .attributes
            .iter()
            .map(|(name, attr_type)| AttributeDefinition::new(name.clone(), *attr_type))
            .collect())
    }

    fn encoding(&self) -> &'static str {
        "3D Tiles"
    }

    fn version(&self) -> Result<String> {
        self.summary
            .version
            .clone()
            .ok_or_else(|| CityJsonStacError::MissingField("asset.version".to_string()))
    }

    fn file_path(&self) -> &Path {
        &self.file_path
    }

    fn transform(&self) -> Result<Option<Transform>> {
        Ok(None)
    }

    fn metadata(&self) -> Result<Option<Value>> {
        Ok(self.summary.metadata.clone())
    }

    fn extensions(&self) -> Result<Vec<String>> {
        Ok(self.summary.extensions.iter().cloned().collect())
    }

    fn semantic_surfaces(&self) -> Result<bool> {
        Err(CityJsonStacError::MetadataError(
            "Semantic surfaces are not described by a tileset".to_string(),
        ))
    }

    fn textures(&self) -> Result<bool> {
        self.summary
            .textures
            .ok_or_else(|| CityJsonStacError::MetadataError("No tile content read".to_string()))
    }

    fn materials(&self) -> Result<bool> {
        self.summary
            .materials
            .ok_or_else(|| CityJsonStacError::MetadataError("No tile content read".to_string()))
    }
}
