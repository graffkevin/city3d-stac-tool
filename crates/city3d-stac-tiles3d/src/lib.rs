//! Metadata of a 3D Tiles tileset, for STAC Items
//!
//! Reads what a tileset publishes about its content without decoding any
//! geometry: the extent from the root bounding volume, the object types and
//! attributes from the metadata schemas, the object count from the property
//! tables of the tile contents. The crate does no I/O: files are read through a
//! [`Source`], so that the same code runs on a local folder, over HTTP, or in a
//! browser through WebAssembly, where reads are answered in rounds (see
//! [`Summary::missing`]).
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
//! Mapping to the `city3d` STAC fields:
//!
//! | Field | Source |
//! |---|---|
//! | bbox | root `boundingVolume`, which encloses all the content of the tileset ([3DT] "Bounding volume spatial coherence"), as WGS 84 longitude, latitude and ellipsoidal height, the CRS of `region` bounding volumes (EPSG:4979, [3DT] "Coordinate reference system (CRS)") |
//! | `city3d:version` | `asset.version` |
//! | `city3d:co_types` | ids of the metadata classes ([3DT-META] "Classes") |
//! | `city3d:attributes` | properties of those classes, or keys of a b3dm Batch Table |
//! | `city3d:city_objects` | sum of the property table counts ([EXT-SM]) or of `BATCH_LENGTH` ([3DT-B3DM]) over the contents rendered together |
//! | `city3d:textures`, `city3d:materials` | non-empty `textures` and `materials` arrays of the glTF contents ([GLTF] 5.29, 5.19) |
//! | `city3d:lods` | none: 3D Tiles levels are refinement levels, not CityGML levels of detail |
//! | `city3d:semantic_surfaces` | unknown: not described by a tileset |
//!
//! Implicit tilesets ([3DT] "Implicit Tiling") give their schema and extent but
//! no object count, since their contents are not listed in the tileset.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// WGS 84 semi-major axis, in meters ([WGS84] 3.4).
const WGS84_A: f64 = 6_378_137.0;
/// WGS 84 flattening ([WGS84] 3.4).
const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// Distances from the Earth's center, in meters, inside which ECEF coordinates
/// are taken as georeferenced: from 1,000 km below the surface to 100 km above.
const ECEF_RANGE: (f64, f64) = (5.4e6, 6.5e6);
/// Depth limit when following external tilesets, against reference cycles.
const MAX_TILESET_DEPTH: usize = 32;
/// Bytes read to tell a tileset from CityJSON.
const SNIFF_BYTES: usize = 64 * 1024;

/// Answer of a [`Source`] to a read
#[derive(Debug, Clone, PartialEq)]
pub enum Fetch<T> {
    /// The data
    Ready(T),
    /// Not read yet: the read is listed in [`Summary::missing`], to be answered in a next round
    Missing,
    /// Cannot be read (absent file, remote URI from a local source, budget exhausted…)
    Unavailable,
}

/// Where the files of a tileset are read.
///
/// URIs are resolved against the folder of the root tileset ([`resolve`]);
/// absolute URIs are passed unchanged.
pub trait Source {
    /// A JSON file: an external tileset or a metadata schema
    fn json(&self, uri: &str) -> Fetch<Value>;
    /// Bytes of a tile content, from `offset`; may return fewer than `len` at the end of the file
    fn bytes(&self, uri: &str, offset: u64, len: u64) -> Fetch<Vec<u8>>;
}

/// A read a [`Source`] answered with [`Fetch::Missing`]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Request {
    /// A whole JSON file
    Json { uri: String },
    /// A byte range of a tile content
    Bytes { uri: String, offset: u64, len: u64 },
}

/// A [`Source`] answering from reads already made, for sources that answer in rounds.
///
/// Reads not in the cache are [`Fetch::Missing`] until [`Cache::close`] is called,
/// after which they are [`Fetch::Unavailable`]: call [`summarize`], perform the
/// [`Summary::missing`] reads and [`Cache::answer`] them, then summarize again,
/// and close the cache on the last round.
#[derive(Debug, Default, Clone)]
pub struct Cache {
    json: HashMap<String, Option<Value>>,
    bytes: HashMap<(String, u64, u64), Option<Vec<u8>>>,
    closed: bool,
}

impl Cache {
    /// Record the answer to a read: `None` when it failed
    pub fn answer_json(&mut self, uri: &str, value: Option<Value>) {
        self.json.insert(uri.to_string(), value);
    }

    /// Record the answer to a byte range read: `None` when it failed
    pub fn answer_bytes(&mut self, uri: &str, offset: u64, len: u64, bytes: Option<Vec<u8>>) {
        self.bytes.insert((uri.to_string(), offset, len), bytes);
    }

    /// Record the answer to a [`Request`]: the JSON value or the bytes, `None` when it failed
    pub fn answer(&mut self, request: &Request, data: Option<Vec<u8>>) {
        match request {
            Request::Json { uri } => {
                let value = data.and_then(|d| serde_json::from_slice(&d).ok());
                self.answer_json(uri, value);
            }
            Request::Bytes { uri, offset, len } => self.answer_bytes(uri, *offset, *len, data),
        }
    }

    /// Treat reads still unanswered as unavailable, for a last round
    pub fn close(&mut self) {
        self.closed = true;
    }

    fn unanswered<T>(&self) -> Fetch<T> {
        if self.closed {
            Fetch::Unavailable
        } else {
            Fetch::Missing
        }
    }
}

impl Source for Cache {
    fn json(&self, uri: &str) -> Fetch<Value> {
        match self.json.get(uri) {
            Some(Some(v)) => Fetch::Ready(v.clone()),
            Some(None) => Fetch::Unavailable,
            None => self.unanswered(),
        }
    }

    fn bytes(&self, uri: &str, offset: u64, len: u64) -> Fetch<Vec<u8>> {
        match self.bytes.get(&(uri.to_string(), offset, len)) {
            Some(Some(b)) => Fetch::Ready(b.clone()),
            Some(None) => Fetch::Unavailable,
            None => self.unanswered(),
        }
    }
}

/// Attribute type, as the STAC `city3d:attributes` field names them
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttrType {
    String,
    Number,
    Boolean,
    Array,
    Object,
}

impl AttrType {
    /// Name used in `city3d:attributes`
    pub fn as_str(self) -> &'static str {
        match self {
            AttrType::String => "String",
            AttrType::Number => "Number",
            AttrType::Boolean => "Boolean",
            AttrType::Array => "Array",
            AttrType::Object => "Object",
        }
    }

    fn of_json(value: &Value) -> Self {
        match value {
            Value::Bool(_) => AttrType::Boolean,
            Value::Number(_) => AttrType::Number,
            Value::Array(_) => AttrType::Array,
            Value::Object(_) => AttrType::Object,
            _ => AttrType::String,
        }
    }
}

/// Options of [`summarize`]
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Maximum number of tile contents to read; beyond it the object count is unknown
    pub max_contents: Option<usize>,
    /// Maximum number of external tilesets to read; beyond it the object count is unknown
    pub max_tilesets: Option<usize>,
}

/// What a tileset says about its content
#[derive(Debug, Clone, Default)]
pub struct Summary {
    /// `asset.version`
    pub version: Option<String>,
    /// `[west, south, min height, east, north, max height]`, degrees and meters (EPSG:4979)
    pub bbox: Option<[f64; 6]>,
    /// Metadata class ids
    pub classes: BTreeSet<String>,
    /// Attributes by name
    pub attributes: BTreeMap<String, AttrType>,
    /// Objects rendered together, when every counted content gives its count
    pub object_count: Option<usize>,
    /// Some glTF content has textures (unknown when no content was read)
    pub textures: Option<bool>,
    /// Some glTF content has materials (unknown when no content was read)
    pub materials: Option<bool>,
    /// Extensions used (`extensionsUsed`)
    pub extensions: BTreeSet<String>,
    /// Tileset metadata (`metadata`)
    pub metadata: Option<Value>,
    /// Reads still to answer: when not empty, the summary is partial and
    /// [`summarize`] should be called again once the source can answer them
    pub missing: Vec<Request>,
    /// Number of tile contents read
    pub contents_read: usize,
}

/// Whether a JSON document is a 3D Tiles tileset: an `asset` object and a `root` tile ([3DT] "Tileset JSON")
pub fn is_tileset(value: &Value) -> bool {
    value.get("asset").is_some_and(Value::is_object)
        && value.get("root").is_some_and(Value::is_object)
}

/// Whether the start of a `.json` file looks like a tileset rather than CityJSON.
///
/// Only the first 64 KiB are looked at, so that large CityJSON files are not
/// parsed twice: CityJSON declares `"type": "CityJSON"`, a tileset declares
/// `"asset"`, usually first.
pub fn looks_like_tileset(head: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&head[..head.len().min(SNIFF_BYTES)]);
    !head.contains("\"CityJSON\"") && head.contains("\"asset\"")
}

/// Resolve a content URI against the folder of the tileset that references it ([3DT] "URIs").
///
/// `base` is that folder, relative to the root tileset folder (`""` for the root
/// tileset), ending with `/` otherwise. Absolute URIs (`scheme://`) are returned
/// unchanged; data URIs and absolute paths give `None`.
pub fn resolve(base: &str, uri: &str) -> Option<String> {
    if uri.contains("://") {
        return Some(uri.to_string());
    }
    if uri.starts_with("data:") || uri.starts_with('/') {
        return None;
    }
    let path = uri.split(['?', '#']).next().unwrap_or(uri);
    let mut parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            p => parts.push(p),
        }
    }
    Some(percent_decode(&parts.join("/")))
}

fn folder_of(uri: &str) -> String {
    match uri.rfind('/') {
        Some(i) => uri[..=i].to_string(),
        None => String::new(),
    }
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

/// Summarize a tileset, reading its schema files, external tilesets and tile contents through `source`
pub fn summarize(
    tileset: &Value,
    source: &dyn Source,
    options: &Options,
) -> Result<Summary, String> {
    if !is_tileset(tileset) {
        return Err("Not a 3D Tiles tileset: missing \"asset\" or \"root\"".to_string());
    }
    let mut walk = Walk {
        source,
        options,
        summary: Summary {
            version: tileset["asset"]["version"].as_str().map(str::to_string),
            bbox: root_bbox(&tileset["root"]),
            metadata: tileset.get("metadata").cloned(),
            ..Summary::default()
        },
        stats: ContentStats::default(),
        visited: HashSet::new(),
        missing: BTreeSet::new(),
    };
    walk.tileset(tileset, "", 0);

    let mut summary = walk.summary;
    let stats = walk.stats;
    summary.object_count = (stats.read && !stats.count_unknown).then_some(stats.objects);
    summary.textures = stats.read.then_some(stats.textures);
    summary.materials = stats.read.then_some(stats.materials);
    summary.missing = walk.missing.into_iter().collect();
    Ok(summary)
}

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
    attributes: Vec<(String, AttrType)>,
}

struct Walk<'a> {
    source: &'a dyn Source,
    options: &'a Options,
    summary: Summary,
    stats: ContentStats,
    visited: HashSet<String>,
    missing: BTreeSet<Request>,
}

impl Walk<'_> {
    fn fetch<T>(&mut self, fetch: Fetch<T>, request: Request) -> Option<T> {
        match fetch {
            Fetch::Ready(value) => Some(value),
            Fetch::Missing => {
                self.missing.insert(request);
                None
            }
            Fetch::Unavailable => None,
        }
    }

    fn json(&mut self, uri: &str) -> Option<Value> {
        let fetch = self.source.json(uri);
        self.fetch(
            fetch,
            Request::Json {
                uri: uri.to_string(),
            },
        )
    }

    fn bytes(&mut self, uri: &str, offset: u64, len: u64) -> Option<Vec<u8>> {
        let fetch = self.source.bytes(uri, offset, len);
        self.fetch(
            fetch,
            Request::Bytes {
                uri: uri.to_string(),
                offset,
                len,
            },
        )
    }

    /// A tileset at `base` (folder of its file): extensions, schema, then its tree
    fn tileset(&mut self, tileset: &Value, base: &str, depth: usize) {
        if let Some(used) = tileset["extensionsUsed"].as_array() {
            self.summary
                .extensions
                .extend(used.iter().filter_map(Value::as_str).map(str::to_string));
        }
        // Schema, inline or in a file ([3DT] "Metadata Schema")
        let schema = match (tileset.get("schema"), tileset["schemaUri"].as_str()) {
            (Some(schema), _) => Some(schema.clone()),
            (None, Some(uri)) => resolve(base, uri).and_then(|uri| self.json(&uri)),
            (None, None) => None,
        };
        if let Some(schema) = schema {
            add_schema(&schema, &mut self.summary);
        }
        let refine = tileset["root"]["refine"].as_str().unwrap_or("REPLACE");
        self.tile(&tileset["root"], refine, base, depth);
    }

    /// Visit a tile and its children.
    ///
    /// Objects are counted once: with replacement refinement the children
    /// replace the parent, so only the leaves count; with additive refinement
    /// the children are rendered with the parent, so every tile counts. The
    /// refinement is inherited when omitted ([3DT] "Refinement").
    fn tile(&mut self, tile: &Value, parent_refine: &str, base: &str, depth: usize) {
        let refine = tile["refine"].as_str().unwrap_or(parent_refine);
        let children = tile["children"].as_array().filter(|c| !c.is_empty());
        let counted = children.is_none() || refine == "ADD";

        if tile.get("implicitTiling").is_some() {
            // Contents are given by a URI template and subtree files, not listed
            // in the tileset ([3DT] "Implicit Tiling")
            self.stats.count_unknown = true;
            return;
        }

        // One `content` or several `contents` ([3DT] "Tile Content")
        let contents: Vec<Value> = match (tile.get("content"), tile["contents"].as_array()) {
            (Some(c), _) => vec![c.clone()],
            (None, Some(list)) => list.clone(),
            (None, None) => Vec::new(),
        };
        for content in &contents {
            let Some(uri) = content["uri"].as_str().and_then(|u| resolve(base, u)) else {
                continue;
            };
            if uri.to_ascii_lowercase().ends_with(".json") {
                // External tileset ([3DT] "External tilesets")
                self.external(&uri, refine, depth);
                continue;
            }
            self.content(&uri, counted);
        }

        for child in children.into_iter().flatten() {
            self.tile(child, refine, base, depth);
        }
    }

    fn external(&mut self, uri: &str, refine: &str, depth: usize) {
        if depth >= MAX_TILESET_DEPTH || self.visited.contains(uri) {
            return;
        }
        if self
            .options
            .max_tilesets
            .is_some_and(|max| self.visited.len() >= max)
        {
            self.stats.count_unknown = true;
            return;
        }
        let Some(external) = self.json(uri) else {
            self.stats.count_unknown = true;
            return;
        };
        self.visited.insert(uri.to_string());
        if !is_tileset(&external) {
            self.stats.count_unknown = true;
            return;
        }
        // The external root inherits the refinement of the tile that references it
        let mut external = external;
        if external["root"].get("refine").is_none() {
            external["root"]["refine"] = Value::String(refine.to_string());
        }
        self.tileset(&external, &folder_of(uri), depth + 1);
    }

    fn content(&mut self, uri: &str, counted: bool) {
        if self
            .options
            .max_contents
            .is_some_and(|max| self.summary.contents_read >= max)
        {
            if counted {
                self.stats.count_unknown = true;
            }
            return;
        }
        match self.read_content(uri) {
            Some(info) => {
                self.summary.contents_read += 1;
                self.stats.read = true;
                self.stats.textures |= info.textures;
                self.stats.materials |= info.materials;
                if counted {
                    match info.object_count {
                        Some(n) => self.stats.objects += n,
                        None => self.stats.count_unknown = true,
                    }
                }
                self.summary.classes.extend(info.classes);
                if let Some(schema) = &info.schema {
                    add_schema(schema, &mut self.summary);
                }
                for (name, attr_type) in info.attributes {
                    self.summary.attributes.entry(name).or_insert(attr_type);
                }
            }
            None if counted => self.stats.count_unknown = true,
            None => {}
        }
    }

    /// Read what a content says, identified by its magic rather than its
    /// extension ([3DT-B3DM] "File extension and media type")
    fn read_content(&mut self, uri: &str) -> Option<ContentInfo> {
        let header = self.bytes(uri, 0, 28)?;
        match header.get(0..4)? {
            b"glTF" => self.read_glb_json(uri, 0).map(|gltf| gltf_info(&gltf)),
            b"b3dm" => self.read_b3dm(uri, &header),
            _ => None,
        }
    }

    /// JSON chunk of a binary glTF starting at `offset` ([GLTF] 4.4 "Binary glTF Layout").
    ///
    /// Header: magic `glTF`, version, length (three uint32); then the first
    /// chunk, which shall be the JSON chunk: length, type `JSON` (0x4E4F534A), data.
    fn read_glb_json(&mut self, uri: &str, offset: u64) -> Option<Value> {
        let header = self.bytes(uri, offset, 20)?;
        if header.len() < 20 || &header[0..4] != b"glTF" || u32_at(&header, 16) != 0x4E4F_534A {
            return None;
        }
        let len = u64::from(u32_at(&header, 12));
        let json = self.bytes(uri, offset + 20, len)?;
        serde_json::from_slice(trim_padding(&json)).ok()
    }

    /// Batched 3D Model ([3DT-B3DM] "Layout"): a 28-byte header (magic, version,
    /// byteLength, then the JSON and binary lengths of the Feature Table and of
    /// the Batch Table, all uint32), the two tables, then a binary glTF.
    fn read_b3dm(&mut self, uri: &str, header: &[u8]) -> Option<ContentInfo> {
        if header.len() < 28 {
            return None;
        }
        let feature_json_len = u64::from(u32_at(header, 12));
        let feature_bin_len = u64::from(u32_at(header, 16));
        let batch_json_len = u64::from(u32_at(header, 20));
        let batch_bin_len = u64::from(u32_at(header, 24));
        let batch_offset = 28 + feature_json_len + feature_bin_len;
        let glb_offset = batch_offset + batch_json_len + batch_bin_len;

        // Ask for the three parts at once, so that a source answering in rounds needs fewer rounds
        let feature = self.bytes(uri, 28, feature_json_len);
        let batch = if batch_json_len > 0 {
            self.bytes(uri, batch_offset, batch_json_len)
        } else {
            Some(Vec::new())
        };
        let gltf = self.read_glb_json(uri, glb_offset);
        let feature_table: Value = serde_json::from_slice(trim_padding(&feature?)).ok()?;
        let batch_table: Option<Value> = serde_json::from_slice(trim_padding(&batch?)).ok();

        let mut info = gltf.map(|g| gltf_info(&g)).unwrap_or_default();
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
}

/// Add the classes of a schema and their properties ([3DT-META] "Classes", "Properties")
fn add_schema(schema: &Value, summary: &mut Summary) {
    let Some(classes) = schema["classes"].as_object() else {
        return;
    };
    for (class_id, class) in classes {
        summary.classes.insert(class_id.clone());
        if let Some(properties) = class["properties"].as_object() {
            for (name, property) in properties {
                summary
                    .attributes
                    .entry(name.clone())
                    .or_insert_with(|| metadata_attribute_type(property));
            }
        }
    }
}

/// Attribute type of a metadata property ([3DT-META] "Component Type", "Arrays")
fn metadata_attribute_type(property: &Value) -> AttrType {
    if property["array"].as_bool() == Some(true) {
        return AttrType::Array;
    }
    match property["type"].as_str() {
        Some("SCALAR") => AttrType::Number,
        Some("BOOLEAN") => AttrType::Boolean,
        Some("STRING") | Some("ENUM") => AttrType::String,
        Some(t) if t.starts_with("VEC") || t.starts_with("MAT") => AttrType::Array,
        _ => AttrType::String,
    }
}

/// Attributes of a b3dm Batch Table: one key per property, values given as a
/// JSON array or as a reference into the binary body ([3DT-B3DM] "Batch Table").
fn batch_attributes(batch: &Value) -> Vec<(String, AttrType)> {
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
                    .map(AttrType::of_json)
                    .unwrap_or(AttrType::String),
                // Binary property: { byteOffset, componentType, type }
                _ if value["type"].as_str() == Some("SCALAR") => AttrType::Number,
                _ => AttrType::Array,
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
/// A tileset in a local frame, not georeferenced, gives no bbox.
fn root_bbox(root: &Value) -> Option<[f64; 6]> {
    let volume = &root["boundingVolume"];
    if let Some(region) = numbers(&volume["region"], 6) {
        // [west, south, east, north, minimum height, maximum height] ([3DT] "Region")
        return Some([
            region[0].to_degrees(),
            region[1].to_degrees(),
            region[4],
            region[2].to_degrees(),
            region[3].to_degrees(),
            region[5],
        ]);
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

    let mut bbox: Option<[f64; 6]> = None;
    for point in points {
        let ecef = apply(&transform, point);
        let norm = (ecef[0].powi(2) + ecef[1].powi(2) + ecef[2].powi(2)).sqrt();
        if !(ECEF_RANGE.0..=ECEF_RANGE.1).contains(&norm) {
            return None;
        }
        let (lon, lat, h) = ecef_to_geodetic(ecef);
        bbox = Some(match bbox {
            Some(b) => [
                b[0].min(lon),
                b[1].min(lat),
                b[2].min(h),
                b[3].max(lon),
                b[4].max(lat),
                b[5].max(h),
            ],
            None => [lon, lat, h, lon, lat, h],
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolve_follows_folders() {
        assert_eq!(resolve("", "tiles/0.glb").unwrap(), "tiles/0.glb");
        assert_eq!(resolve("city/", "../0.glb").unwrap(), "0.glb");
        assert_eq!(resolve("a/b/", "./c%20d.glb?v=1").unwrap(), "a/b/c d.glb");
        assert_eq!(resolve("", "https://x/y.glb").unwrap(), "https://x/y.glb");
        assert!(resolve("", "../outside.glb").is_none());
        assert!(resolve("", "data:application/octet-stream;base64,AA").is_none());
    }

    #[test]
    fn sniffing_does_not_split_characters() {
        let mut head = vec![b' '; SNIFF_BYTES - 1];
        head.extend_from_slice("é\"asset\"".as_bytes());
        assert!(!looks_like_tileset(&head));
        assert!(looks_like_tileset(
            br#"{"asset":{"version":"1.1"},"root":{}}"#
        ));
        assert!(!looks_like_tileset(br#"{"type":"CityJSON","asset":1}"#));
    }

    #[test]
    fn missing_reads_are_listed_then_answered() {
        let tileset = json!({
            "asset": { "version": "1.1" },
            "geometricError": 10,
            "root": { "boundingVolume": { "region": [0.0, 0.0, 0.01, 0.01, 0.0, 10.0] }, "geometricError": 0,
                      "refine": "ADD", "content": { "uri": "city/tileset.json" } }
        });
        let mut source = Cache::default();
        let first = summarize(&tileset, &source, &Options::default()).unwrap();
        assert_eq!(
            first.missing,
            vec![Request::Json {
                uri: "city/tileset.json".into()
            }]
        );

        source.answer_json(
            "city/tileset.json",
            Some(json!({ "asset": { "version": "1.1" }, "geometricError": 1,
                    "root": { "boundingVolume": { "region": [0.0, 0.0, 0.01, 0.01, 0.0, 10.0] }, "geometricError": 0,
                              "content": { "uri": "0.glb" } } })),
        );
        let second = summarize(&tileset, &source, &Options::default()).unwrap();
        assert_eq!(
            second.missing,
            vec![Request::Bytes {
                uri: "city/0.glb".into(),
                offset: 0,
                len: 28
            }]
        );

        // Last round: what is still unanswered is unavailable, and the count unknown
        source.close();
        let last = summarize(&tileset, &source, &Options::default()).unwrap();
        assert!(last.missing.is_empty());
        assert_eq!(last.object_count, None);
    }
}
