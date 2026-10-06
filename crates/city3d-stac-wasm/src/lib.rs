//! STAC Items of 3D Tiles tilesets in the browser
//!
//! A WebAssembly module with a JSON in, JSON out interface and no tooling: the
//! page fetches the tileset, then answers in rounds the reads that
//! [`city3d_stac_tiles3d::summarize`] lists as missing (external tilesets,
//! schema files, byte ranges of tile contents), until none is left.
//!
//! Exports: `alloc(len)` and `dealloc(ptr, len)` for buffers, and
//! `analyze(ptr, len)`, which reads a request at `ptr` and returns a pointer to
//! a 4-byte little-endian length followed by the response, both UTF-8 JSON.
//!
//! Request:
//!
//! ```json
//! { "tileset": {…}, "href": "https://…/tileset.json", "id": "lod22",
//!   "json": { "city/tileset.json": {…} or null },
//!   "bytes": [ { "uri": "0.glb", "offset": 0, "len": 28, "data": "<base64>" or null } ],
//!   "closed": false, "max_contents": 500, "max_tilesets": 50 }
//! ```
//!
//! Response: `{ "missing": [ { "uri", "json": true } or { "uri", "offset", "len" } ], "item": {…} }`,
//! or `{ "error": "…" }`.

use city3d_stac_tiles3d::{summarize, Cache, Options, Request, Summary};
use serde_json::{json, Map, Value};

/// Schema of the STAC 3D City Models extension, as `city3d-stac-types` pins it
const CITY3D_EXTENSION: &str = "https://cityjson.github.io/stac-city3d/v0.2.0/schema.json";
const PROJECTION_EXTENSION: &str =
    "https://stac-extensions.github.io/projection/v2.0.0/schema.json";

/// Allocate `len` bytes for the request
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Free a buffer from [`alloc`] or [`analyze`]
///
/// # Safety
/// `ptr` and `len` shall come from [`alloc`], or from [`analyze`] with `len`
/// the response length plus 4.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

/// Analyze a request (see the module documentation)
///
/// # Safety
/// `ptr` shall point to `len` bytes written after an [`alloc`] of `len`.
#[no_mangle]
pub unsafe extern "C" fn analyze(ptr: *mut u8, len: usize) -> *mut u8 {
    let request = Vec::from_raw_parts(ptr, len, len);
    let response = match serde_json::from_slice::<Value>(&request) {
        Ok(request) => respond(&request),
        Err(e) => json!({ "error": format!("Invalid request: {e}") }),
    };
    let body = serde_json::to_vec(&response).unwrap_or_default();
    let mut out = Vec::with_capacity(body.len() + 4);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

/// The response to a request, as JSON
pub fn respond(request: &Value) -> Value {
    let mut cache = Cache::default();
    if let Some(files) = request["json"].as_object() {
        for (uri, value) in files {
            cache.answer_json(uri, (!value.is_null()).then(|| value.clone()));
        }
    }
    for read in request["bytes"].as_array().into_iter().flatten() {
        let (Some(uri), Some(offset), Some(len)) = (
            read["uri"].as_str(),
            read["offset"].as_u64(),
            read["len"].as_u64(),
        ) else {
            continue;
        };
        cache.answer_bytes(
            uri,
            offset,
            len,
            read["data"].as_str().and_then(base64_decode),
        );
    }
    if request["closed"].as_bool() == Some(true) {
        cache.close();
    }
    let options = Options {
        max_contents: request["max_contents"].as_u64().map(|n| n as usize),
        max_tilesets: request["max_tilesets"].as_u64().map(|n| n as usize),
    };
    match summarize(&request["tileset"], &cache, &options) {
        Ok(summary) => json!({
            "missing": summary.missing.iter().map(request_json).collect::<Vec<_>>(),
            "contents_read": summary.contents_read,
            "item": item(
                &summary,
                request["id"].as_str().unwrap_or("tileset"),
                request["href"].as_str().unwrap_or("tileset.json"),
            ),
        }),
        Err(e) => json!({ "error": e }),
    }
}

fn request_json(request: &Request) -> Value {
    match request {
        Request::Json { uri } => json!({ "uri": uri, "json": true }),
        Request::Bytes { uri, offset, len } => json!({ "uri": uri, "offset": offset, "len": len }),
    }
}

/// A STAC Item shaped as `city3dstac item` writes it for a tileset
pub fn item(summary: &Summary, id: &str, href: &str) -> Value {
    let mut properties = Map::new();
    properties.insert("datetime".into(), Value::Null);
    if let Some(version) = &summary.version {
        properties.insert("city3d:version".into(), json!(version));
    }
    if let Some(count) = summary.object_count {
        properties.insert("city3d:city_objects".into(), json!(count));
    }
    if !summary.classes.is_empty() {
        properties.insert("city3d:co_types".into(), json!(summary.classes));
    }
    if !summary.attributes.is_empty() {
        let attributes: Vec<Value> = summary
            .attributes
            .iter()
            .map(|(name, t)| json!({ "name": name, "type": t.as_str() }))
            .collect();
        properties.insert("city3d:attributes".into(), json!(attributes));
    }
    if let Some(textures) = summary.textures {
        properties.insert("city3d:textures".into(), json!(textures));
    }
    if let Some(materials) = summary.materials {
        properties.insert("city3d:materials".into(), json!(materials));
    }
    properties.insert("proj:code".into(), json!("EPSG:4979"));

    let (geometry, bbox) = match summary.bbox {
        Some([w, s, zmin, e, n, zmax]) => (
            json!({ "type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]] }),
            json!([w, s, zmin, e, n, zmax]),
        ),
        None => (Value::Null, Value::Null),
    };
    let mut item = json!({
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [CITY3D_EXTENSION, PROJECTION_EXTENSION],
        "id": id,
        "geometry": geometry,
        "properties": properties,
        "links": [{ "href": href, "rel": "city-model", "type": "application/json" }],
        "assets": { "data": {
            "href": href, "title": "3D city model data", "type": "application/json", "roles": ["data"]
        } }
    });
    if !bbox.is_null() {
        item["bbox"] = bbox;
    }
    item
}

/// Standard base64 (RFC 4648, section 4), padding optional
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let bytes: Vec<u8> = text.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= value(c)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        assert_eq!(base64_decode("Z2xURg==").unwrap(), b"glTF");
        assert_eq!(base64_decode("YjNkbQ").unwrap(), b"b3dm");
        assert!(base64_decode("@@").is_none());
    }

    #[test]
    fn rounds_until_nothing_is_missing() {
        let tileset = json!({
            "asset": { "version": "1.1" }, "geometricError": 10,
            "root": { "boundingVolume": { "region": [0.0, 0.0, 0.01, 0.01, 0.0, 10.0] },
                      "geometricError": 0, "refine": "ADD", "content": { "uri": "0.glb" } }
        });
        let first =
            respond(&json!({ "tileset": tileset, "id": "t", "href": "https://x/tileset.json" }));
        assert_eq!(
            first["missing"],
            json!([{ "uri": "0.glb", "offset": 0, "len": 28 }])
        );
        assert_eq!(first["item"]["bbox"][3], json!(0.01f64.to_degrees()));

        // The content is not a tile format: the last round gives the item without count
        let last = respond(&json!({
            "tileset": tileset, "id": "t", "href": "https://x/tileset.json",
            "bytes": [{ "uri": "0.glb", "offset": 0, "len": 28, "data": "AAAA" }], "closed": true
        }));
        assert_eq!(last["missing"], json!([]));
        assert!(last["item"]["properties"]
            .get("city3d:city_objects")
            .is_none());
    }
}
