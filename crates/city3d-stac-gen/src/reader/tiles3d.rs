//! 3D Tiles reader (`tileset.json`)
//!
//! Reads a tileset from a local folder or a URL. What is read and how it maps
//! to the `city3d` fields, with the references to the specifications, is in the
//! `city3d-stac-tiles3d` crate; this module only provides the files: from disk,
//! or over the network with ranged reads, in rounds.

use crate::error::{CityJsonStacError, Result};
use crate::metadata::{AttributeDefinition, AttributeType, BBox3D, Transform, CRS};
use crate::reader::CityModelMetadataReader;
use crate::remote::{download_from_url, download_range};

use city3d_stac_tiles3d::{self as tiles3d, AttrType, Cache, Fetch, Options, Request, Source};
use futures::stream::{self, StreamExt};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub use city3d_stac_tiles3d::is_tileset;

/// Rounds of ranged reads for a remote tileset: external tilesets, content
/// headers, then the JSON they point to
const REMOTE_ROUNDS: usize = 6;
/// Tile contents and external tilesets read for a remote tileset; beyond, the
/// object count is unknown (a national tileset holds hundreds of thousands)
const REMOTE_MAX_CONTENTS: usize = 500;
const REMOTE_MAX_TILESETS: usize = 50;
/// Concurrent requests for a remote tileset
const REMOTE_CONCURRENCY: usize = 16;

/// Reader for 3D Tiles tilesets (`tileset.json`)
pub struct Tiles3DReader {
    file_path: PathBuf,
    summary: tiles3d::Summary,
}

impl Tiles3DReader {
    /// Create a reader for a local tileset; its tile contents are read too
    pub fn new(file_path: &Path) -> Result<Self> {
        let tileset: Value = serde_json::from_str(&fs::read_to_string(file_path)?)?;
        let source = FolderSource {
            folder: file_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
        };
        Ok(Self {
            file_path: file_path.to_path_buf(),
            summary: summarize(&tileset, &source, &Options::default())?,
        })
    }

    /// Create a reader from the content of a tileset alone, without reading its tile contents
    pub fn from_content(content: &str, virtual_path: PathBuf) -> Result<Self> {
        let tileset: Value = serde_json::from_str(content)?;
        let mut cache = Cache::default();
        cache.close();
        Ok(Self {
            file_path: virtual_path,
            summary: summarize(&tileset, &cache, &Options::default())?,
        })
    }

    /// Create a reader for a remote tileset whose content is already downloaded.
    ///
    /// External tilesets, schema files and the headers of the tile contents are
    /// fetched with ranged reads, at most [`REMOTE_MAX_CONTENTS`] contents and
    /// [`REMOTE_MAX_TILESETS`] external tilesets.
    pub async fn from_url(url: &str, content: &str, virtual_path: PathBuf) -> Result<Self> {
        let tileset: Value = serde_json::from_str(content)?;
        let base = url.rsplit_once('/').map_or(url, |(base, _)| base);
        let options = Options {
            max_contents: Some(REMOTE_MAX_CONTENTS),
            max_tilesets: Some(REMOTE_MAX_TILESETS),
        };
        let mut cache = Cache::default();
        for round in 0..REMOTE_ROUNDS {
            let summary = summarize(&tileset, &cache, &options)?;
            if summary.missing.is_empty() {
                break;
            }
            log::debug!(
                "3D Tiles round {}: {} reads",
                round + 1,
                summary.missing.len()
            );
            let answers: Vec<(Request, Option<Vec<u8>>)> = stream::iter(summary.missing)
                .map(|request| async move {
                    let data = fetch_request(base, &request).await;
                    (request, data)
                })
                .buffer_unordered(REMOTE_CONCURRENCY)
                .collect()
                .await;
            for (request, data) in answers {
                cache.answer(&request, data);
            }
        }
        cache.close();
        Ok(Self {
            file_path: virtual_path,
            summary: summarize(&tileset, &cache, &options)?,
        })
    }
}

/// Perform a read of a remote tileset; `None` when it fails
async fn fetch_request(base: &str, request: &Request) -> Option<Vec<u8>> {
    let url_of = |uri: &str| {
        if uri.contains("://") {
            uri.to_string()
        } else {
            format!("{base}/{uri}")
        }
    };
    let result = match request {
        Request::Json { uri } => download_from_url(&url_of(uri)).await,
        Request::Bytes { uri, offset, len } => download_range(&url_of(uri), *offset, *len).await,
    };
    match result {
        Ok(bytes) => Some(bytes.to_vec()),
        Err(e) => {
            log::debug!("3D Tiles read failed: {e}");
            None
        }
    }
}

fn summarize(tileset: &Value, source: &dyn Source, options: &Options) -> Result<tiles3d::Summary> {
    tiles3d::summarize(tileset, source, options).map_err(CityJsonStacError::InvalidCityJson)
}

/// Files of a local tileset, relative to its folder
struct FolderSource {
    folder: PathBuf,
}

impl FolderSource {
    fn path(&self, uri: &str) -> Option<PathBuf> {
        (!uri.contains("://")).then(|| self.folder.join(uri))
    }
}

impl Source for FolderSource {
    fn json(&self, uri: &str) -> Fetch<Value> {
        self.path(uri)
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|text| serde_json::from_str(&text).ok())
            .map_or(Fetch::Unavailable, Fetch::Ready)
    }

    fn bytes(&self, uri: &str, offset: u64, len: u64) -> Fetch<Vec<u8>> {
        let read = || -> std::io::Result<Vec<u8>> {
            let mut file = File::open(self.path(uri).ok_or(std::io::ErrorKind::NotFound)?)?;
            file.seek(SeekFrom::Start(offset))?;
            let mut bytes = Vec::new();
            file.take(len).read_to_end(&mut bytes)?;
            Ok(bytes)
        };
        read().map_or(Fetch::Unavailable, Fetch::Ready)
    }
}

/// Whether a local `.json` file looks like a tileset rather than CityJSON (only its start is read)
pub fn looks_like_tileset(path: &Path) -> Result<bool> {
    let mut head = Vec::with_capacity(64 * 1024);
    File::open(path)?.take(64 * 1024).read_to_end(&mut head)?;
    Ok(tiles3d::looks_like_tileset(&head))
}

/// Same test as [`looks_like_tileset`], on content already in memory
pub fn head_looks_like_tileset(head: &str) -> bool {
    tiles3d::looks_like_tileset(head.as_bytes())
}

fn attribute_type(attr_type: AttrType) -> AttributeType {
    match attr_type {
        AttrType::String => AttributeType::String,
        AttrType::Number => AttributeType::Number,
        AttrType::Boolean => AttributeType::Boolean,
        AttrType::Array => AttributeType::Array,
        AttrType::Object => AttributeType::Object,
    }
}

impl CityModelMetadataReader for Tiles3DReader {
    fn bbox(&self) -> Result<BBox3D> {
        let b = self.summary.bbox.ok_or_else(|| {
            CityJsonStacError::MetadataError(
                "Tileset root bounding volume missing or not georeferenced".to_string(),
            )
        })?;
        Ok(BBox3D::new(b[0], b[1], b[2], b[3], b[4], b[5]))
    }

    fn crs(&self) -> Result<CRS> {
        // The bbox is in WGS 84 longitude, latitude and ellipsoidal height
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
            .map(|(name, t)| AttributeDefinition::new(name.clone(), attribute_type(*t)))
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
