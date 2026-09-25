"""Minimaler Nachbau des Lightroom-Katalogschemas für Tests.

Das ist KEIN echter Lightroom-Katalog, sondern bildet nur die Tabellen und Spalten nach,
die Imagomat liest bzw. schreibt. Die Kompatibilität mit echten Katalogen muss mit einem
leeren, von Lightroom erzeugten Katalog geprüft werden (siehe docs/lightroom-roundtrip.md).
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE Adobe_variablesTable (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, name, type, value NOT NULL DEFAULT '');
CREATE TABLE AgLibraryRootFolder (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, absolutePath UNIQUE NOT NULL DEFAULT '', name NOT NULL DEFAULT '', relativePathFromCatalog);
CREATE TABLE AgLibraryFolder (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, parentId INTEGER, pathFromRoot NOT NULL DEFAULT '', rootFolder INTEGER NOT NULL DEFAULT 0, visibility INTEGER);
CREATE TABLE AgLibraryFile (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, baseName NOT NULL DEFAULT '', errorMessage, errorTime, extension NOT NULL DEFAULT '', externalModTime, folder INTEGER NOT NULL DEFAULT 0, idx_filename NOT NULL DEFAULT '', importHash, lc_idx_filename NOT NULL DEFAULT '', lc_idx_filenameExtension NOT NULL DEFAULT '', md5, modTime, originalFilename NOT NULL DEFAULT '', sidecarExtensions);
CREATE TABLE Adobe_images (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, aspectRatioCache NOT NULL DEFAULT -1, bitDepth NOT NULL DEFAULT 0, captureTime, colorChannels NOT NULL DEFAULT 0, colorLabels NOT NULL DEFAULT '', colorMode NOT NULL DEFAULT -1, copyCreationTime NOT NULL DEFAULT -63113817600, copyName, copyReason, developSettingsIDCache, editLock INTEGER NOT NULL DEFAULT 0, fileFormat NOT NULL DEFAULT 'unset', fileHeight, fileWidth, hasMissingSidecars INTEGER, masterImage INTEGER, orientation, originalCaptureTime, originalRootEntity INTEGER, panningDistanceH, panningDistanceV, pick NOT NULL DEFAULT 0, positionInFolder NOT NULL DEFAULT 'z', previewApplication, previewApplicationVersion, previewSize, pyramidIDCache, rating, rootFile INTEGER NOT NULL DEFAULT 0, sidecarStatus, touchCount NOT NULL DEFAULT 0, touchTime NOT NULL DEFAULT 0);
CREATE TABLE Adobe_imageDevelopSettings (id_local INTEGER PRIMARY KEY, allowFastRender INTEGER, beforeSettingsIDCache, croppedHeight, croppedWidth, digest, fileHeight, fileWidth, grayscale INTEGER, hasDevelopAdjustments INTEGER, hasDevelopAdjustmentsEx, historySettingsID, image INTEGER, processVersion, settingsID, snapshotID, text, validatedForVersion, whiteBalance);
CREATE TABLE Adobe_libraryImageDevelopHistoryStep (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, dateCreated, digest, hasDevelopAdjustments, image INTEGER, name, relValueString, text, valueString);
CREATE TABLE Adobe_AdditionalMetadata (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, additionalInfoSet INTEGER NOT NULL DEFAULT 0, embeddedXmp INTEGER NOT NULL DEFAULT 0, externalXmpIsDirty INTEGER NOT NULL DEFAULT 0, image INTEGER, incrementalWhiteBalance INTEGER NOT NULL DEFAULT 0, internalXmpDigest, isRawFile INTEGER NOT NULL DEFAULT 0, lastSynchronizedHash, lastSynchronizedTimestamp NOT NULL DEFAULT -63113817600, metadataPresetID, metadataVersion, monochrome INTEGER NOT NULL DEFAULT 0, xmp NOT NULL DEFAULT '');
CREATE TABLE AgHarvestedExifMetadata (id_local INTEGER PRIMARY KEY, image INTEGER, aperture, cameraModelRef INTEGER, cameraSNRef INTEGER, dateDay, dateMonth, dateYear, flashFired INTEGER, focalLength, gpsLatitude, gpsLongitude, gpsSequence NOT NULL DEFAULT 0, hasGPS INTEGER, isoSpeedRating, lensRef INTEGER, shutterSpeed);
CREATE TABLE AgInternedExifCameraModel (id_local INTEGER PRIMARY KEY, searchIndex, value);
CREATE TABLE AgInternedExifLens (id_local INTEGER PRIMARY KEY, searchIndex, value);
CREATE TABLE AgLibraryKeyword (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, dateCreated NOT NULL DEFAULT '', genealogy NOT NULL DEFAULT '', imageCountCache DEFAULT -1, includeOnExport INTEGER NOT NULL DEFAULT 1, includeParents INTEGER NOT NULL DEFAULT 1, includeSynonyms INTEGER NOT NULL DEFAULT 1, keywordType, lastApplied, lc_name, name, parent INTEGER);
CREATE TABLE AgLibraryKeywordImage (id_local INTEGER PRIMARY KEY, image INTEGER NOT NULL DEFAULT 0, tag INTEGER NOT NULL DEFAULT 0);
CREATE TABLE AgLibraryCollection (id_local INTEGER PRIMARY KEY, creationId NOT NULL DEFAULT '', genealogy NOT NULL DEFAULT '', imageCount, name NOT NULL DEFAULT '', parent INTEGER, systemOnly NOT NULL DEFAULT '');
CREATE TABLE AgLibraryCollectionImage (id_local INTEGER PRIMARY KEY, collection INTEGER NOT NULL DEFAULT 0, image INTEGER NOT NULL DEFAULT 0, pick NOT NULL DEFAULT 0, positionInCollection);
CREATE TABLE AgLibraryFolderStack (id_local INTEGER PRIMARY KEY, id_global UNIQUE NOT NULL, collapsed INTEGER NOT NULL DEFAULT 0, text NOT NULL DEFAULT '');
CREATE TABLE AgLibraryFolderStackImage (id_local INTEGER PRIMARY KEY, collapsed INTEGER NOT NULL DEFAULT 0, image INTEGER NOT NULL DEFAULT 0, position NOT NULL DEFAULT '', stack INTEGER NOT NULL DEFAULT 0);
"""


def make_template(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO AgLibraryKeyword (id_local, id_global, genealogy, name, parent) VALUES (20, 'ROOT', '/220', NULL, NULL)")
    conn.execute("INSERT INTO Adobe_variablesTable (id_local, id_global, name, value) VALUES (1, 'V1', 'AgLibraryKeyword_rootTagID', '20')")
    conn.execute("INSERT INTO Adobe_variablesTable (id_local, id_global, name, value) VALUES (2, 'V2', 'Adobe_entityIDCounter', '100')")
    conn.execute("INSERT INTO Adobe_variablesTable (id_local, id_global, name, value) VALUES (3, 'V3', 'Adobe_DBVersion', '1500000')")
    conn.commit()
    conn.close()
    return path
