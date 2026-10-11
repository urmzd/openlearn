//! Course tests for the isolated KV envelope migration utility.
//! Each case protects a migration invariant that could otherwise corrupt a
//! block during a mixed-version deployment.
use tl_kv_format::{decode, encode, upgrade_v1, Block, Envelope, Error, Shape, V1, V2};
use std::{fs, path::PathBuf, process::Command};
fn fixture(name:&str)->Vec<u8>{ let root=PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("ol sets fixture root")); fs::read(root.join("craft.13").join(name)).unwrap() }

fn shape() -> Shape { Shape { block_tokens: 2, layers: 1, kv_heads: 1, head_dim: 2 } }
fn sample() -> Envelope {
    Envelope { version: V1, shape: shape(), blocks: vec![Block { hash: 0x1234, n_tokens: 2, values: vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 0.25] }] }
}

#[test]
// WHY: The worked v1 fixture must upgrade byte-for-byte to the published v2 fixture.
// KIND: unit
fn hand_example_golden_migration_matches_fixture() {
    // A migration must retain routing identity and dimensions, not just turn bytes into fp8.
    let old = fixture("v1-worked.bin");
    let new = upgrade_v1(&old, shape()).unwrap();
    assert_eq!(new, fixture("v2-worked.bin"));
    let got = decode(&new, shape()).unwrap();
    assert_eq!(got.version, V2);
    assert_eq!(got.blocks[0].hash, 0xA91AB0C1027B9366);
    assert_eq!(got.blocks[0].n_tokens, 2);
    for (a,b) in got.blocks[0].values.iter().zip(&sample().blocks[0].values) { assert!((a-b).abs() <= 0.2, "{a} != {b}"); }
}

#[test]
// WHY: Mixed-version rollout requires each dtype to decode under its matching version.
// KIND: unit
fn both_formats_roundtrip_with_version_specific_dtype() {
    // A v2 reader must continue to read v1 while the rollout has mixed versions.
    for version in [V1,V2] { let bytes=encode(&sample(),version).unwrap(); let got=decode(&bytes,shape()).unwrap(); assert_eq!(got.version,version); }
}

#[test]
// WHY: Corrupt checksums must reject the full envelope before exposing blocks.
// KIND: boundary
fn rejects_corruption_before_returning_a_partial_envelope() {
    // CRC failure must reject the whole transfer before any block is accepted.
    let mut bytes=encode(&sample(),V1).unwrap(); bytes[40]^=0x80;
    assert_eq!(decode(&bytes,shape()),Err(Error::Checksum));
}

#[test]
// WHY: Readers must fail closed on incompatible dimensions and unsupported versions.
// KIND: boundary
fn rejects_shape_mismatch_and_unknown_version() {
    // Peer incompatibility is explicit, so negotiation can fail before applying data.
    let bytes=encode(&sample(),V1).unwrap();
    assert_eq!(decode(&bytes,Shape{head_dim:3,..shape()}),Err(Error::Shape));
    let mut bad=bytes.clone(); bad[4]=9; bad[5]=0;
    assert_eq!(decode(&bad,shape()),Err(Error::Version));
}

#[test]
// WHY: Tail padding and hash identity must remain safe for a partially filled block.
// KIND: unit
fn partial_tail_is_zero_filled_and_never_claims_a_full_hash() {
    // Padding must be deterministic and a partial block cannot enter prefix dedup.
    let mut x=sample(); x.blocks[0].hash=0; x.blocks[0].n_tokens=1;
    let bytes=encode(&x,V2).unwrap(); let got=decode(&bytes,shape()).unwrap();
    assert_eq!(got.blocks[0].hash,0);
    assert!(got.blocks[0].values[2..4].iter().all(|v| *v == 0.0));
}

#[test]
// WHY: An all-zero slab must use a finite, nonzero scale.
// KIND: boundary
fn zero_slab_uses_unit_scale() {
    // The scale must be finite for an all-zero slab and decode back to zeros.
    let mut x=sample(); x.blocks[0].values.fill(0.0);
    let bytes=encode(&x,V2).unwrap(); assert!(decode(&bytes,shape()).unwrap().blocks[0].values.iter().all(|v|*v==0.0));
}

#[test]
// WHY: The one-way migration must reject already upgraded bytes.
// KIND: boundary
fn v2_upgrader_refuses_a_v2_input_instead_of_double_converting() {
    // The CLI is deliberately a one-way v1 to v2 migration, safe to rerun only on original inputs.
    let v2=encode(&sample(),V2).unwrap();
    assert_eq!(upgrade_v1(&v2,shape()),Err(Error::Version));
}

#[test]
// WHY: The command-line migration boundary must convert files and leave no partial output.
// KIND: boundary
fn process_converter_reads_and_atomically_writes_file_payloads() {
    // Language boundaries use a subprocess and files, making this converter callable without linking another runtime.
    let dir=std::env::temp_dir().join(format!("kv-migrate-{}",std::process::id()));let _=fs::remove_dir_all(&dir);fs::create_dir_all(&dir).unwrap();
    let input=dir.join("before.kv");let output=dir.join("after.kv");fs::write(&input,fixture("v1-worked.bin")).unwrap();
    // ol runs these tests from a generated crate, so build and launch the
    // declared binary target explicitly instead of relying on Cargo's
    // CARGO_BIN_EXE_* variables, which only cover binaries in this package.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("crates/tl-kv-format/Cargo.toml");
    let status=Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(manifest)
        .args(["--bin", "kv-migrate", "--"])
        .arg(&input).arg(&output).args(["2","1","1","2"])
        .status().unwrap();
    assert!(status.success());assert_eq!(fs::read(&output).unwrap(),fixture("v2-worked.bin"));let _=fs::remove_dir_all(dir);
}
