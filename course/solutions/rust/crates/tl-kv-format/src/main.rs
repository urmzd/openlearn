//! Offline process boundary for rolling format changes: read one exported KV
//! envelope and write the converted envelope to another file.
use std::{env, fs, process};
use tl_kv_format::{upgrade_v1, Shape};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 { return Err("usage: kv-migrate <input> <output> <block_tokens> <layers> <kv_heads> <head_dim>".into()); }
    let n = |i:usize| args[i].parse::<u32>().map_err(|e| format!("invalid dimension {}: {e}",args[i]));
    let shape=Shape{block_tokens:n(3)?,layers:n(4)?,kv_heads:n(5)?,head_dim:n(6)?};
    let input=fs::read(&args[1])?;
    let output=upgrade_v1(&input,shape)?;
    let tmp=args[2].to_owned()+".tmp";
    fs::write(&tmp,output)?;
    fs::rename(tmp,&args[2])?;
    Ok(())

}
fn main(){
        if let Err(e)=run(){eprintln!("kv-migrate: {e}");process::exit(2)}
}
