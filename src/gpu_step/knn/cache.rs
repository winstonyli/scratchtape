//! A built `KnnStore` on disk: a long eval spends most of its time building its memory (forward passes over the train text,
//! then the IVF clustering), and the memory depends only on the model and the build options, so it is built once and reloaded.
//! The file starts with a key string naming everything the contents depend on; `load` returns None when it differs.
use super::ivf::{Ivf, Piece};
use super::*;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: u64 = 0x5343_5254_4b4e_4e02;

fn put(w: &mut impl Write, x: u64) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn put_bytes(w: &mut impl Write, b: &[u8]) -> std::io::Result<()> {
    put(w, b.len() as u64)?;
    w.write_all(b)
}

fn get(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn get_bytes(r: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut b = vec![0u8; get(r)? as usize];
    r.read_exact(&mut b)?;
    Ok(b)
}

impl KnnStore {
    /// Writes the uploaded tiles, the key sample and the IVF index (the store must be `finish`ed) with `key` and a caller value
    /// `extra`, through a temporary file so a crash never leaves a half-written cache.
    pub fn save(&self, path: &Path, key: &str, extra: u64) -> std::io::Result<()> {
        assert!(self.pending_vals.is_empty(), "save after finish");
        let tmp = path.with_extension("part");
        let mut w = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(&tmp)?);
        put(&mut w, MAGIC)?;
        put_bytes(&mut w, key.as_bytes())?;
        for x in [extra, self.d as u64, self.tile as u64, self.f16 as u64, self.kbits as u64, self.tiles.len() as u64] {
            put(&mut w, x)?;
        }
        for t in &self.tiles {
            put(&mut w, t.len as u64)?;
            put_bytes(&mut w, &client().read_one(t.keys.clone()).unwrap())?;
            if let Some(s) = &t.scales {
                put_bytes(&mut w, f32::as_bytes(&read(s)))?;
            }
            put_bytes(&mut w, f32::as_bytes(&read(&t.norms)))?;
            put_bytes(&mut w, f32::as_bytes(&read(&t.vals)))?;
        }
        put_bytes(&mut w, &[])?; // the key sample (older files carry one; it is derived from the tiles now)
        match &self.ivf {
            None => put(&mut w, 0)?,
            Some(ivf) => {
                put(&mut w, 1)?;
                put(&mut w, ivf.nlist as u64)?;
                put(&mut w, ivf.n as u64)?;
                put_bytes(&mut w, &client().read_one(ivf.keys.clone()).unwrap())?;
                put_bytes(&mut w, f32::as_bytes(&ivf.norm))?;
                for cell in &ivf.cells {
                    put(&mut w, cell.len() as u64)?;
                    for p in cell {
                        for x in [p.tile, p.a, p.b, p.g0] {
                            put(&mut w, x as u64)?;
                        }
                    }
                }
            }
        }
        w.flush()?;
        drop(w);
        std::fs::rename(tmp, path)
    }

    /// The store saved at `path` and its `extra`, or None if there is no such file or it was saved under another `key`.
    pub fn load(path: &Path, key: &str) -> std::io::Result<Option<(KnnStore, u64)>> {
        let Ok(file) = std::fs::File::open(path) else { return Ok(None) };
        let mut r = std::io::BufReader::with_capacity(1 << 24, file);
        if get(&mut r)? != MAGIC || get_bytes(&mut r)? != key.as_bytes() {
            return Ok(None);
        }
        let extra = get(&mut r)?;
        let (d, tile, f16, kbits, ntiles) = (get(&mut r)? as usize, get(&mut r)? as usize, get(&mut r)? != 0, get(&mut r)? as u32, get(&mut r)? as usize);
        let mut s = KnnStore::with_precision(d, tile, f16);
        s.kbits = kbits;
        let (mut all_norms, mut all_vals): (Vec<f32>, Vec<f32>) = (vec![], vec![]);
        // A reader thread stays a few tiles ahead of the uploads, so the file reads overlap the host-to-device copies.
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let read_err = std::thread::scope(|sc| {
            let rr = &mut r;
            let reader = sc.spawn(move || -> std::io::Result<()> {
                for _ in 0..ntiles {
                    let len = get(rr)? as usize;
                    let (kb, sb, nb, vb) = (get_bytes(rr)?, if kbits > 0 { Some(get_bytes(rr)?) } else { None }, get_bytes(rr)?, get_bytes(rr)?);
                    if tx.send((len, kb, sb, nb, vb)).is_err() {
                        break;
                    }
                }
                Ok(())
            });
            for (len, kb, sb, nb, vb) in rx {
                all_norms.extend_from_slice(f32::from_bytes(&nb));
                all_vals.extend_from_slice(f32::from_bytes(&vb));
                let keys = client().create_from_slice(&kb);
                let scales = sb.map(|b| client().create_from_slice(&b));
                let norms = client().create_from_slice(&nb);
                let vals = client().create_from_slice(&vb);
                s.tiles.push(Tile { keys, scales, norms, vals, len });
            }
            reader.join().unwrap()
        });
        read_err?;
        assert_eq!(s.tiles.len(), ntiles);
        let skip = get(&mut r)? as i64;
        r.seek_relative(skip)?;
        if get(&mut r)? == 1 {
            let (nlist, n) = (get(&mut r)? as usize, get(&mut r)? as usize);
            let keys = client().create_from_slice(&get_bytes(&mut r)?);
            let norm = f32::from_bytes(&get_bytes(&mut r)?).to_vec();
            let mut cells = Vec::with_capacity(nlist);
            for _ in 0..nlist {
                let np = get(&mut r)? as usize;
                let mut cell = Vec::with_capacity(np);
                for _ in 0..np {
                    let v = [get(&mut r)?, get(&mut r)?, get(&mut r)?, get(&mut r)?].map(|x| x as usize);
                    cell.push(Piece { tile: v[0], a: v[1], b: v[2], g0: v[3] });
                }
                cells.push(cell);
            }
            s.ivf = Some(Ivf { keys, norm, nlist, n, norms: upload_f32(&all_norms), vals: upload_f32(&all_vals), cells });
        }
        Ok(Some((s, extra)))
    }
}
