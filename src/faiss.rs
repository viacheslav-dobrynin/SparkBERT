use std::ffi::{CStr, CString};
use std::os::raw::c_int;

use anyhow::anyhow;
use faiss_sys as ffi;

pub mod error {
    pub type Error = anyhow::Error;
    pub type Result<T> = anyhow::Result<T>;
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct Idx(ffi::idx_t);

impl Idx {
    pub fn new(idx: u64) -> Self {
        assert!(
            idx < 0x8000_0000_0000_0000,
            "too large index value provided to Idx::new"
        );
        let idx = idx as ffi::idx_t;
        Idx(idx)
    }

    pub fn none() -> Self {
        Idx(-1)
    }

    pub fn is_none(self) -> bool {
        self.0 == -1
    }

    pub fn is_some(self) -> bool {
        self.0 != -1
    }

    pub fn get(self) -> Option<u64> {
        match self.0 {
            -1 => None,
            x => Some(x as u64),
        }
    }

    pub fn to_native(self) -> ffi::idx_t {
        self.0
    }
}

pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<Idx>,
}

pub trait Index {
    fn d(&self) -> u32;
    fn ntotal(&self) -> u64;
    fn search(&mut self, q: &[f32], k: usize) -> error::Result<SearchResult>;
    fn reconstruct(&self, key: Idx, output: &mut [f32]) -> error::Result<()>;
}

fn faiss_try(code: c_int) -> error::Result<()> {
    if code == 0 {
        return Ok(());
    }
    unsafe {
        let ptr = ffi::faiss_get_last_error();
        if ptr.is_null() {
            return Err(anyhow!("Faiss error (code {}), but no message", code));
        }
        let cstr = CStr::from_ptr(ptr);
        let msg: String = cstr.to_string_lossy().into_owned();
        Err(anyhow!("Faiss error (code {}): {}", code, msg))
    }
}

#[derive(Debug)]
pub struct IndexImpl {
    inner: *mut ffi::FaissIndex,
}

unsafe impl Send for IndexImpl {}
unsafe impl Sync for IndexImpl {}

impl Drop for IndexImpl {
    fn drop(&mut self) {
        unsafe {
            ffi::faiss_Index_free(self.inner);
        }
    }
}

impl IndexImpl {
    pub fn d(&self) -> u32 {
        unsafe { ffi::faiss_Index_d(self.inner as *const _) as u32 }
    }

    pub fn ntotal(&self) -> u64 {
        unsafe { ffi::faiss_Index_ntotal(self.inner as *const _) as u64 }
    }

    pub fn search(&mut self, q: &[f32], k: usize) -> error::Result<SearchResult> {
        let d = self.d() as usize;
        if d == 0 {
            return Err(anyhow!("Faiss index has zero dimension"));
        }
        if q.len() % d != 0 {
            return Err(anyhow!(
                "Input vector length {} is not divisible by index dimension {}",
                q.len(),
                d
            ));
        }
        let n = q.len() / d;
        let mut distances = vec![0f32; n * k];
        let mut labels_raw = vec![0 as ffi::idx_t; n * k];
        unsafe {
            faiss_try(ffi::faiss_Index_search(
                self.inner as *const _,
                n as ffi::idx_t,
                q.as_ptr(),
                k as ffi::idx_t,
                distances.as_mut_ptr(),
                labels_raw.as_mut_ptr(),
            ))?;
        }
        let labels = labels_raw.into_iter().map(Idx).collect();
        Ok(SearchResult { distances, labels })
    }

    pub fn reconstruct(&self, key: Idx, output: &mut [f32]) -> error::Result<()> {
        let d = self.d() as usize;
        if output.len() != d {
            return Err(anyhow!(
                "Output vector length {} does not match index dimension {}",
                output.len(),
                d
            ));
        }
        unsafe {
            faiss_try(ffi::faiss_Index_reconstruct(
                self.inner as *const _,
                key.to_native(),
                output.as_mut_ptr(),
            ))
        }
    }
}

impl Index for IndexImpl {
    fn d(&self) -> u32 {
        self.d()
    }

    fn ntotal(&self) -> u64 {
        self.ntotal()
    }

    fn search(&mut self, q: &[f32], k: usize) -> error::Result<SearchResult> {
        self.search(q, k)
    }

    fn reconstruct(&self, key: Idx, output: &mut [f32]) -> error::Result<()> {
        self.reconstruct(key, output)
    }
}

pub fn read_index<P>(file_name: P) -> error::Result<IndexImpl>
where
    P: AsRef<str>,
{
    let f = file_name.as_ref();
    let f = CString::new(f).map_err(|_| anyhow!("Invalid file path"))?;
    let mut inner = std::ptr::null_mut();
    unsafe {
        faiss_try(ffi::faiss_read_index_fname(
            f.as_ptr(),
            0, // IoFlags::MEM_RESIDENT
            &mut inner,
        ))?;
    }
    if inner.is_null() {
        Err(anyhow!("Faiss returned a null index pointer"))
    } else {
        Ok(IndexImpl { inner })
    }
}

pub mod index {
    pub use super::{Idx, Index, IndexImpl, SearchResult};
}

