use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};
use tantivy::directory::{Directory, RamDirectory};

const TANTIVY_LOCK_FILES: &[&str] = &[".tantivy-meta.lock", ".tantivy-writer.lock"];

pub fn ram_directory_from_mmap_dir<P: AsRef<Path>>(source_dir: P) -> Result<RamDirectory> {
    let source_dir = source_dir.as_ref();
    ensure!(
        source_dir.exists(),
        "mmap directory {} does not exist",
        source_dir.display()
    );
    ensure!(
        source_dir.is_dir(),
        "mmap directory {} is not a directory",
        source_dir.display()
    );
    let ram_directory = RamDirectory::create();
    copy_into_ram_directory(&ram_directory, source_dir, source_dir)?;
    Ok(ram_directory)
}

fn copy_into_ram_directory(
    ram_directory: &RamDirectory,
    base_path: &Path,
    current_path: &Path,
) -> Result<()> {
    for entry in fs::read_dir(current_path)
        .with_context(|| format!("reading directory {}", current_path.display()))?
    {
        let entry = entry.with_context(|| {
            format!("reading entry inside directory {}", current_path.display())
        })?;
        let entry_path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("reading file type for {}", entry_path.display()))?;
        if file_type.is_dir() {
            copy_into_ram_directory(ram_directory, base_path, &entry_path)?;
            continue;
        }
        if file_type.is_file() {
            let relative_path = entry_path
                .strip_prefix(base_path)
                .with_context(|| format!("computing relative path for {}", entry_path.display()))?;
            if TANTIVY_LOCK_FILES
                .iter()
                .any(|lock| relative_path == Path::new(lock))
            {
                continue;
            }
            let data = fs::read(&entry_path)
                .with_context(|| format!("reading {}", entry_path.display()))?;
            ram_directory
                .atomic_write(relative_path, &data)
                .with_context(|| {
                    format!("writing {} into ram directory", relative_path.display())
                })?;
        }
    }
    Ok(())
}
