use std::path::Path;

use anyhow::Result;
use aphelios_core::utils::core_utils::Error;

#[test]
fn filename_search_init_test() -> Result<(), Error> {
    let dir_path = "/Volumes/sw/books";

    let dir_p = Path::new(dir_path);
    let _ = dir_p.try_exists().map_err(|e| Error::FileNotExists {
        msg: "path must exists",
    });
    if !dir_p.is_dir() {
        return Err(Error::PathMustDir { msg: "must a dir" });
    }
    Ok(())
}

#[test]
fn filename_search_test() -> Result<()> {
    let text_to_search = "东欧";
    let text_to_search2 = "斯大林";

    Ok(())
}
