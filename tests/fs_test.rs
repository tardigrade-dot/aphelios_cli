use std::fs;
use std::path::Path;
use anyhow::Result;
use epub::doc::EpubDoc;
use tracing::info;

#[test]
fn fs_test_1() -> Result<()>{
    // let res = fs::create_dir_all("/Users/larry/coderesp/aphelios_cli/output/a/b/c");

    let res = fs::write("/Users/larry/coderesp/aphelios_cli/output/a.txt", "hello world");
    Ok(res?)
    
}

#[test]
fn fs_test_2() -> Result<()>{

    let path = Path::new("/Volumes/sw/books");

    // 获取当前目录下所有的文件和子目录的路径列表。
    let entries = fs::read_dir(path).unwrap();

    let mut i: u32 = 0;
    for entry in entries {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                if path.is_file() { // 如果是普通文件，则打印其名称。

                    i = i + 1;
                    println!("{} - {}",i , path.file_name().unwrap().to_str().unwrap());
                } else if path.is_dir() { // 如果是子目录，则递归遍历该子目录。
                    println!("Directory: {}", path.to_str().unwrap());
                    fs::read_dir(&path).unwrap(); // 递归地读取该子目录中的内容。
                }
            }
            Err(e) => eprintln!("Error reading {}: {}", path.display(), e),
        }
    }
    Ok(())
}

#[test]
fn get_epub_desc() -> Result<()>{


    let doc = EpubDoc::new("/Users/larry/Downloads/俄罗斯：一千年的狂野纪事（全译本） 自制 (马丁·西克史密斯(Martin Sixsmith) 著, 周全 译) (Z-Library).epub").ok();
    let r = doc.unwrap().metadata;
    
    for i in r{
        println!("{} {}", i.property, i.value);
        if i.property == "description"{
            println!("description is {}", i.value);
        }
    }
    Ok(())
}