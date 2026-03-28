use candle_core::Device;

fn main() {
    #[cfg(feature = "metal")]
    {
        println!("Probing Metal device via candle_core::Device::new_metal(0) ...");
        let dev = Device::new_metal(0);
        match dev {
            Ok(_) => println!("Metal device init: OK"),
            Err(e) => println!("Metal device init: ERR: {e}"),
        }
    }

    #[cfg(not(feature = "metal"))]
    {
        println!("Build without `--features metal`, skipping probe.");
    }
}

