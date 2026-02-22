use candle_core::backend::BackendDevice;
use candle_core::Device;
use candle_core::MetalDevice;

pub mod c1;

pub fn metal_device() -> Result<Device, candle_core::Error> {
    Ok(Device::Metal(MetalDevice::new(0)?))
}
