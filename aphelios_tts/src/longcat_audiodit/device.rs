use anyhow::Result;
use candle_core::{DType, Device};

pub fn select_device(cpu: bool) -> Result<Device> {
    aphelios_core::utils::base::get_default_device(cpu)
}

pub fn default_dtype(_device: &Device) -> DType {
    DType::F32
}
