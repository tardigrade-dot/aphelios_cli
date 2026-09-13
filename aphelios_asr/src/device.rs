#[allow(unused_imports)]
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use ort::ep::{ExecutionProviderDispatch, CPU};
use tracing::{info, warn};

pub fn get_cpu_ep() -> Vec<ExecutionProviderDispatch> {
    vec![CPU::default().build().into()]
}

pub fn get_available_ep() -> Vec<ExecutionProviderDispatch> {
    let mut execution_providers = Vec::new();
    #[cfg(feature = "metal")]
    {
        use ort::ep::CoreML;
        execution_providers.push(CoreML::default().build().into());
    }
    #[cfg(feature = "cuda")]
    {
        use ort::ep::CUDA;
        execution_providers.push(CUDA::default().build().into());
    }
    execution_providers.push(CPU::default().build().into());
    execution_providers
}

pub fn get_default_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        return try_metal_device().with_context(|| {
            "Metal support is compiled in, but the current process could not initialize a Metal device"
        });
    }
    #[cfg(not(feature = "metal"))]
    #[cfg(feature = "cuda")]
    {
        use candle_core::utils::cuda_is_available;
        if cuda_is_available() {
            return Ok(Device::Cuda(0)?);
        }
    }
    #[cfg(not(feature = "metal"))]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        info!("Running on CPU, to run on GPU(metal), build with `--features metal`");
    }
    #[cfg(not(feature = "metal"))]
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        info!("Running on CPU, to run on GPU, build with `--features cuda`");
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::Cpu)
    }
}

#[cfg(feature = "metal")]
fn try_metal_device() -> Result<Device> {
    use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

    let previous_hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| Device::new_metal(0)));
    set_hook(previous_hook);

    match result {
        Ok(Ok(device)) => Ok(device),
        Ok(Err(err)) => Err(anyhow!("Metal device init failed: {err}")),
        Err(_) => Err(anyhow!(
            "Metal device init panicked, likely because no default Metal device was exposed"
        )),
    }
}

pub fn get_device() -> Device {
    get_default_device(false).unwrap_or_else(|err| {
        panic!("Device initialization failed: {err}");
    })
}

pub fn get_device_dtype() -> (Device, DType) {
    let device = get_default_device(false).unwrap_or_else(|err| {
        panic!("Device initialization failed: {err}");
    });
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    (device, dtype)
}

pub fn get_device_fallback() -> Device {
    get_default_device(false).unwrap_or_else(|err| {
        warn!("Device initialization failed, falling back to CPU: {err}");
        Device::Cpu
    })
}
