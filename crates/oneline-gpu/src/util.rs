//! Small helpers shared by the GPU stages: bind-group plumbing, sizing, readback.

use anyhow::{Result, anyhow};
use bytemuck::Pod;

/// Longest input the GPU path accepts: offsets and word indices are `u32` inside
/// the shaders, and the text is padded to a multiple of 4 bytes.
pub const MAX_GPU_INPUT_LEN: usize = u32::MAX as usize - 3;

pub(crate) fn check_gpu_len(len: usize) -> Result<()> {
    if len > MAX_GPU_INPUT_LEN {
        return Err(anyhow!(
            "input is {len} bytes; the GPU path addresses at most {MAX_GPU_INPUT_LEN} bytes (32-bit offsets)"
        ));
    }
    Ok(())
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// 2D dispatch size covering `total_workgroups` (one dimension is capped at 65535).
/// The shaders recombine it as `y * 65535 + x`.
pub(crate) fn dispatch_dims(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= 65535 {
        (total_workgroups.max(1), 1)
    } else {
        (65535, total_workgroups.div_ceil(65535))
    }
}

pub(crate) fn compute_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &'static str,
    timer: Option<&crate::timing::StageTimer>,
) -> wgpu::ComputePass<'e> {
    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: timer.and_then(|t| t.writes(label)),
    })
}

/// A `MAP_READ` buffer to copy results into.
pub(crate) fn staging_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Maps all `buffers` for reading and blocks until the GPU is done with them.
pub(crate) fn map_all(device: &wgpu::Device, buffers: &[&wgpu::Buffer]) -> Result<()> {
    let receivers: Vec<_> = buffers
        .iter()
        .map(|buffer| {
            let (tx, rx) = std::sync::mpsc::channel();
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            rx
        })
        .collect();
    device.poll(wgpu::Maintain::wait());
    for rx in receivers {
        rx.recv()?
            .map_err(|e| anyhow!("failed to map staging buffer: {e:?}"))?;
    }
    Ok(())
}

/// Copies the first `count` elements out of a mapped staging buffer and unmaps it.
pub(crate) fn take_mapped<T: Pod>(buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let view = buffer.slice(..).get_mapped_range();
    let out = bytemuck::cast_slice::<u8, T>(&view)[..count].to_vec();
    drop(view);
    buffer.unmap();
    out
}
