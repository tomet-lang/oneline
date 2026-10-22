use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use oneline_core::{FlatNode, FlatTree};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShaderConfig {
    line_count: u32,
    text_len: u32,
}

/// Massively parallel GPU parser powered by wgpu Compute Shaders.
pub struct GpuParser {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuParser {
    /// Initializes GPU device and compute pipeline.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let adapter = adapters
            .into_iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .or_else(|| {
                instance.enumerate_adapters(wgpu::Backends::all()).into_iter().next()
            })
            .ok_or_else(|| anyhow!("Failed to find suitable GPU adapter"))?;

        eprintln!("==> [wgpu] Selected GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Oneline GPU Parser Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        let shader_src = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Oneline Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Oneline Parser Bind Group Layout"),
            entries: &[
                // 0: input_text
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: line_offsets
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: config
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: nodes output
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Oneline Parser Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Oneline Parser Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("parse_lines"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Parses text in parallel using the GPU.
    pub fn parse(&self, input: &str, line_offsets: &[u32]) -> Result<FlatTree> {
        pollster::block_on(self.parse_async(input, line_offsets))
    }

    pub async fn parse_async(&self, input: &str, line_offsets: &[u32]) -> Result<FlatTree> {
        let line_count = line_offsets.len() as u32;
        if line_count == 0 {
            return Ok(FlatTree::new(input.to_string(), Vec::new(), Vec::new()));
        }

        let input_bytes = input.as_bytes();
        let text_len = input_bytes.len() as u32;

        // Pad byte array to 4-byte boundaries for u32 storage vector.
        let u32_count = (text_len as usize + 3) / 4;
        let mut u32_bytes = vec![0u8; u32_count * 4];
        u32_bytes[..input_bytes.len()].copy_from_slice(input_bytes);

        use wgpu::util::DeviceExt;

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Text Buffer"),
            contents: &u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let offset_bytes = bytemuck::cast_slice(line_offsets);
        let offset_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line Offset Buffer"),
            contents: offset_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let config = ShaderConfig {
            line_count,
            text_len,
        };
        let config_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_byte_size = (line_count as usize * std::mem::size_of::<FlatNode>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Nodes GPU Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Oneline Parser Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Oneline Parser Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Oneline Parse Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let total_workgroups = (line_count + 63) / 64;
            let (wg_x, wg_y) = if total_workgroups <= 65535 {
                (total_workgroups, 1)
            } else {
                (65535, (total_workgroups + 65534) / 65535)
            };
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_byte_size);
        self.queue.submit(Some(encoder.finish()));

        // Read results back from GPU.
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });

        self.device.poll(wgpu::Maintain::wait());
        receiver
            .recv()
            .map_err(|e| anyhow!("Failed to receive mapping notification: {:?}", e))?
            .map_err(|e| anyhow!("Buffer mapping failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let nodes: &[FlatNode] = bytemuck::cast_slice(&data);
        let result_nodes = nodes.to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(FlatTree::new(
            input.to_string(),
            result_nodes,
            line_offsets.to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oneline_core::NodeKind;

    #[test]
    fn test_gpu_parser_basic() {
        let parser = match GpuParser::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping GPU test (no adapter or headless): {:?}", e);
                return;
            }
        };

        let text = "|>[ Header ]\n| Body text\n||>[ Callout ]\n|| Content.\n";
        let offsets = oneline_simd::scan_line_offsets(text.as_bytes());
        let tree = parser.parse(text, &offsets).unwrap();

        assert_eq!(tree.nodes.len(), 4);
        assert_eq!(tree.nodes[0].depth, 1);
        assert_eq!(tree.nodes[0].kind(), NodeKind::Callout);
        assert_eq!(tree.nodes[1].depth, 1);
        assert_eq!(tree.nodes[1].kind(), NodeKind::Text);
        assert_eq!(tree.nodes[2].depth, 2);
        assert_eq!(tree.nodes[2].kind(), NodeKind::Callout);
        assert_eq!(tree.nodes[3].depth, 2);
        assert_eq!(tree.nodes[3].kind(), NodeKind::Text);
    }
}
