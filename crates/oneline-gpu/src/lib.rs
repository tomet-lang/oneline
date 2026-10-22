use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use oneline_core::{FlatNode, FlatTree, NodeKind};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShaderConfig {
    line_count: u32,
    text_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QueryConfig {
    target_kind: u32,
    target_depth: u32,
    max_matches: u32,
    _pad: u32,
}

/// Massively parallel GPU parser powered by wgpu Compute Shaders.
pub struct GpuParser {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    filter_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    query_bind_group_layout: wgpu::BindGroupLayout,
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
                    required_limits: adapter.limits(),
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

        let query_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Oneline Query Bind Group Layout"),
            entries: &[
                // 0: query config
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: match_count (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: matched_indices
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let filter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Oneline Filter Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &query_bind_group_layout],
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

        let filter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Oneline Filter Compute Pipeline"),
            layout: Some(&filter_pipeline_layout),
            module: &shader,
            entry_point: Some("filter_nodes"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            filter_pipeline,
            bind_group_layout,
            query_bind_group_layout,
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

    /// Queries the document entirely inside GPU VRAM without downloading the full tree.
    /// Returns matched line indices.
    pub fn query(
        &self,
        input: &str,
        line_offsets: &[u32],
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        pollster::block_on(self.query_async(input, line_offsets, kind, depth, max_matches))
    }

    pub async fn query_async(
        &self,
        input: &str,
        line_offsets: &[u32],
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        let line_count = line_offsets.len() as u32;
        if line_count == 0 {
            return Ok(Vec::new());
        }

        let input_bytes = input.as_bytes();
        let text_len = input_bytes.len() as u32;

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
            usage: wgpu::BufferUsages::STORAGE,
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

        // Query buffers setup
        let q_config = QueryConfig {
            target_kind: kind.map(|k| k as u32).unwrap_or(0),
            target_depth: depth.unwrap_or(0xFFFFFFFF),
            max_matches: max_matches as u32,
            _pad: 0,
        };
        let q_config_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query Config Buffer"),
            contents: bytemuck::bytes_of(&q_config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let initial_count = [0u32; 1];
        let match_count_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Match Count Buffer"),
            contents: bytemuck::cast_slice(&initial_count),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let indices_byte_size = (max_matches * std::mem::size_of::<u32>()) as u64;
        let matched_indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matched Indices Buffer"),
            size: indices_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let query_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Query Bind Group"),
            layout: &self.query_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: match_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matched_indices_buffer.as_entire_binding(),
                },
            ],
        });

        let staging_count_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Indices Buffer"),
            size: indices_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Oneline Query Encoder"),
        });

        let total_workgroups = (line_count + 63) / 64;
        let (wg_x, wg_y) = if total_workgroups <= 65535 {
            (total_workgroups, 1)
        } else {
            (65535, (total_workgroups + 65534) / 65535)
        };

        // Pass 1: Parse lines into GPU VRAM
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Oneline Parse Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2: Filter nodes entirely in GPU VRAM
        {
            let mut filter_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Oneline Filter Pass"),
                timestamp_writes: None,
            });
            filter_pass.set_pipeline(&self.filter_pipeline);
            filter_pass.set_bind_group(0, &bind_group, &[]);
            filter_pass.set_bind_group(1, &query_bind_group, &[]);
            filter_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&match_count_buffer, 0, &staging_count_buffer, 0, 4);
        encoder.copy_buffer_to_buffer(&matched_indices_buffer, 0, &staging_indices_buffer, 0, indices_byte_size);
        self.queue.submit(Some(encoder.finish()));

        // Download only the small query result
        let count_slice = staging_count_buffer.slice(..);
        let indices_slice = staging_indices_buffer.slice(..);

        let (sender1, receiver1) = std::sync::mpsc::channel();
        count_slice.map_async(wgpu::MapMode::Read, move |r| sender1.send(r).unwrap());

        let (sender2, receiver2) = std::sync::mpsc::channel();
        indices_slice.map_async(wgpu::MapMode::Read, move |r| sender2.send(r).unwrap());

        self.device.poll(wgpu::Maintain::wait());
        receiver1.recv()?.map_err(|e| anyhow!("Failed count map: {:?}", e))?;
        receiver2.recv()?.map_err(|e| anyhow!("Failed indices map: {:?}", e))?;

        let count_data = count_slice.get_mapped_range();
        let total_matches = bytemuck::cast_slice::<u8, u32>(&count_data)[0] as usize;
        drop(count_data);
        staging_count_buffer.unmap();

        let valid_count = total_matches.min(max_matches);
        let indices_data = indices_slice.get_mapped_range();
        let indices: &[u32] = bytemuck::cast_slice(&indices_data);
        let result = indices[..valid_count].to_vec();
        drop(indices_data);
        staging_indices_buffer.unmap();

        Ok(result)
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

    #[test]
    fn test_gpu_query_resident() {
        let parser = match GpuParser::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping GPU test (no adapter or headless): {:?}", e);
                return;
            }
        };

        let text = "| @card(title:\"A\")\n|| Content\n| @card(title:\"B\")\n|| More\n|=[ Header ]\n";
        let offsets = oneline_simd::scan_line_offsets(text.as_bytes());

        // Find only Element nodes
        let matches = parser
            .query(text, &offsets, Some(NodeKind::Element), None, 100)
            .unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches, vec![0, 2]);

        // Find depth 2 nodes
        let depth2 = parser
            .query(text, &offsets, None, Some(2), 100)
            .unwrap();

        assert_eq!(depth2.len(), 2);
        assert_eq!(depth2, vec![1, 3]);
    }
}
