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
struct ScanConfig {
    num_blocks: u32,
    text_len: u32,
    _pad1: u32,
    _pad2: u32,
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

    // Stage 1: GPU Parallel Prefix Scan pipelines
    scan_pipeline_count: wgpu::ComputePipeline,
    scan_pipeline_scan: wgpu::ComputePipeline,
    scan_pipeline_scatter: wgpu::ComputePipeline,
    scan_bind_group_layout: wgpu::BindGroupLayout,
    scatter_bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuParser {
    /// Initializes GPU device and compute pipelines.
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

        // 1. Stage 2 Parser & Filter Shader Modules
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

        // 2. Stage 1 Parallel Prefix Scan Shaders & Pipelines
        let scan_shader_src = include_str!("scan.wgsl");
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Oneline Scan Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(scan_shader_src.into()),
        });

        let scan_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Oneline Scan Bind Group Layout 0"),
            entries: &[
                // 0: scan_input
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
                // 1: scan_cfg
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: block_counts
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
                // 3: block_offsets
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
                // 4: scan_meta
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let scatter_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Oneline Scatter Bind Group Layout 1"),
            entries: &[
                // 0: line_offsets
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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

        let scan_count_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scan Count Pipeline Layout"),
            bind_group_layouts: &[&scan_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scan_scan_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scan Scan Pipeline Layout"),
            bind_group_layouts: &[&scan_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scan_scatter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scan Scatter Pipeline Layout"),
            bind_group_layouts: &[&scan_bind_group_layout, &scatter_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scan_pipeline_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scan Count Pipeline"),
            layout: Some(&scan_count_pipeline_layout),
            module: &scan_shader,
            entry_point: Some("count_block_newlines"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scan_pipeline_scan = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scan Scan Pipeline"),
            layout: Some(&scan_scan_pipeline_layout),
            module: &scan_shader,
            entry_point: Some("scan_block_counts"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scan_pipeline_scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scan Scatter Pipeline"),
            layout: Some(&scan_scatter_pipeline_layout),
            module: &scan_shader,
            entry_point: Some("scatter_line_offsets"),
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
            scan_pipeline_count,
            scan_pipeline_scan,
            scan_pipeline_scatter,
            scan_bind_group_layout,
            scatter_bind_group_layout,
        })
    }

    /// Internal GPU prefix scan executing Pass 1, Pass 2, and Pass 3 completely inside VRAM.
    /// Returns GPU buffer containing strictly ordered line offsets and total line count.
    async fn scan_line_offsets_gpu_buffer(
        &self,
        input_buffer: &wgpu::Buffer,
        text_len: u32,
    ) -> Result<(wgpu::Buffer, u32)> {
        use wgpu::util::DeviceExt;

        let num_blocks = (text_len + 16383) / 16384;
        let num_blocks = num_blocks.max(1);

        let scan_cfg = ScanConfig {
            num_blocks,
            text_len,
            _pad1: 0,
            _pad2: 0,
        };
        let scan_cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scan Config Buffer"),
            contents: bytemuck::bytes_of(&scan_cfg),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let block_counts_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Counts Buffer"),
            size: (num_blocks as u64 * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let block_offsets_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Offsets Buffer"),
            size: (num_blocks as u64 * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let scan_meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Meta Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Meta Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scan_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Bind Group 0"),
            layout: &self.scan_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scan_cfg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: block_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scan_meta_buf.as_entire_binding(),
                },
            ],
        });

        // Step A: Run Pass 1 (count block newlines) and Pass 2 (scan block counts)
        let mut encoder1 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Scan Pass 1 & 2 Encoder"),
        });

        {
            let mut pass1 = encoder1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scan Pass 1 (Count Newlines)"),
                timestamp_writes: None,
            });
            pass1.set_pipeline(&self.scan_pipeline_count);
            pass1.set_bind_group(0, &scan_bind_group, &[]);
            let (wg_x, wg_y) = if num_blocks <= 65535 {
                (num_blocks, 1)
            } else {
                (65535, (num_blocks + 65534) / 65535)
            };
            pass1.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass2 = encoder1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scan Pass 2 (Scan Block Counts)"),
                timestamp_writes: None,
            });
            pass2.set_pipeline(&self.scan_pipeline_scan);
            pass2.set_bind_group(0, &scan_bind_group, &[]);
            pass2.dispatch_workgroups(1, 1, 1);
        }

        encoder1.copy_buffer_to_buffer(&scan_meta_buf, 0, &staging_meta_buf, 0, 16);
        self.queue.submit(Some(encoder1.finish()));

        // Read total lines from meta buffer
        let meta_slice = staging_meta_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        meta_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::wait());
        rx.recv()?.map_err(|e| anyhow!("Failed to map meta buffer: {:?}", e))?;
        let meta_data = meta_slice.get_mapped_range();
        let total_lines = bytemuck::cast_slice::<u8, u32>(&meta_data)[0];
        drop(meta_data);
        staging_meta_buf.unmap();

        let line_offsets_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line Offsets Buffer"),
            size: (total_lines as u64 * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Step B: Run Pass 3 (scatter line offsets into ordered array)
        let scatter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter Bind Group 1"),
            layout: &self.scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: line_offsets_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Scan Pass 3 Encoder"),
        });

        {
            let mut pass3 = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scan Pass 3 (Scatter Line Offsets)"),
                timestamp_writes: None,
            });
            pass3.set_pipeline(&self.scan_pipeline_scatter);
            pass3.set_bind_group(0, &scan_bind_group, &[]);
            pass3.set_bind_group(1, &scatter_bind_group, &[]);
            let (wg_x, wg_y) = if num_blocks <= 65535 {
                (num_blocks, 1)
            } else {
                (65535, (num_blocks + 65534) / 65535)
            };
            pass3.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.queue.submit(Some(encoder2.finish()));

        Ok((line_offsets_buf, total_lines))
    }

    /// Scans line breaks entirely on GPU using 3-pass parallel prefix scan and returns starting byte offsets.
    pub fn scan_line_offsets(&self, input: &str) -> Result<Vec<u32>> {
        pollster::block_on(self.scan_line_offsets_async(input))
    }

    pub async fn scan_line_offsets_async(&self, input: &str) -> Result<Vec<u32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let input_bytes = input.as_bytes();
        let text_len = input_bytes.len() as u32;

        let u32_count = (text_len as usize + 3) / 4;
        let mut u32_bytes = vec![0u8; (u32_count * 4).max(4)];
        u32_bytes[..input_bytes.len()].copy_from_slice(input_bytes);

        use wgpu::util::DeviceExt;

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Text Buffer"),
            contents: &u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let (line_offsets_buf, total_lines) = self.scan_line_offsets_gpu_buffer(&input_buffer, text_len).await?;

        if total_lines == 0 {
            return Ok(Vec::new());
        }

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Line Offsets Buffer"),
            size: total_lines as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Line Offsets Encoder"),
        });
        encoder.copy_buffer_to_buffer(&line_offsets_buf, 0, &staging_buf, 0, total_lines as u64 * 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::wait());
        rx.recv()?.map_err(|e| anyhow!("Failed to map line offsets buffer: {:?}", e))?;

        let data = slice.get_mapped_range();
        let offsets: &[u32] = bytemuck::cast_slice(&data);
        let result = offsets.to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }

    /// Parses raw text completely on GPU without precomputing line offsets on CPU.
    pub fn parse_raw(&self, input: &str) -> Result<FlatTree> {
        pollster::block_on(self.parse_raw_async(input))
    }

    pub async fn parse_raw_async(&self, input: &str) -> Result<FlatTree> {
        if input.is_empty() {
            return Ok(FlatTree::new(input.to_string(), Vec::new(), Vec::new()));
        }

        let input_bytes = input.as_bytes();
        let text_len = input_bytes.len() as u32;

        let u32_count = (text_len as usize + 3) / 4;
        let mut u32_bytes = vec![0u8; (u32_count * 4).max(4)];
        u32_bytes[..input_bytes.len()].copy_from_slice(input_bytes);

        use wgpu::util::DeviceExt;

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Text Buffer"),
            contents: &u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Step A: GPU Parallel Prefix Scan to compute line offsets in VRAM
        let (line_offsets_buf, line_count) = self.scan_line_offsets_gpu_buffer(&input_buffer, text_len).await?;

        if line_count == 0 {
            return Ok(FlatTree::new(input.to_string(), Vec::new(), Vec::new()));
        }

        // Step B: Stage 2 Parse Lines on GPU
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

        let offsets_byte_size = line_count as u64 * 4;
        let staging_offsets_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Offsets Buffer"),
            size: offsets_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_nodes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Nodes Buffer"),
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
                    resource: line_offsets_buf.as_entire_binding(),
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
            label: Some("Oneline Parse Raw Encoder"),
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

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_nodes_buffer, 0, output_byte_size);
        encoder.copy_buffer_to_buffer(&line_offsets_buf, 0, &staging_offsets_buffer, 0, offsets_byte_size);
        self.queue.submit(Some(encoder.finish()));

        // Download results from GPU
        let nodes_slice = staging_nodes_buffer.slice(..);
        let offsets_slice = staging_offsets_buffer.slice(..);

        let (tx1, rx1) = std::sync::mpsc::channel();
        nodes_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx1.send(res);
        });

        let (tx2, rx2) = std::sync::mpsc::channel();
        offsets_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx2.send(res);
        });

        self.device.poll(wgpu::Maintain::wait());
        rx1.recv()?.map_err(|e| anyhow!("Failed nodes map: {:?}", e))?;
        rx2.recv()?.map_err(|e| anyhow!("Failed offsets map: {:?}", e))?;

        let nodes_data = nodes_slice.get_mapped_range();
        let nodes: &[FlatNode] = bytemuck::cast_slice(&nodes_data);
        let result_nodes = nodes.to_vec();
        drop(nodes_data);
        staging_nodes_buffer.unmap();

        let offsets_data = offsets_slice.get_mapped_range();
        let offsets: &[u32] = bytemuck::cast_slice(&offsets_data);
        let result_offsets = offsets.to_vec();
        drop(offsets_data);
        staging_offsets_buffer.unmap();

        Ok(FlatTree::new(
            input.to_string(),
            result_nodes,
            result_offsets,
        ))
    }

    /// Queries the document entirely inside GPU VRAM directly from raw text.
    /// Detects lines, parses AST nodes, filters matches, and only downloads match indices.
    pub fn query_raw(
        &self,
        input: &str,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        pollster::block_on(self.query_raw_async(input, kind, depth, max_matches))
    }

    pub async fn query_raw_async(
        &self,
        input: &str,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let input_bytes = input.as_bytes();
        let text_len = input_bytes.len() as u32;

        let u32_count = (text_len as usize + 3) / 4;
        let mut u32_bytes = vec![0u8; (u32_count * 4).max(4)];
        u32_bytes[..input_bytes.len()].copy_from_slice(input_bytes);

        use wgpu::util::DeviceExt;

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Text Buffer"),
            contents: &u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let (line_offsets_buf, line_count) = self.scan_line_offsets_gpu_buffer(&input_buffer, text_len).await?;

        if line_count == 0 {
            return Ok(Vec::new());
        }

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
                    resource: line_offsets_buf.as_entire_binding(),
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
            label: Some("Oneline Query Raw Encoder"),
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

    /// Parses text in parallel using the GPU with precomputed line offsets.
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
        let mut u32_bytes = vec![0u8; (u32_count * 4).max(4)];
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
        let mut u32_bytes = vec![0u8; (u32_count * 4).max(4)];
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
    fn test_gpu_scan_line_offsets_various() {
        let parser = match GpuParser::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping GPU test: {:?}", e);
                return;
            }
        };

        let cases = [
            "hello",
            "hello\n",
            "line1\nline2\nline3\n",
            "line1\r\nline2\r\nline3\r\n",
            "\n\n\n",
            "|>[ Header ]\n| Paragraph\n|| Content\n",
        ];

        for &case in &cases {
            let cpu_offsets = oneline_simd::scan_line_offsets(case.as_bytes());
            let gpu_offsets = parser.scan_line_offsets(case).unwrap();
            assert_eq!(gpu_offsets, cpu_offsets, "Failed matching for case: {:?}", case);
        }
    }

    #[test]
    fn test_gpu_scan_line_offsets_large() {
        let parser = match GpuParser::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping GPU test: {:?}", e);
                return;
            }
        };

        let sample = "|>[ Header ]\n| Body paragraph\n||>[ Callout ]\n|| Content line\n";
        let mut large_text = String::with_capacity(sample.len() * 5000);
        for _ in 0..5000 {
            large_text.push_str(sample);
        }

        let cpu_offsets = oneline_simd::scan_line_offsets(large_text.as_bytes());
        let gpu_offsets = parser.scan_line_offsets(&large_text).unwrap();

        assert_eq!(gpu_offsets.len(), cpu_offsets.len());
        assert_eq!(gpu_offsets, cpu_offsets);
    }

    #[test]
    fn test_gpu_parse_raw() {
        let parser = match GpuParser::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping GPU test: {:?}", e);
                return;
            }
        };

        let text = "|>[ Header ]\n| Body text\n||>[ Callout ]\n|| Content.\n";
        let tree = parser.parse_raw(text).unwrap();

        assert_eq!(tree.nodes.len(), 4);
        assert_eq!(tree.nodes[0].depth, 1);
        assert_eq!(tree.nodes[0].kind(), NodeKind::Callout);
        assert_eq!(tree.nodes[1].depth, 1);
        assert_eq!(tree.nodes[1].kind(), NodeKind::Text);
        assert_eq!(tree.nodes[2].depth, 2);
        assert_eq!(tree.nodes[2].kind(), NodeKind::Callout);
        assert_eq!(tree.nodes[3].depth, 2);
        assert_eq!(tree.nodes[3].kind(), NodeKind::Text);
        assert_eq!(tree.line_offsets, vec![0, 13, 25, 40]);
    }

    #[test]
    fn test_gpu_query_raw() {
        let parser = match GpuParser::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping GPU test: {:?}", e);
                return;
            }
        };

        let text = "| @card(title:\"A\")\n|| Content\n| @card(title:\"B\")\n|| More\n|=[ Header ]\n";
        let matches = parser
            .query_raw(text, Some(NodeKind::Element), None, 100)
            .unwrap();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches, vec![0, 2]);

        let depth2 = parser
            .query_raw(text, None, Some(2), 100)
            .unwrap();

        assert_eq!(depth2.len(), 2);
        assert_eq!(depth2, vec![1, 3]);
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
