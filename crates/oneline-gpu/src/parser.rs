use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;

use crate::timing::{StageTimer, mark};
use crate::util::{check_gpu_len, storage_entry, uniform_entry};

/// Massively parallel GPU parser powered by wgpu Compute Shaders.
pub struct GpuParser {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) limits: wgpu::Limits,
    adapter_name: String,
    software: bool,
    pub(crate) timestamps: bool,
    /// Reused buffers for queries, so a query does not allocate. Guarded because
    /// a query reads results back through the shared staging buffer.
    pub(crate) query_scratch: std::sync::Mutex<Option<crate::query::QueryScratch>>,
    /// Persistent staging buffers for pipelined uploads (see `staging.rs`).
    pub(crate) staging: std::sync::Mutex<Option<crate::staging::StagingRing>>,
    /// Tests only: caps the compact strategy's candidate buffer so its overflow path can be reached with a
    /// small text (with the real capacity, a full workgroup tile is always hit first below 512 MB). 0 = off.
    #[cfg(test)]
    pub(crate) candidate_capacity_cap: std::sync::atomic::AtomicUsize,

    // Stage 2/3 pipelines (`shader.wgsl`).
    pub(crate) parse_pipeline: wgpu::ComputePipeline,
    /// Query over resident nodes.
    pub(crate) filter_pipeline: wgpu::ComputePipeline,
    /// Parse + query in one kernel, no nodes buffer.
    pub(crate) fused_pipeline: wgpu::ComputePipeline,
    /// Group 0 with all four bindings (input, offsets, config, nodes).
    pub(crate) parse_layout: wgpu::BindGroupLayout,
    /// Group 0 without the nodes binding, for the fused kernel.
    pub(crate) fused_layout: wgpu::BindGroupLayout,
    pub(crate) query_layout: wgpu::BindGroupLayout,
    /// Substring search (`search.wgsl`): group 0 is just the text; group 1 reuses `query_layout`.
    pub(crate) search_pipeline: wgpu::ComputePipeline,
    pub(crate) text_layout: wgpu::BindGroupLayout,
    /// Dictionary search (`many.wgsl`); `None` if the adapter lacks the shared memory it needs.
    pub(crate) many: crate::many::ManyKernel,
    pub(crate) many_run_layout: wgpu::BindGroupLayout,
    pub(crate) many_set_layout: wgpu::BindGroupLayout,
    pub(crate) many_compact_layout: wgpu::BindGroupLayout,
    /// Tree construction (`tree.wgsl`).
    pub(crate) tree: crate::tree::TreePipelines,
    pub(crate) tree_layout0: wgpu::BindGroupLayout,
    pub(crate) tree_layout1: wgpu::BindGroupLayout,

    // Stage 1 prefix-scan pipelines (`scan.wgsl`).
    pub(crate) scan_count_pipeline: wgpu::ComputePipeline,
    pub(crate) scan_blocks_pipeline: wgpu::ComputePipeline,
    pub(crate) scan_scatter_pipeline: wgpu::ComputePipeline,
    pub(crate) scan_layout: wgpu::BindGroupLayout,
    pub(crate) scatter_layout: wgpu::BindGroupLayout,
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn bind_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
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
            .or_else(|| instance.enumerate_adapters(wgpu::Backends::all()).into_iter().next())
            .ok_or_else(|| anyhow!("Failed to find suitable GPU adapter"))?;

        eprintln!(
            "==> [wgpu] Selected GPU: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let info = adapter.get_info();
        let (adapter_name, software) = (info.name.clone(), info.device_type == wgpu::DeviceType::Cpu);
        let limits = adapter.limits();
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Oneline GPU Parser Device"),
                    required_features: if timestamps { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
                    required_limits: limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        // Stage 2/3: parse + query.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Oneline Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let parse_layout = bind_layout(
            &device,
            "Parse Bind Group Layout",
            &[
                storage_entry(0, true),  // input_text
                storage_entry(1, true),  // line_offsets
                uniform_entry(2),        // config
                storage_entry(3, false), // nodes
            ],
        );
        let fused_layout = bind_layout(
            &device,
            "Fused Query Bind Group Layout",
            &[storage_entry(0, true), storage_entry(1, true), uniform_entry(2)],
        );
        let query_layout = bind_layout(
            &device,
            "Query Bind Group Layout",
            &[
                uniform_entry(0),        // query config
                storage_entry(1, false), // match_count (atomic)
                storage_entry(2, false), // matched_indices
            ],
        );

        let parse_pipeline = pipeline(&device, "Parse", &shader, "parse_lines", &[&parse_layout]);
        let filter_pipeline = pipeline(
            &device,
            "Filter",
            &shader,
            "filter_nodes",
            &[&parse_layout, &query_layout],
        );
        let fused_pipeline = pipeline(
            &device,
            "Fused Filter",
            &shader,
            "filter_fused",
            &[&fused_layout, &query_layout],
        );


        // Substring search.
        let search_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Oneline Search Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("search.wgsl").into()),
        });
        let text_layout = bind_layout(&device, "Search Text Bind Group Layout", &[storage_entry(0, true)]);
        let search_pipeline = pipeline(&device, "Search", &search_shader, "search", &[&text_layout, &query_layout]);



        // Dictionary search. The kernels are specialized per pattern set (filter and histogram
        // sizes are pipeline constants), so only the shader modules and layouts are built here.
        let many_run_layout = bind_layout(
            &device,
            "Many Bind Group Layout 1",
            &[uniform_entry(0), storage_entry(1, false), storage_entry(2, false), storage_entry(3, false)],
        );
        let many_set_layout = bind_layout(
            &device,
            "Many Bind Group Layout 2",
            &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, true), storage_entry(3, true)],
        );
        let many_compact_layout = bind_layout(
            &device,
            "Many Bind Group Layout 3",
            &[storage_entry(0, false), storage_entry(1, false), storage_entry(2, false), storage_entry(3, false)],
        );
        let many_pipeline_layout = |label: &str, groups: &[&wgpu::BindGroupLayout]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: groups,
                push_constant_ranges: &[],
            })
        };
        let many = crate::many::ManyKernel {
            shader_fused: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Oneline Many-Pattern Fused Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(include_str!("many_common.wgsl"), include_str!("many_fused.wgsl")).into(),
                ),
            }),
            shader_compact: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Oneline Many-Pattern Compact Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(include_str!("many_common.wgsl"), include_str!("many_compact.wgsl")).into(),
                ),
            }),
            layout_fused: many_pipeline_layout("Many Fused Pipeline Layout", &[&text_layout, &many_run_layout, &many_set_layout]),
            layout_compact: many_pipeline_layout(
                "Many Compact Pipeline Layout",
                &[&text_layout, &many_run_layout, &many_set_layout, &many_compact_layout],
            ),
            pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
        };

        // Tree construction.
        let tree_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Oneline Tree Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tree.wgsl").into()),
        });
        let tree_layout0 = bind_layout(
            &device,
            "Tree Bind Group Layout 0",
            &[
                storage_entry(0, true),  // nodes
                uniform_entry(1),        // config
                storage_entry(2, false), // run summaries
                storage_entry(3, false), // run start states
                storage_entry(4, false), // status
                storage_entry(5, false), // keys
            ],
        );
        let tree_layout1 = bind_layout(&device, "Tree Bind Group Layout 1", &[storage_entry(0, false)]);
        let tree_pipeline = |label: &str, entry: &str| pipeline(&device, label, &tree_shader, entry, &[&tree_layout0, &tree_layout1]);
        let tree = crate::tree::TreePipelines {
            extract_keys: tree_pipeline("Tree Extract Keys", "extract_keys"),
            summarize_forward: tree_pipeline("Tree Summarize Forward", "summarize_forward"),
            scan_forward: tree_pipeline("Tree Scan Forward", "scan_forward"),
            assign_parent: tree_pipeline("Tree Assign Parent", "assign_parent"),
            summarize_backward: tree_pipeline("Tree Summarize Backward", "summarize_backward"),
            scan_backward: tree_pipeline("Tree Scan Backward", "scan_backward"),
            assign_end: tree_pipeline("Tree Assign End", "assign_end"),
        };

        // Stage 1: line-offset prefix scan.
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Oneline Scan Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scan.wgsl").into()),
        });
        let scan_layout = bind_layout(
            &device,
            "Scan Bind Group Layout 0",
            &[
                storage_entry(0, true),  // scan_input
                uniform_entry(1),        // scan_cfg
                storage_entry(2, false), // block_counts
                storage_entry(3, false), // block_offsets
                storage_entry(4, false), // scan_meta
            ],
        );
        let scatter_layout = bind_layout(&device, "Scatter Bind Group Layout 1", &[storage_entry(0, false)]);

        let scan_count_pipeline =
            pipeline(&device, "Scan Count", &scan_shader, "count_block_newlines", &[&scan_layout]);
        let scan_blocks_pipeline =
            pipeline(&device, "Scan Blocks", &scan_shader, "scan_block_counts", &[&scan_layout]);
        let scan_scatter_pipeline = pipeline(
            &device,
            "Scan Scatter",
            &scan_shader,
            "scatter_line_offsets",
            &[&scan_layout, &scatter_layout],
        );

        Ok(Self {
            device,
            queue,
            limits,
            adapter_name,
            software,
            timestamps,
            query_scratch: std::sync::Mutex::new(None),
            staging: std::sync::Mutex::new(None),
            #[cfg(test)]
            candidate_capacity_cap: std::sync::atomic::AtomicUsize::new(0),
            parse_pipeline,
            filter_pipeline,
            fused_pipeline,
            parse_layout,
            fused_layout,
            query_layout,
            search_pipeline,
            text_layout,
            many,
            many_run_layout,
            many_set_layout,
            many_compact_layout,
            tree,
            tree_layout0,
            tree_layout1,
            scan_count_pipeline,
            scan_blocks_pipeline,
            scan_scatter_pipeline,
            scan_layout,
            scatter_layout,
        })
    }

    /// Fails if a single buffer of `bytes` bytes exceeds what this device can bind.
    pub(crate) fn check_fits(&self, bytes: u64, what: &str) -> Result<()> {
        let max = self
            .limits
            .max_buffer_size
            .min(u64::from(self.limits.max_storage_buffer_binding_size));
        if bytes > max {
            return Err(anyhow!(
                "{what} needs a {bytes}-byte buffer but this device allows at most {max} bytes"
            ));
        }
        Ok(())
    }

    /// Uploads the text as a `u32`-packed storage buffer, streamed through the staging ring.
    pub(crate) fn upload_text(&self, input: &str, timer: Option<&StageTimer>) -> Result<wgpu::Buffer> {
        let len = input.len();
        check_gpu_len(len)?;
        let padded = ((len + 3) & !3).max(4);
        self.check_fits(padded as u64, "the input text")?;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Text Buffer"),
            size: padded as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        mark(timer, "upload_text: create device buffer");
        // The text, then zeros up to the next word boundary (the shaders read whole words).
        self.stream_to_buffer(&buffer, 0, input.as_bytes(), padded - len)?;
        mark(timer, "upload_text: stream through the staging ring");
        Ok(buffer)
    }

    pub(crate) fn upload_u32s(&self, label: &str, data: &[u32]) -> Result<wgpu::Buffer> {
        self.check_fits(std::mem::size_of_val(data) as u64, label)?;
        Ok(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    }

    pub(crate) fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
}

impl GpuParser {
    /// Name of the selected adapter.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// True for CPU-emulated adapters (e.g. llvmpipe). Their "VRAM" is system
    /// memory, so timings say nothing about a real GPU.
    pub fn is_software(&self) -> bool {
        self.software
    }
}
