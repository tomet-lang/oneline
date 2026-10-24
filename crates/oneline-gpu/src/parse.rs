use anyhow::Result;
use oneline_core::{FlatNode, FlatTree};

use crate::ShaderConfig;
use crate::parser::GpuParser;
use crate::timing::{StageTimer, mark};
use crate::util::{
    bind, check_gpu_len, compute_pass, dispatch_dims, map_all, staging_buffer, take_mapped,
};

/// Nodes in VRAM plus the bind group that exposes them to the kernels.
pub(crate) struct ParsedNodes {
    pub nodes: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl GpuParser {
    /// Stage 2: classify every line on the GPU. The result stays in VRAM.
    pub(crate) fn parse_to_buffer(
        &self,
        input_buffer: &wgpu::Buffer,
        line_offsets: &wgpu::Buffer,
        line_count: u32,
        text_len: u32,
        timer: Option<&StageTimer>,
    ) -> Result<ParsedNodes> {
        let node_bytes = u64::from(line_count) * std::mem::size_of::<FlatNode>() as u64;
        self.check_fits(node_bytes, "the parsed nodes")?;

        let config = self.uniform(
            "Config Buffer",
            &ShaderConfig {
                line_count,
                text_len,
            },
        );
        let nodes = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Nodes GPU Buffer"),
            size: node_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Oneline Parser Bind Group"),
            layout: &self.parse_layout,
            entries: &[
                bind(0, input_buffer),
                bind(1, line_offsets),
                bind(2, &config),
                bind(3, &nodes),
            ],
        });
        mark(timer, "parse: create buffers + bind group");

        let (wg_x, wg_y) = dispatch_dims(line_count.div_ceil(64));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = compute_pass(&mut encoder, "parse lines", timer);
            pass.set_pipeline(&self.parse_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        mark(timer, "parse: encode + submit");

        Ok(ParsedNodes { nodes, bind_group })
    }

    pub(crate) fn download_nodes(
        &self,
        nodes: &wgpu::Buffer,
        line_count: usize,
    ) -> Result<Vec<FlatNode>> {
        let bytes = (line_count * std::mem::size_of::<FlatNode>()) as u64;
        let staging = staging_buffer(&self.device, "Staging Nodes Buffer", bytes);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(nodes, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        map_all(&self.device, &[&staging])?;
        Ok(take_mapped::<FlatNode>(&staging, line_count))
    }

    /// Parses raw text completely on the GPU without precomputing line offsets on the CPU.
    ///
    /// The returned tree borrows `input`; the text is not copied.
    pub fn parse_raw<'a>(&self, input: &'a str) -> Result<FlatTree<'a>> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(FlatTree::new(input, Vec::new(), Vec::new()));
        }
        let text_len = input.len() as u32;

        let input_buffer = self.upload_text(input, None)?;
        let (line_offsets, line_count) = self.scan_to_buffer(&input_buffer, text_len, None)?;
        let parsed =
            self.parse_to_buffer(&input_buffer, &line_offsets, line_count, text_len, None)?;

        let nodes = self.download_nodes(&parsed.nodes, line_count as usize)?;
        let offsets = self.download_u32s(&line_offsets, line_count as usize)?;
        Ok(FlatTree::new(input, nodes, offsets))
    }

    /// Parses text in parallel on the GPU with precomputed line offsets.
    pub fn parse<'a>(&self, input: &'a str, line_offsets: &[u32]) -> Result<FlatTree<'a>> {
        check_gpu_len(input.len())?;
        let line_count = line_offsets.len();
        if line_count == 0 {
            return Ok(FlatTree::new(input, Vec::new(), Vec::new()));
        }
        let text_len = input.len() as u32;

        let input_buffer = self.upload_text(input, None)?;
        let offsets_buffer = self.upload_u32s("Line Offset Buffer", line_offsets)?;
        let parsed = self.parse_to_buffer(
            &input_buffer,
            &offsets_buffer,
            line_count as u32,
            text_len,
            None,
        )?;

        let nodes = self.download_nodes(&parsed.nodes, line_count)?;
        Ok(FlatTree::new(input, nodes, line_offsets.to_vec()))
    }
}
