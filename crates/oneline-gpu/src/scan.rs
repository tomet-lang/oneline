use anyhow::Result;

use crate::ScanConfig;
use crate::parser::GpuParser;
use crate::timing::{StageTimer, mark};
use crate::util::{
    bind, check_gpu_len, compute_pass, dispatch_dims, map_all, staging_buffer, take_mapped,
};

/// Bytes of text handled by one scan workgroup (256 threads x 16 words x 4 bytes).
const BLOCK_BYTES: u32 = 16384;

impl GpuParser {
    /// Runs the 3-pass prefix scan inside VRAM.
    ///
    /// Returns the buffer of line start offsets and the line count. The only CPU
    /// round trip is a 4-byte read of the line count between pass 2 and pass 3,
    /// needed to size the output buffer.
    pub(crate) fn scan_to_buffer(
        &self,
        input_buffer: &wgpu::Buffer,
        text_len: u32,
        timer: Option<&StageTimer>,
    ) -> Result<(wgpu::Buffer, u32)> {
        let num_blocks = text_len.div_ceil(BLOCK_BYTES).max(1);

        let scan_cfg_buf = self.uniform(
            "Scan Config Buffer",
            &ScanConfig {
                num_blocks,
                text_len,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let storage = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let block_counts = storage(
            "Block Counts Buffer",
            u64::from(num_blocks) * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let block_offsets = storage(
            "Block Offsets Buffer",
            u64::from(num_blocks) * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let scan_meta = storage(
            "Scan Meta Buffer",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging_meta = staging_buffer(&self.device, "Staging Meta Buffer", 16);

        let scan_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Bind Group 0"),
            layout: &self.scan_layout,
            entries: &[
                bind(0, input_buffer),
                bind(1, &scan_cfg_buf),
                bind(2, &block_counts),
                bind(3, &block_offsets),
                bind(4, &scan_meta),
            ],
        });
        mark(timer, "scan: create buffers + bind group");

        let (wg_x, wg_y) = dispatch_dims(num_blocks);

        // Pass 1 (per-block newline counts) and pass 2 (scan of the block counts).
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = compute_pass(&mut encoder, "scan pass 1 (count newlines)", timer);
            pass.set_pipeline(&self.scan_count_pipeline);
            pass.set_bind_group(0, &scan_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut pass = compute_pass(&mut encoder, "scan pass 2 (scan block counts)", timer);
            pass.set_pipeline(&self.scan_blocks_pipeline);
            pass.set_bind_group(0, &scan_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&scan_meta, 0, &staging_meta, 0, 16);
        self.queue.submit(Some(encoder.finish()));
        mark(timer, "scan: encode + submit passes 1-2");

        map_all(&self.device, &[&staging_meta])?;
        let total_lines = take_mapped::<u32>(&staging_meta, 1)[0];
        mark(timer, "scan: wait for passes 1-2 + read line count");

        self.check_fits(u64::from(total_lines) * 4, "the line offsets")?;
        let line_offsets = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line Offsets Buffer"),
            size: (u64::from(total_lines) * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scatter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter Bind Group 1"),
            layout: &self.scatter_layout,
            entries: &[bind(0, &line_offsets)],
        });

        // Pass 3: scatter offsets into their final slots.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = compute_pass(&mut encoder, "scan pass 3 (scatter offsets)", timer);
            pass.set_pipeline(&self.scan_scatter_pipeline);
            pass.set_bind_group(0, &scan_bind_group, &[]);
            pass.set_bind_group(1, &scatter_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        mark(timer, "scan: allocate offsets + submit pass 3");

        Ok((line_offsets, total_lines))
    }

    /// Scans line breaks entirely on the GPU and returns starting byte offsets.
    pub fn scan_line_offsets(&self, input: &str) -> Result<Vec<u32>> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let input_buffer = self.upload_text(input, None)?;
        let (line_offsets, total_lines) =
            self.scan_to_buffer(&input_buffer, input.len() as u32, None)?;
        self.download_u32s(&line_offsets, total_lines as usize)
    }

    pub(crate) fn download_u32s(&self, src: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
        let bytes = count as u64 * 4;
        let staging = staging_buffer(&self.device, "Staging Download Buffer", bytes);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        map_all(&self.device, &[&staging])?;
        Ok(take_mapped::<u32>(&staging, count))
    }
}
