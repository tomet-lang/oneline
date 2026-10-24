//! Pipelined host -> GPU upload through a ring of persistent staging buffers.
//!
//! Measured on the development machine (RTX 3060, PCIe Gen4 x16), moving 256 MB:
//!
//! | approach | rate |
//! |---|---|
//! | fresh `mapped_at_creation` buffer each time (what this crate used to do) | 2.8 GB/s |
//! | `queue.write_buffer` | 3.0 GB/s |
//! | persistent staging buffer: memcpy 20 GB/s, then DMA 15.8 GB/s, one after the other | 8.4 GB/s |
//! | ring of persistent buffers, copy of chunk i+1 overlapping the DMA of chunk i | ~10.5 GB/s |
//!
//! Most of the old cost was allocating (and page-faulting) fresh host memory for every
//! upload: 54 of 94 ms. The ring is allocated once and reused. The remaining gap to the
//! PCIe rate is DRAM traffic: each byte is read from the source, written to staging and
//! read again by the DMA engine, which saturates memory bandwidth before the link.

use anyhow::{Result, anyhow};

use crate::parser::GpuParser;

/// Bytes per staging buffer. 16 MiB was within noise of the best size (see above) and
/// keeps the ring (and its one-time page faults) small.
const CHUNK: usize = 16 << 20;
/// Buffers in the ring: how many chunks can be in flight.
const RING: usize = 4;

pub(crate) struct StagingRing {
    buffers: Vec<wgpu::Buffer>,
    /// The submission that last read buffer `i`, if it has not been waited for yet. A
    /// buffer with `None` is mapped and ready to be written.
    in_flight: Vec<Option<wgpu::SubmissionIndex>>,
}

impl StagingRing {
    fn new(device: &wgpu::Device) -> Self {
        let buffers = (0..RING)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Upload Staging Ring Buffer"),
                    size: CHUNK as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                })
            })
            .collect();
        Self {
            buffers,
            in_flight: vec![None; RING],
        }
    }
}

impl GpuParser {
    /// Copies `data` into `dst` at `dst_offset`, followed by `zero_tail` zero bytes, through the
    /// staging ring. The copies are submitted but not waited for: later submissions are ordered
    /// after them, and the ring waits for each buffer before reusing it.
    ///
    /// `dst_offset`, `data.len() + zero_tail` must be multiples of 4 (buffer copy alignment).
    pub(crate) fn stream_to_buffer(
        &self,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        data: &[u8],
        zero_tail: usize,
    ) -> Result<()> {
        debug_assert_eq!(dst_offset % 4, 0);
        let total = data.len() + zero_tail;
        debug_assert_eq!(total % 4, 0);

        let mut guard = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        let ring = guard.get_or_insert_with(|| StagingRing::new(&self.device));
        // A failure part-way leaves buffers mapped or half-mapped; start from scratch next time.
        let result = self.stream_chunks(ring, dst, dst_offset, data, total);
        if result.is_err() {
            *guard = None;
        }
        result
    }

    fn stream_chunks(
        &self,
        ring: &mut StagingRing,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        data: &[u8],
        total: usize,
    ) -> Result<()> {
        let mut done = 0usize;
        let mut slot = 0usize;
        while done < total {
            let n = CHUNK.min(total - done);
            let buffer = &ring.buffers[slot];

            if let Some(submission) = ring.in_flight[slot].take() {
                // The DMA that read this buffer must finish before the CPU writes it again.
                self.device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
                self.remap_for_write(buffer)?;
            }

            {
                let mut view = buffer.slice(..n as u64).get_mapped_range_mut();
                let from_data = data.len().saturating_sub(done).min(n);
                view[..from_data].copy_from_slice(&data[done..done + from_data]);
                view[from_data..].fill(0); // only the final chunk can reach the zero tail
            }
            buffer.unmap();

            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, dst, dst_offset + done as u64, n as u64);
            ring.in_flight[slot] = Some(self.queue.submit(Some(encoder.finish())));

            done += n;
            slot = (slot + 1) % RING;
        }
        Ok(())
    }

    fn remap_for_write(&self, buffer: &wgpu::Buffer) -> Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Write, move |r| {
            let _ = tx.send(r);
        });
        // The GPU is already done with the buffer, so one non-blocking poll normally resolves
        // the request; fall back to blocking only if it did not.
        self.device.poll(wgpu::Maintain::Poll);
        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(_) => {
                self.device.poll(wgpu::Maintain::wait());
                rx.recv()?
            }
        };
        result.map_err(|e| anyhow!("failed to map a staging buffer for writing: {e:?}"))
    }
}
