//! Opt-in profiling: GPU pass durations from timestamp queries plus host-side
//! step timings. Normal calls pass `None` and pay nothing.

use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::parser::GpuParser;
use crate::util::{map_all, staging_buffer, take_mapped};

/// Two timestamps per compute pass.
const MAX_QUERIES: u32 = 32;

/// Collects timings for one profiled run.
pub(crate) struct StageTimer {
    query_set: Option<wgpu::QuerySet>,
    next: Cell<u32>,
    gpu_labels: RefCell<Vec<&'static str>>,
    marks: RefCell<Vec<(&'static str, Instant)>>,
}

/// `timer.mark(label)` if profiling; a no-op otherwise.
pub(crate) fn mark(timer: Option<&StageTimer>, label: &'static str) {
    if let Some(t) = timer {
        t.marks.borrow_mut().push((label, Instant::now()));
    }
}

impl StageTimer {
    pub(crate) fn new(parser: &GpuParser) -> Self {
        let query_set = parser.timestamps.then(|| {
            parser.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Stage Timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_QUERIES,
            })
        });
        let timer = Self {
            query_set,
            next: Cell::new(0),
            gpu_labels: RefCell::new(Vec::new()),
            marks: RefCell::new(Vec::new()),
        };
        mark(Some(&timer), "start");
        timer
    }

    /// Timestamp slots for the next compute pass (`None` without timestamp support).
    pub(crate) fn writes(
        &self,
        label: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let query_set = self.query_set.as_ref()?;
        let i = self.next.get();
        if i + 2 > MAX_QUERIES {
            return None;
        }
        self.next.set(i + 2);
        self.gpu_labels.borrow_mut().push(label);
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }

    pub(crate) fn finish(self, parser: &GpuParser) -> Result<Profile> {
        let marks = self.marks.borrow();
        let wall = marks.last().unwrap().1 - marks.first().unwrap().1;
        let host_steps = marks
            .windows(2)
            .map(|w| (w[1].0, w[1].1 - w[0].1))
            .collect();

        let mut gpu_passes = Vec::new();
        let n = self.next.get();
        if let (Some(query_set), true) = (&self.query_set, n > 0) {
            let bytes = u64::from(n) * 8;
            let resolve = parser.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Resolve Buffer"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = staging_buffer(&parser.device, "Timestamp Staging Buffer", bytes);
            let mut encoder = parser.device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(query_set, 0..n, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
            parser.queue.submit(Some(encoder.finish()));
            map_all(&parser.device, &[&staging])?;
            let ticks = take_mapped::<u64>(&staging, n as usize);
            let period = f64::from(parser.queue.get_timestamp_period());
            for (i, label) in self.gpu_labels.borrow().iter().enumerate() {
                let dt = ticks[2 * i + 1].saturating_sub(ticks[2 * i]);
                gpu_passes.push((*label, Duration::from_nanos((dt as f64 * period) as u64)));
            }
        }
        Ok(Profile {
            wall,
            host_steps,
            gpu_passes,
        })
    }
}

/// Where the time of one pipeline run went.
#[derive(Debug, Clone)]
pub struct Profile {
    /// Host wall-clock from start to the last host step.
    pub wall: Duration,
    /// Time since the previous step, in execution order. GPU work overlaps these
    /// steps; the waiting shows up in the step that blocks on the GPU.
    pub host_steps: Vec<(&'static str, Duration)>,
    /// Real GPU execution time per compute pass (empty if the adapter lacks
    /// `TIMESTAMP_QUERY`).
    pub gpu_passes: Vec<(&'static str, Duration)>,
}

impl Profile {
    pub fn gpu_total(&self) -> Duration {
        self.gpu_passes.iter().map(|&(_, d)| d).sum()
    }
}
